import os
import json
import copy
import warnings
import base64
import gzip
import traceback
from typing import List, Union
from datetime import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd
from azure.storage.queue import QueueClient
from qdrant_client import QdrantClient, models
from pymongo import MongoClient

from ptinfra import intialize, get_logger
from ptinfra.pt_queue import  MessageQueue, MemoryQueue, RoundRobinReader
from ptinfra.config import get_variable
from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, Message
from ptinfra import  AbortRequested

from src.core.photos import update_photos_ranks
from src.smart_cropping import process_crop_images
from src.pipeline import (AlbumContext, album_group, build_select,
                          compose_albums)
from src.core.key_pages import generate_first_last_pages
from src.album_processing import album_processing
from src.core.models import SpreadSearchParams
from src.predefined.models import PredefinedLayoutInput
from src.predefined.processing import predefined_layout_processing, build_first_last_pages
from src.album_response import build_reply, encoded_size
from src.pipeline.album_requests import UNFULFILLED_KEY, record_failure
from src.request_processing import read_messages, assembly_output
from utils.time_processing import generate_time_clusters
from utils.configs import CONFIGS


if os.environ.get('PTEnvironment') == 'dev' or os.environ.get('PTEnvironment') is None:
    os.environ['ConfigServiceURL'] = 'https://devqa.pic-time.com/config/'

warnings.filterwarnings('ignore')
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

read_time_list = list()
processing_time_list = list()
reporting_time_list = list()


def push_report_error(one_msg, az_connection_string, logger=None):
    '''Push result to the report queue'''

    if type(one_msg.error) is str:
        error_report = {
            'requestId': one_msg.content['conditionId'],
            'error': one_msg.error,
            'composition': None
        }
    else:
        error_report = {
            'requestId': one_msg.content['conditionId'],
            'error': str(one_msg.error),
            'composition': None
        }
    try:
        q_client = QueueClient.from_connection_string(az_connection_string, one_msg.content['replyQueueName'])
        q_client.create_queue()
    except Exception as ex:
        pass
    # q_name = one_msg.content['replyQueueName']
    result_doc = error_report
    jsonContent = json.dumps(result_doc)
    compressed = gzip.compress(jsonContent.encode("ascii"))
    base64Content = base64.b64encode(compressed).decode("ascii")
    try:
        q_client = QueueClient.from_connection_string(az_connection_string, one_msg.content['replyQueueName'])
        q_client.send_message(base64Content)
        if logger is not None:
            logger.info('Message was sent to the report queue {}'.format(result_doc))
    except Exception as ex:
        tb = traceback.extract_tb(ex.__traceback__)
        filename, lineno, func, text = tb[-1]
        raise Exception(f'Report queue error, message not sent, error: {ex}. Exception in function: {func}, line {lineno}, file {filename}.')


def push_report_msg(one_msg, az_connection_string, logger=None, result_doc=None):
    '''Push result to the report queue.

    ``result_doc`` overrides the message's own `album_doc`, which is how a
    multi-album reply -- one payload covering every album of the request -- is
    sent without giving any single album's message a doc it did not produce.
    '''

    try:
        q_client = QueueClient.from_connection_string(az_connection_string, one_msg.content['replyQueueName'])
        q_client.create_queue()
    except Exception as ex:
        pass
    # q_name = one_msg.content['replyQueueName']
    result_doc = one_msg.album_doc if result_doc is None else result_doc
    jsonContent = json.dumps(result_doc)
    compressed = gzip.compress(jsonContent.encode("ascii"))
    base64Content = base64.b64encode(compressed).decode("ascii")
    try:
        q_client = QueueClient.from_connection_string(az_connection_string, one_msg.content['replyQueueName'])
        q_client.send_message(base64Content)
        if logger is not None:
            logger.info('Message was sent to the report queue {}'.format(result_doc))
    except Exception as ex:
        tb = traceback.extract_tb(ex.__traceback__)
        filename, lineno, func, text = tb[-1]
        raise Exception(f'Report queue error, message not sent, error: {ex}. Exception in function: {func}, line {lineno}, file {filename}.')


class ReadStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: MemoryQueue = None, err_q: MemoryQueue = None, logger = None):
        super().__init__('ReadStage', self.read_messages, in_q, out_q, err_q, batch_size=1, max_threads=1)
        self.logger = logger
        try:
            connection_string = get_variable(CONFIGS["DB_CONNECTION_STRING_VAR"])
            client = MongoClient(connection_string)
            db = client[CONFIGS["DB_NAME"]]
            self.project_status_collection = db[CONFIGS["STATUS_COLLECTION_NAME"]]
        except Exception as ex:
            self.logger.error(f"Failed to connect to database: {ex}")
        try:
            self.qdrant_client = QdrantClient(host=CONFIGS["QDRANT_HOST"],
                                              port=6333,
                                              # The HTTP port is often used for general access if not explicitly setting grpc_port
                                              grpc_port=6334,  # Explicitly define the gRPC port
                                              prefer_grpc=True,
                                              # This forces the client to use gRPC for large operations like upsert
                                              timeout=30,
                                              )
            self.logger.info(f'Initialize qdrant client, host {CONFIGS["QDRANT_HOST"]}, port 6333, grpc_port 6334')
        except Exception as ex:
            self.logger.error(f"Failed to connect to Qdrant: {ex}")

    def read_messages(self, msgs: Union[Message, List[Message], AbortRequested]):
        if isinstance(msgs, AbortRequested):
            self.logger.info("Abort requested.")
            return []

        messages = msgs if isinstance(msgs, list) else [msgs]
        start = datetime.now()
        # Read messages using a helper function
        try:
            messages, reading_error = read_messages(messages,self.project_status_collection,self.qdrant_client, self.logger)
            if reading_error is not None:
                self.logger.error(f"Error reading messages: {reading_error}")
                raise Exception(f"Error reading messages: {reading_error}")
        except Exception as ex:
            tb = traceback.extract_tb(ex.__traceback__)
            filename, lineno, func, text = tb[-1]
            self.logger.error(f"Error reading messages: {ex}. Exception in function: {func}, line {lineno}, file {filename}.")
            raise Exception(f"Error reading messages: {ex}. Exception in function: {func}, line {lineno}, file {filename}.")

        handling_time = (datetime.now() - start) / max(len(messages), 1)
        read_time_list.append(handling_time)
        self.logger.info(f"READING Stage for {len(messages)} messages. Average time: {handling_time}")
        return messages


class SelectionStage(Stage):
    """Choose which photos make the album.

    A thin driver over the selection substages; the logic that used to be
    inlined here now lives in `src/pipeline/select` (route -> budget -> pick ->
    publish), with each content category's rule a separate strategy. See
    `src.pipeline.registry.SELECT`.
    """

    def __init__(self, in_q: MemoryQueue = None, out_q: MemoryQueue = None, err_q: MemoryQueue = None, logger = None):
        super().__init__('SelectionStage', self.get_selection, in_q, out_q, err_q, batch_size=1, max_threads=1)
        self.logger = logger
        self.pipeline = build_select(logger=logger)

    def get_selection(self, msgs: Union[Message, List[Message], AbortRequested]):
        if isinstance(msgs, AbortRequested):
            self.logger.info("Abort requested")
            return []

        updated_messages = []
        messages = msgs if isinstance(msgs, list) else [msgs]
        start = datetime.now()

        try:
            for _msg in messages:
                # Third route, beside manual and AI: an external service fixed the
                # spreads, so selection is not run at all. Checked here rather
                # than inside `select.route` because the whole SELECT composition
                # is skipped, not just steered -- there is no budget, no
                # preselect and no pick to reach.
                predefined = PredefinedLayoutInput.from_request(_msg.content)
                if predefined is not None:
                    df = _msg.content.get('gallery_photos_info', pd.DataFrame())
                    if df.empty:
                        raise Exception(f"Gallery photos info DataFrame is empty for message {_msg}")
                    _msg.content['predefined_layout'] = predefined
                    _msg.content['gallery_all_photos_info'] = df.copy()
                    _msg.content['gallery_photos_info'] = df[df['image_id'].isin(predefined.all_photo_ids())]
                    self.logger.info(f"Predefined layout: {len(predefined.spreads)} spreads, skipping selection.")
                    updated_messages.append(_msg)
                    continue

                # The manual/AI split, the empty-frame check and the narrowing to
                # the request's own `photos` all live in `select.route` now, so
                # the branch's inline copies of them are dropped rather than
                # merged: `route._manual` aligns the frame with the user's list
                # and widens the lookup table, and `route._ai` does the pool
                # narrowing that used to sit here.
                # One album per planned variant, each composed from its own
                # copy of the read's output. At N=1 -- which is every request
                # until `enrich.variants` lands -- this is exactly the single
                # `pipeline.run(for_message(...))` it replaces; the point is
                # that N>1 cannot contaminate, because albums never share the
                # frame selection narrows.
                # N comes from `enrich.variants` -- a fact about the gallery,
                # carried on the context the read attached. `count` is only the
                # fallback for a message that never went through ENRICH.
                settings = CONFIGS.get('albums', {})
                runs = compose_albums(_msg, self.pipeline,
                                      count=int(settings.get('count', 1)),
                                      logger=self.logger,
                                      seed=settings.get('seed'))

                # One message per album -- siblings after the first -- so
                # ProcessStage lays each one out without them colliding. At
                # count=1 this is the original message and nothing changes.
                for run in runs:
                    if run.context.failed:
                        self.logger.error(
                            f"Error for Selection images for this message {_msg}"
                            + (f" (album {run.index})" if len(runs) > 1 else ""))
                        run.message.content['error'] = (
                            f"Error for Selection images for this message {_msg}")
                        updated_messages.append(run.message)
                        continue
                    updated_messages.append(run.context.sync_to_message())

        except Exception as ex:
            tb = traceback.extract_tb(ex.__traceback__)
            filename, lineno, func, text = tb[-1]
            self.logger.error(f"Error selection stage: {ex}. Exception in function: {func}, line {lineno}, file {filename}.")
            raise Exception(f"Error selection stage: {ex}. Exception in function: {func}, line {lineno}, file {filename}.")

        handling_time = (datetime.now() - start) / max(len(messages), 1)
        read_time_list.append(handling_time)
        self.logger.info(f"Selection Stage for {len(messages)} messages. Average time: {handling_time}")

        return updated_messages


class ProcessStage(Stage):
    def __init__(self, in_q: MemoryQueue = None, out_q: MemoryQueue = None, err_q: MemoryQueue = None, logger = None):
        super().__init__('ProcessingStage', self.process_message, in_q, out_q, err_q, batch_size=1, max_threads=1,
        batch_wait_time=5)
        self.logger = logger
        self.q = mp.Queue()

    #: What `process_crop_images` needs, and all it needs. Every one of them is
    #: a property of the photo, which is why one crop pass can serve N albums.
    CROP_COLUMNS = ['image_id', 'faces_info', 'background_centroid', 'diameter',
                    'image_as']

    def _crop_once(self, messages):
        """Crop every photo any album needs, in one subprocess.

        A crop depends only on the photo -- its faces, saliency blob and aspect
        -- never on which album chose it, so N albums cropping their own
        selections is N passes over largely the same photos. The union is
        deduplicated on `image_id` and cropped once.

        Returns None for a single album, which keeps that path on exactly the
        code it has always used.
        """
        if len(messages) < 2:
            return None

        frames = []
        for message in messages:
            for key in ('gallery_photos_info', 'bride and groom'):
                part = message.content.get(key)
                if part is None or getattr(part, 'empty', True):
                    continue
                if not set(self.CROP_COLUMNS).issubset(part.columns):
                    self.logger.warning(
                        f"Cannot share crops: {key} lacks "
                        f"{sorted(set(self.CROP_COLUMNS) - set(part.columns))}")
                    return None
                frames.append(part[self.CROP_COLUMNS])
        if not frames:
            return None

        union = pd.concat(frames).drop_duplicates(subset='image_id')
        wanted = sum(len(f) for f in frames)
        self.logger.info(
            f"Cropping once for {len(messages)} albums: {len(union)} distinct "
            f"photos of {wanted} selected ({wanted - len(union)} shared)")

        worker = mp.Process(target=process_crop_images, args=(self.q, union))
        worker.start()
        try:
            cropped = self.q.get(timeout=200)
        except Exception as ex:
            worker.terminate()
            raise Exception('shared cropping process not completed: {}'.format(ex))
        worker.join(timeout=5)
        if worker.is_alive():
            worker.terminate()
            self.logger.error('shared cropping process not completed')
            raise Exception('shared cropping process not completed.')
        return cropped

    def _album_failed(self, messages, message, detail):
        """Lose one album of a request, not the request.

        One queue message now carries every album of a gallery, so an
        exception composing one of them used to take its siblings with it --
        the failure shape that cost 53009168 two albums for one bad group.

        The brief still gets an answer: a decline recorded on the *group's
        first* message, which is the one ReportStage builds the reply from.
        Siblings carry a shallow copy of the body (`sibling_message`), so a
        note left on the album that failed would never be read.
        """
        message.content['error'] = detail
        record_failure(messages[0].content,
                       getattr(message, "album_request_id", None), detail)
        self.logger.error(
            f"album {getattr(message, 'album_index', 0)} of "
            f"{len(messages)} failed; the others continue: {detail}")

    def process_message(self, msgs: Union[Message, List[Message]]):
        # check if its single message or list
        messages = msgs if isinstance(msgs, list) else [msgs]
        whole_messages_start = datetime.now()

        params = SpreadSearchParams()

        # One crop pass for every album of this request, or None for one album.
        shared_cropped = self._crop_once(messages)

        for i,message in enumerate(messages):
            self.logger.debug("Params for this Gallery are: {}".format(params))
            df = message.content.get('gallery_photos_info', pd.DataFrame())
            ai_metadata = message.content.get('aiMetadata', {})
            if ai_metadata is not None:
                chosen_photos = ai_metadata.get('photoIds', [])
            else:
                chosen_photos = []
            df = update_photos_ranks(df, chosen_photos)
            if df.empty:
                detail = f"Gallery photos info DataFrame is empty for message {message}"
                self.logger.error(detail)
                if len(messages) == 1:
                    message.content['error'] = detail
                    raise Exception(detail)
                self._album_failed(messages, message, detail)
                continue

            bride_and_groom_df = message.content.get('bride and groom', pd.DataFrame())
            df_serializable = pd.concat([df.copy(), bride_and_groom_df])  # Make a copy to avoid modifying original
            df_serializable = df_serializable[['image_id', 'faces_info', 'background_centroid', 'diameter', 'image_as']]

            p = None
            if shared_cropped is None:
                p = mp.Process(target=process_crop_images, args=(self.q, df_serializable))
                p.start()

            try:
                stage_start = datetime.now()
                # Sorting the DataFrame by "image_order" column
                sorted_df = df.sort_values(by="image_order", ascending=False)

                # generate time clusters for the gallery photos
                sorted_df = generate_time_clusters(message, sorted_df, self.logger)

                predefined = message.content.get('predefined_layout', None)
                if predefined is not None:
                    # Covers come from the input, not from choose_good_wedding_images —
                    # that removes cover photos from df, which would steal photos from
                    # the fixed spreads. Body spreads are explicit, so no removal here.
                    first_last_pages_data_dict = build_first_last_pages(predefined, sorted_df, message, self.logger)
                    df = sorted_df
                else:
                    df, first_last_pages_data_dict = generate_first_last_pages(message, sorted_df, self.logger)

                # Handle the processing time logging
                start = datetime.now()

                if message.content.get('aiMetadata', None) is not None:
                    density = message.content['aiMetadata'].get('density', 3)
                else:
                    density = 3

                modified_lut = message.content['modified_lut'] if message.content.get('modified_lut', None) is not None else None

                manual_selection = message.content.get('manual_selection', False)

                all_gallery_df = message.content.get('gallery_all_photos_info', None)
                selection_min_total_spreads = message.content.get('min_total_spreads', None)
                selection_max_total_spreads = message.content.get('max_total_spreads', None)
                if predefined is not None:
                    # Stages 1+2 (partitions/combinations) are given by the input;
                    # only stage 3 (layout + page split + box assignment) runs.
                    album_result, df = predefined_layout_processing(df, message.designsInfo, predefined, params,
                                                                   message.content['is_wedding'], self.logger)
                else:
                    album_result, df = album_processing(df, message.designsInfo, message.content['is_wedding'], modified_lut, params,
                                                logger=self.logger,density=density, manual_selection=manual_selection,
                                                all_gallery_df=all_gallery_df,
                                                selection_min_total_spreads=selection_min_total_spreads,
                                                selection_max_total_spreads=selection_max_total_spreads,
                                                is_artificial_time=message.content.get('is_artificial_time', False))

                wait_start = datetime.now()
                if p is None:
                    # Already cropped for every album of this request.
                    cropped_df = shared_cropped
                else:
                    try:
                        cropped_df = self.q.get(timeout=200)
                    except Exception as e:
                        p.terminate()
                        raise Exception('cropping process not completed: {}'.format(e))
                    p.join(timeout=5)
                    if p.is_alive():
                        p.terminate()
                        self.logger.error('cropping process not completed 2')
                        raise Exception('cropping process not completed.')

                df = df.merge(cropped_df, how='left', on='image_id')

                # Fill missing crop values (photos added by singleton resolution
                # that weren't in the original cropping batch) with centered crop
                crop_missing = df['cropped_x'].isna()
                if crop_missing.any():
                    ar = df.loc[crop_missing, 'image_as'].astype(float)
                    landscape = ar > 1
                    portrait = ar <= 1
                    df.loc[crop_missing & landscape, 'cropped_x'] = ((1 - 1 / ar[landscape]) / 2).values
                    df.loc[crop_missing & landscape, 'cropped_y'] = 0.0
                    df.loc[crop_missing & landscape, 'cropped_w'] = (1 / ar[landscape]).values
                    df.loc[crop_missing & landscape, 'cropped_h'] = 1.0
                    df.loc[crop_missing & portrait, 'cropped_x'] = 0.0
                    df.loc[crop_missing & portrait, 'cropped_y'] = ((1 - ar[portrait]) / 2).values
                    df.loc[crop_missing & portrait, 'cropped_w'] = 1.0
                    df.loc[crop_missing & portrait, 'cropped_h'] = ar[portrait].values

                # for key, value in first_last_pages_data_dict.items():
                #     if first_last_pages_data_dict[key]['last_images_df'] is not None or first_last_pages_data_dict[key][
                #         'first_images_df'] is not None:
                #         if len(first_last_pages_data_dict[key]['last_images_df']) != 0 or len(
                #                 first_last_pages_data_dict[key]['first_images_df']) != 0:
                #             first_last_pages_data_dict[key]['last_images_df'] = value['last_images_df'].merge(
                #                 cropped_df, how='inner', on='image_id')
                #             first_last_pages_data_dict[key]['first_images_df'] = value['first_images_df'].merge(
                #                 cropped_df, how='inner', on='image_id')

                _IMAGE_DF_FIELDS = ("first_images_df", "last_images_df")
                for page_key, page_data in first_last_pages_data_dict.items():
                    for field in _IMAGE_DF_FIELDS:
                        if field in page_data:
                            if not page_data[field].empty:
                                page_data[field] = page_data[field].merge(cropped_df, how="inner", on="image_id")

                self.logger.debug('waited for cropping process: {}'.format(datetime.now() - wait_start))

                final_response = assembly_output(album_result, message, df, first_last_pages_data_dict, message.content.get('album_ar',
                                                                                                                   {'anyPage':2})['anyPage'],self.logger)

                message.content['gallery_photos_info'] = df
                message.album_doc = copy.deepcopy(final_response)
                processing_time = datetime.now() - start

                self.logger.debug('Lay-outing time: {}.For Processed album id: {}'.format(processing_time,
                                                                                          message.content.get(
                                                                                              'projectURL', True)))
                self.logger.debug(
                    'Processing Stage time: {}.For Processed album id: {}'.format(datetime.now() - stage_start,
                                                                                  message.content.get('projectURL',
                                                                                                          True)))

            except Exception as ex:
                tb = traceback.extract_tb(ex.__traceback__)
                filename, lineno, func, text = tb[-1]
                detail = (f"Error processing stage: {ex}. Exception in function: "
                          f"{func}, line {lineno}, file {filename}.")
                self.logger.error(detail)
                # One album is one album. A request carrying several loses only
                # the one that failed, and says why; a request carrying one has
                # nothing left to deliver, so it fails as it always did.
                if len(messages) == 1:
                    raise Exception(detail)
                self._album_failed(messages, message, detail)
                continue

        # Every album failed. There is nothing to reply with, so this is a
        # failed request rather than a successful one that happens to be empty
        # -- the caller can tell those apart and would retry only the first.
        if len(messages) > 1 and not any(
                getattr(m, 'album_doc', None) for m in messages):
            raise Exception(
                messages[0].content.get('error')
                or "every album of this request failed")

        processing_time = (datetime.now() - whole_messages_start) / max(len(messages), 1)
        processing_time_list.append(processing_time)
        self.logger.debug('Average Processing Stage time: {}. For : {} messages '.format(processing_time, len(messages)))
        return msgs


class ReportStage(Stage):
    def __init__(self, in_q: MemoryQueue = None, logger = None):
        super().__init__('ReportMessage', self.report_message, in_q, None, None, batch_size=1, max_threads=1)
        self.az_connection_string = get_variable("QueueConnectionString")
        self.global_start_time = datetime.now()
        self.global_number_of_msgs = 0
        self.number_of_reports = 0
        self.logger = logger

    def print_time_summary(self, period=3):
        self.number_of_reports += 1
        if self.number_of_reports % period != 0:
            return

        def avg_time(datetimes):
            if len(datetimes) == 0:
                return 1000000
            total = sum(dt.seconds * 1000000 + dt.microseconds for dt in datetimes)
            avg = total / len(datetimes)
            avg = float(avg) / 1000000
            return avg

        global_average = (datetime.now() - self.global_start_time) / self.global_number_of_msgs \
            if self.global_number_of_msgs > 0 else 1000000

        self.logger.debug('**********. '
                          'Average processing time for last {} requests: '
                          'Handling messages and data loading average time: {}. '
                          'Processing average time: {}. '
                          'Reporting average time: {}. '
                          'General average time: {}. '
                          '**********.'.format(self.global_number_of_msgs, avg_time(read_time_list),
                                               avg_time(processing_time_list), avg_time(reporting_time_list),
                                               global_average))

    def report_one_message(self, one_msg, result_doc=None):
        if one_msg.error:
            push_report_error(one_msg,self.az_connection_string,self.logger)
            self.logger.debug('REPORT ERROR MESSAGE  {}.'.format(one_msg.error))
        else:
            push_report_msg(one_msg, self.az_connection_string, self.logger,
                            result_doc=result_doc)
            self.logger.debug('Message was reported to the queue: {}/{}. '.format(one_msg.content['projectId'], one_msg.content['conditionId']))

    def report_message(self, msgs: Union[Message, List[Message]]):
        start = datetime.now()
        if isinstance(msgs, Message):
            self.report_one_message(msgs)
            try:
                self.logger.debug('deleting message id  {}.'.format(msgs.source.id))
                msgs.delete()
            except Exception as e:
                self.logger.error('Error while deleting message: {}. Exception: {}'.format(msgs, e))
        elif isinstance(msgs, list):
            # Grouped by the queue message they came from, because N albums of
            # one gallery arrive as N sibling messages sharing one `source`.
            # Reporting each would send N results for one request and, worse,
            # delete the same underlying queue message N times. So: one report
            # and one delete per group.
            for group in album_group(msgs):
                first = group[0]
                reply = None
                # Briefs the gallery declined. Owed to the caller even when a
                # single album came back, so this is not gated on the count.
                unfulfilled = first.content.get('albumsUnfulfilled') or []
                if len(group) > 1 or unfulfilled:
                    # One payload for the whole request: `composition` is the
                    # first album, `albums` carries them all, and the size
                    # guard drops any that will not fit rather than letting
                    # the send fail opaquely at the queue.
                    reply = build_reply(
                        [getattr(m, 'album_doc', None) for m in group],
                        [getattr(m, 'variant_name', None) for m in group],
                        [getattr(m, 'album_request_id', None) for m in group],
                        [getattr(m, 'derived_from', None) for m in group],
                        unfulfilled,
                        logger=self.logger)
                    self.logger.info(
                        f"Reporting {len(group)} albums for request "
                        f"{first.content.get('conditionId')}"
                        + (f", {encoded_size(reply)} bytes encoded"
                           if reply else ""))
                self.report_one_message(first, result_doc=reply)
                try:
                    self.logger.debug('deleting message id  {}.'.format(first.source.id))
                    first.delete()
                except Exception as e:
                    self.logger.error('Error while deleting message: {}. Exception: {}'.format(first, e))

        reporting_time = (datetime.now() - start) / (len(msgs) if isinstance(msgs, list) and len(msgs) > 0 else 1)
        reporting_time_list.append(reporting_time)

        # photo_ids = msgs.content['photoId'] if isinstance(msgs, Message) else [msg.content["photoId"] for msg in msgs]
        # self.logger.debug('Deleted images: {}. Reporting time: {}.'.format(photo_ids, reporting_time))

        self.global_number_of_msgs += len(msgs) if isinstance(msgs, list) else 1
        self.print_time_summary()


def _get_azure_input_queue(logger):
    try:
        prefix = get_variable('PTEnvironment')
    except:
        prefix = 'dev'

    input_queue = CONFIGS['collection_name']
    print(prefix + input_queue)

    if prefix == 'dev':
        dev_queue = MessageQueue(prefix + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                 max_dequeue_allowed=1000)
        dev3_queue = MessageQueue('dev3' + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                  max_dequeue_allowed=1000)
        azure_input_q = RoundRobinReader([dev_queue, dev3_queue])
    elif prefix == 'production':
        logger.info('PRODUCTION environment set, queue name: ' + input_queue)
        ep_queue = MessageQueue('ep' + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                max_dequeue_allowed=1000)
        prod_queue = MessageQueue(input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                     max_dequeue_allowed=1000)
        azure_input_q = RoundRobinReader([prod_queue, ep_queue])
    else:
        logger.info(prefix + ' environment, queue name: ' + prefix + input_queue)
        azure_input_q = MessageQueue(prefix + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                     max_dequeue_allowed=1000)

    return azure_input_q


def main():
    logger = get_logger(__name__, 'DEBUG')

    # Initialize
    settings_filename = os.environ.get('HostingSettingsPath',
                                       '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt')
    intialize('AlbumDesigner', settings_filename)

    private_key = get_variable('PtKey')
    logger.debug('Private key: {}'.format(private_key))

    # Define message queues
    azure_input_q = _get_azure_input_queue(logger)
    read_q = MemoryQueue(1)
    selection_q = MemoryQueue(1)
    report_q = MemoryQueue(1)

    # Define stages
    read_stage = ReadStage(azure_input_q, read_q, report_q, logger=logger)
    selection_stage = SelectionStage(read_q, selection_q, report_q, logger=logger)
    process_stage = ProcessStage(selection_q, report_q, report_q, logger=logger)
    report_stage = ReportStage(report_q, logger=logger)

    # Run
    report_stage.start()
    selection_stage.start()
    process_stage.start()
    read_stage.start()


if __name__ == '__main__':
    main()
