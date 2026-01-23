import os
import json
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
from src.selection.auto_selection import ai_selection
from src.core.key_pages import generate_first_last_pages
from src.album_processing import album_processing
from src.request_processing import read_messages, assembly_output
from utils.time_processing import generate_time_clusters
from utils.configs import CONFIGS
from utils.lookup_table_tools import wedding_lookup_table


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


def push_report_msg(one_msg, az_connection_string, logger=None):
    '''Push result to the report queue'''

    try:
        q_client = QueueClient.from_connection_string(az_connection_string, one_msg.content['replyQueueName'])
        q_client.create_queue()
    except Exception as ex:
        pass
    # q_name = one_msg.content['replyQueueName']
    result_doc = one_msg.album_doc
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
                                              prefer_grpc=True
                                              # This forces the client to use gRPC for large operations like upsert
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
    def __init__(self, in_q: MemoryQueue = None, out_q: MemoryQueue = None, err_q: MemoryQueue = None, logger = None):
        super().__init__('SelectionStage', self.get_selection, in_q, out_q, err_q, batch_size=1, max_threads=1)
        self.logger = logger

    def get_selection(self, msgs: Union[Message, List[Message], AbortRequested]):
        if isinstance(msgs, AbortRequested):
            self.logger.info("Abort requested")
            return []

        updated_messages = []
        messages = msgs if isinstance(msgs, list) else [msgs]
        start = datetime.now()
        #Iterate over message and start the selection process
        try:
            for _msg in messages:
                ai_metadata = _msg.content.get('aiMetadata', None)
                if ai_metadata is None or ai_metadata['photoIds'] is None:
                    self.logger.info(f"aiMetadata not found for message {_msg}. Continue with chosen photos.")
                    photos = _msg.content.get('photos', [])
                    df = pd.DataFrame(photos, columns=['image_id'])
                    _msg.content['gallery_photos_info'] = df.merge(_msg.content['gallery_photos_info'], how='inner', on='image_id')
                    # handle LUT for manual selection
                    is_wedding = _msg.content.get('is_wedding', False)
                    if is_wedding:
                        modified_lut = wedding_lookup_table.copy()  # Create a copy to avoid modifying the original LUT
                        modified_lut['Other'] = (24, 4)  # Set 'Other' event to have max spreads
                        modified_lut['None'] = (24, 4)
                        _msg.content['modified_lut'] = modified_lut
                    _msg.content['manual_selection'] = True
                    updated_messages.append(_msg)
                    continue
                available_photos = _msg.content.get('photos', [])
                df = _msg.content.get('gallery_photos_info', pd.DataFrame())
                if df.empty:
                    raise Exception(f"Gallery photos info DataFrame is empty for message {_msg}")
                if len(available_photos) != 0:
                    df = df[df['image_id'].isin(available_photos)]
                    _msg.content['gallery_photos_info'] = df

                _msg.content['gallery_all_photos_info'] = df.copy()

                ten_photos = ai_metadata.get('photoIds', [])
                people_ids = ai_metadata.get('personIds', [])
                focus = ai_metadata.get('focus', ['everyoneElse'])
                tags = ai_metadata.get('subjects', ['Wedding dress', 'ceremony', 'bride', 'dancing', 'bride getting ready', 'groom getting ready', 'table setting', 'flowers', 'decorations', 'family', 'baby', 'kids', 'mother', 'father', 'Romance', 'affection', 'Intimacy', 'Happiness', 'Holding hands', 'smiling', 'Hugging', 'Kissing', 'ring', 'veil', 'soft light', 'portrait'])
                density = ai_metadata.get('density', 3)
                is_wedding = _msg.content.get('is_wedding', False)
                df = _msg.content.get('gallery_photos_info', pd.DataFrame())

                if df.empty:
                    self.logger.error(f"Gallery photos info DataFrame is empty for message {_msg}")
                    _msg.content['error'] = f"Gallery photos info DataFrame is empty for message {_msg}"
                    updated_messages.append(_msg)
                    continue

                if is_wedding:
                    modified_lut = wedding_lookup_table.copy()  # Create a copy to avoid modifying the original LUT

                    density_factor = CONFIGS['density_factors'][density] if density in CONFIGS['density_factors'] else 1
                    for event, pair in modified_lut.items():
                        modified_lut[event] = (min(24, max(1, pair[0] * density_factor)), pair[1])
                else:
                    modified_lut = None
                _msg.content['modified_lut'] = modified_lut

                is_artificial_time = _msg.content['is_artificial_time']
                ai_photos_selected,spreads_dict, errors = ai_selection(df, ten_photos, people_ids, focus, tags, is_wedding, density,is_artificial_time,
                                                  self.logger)

                if errors:
                    self.logger.error(f"Error for Selection images for this message {_msg}")
                    _msg.content['error'] = f"Error for Selection images for this message {_msg}"
                    updated_messages.append(_msg)
                    continue

                filtered_df = df[df['image_id'].isin(ai_photos_selected)]
                _msg.content['gallery_photos_info'] = filtered_df
                _msg.content['photos'] = ai_photos_selected
                _msg.content['spreads_dict'] = spreads_dict

                if _msg.pagesInfo.get("firstPage"):
                    if _msg.content.get('is_wedding', True):
                        all_bride_groom = df[
                            (df["cluster_context"] == "bride and groom")]
                        _msg.content['bride and groom'] = all_bride_groom
                else:
                    _msg.content['bride and groom'] = None

                updated_messages.append(_msg)

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

    def process_message(self, msgs: Union[Message, List[Message]]):
        # check if its single message or list
        messages = msgs if isinstance(msgs, list) else [msgs]
        whole_messages_start = datetime.now()

        params = [0.01, 100, 1000, 100, 300, 12]

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
                self.logger.error(f"Gallery photos info DataFrame is empty for message {message}")
                message.content['error'] = f"Gallery photos info DataFrame is empty for message {message}"
                raise Exception(f"Gallery photos info DataFrame is empty for message {message}")

            bride_and_groom_df = message.content.get('bride and groom', pd.DataFrame())
            df_serializable = pd.concat([df.copy(), bride_and_groom_df])  # Make a copy to avoid modifying original
            df_serializable = df_serializable[['image_id', 'faces_info', 'background_centroid', 'diameter', 'image_as']]

            p = mp.Process(target=process_crop_images, args=(self.q, df_serializable))
            p.start()

            try:
                stage_start = datetime.now()
                # Sorting the DataFrame by "image_order" column
                sorted_df = df.sort_values(by="image_order", ascending=False)

                # generate time clusters for the gallery photos
                sorted_df = generate_time_clusters(message, sorted_df, self.logger)

                df, first_last_pages_data_dict = generate_first_last_pages(message, sorted_df, self.logger)

                # Handle the processing time logging
                start = datetime.now()

                if message.content.get('aiMetadata', None) is not None:
                    density = message.content['aiMetadata'].get('density', 3)
                else:
                    density = 3

                modified_lut = message.content['modified_lut'] if message.content.get('modified_lut', None) is not None else None

                manual_selection = message.content.get('manual_selection', False)

                album_result = album_processing(df, message.designsInfo, message.content['is_wedding'], modified_lut, params,
                                                logger=self.logger,density=density, manual_selection=manual_selection)

                wait_start = datetime.now()
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

                df = df.merge(cropped_df, how='inner', on='image_id')

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

                message.album_doc = final_response
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
                self.logger.error(f"Error processing stage: {ex}. Exception in function: {func}, line {lineno}, file {filename}.")
                raise Exception(f"Error processing stage: {ex}. Exception in function: {func}, line {lineno}, file {filename}.")

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

    def report_one_message(self, one_msg):
        if one_msg.error:
            push_report_error(one_msg,self.az_connection_string,self.logger)
            self.logger.debug('REPORT ERROR MESSAGE  {}.'.format(one_msg.error))
        else:
            push_report_msg(one_msg, self.az_connection_string, self.logger)
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
            for one_msg in msgs:
                self.report_one_message(one_msg)
                try:
                    self.logger.debug('deleting message id  {}.'.format(one_msg.source.id))
                    one_msg.delete()
                except Exception as e:
                    self.logger.error('Error while deleting message: {}. Exception: {}'.format(one_msg, e))

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
