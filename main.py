import os
import json
import warnings
import numpy as np
import base64
import gzip

import pandas as pd
from typing import List, Union
from datetime import datetime
import multiprocessing as mp
from azure.storage.queue import QueueClient

from ptinfra import intialize, get_logger
from ptinfra.pt_queue import  MessageQueue, MemoryQueue, RoundRobinReader
from ptinfra.config import get_variable
from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, QWriter, Message
from ptinfra import  AbortRequested


from src.smart_cropping import process_crop_images
from src.selection.auto_selection import ai_selection
from src.core.key_pages import generate_first_last_pages
from utils.time_processing import process_image_time, get_time_clusters, merge_time_clusters_by_context
from src.album_processing import album_processing
from src.request_processing import read_messages, assembly_output
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
        raise Exception('Report queue error, message not sent, error: {}'.format(ex))

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
        raise Exception('Report queue error, message not sent, error: {}'.format(ex))


class ReadStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        super().__init__('ReadStage', self.read_messages, in_q, out_q, err_q, batch_size=1, max_threads=1)
        self.logger = logger


    def read_messages(self, msgs: Union[Message, List[Message], AbortRequested]):
        if isinstance(msgs, AbortRequested):
            self.logger.info("Abort requested.")
            return []

        messages = msgs if isinstance(msgs, list) else [msgs]
        start = datetime.now()
        # Read messages using a helper function
        try:
            messages = read_messages(messages, self.logger)
        except Exception as e:
            self.logger.error(f"Error reading messages: {e}")
            raise Exception(f"Error reading messages: {e}")

        handling_time = (datetime.now() - start) / max(len(messages), 1)
        read_time_list.append(handling_time)
        self.logger.info(f"READING Stage for {len(messages)} messages. Average time: {handling_time}")
        return messages



class SelectionStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
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
                ai_metadata = _msg.content.get('aiMetadata', {})
                if not ai_metadata:
                    self.logger.info(f"aiMetadata not found for message {_msg}. Continue with chosen photos.")
                    photos = _msg.content.get('photos', [])
                    df = pd.DataFrame(photos, columns=['image_id'])
                    _msg.content['gallery_photos_info'] = df.merge(_msg.content['gallery_photos_info'], how='inner', on='image_id')
                    updated_messages.append(_msg)
                    continue
                photos = _msg.content.get('photos', [])
                if len(photos) != 0:
                    updated_messages.append(_msg)
                    continue

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

                ai_photos_selected,spreads_dict, errors = ai_selection(df, ten_photos, people_ids,focus,tags,is_wedding,density,
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
                updated_messages.append(_msg)

        except Exception as e:
            # self.logger.error(f"Error reading messages: {e}")
            raise(e)
            # return []

        handling_time = (datetime.now() - start) / max(len(messages), 1)
        read_time_list.append(handling_time)
        self.logger.info(f"Selection Stage for {len(messages)} messages. Average time: {handling_time}")

        return updated_messages


class ProcessStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
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
            df_serializable = df.copy()  # Make a copy to avoid modifying original
            df_serializable = df_serializable[['image_id', 'faces_info', 'background_centroid', 'diameter', 'image_as']]


            p = mp.Process(target=process_crop_images, args=(self.q, df_serializable))
            p.start()

            try:
                stage_start = datetime.now()
                # Extract gallery photo info safely
                df = message.content.get('gallery_photos_info', pd.DataFrame())
                if df.empty:
                    self.logger.error(f"Gallery photos info DataFrame is empty for message {message}")
                    message.content['error'] = f"Gallery photos info DataFrame is empty for message {message}"
                    continue

                # Sorting the DataFrame by "image_order" column
                sorted_df = df.sort_values(by="image_order", ascending=False)

                # Process time
                sorted_df, image_id2general_time = process_image_time(sorted_df)
                sorted_df['time_cluster'] = get_time_clusters(sorted_df['general_time'])
                sorted_df = merge_time_clusters_by_context(sorted_df, ['dancing'])

                df, first_last_pages_data_dict = generate_first_last_pages(message, sorted_df, self.logger)

                # Handle the processing time logging
                start = datetime.now()

                if message.content.get('aiMetadata', None) is not None:
                    density = message.content['aiMetadata'].get('density', 3)
                else:
                    density = 3

                album_result = album_processing(df, message.designsInfo, message.content['is_wedding'], params,
                                                logger=self.logger,density=density)

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
                for key, value in first_last_pages_data_dict.items():
                    first_last_pages_data_dict[key]['images_df'] = value['images_df'].merge(cropped_df, how='inner', on='image_id')

                self.logger.debug('waited for cropping process: {}'.format(datetime.now() - wait_start))


                final_response = assembly_output(album_result, message, df, first_last_pages_data_dict, self.logger)

                message.album_doc = final_response
                processing_time = datetime.now() - start

                self.logger.debug('Lay-outing time: {}.For Processed album id: {}'.format(processing_time,
                                                                                          message.content.get(
                                                                                              'projectURL', True)))
                self.logger.debug(
                    'Processing Stage time: {}.For Processed album id: {}'.format(datetime.now() - stage_start,
                                                                                  message.content.get('projectURL',
                                                                                                          True)))

            except Exception as e:
                self.logger.error(f"Unexpected error in message processing: {e}")
                raise Exception(f"Unexpected error in message processing: {e}")

        processing_time = (datetime.now() - whole_messages_start) / max(len(messages), 1)
        processing_time_list.append(processing_time)
        self.logger.debug('Average Processing Stage time: {}. For : {} messages '.format(processing_time, len(messages)))
        return msgs



class ReportStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        super().__init__('ReportMessage', self.report_message, in_q, out_q, err_q, batch_size=1, max_threads=1)
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



class MessageProcessor:
    def __init__(self):
        self.logger = get_logger(__name__, 'DEBUG')

    def run(self):
        settings_filename = os.environ.get('HostingSettingsPath',
                                           '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt')

        intialize('AlbumDesigner', settings_filename)

        private_key = get_variable('PtKey')
        self.logger.debug('Private key: {}'.format(private_key))

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
            # ep_queue = MessageQueue('ep' + input_queue, def_visibility=CONFIGS['visibility_timeout'],
            #                         max_dequeue_allowed=1000)
            azure_input_q = RoundRobinReader([dev_queue, dev3_queue])

            # azure_input_q = dev_queue

        elif prefix == 'production':
            self.logger.info('PRODUCTION environment set, queue name: ' + input_queue)
            azure_input_q = MessageQueue(input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                         max_dequeue_allowed=1000)
        else:
            self.logger.info(prefix + ' environment, queue name: ' + prefix + input_queue)
            azure_input_q = MessageQueue(prefix + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                         max_dequeue_allowed=1000)

        read_q = MemoryQueue(2)
        selection_q = MemoryQueue(2)
        report_q = MemoryQueue(2)

        read_stage = ReadStage(azure_input_q, read_q, report_q, logger=self.logger)
        selection_stage = SelectionStage(read_q, selection_q, report_q, logger=self.logger)
        process_stage = ProcessStage(selection_q, report_q, report_q, logger=self.logger)
        report_stage = ReportStage(report_q, logger=self.logger)

        report_stage.start()
        selection_stage.start()
        process_stage.start()
        read_stage.start()


def main():
    message_processor = MessageProcessor()
    message_processor.run()


if __name__ == '__main__':
    main()
