import os
import torch
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool
from typing import List, Union

from datetime import datetime
from pymongo import MongoClient
from multiprocessing import Pool, Lock, Manager

from ptinfra import intialize, get_logger, AbortRequested
from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, QWriter, MessageQueue, MemoryQueue, Message, RoundRobinReader
from ptinfra.config import get_variable

from testlocally import Message
from utils.parser import CONFIGS
from utils.request_processing import read_messages, organize_one_message_results
from utils.clustering_time import cluster_by_time

from utils.load_layouts import load_layouts
from utils.cover_image import process_non_wedding_cover_image, process_wedding_cover_end_image, get_cover_end_layout
from utils.time_proessing import process_image_time
from utils.load_layouts import get_layouts_data
from src.album_processing import start_processing_album
from utils.parallel_methods import parallel_content_processing
# if os.environ.get('PTEnvironment') == 'dev' or os.environ.get('PTEnvironment') is None:
#     os.environ['ConfigServiceURL'] = 'https://devqa.pic-time.com/config/'


warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

read_time_list = list()
processing_time_list = list()
processing_scores_time_list = list()
storing_time_list = list()
retorting_time_list = list()
general_time_list = list()


class GetStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None, logger=None):
        super().__init__('GetStage', self.get_message, in_q, out_q, err_q, batch_size=1, max_threads=1)
        self.logger = logger

    def get_message(self, msgs: Union[List[Message], AbortRequested]):
        return msgs


class ReadStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        #super().__init__('ReadStage', self.read_messages, in_q, out_q, err_q, batch_size=1, max_threads=2)
        self.logger = logger
        self.queries_file = CONFIGS['queries_file']

    def read_messages(self, msgs: Union[Message, List[Message], AbortRequested]):
        if isinstance(msgs, AbortRequested):
            self.logger.info("Abort requested")
            return []

        messages = msgs if isinstance(msgs, list) else [msgs]
        start = datetime.now()
        #Read messages using a helper function
        try:
            messages = read_messages(messages, self.queries_file, self.logger)
        except Exception as e:
            self.logger.error(f"Error reading messages: {e}")
            return []

        handling_time = (datetime.now() - start) / len(messages) if messages else 0
        self.logger.debug(f"READING {len(messages)} messages. Average time: {handling_time}")
        return messages


class ProcessStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        #super().__init__('ProcessingStage', self.process_message, in_q, out_q, err_q, batch_size=1, max_threads=2,
                         #batch_wait_time=5)
        self.logger = logger

    def process_message(self, msgs: Union[Message, List[Message]]):
        # check if its single message or list
        messages = msgs if isinstance(msgs, list) else [msgs]
        for message in messages:
            try:
                # Extract gallery photo info safely
                df = message.content.get('gallery_photos_info', pd.DataFrame())
                if df.empty:
                    self.logger.error(f"Gallery photos info DataFrame is empty for message {message}")
                    message.content['error'] = f"Gallery photos info DataFrame is empty for message {message}"
                    continue

                if "image_order" not in df.columns:
                    self.logger.error(f"Missing 'image_order' column in DataFrame for message {message}")
                    message.content['error'] = f"Missing 'image_order' column in DataFrame for message {message}"
                    continue

                # Sorting the DataFrame by "image_order" column
                sorted_df = df.sort_values(by="image_order", ascending=False)

                try:
                    if message.content.get('is_wedding', True):
                        processed_content_df = parallel_content_processing(sorted_df)
                        processed_df = sorted_df.merge(processed_content_df[['image_id', 'cluster_context']], how='left', on='image_id')
                        df, cover_end_images_ids, cover_end_imgs_df = process_wedding_cover_end_image(processed_df,
                                                                                                      self.logger)
                    else:
                        df, cover_end_images_ids, cover_end_imgs_df = process_non_wedding_cover_image(sorted_df,
                                                                                                      self.logger)
                except Exception as e:
                    self.logger.error(f"Error in parallel content processing: {e}")
                    message.content['error'] = f"Error in parallel content processing: {e}"
                    continue

                cover_end_imgs_layouts = get_cover_end_layout(message.content['layouts_df'],logger)
                if not cover_end_imgs_layouts:
                        self.logger.error(f"No cover-end layouts found for message {message}")
                        message.content['error'] = f"No cover-end layouts found for message {message}"
                        continue

                sorted_by_time_df, image_id2general_time = process_image_time(df)
                df_time = cluster_by_time(sorted_by_time_df)

                # to ignore the read only memory
                df = pd.DataFrame(df_time.to_dict())

                # Handle the processing time logging
                try:
                    start = datetime.now()
                    album_result = start_processing_album(df, message.content['layouts_df'], message.content['layout_id2data'],
                                                            message.content['is_wedding'], logger=self.logger)

                    if isinstance(album_result, str):  # Check if it's an error message, report it
                        message.content['error'] = album_result
                        continue

                    message.content['album'] = album_result
                    processing_time = datetime.now() - start
                    self.logger.debug('Average processing time: {}. Processed images: {}'.format(processing_time,
                                                                                                 [msg.content.get('photoId')
                                                                                                  for msg in msgs]))
                except Exception as ex:
                    self.logger.error('Exception while processing messages: {}.'.format(ex))
                    message.content['error'] = ex
                    continue

            except Exception as e:
                self.logger.error(f"Unexpected error in message processing: {e}")
                message.content['error'] = f"Unexpected error in message processing: {e}"
                continue

        return msgs


class StoreStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        super().__init__('StoreStage', self.store_message, in_q, out_q, err_q, batch_size=1, max_threads=1)
        db = self.get_database()
        self.bg_segmentation_dal = db[CONFIGS['mongo_db_name']]
        self.logger = logger

    @staticmethod
    def get_database():
        connection_string = get_variable('MongoConnectionString')
        client = MongoClient(connection_string)
        return client['aimongo']

    def store_doc(self, one_msg):
        bg_data_doc = organize_one_message_results(one_msg, CONFIGS['model_version'], self.logger)
        try:
            self.bg_segmentation_dal.update_one({"_id": one_msg.content['photoId']}, {"$set": bg_data_doc},
                                                upsert=True)
            # self.logger.debug('Storing bg data doc: {}'.format(one_msg.content['photoId']))
        except Exception as e:
            print('DATABASE EXCEPTION', e)
            one_msg.error = e

    def store_message(self, msgs: Union[Message, List[Message]]):
        start = datetime.now()
        if isinstance(msgs, Message):
            self.store_doc(msgs)
        elif isinstance(msgs, list):
            for one_msg in msgs:
                self.store_doc(one_msg)

        storing_time = (datetime.now() - start) / (len(msgs) if isinstance(msgs, list) else 1)
        storing_time_list.append(storing_time)

        # photo_ids = msgs.content['photoId'] if isinstance(msgs, Message) else [msg.content["photoId"] for msg in msgs]
        # self.logger.debug('Stored images: {}. Storing time: {}.'.format(photo_ids, storing_time))
        return msgs


class ReportStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        super().__init__('ProcessMessage', self.report_message, in_q, out_q, err_q, batch_size=1, max_threads=1)
        self.global_start_time = datetime.now()
        self.global_number_of_msgs = 0
        self.number_of_reports = 0
        self.logger = logger

    def print_time_summary(self, period=10):
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
                          'Processing scores average time: {}. '
                          'Storing average time: {}. '
                          'Reporting average time: {}. '
                          'General average time: {}. '
                          '**********.'.format(self.global_number_of_msgs, avg_time(read_time_list),
                                               avg_time(processing_time_list), avg_time(processing_scores_time_list),
                                               avg_time(storing_time_list), avg_time(retorting_time_list),
                                               global_average))

    def report_one_message(self, one_msg):
        if one_msg.error:
            self.logger.debug('REPORT ERROR MESSAGE  {}.'.format(one_msg.error))

    def report_message(self, msgs: Union[Message, List[Message]]):
        start = datetime.now()
        if isinstance(msgs, Message):
            self.report_one_message(msgs)
            msgs.delete()
        elif isinstance(msgs, list):
            for one_msg in msgs:
                self.report_one_message(one_msg)
                one_msg.delete()

        reporting_time = (datetime.now() - start) / (len(msgs) if isinstance(msgs, list) else 1)
        retorting_time_list.append(reporting_time)

        # photo_ids = msgs.content['photoId'] if isinstance(msgs, Message) else [msg.content["photoId"] for msg in msgs]
        # self.logger.debug('Deleted images: {}. Reporting time: {}.'.format(photo_ids, reporting_time))

        self.global_number_of_msgs += len(msgs) if isinstance(msgs, list) else 1
        self.print_time_summary()

        return


class MessageProcessor:
    def __init__(self):
        self.logger = get_logger(__name__, 'DEBUG')

    def run(self):
        settings_filename = os.environ.get('HostingSettingsPath',
                                           '/ptinternal/pictures/hosting/ai_settings_audiobeat.json.txt')
        # settings_filename = os.environ.get('HostingSettingsPath',
        #                                    'h:/Projects/pic_time/info/visual_embedding_deployment/visual_emb_settings.json.txt')

        # private_key = get_variable('PtKey')
        # self.logger.debug('Private key: {}'.format(private_key))

        intialize('BackgroundSegmentation', settings_filename)

        try:
            prefix = get_variable('PTEnvironment')
        except:
            prefix = 'dev'

        # input_queue = 'devvisualembeddto'
        input_queue = CONFIGS['collection_name']
        print(prefix + input_queue)
        if prefix == 'dev':
            dev_queue = MessageQueue(prefix + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                     max_dequeue_allowed=100)
            test_queue = MessageQueue('test' + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                      max_dequeue_allowed=100)
            ep_queue = MessageQueue('ep' + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                    max_dequeue_allowed=100)
            azure_input_q = RoundRobinReader([dev_queue, test_queue, ep_queue])
        elif prefix == 'production':
            self.logger.info('PRODUCTION environment set, queue name: ' + input_queue)
            azure_input_q = MessageQueue(input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                         max_dequeue_allowed=100)
        else:
            self.logger.info(prefix + ' environment, queue name: ' + prefix + input_queue)
            azure_input_q = MessageQueue(prefix + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                         max_dequeue_allowed=100)

        get_q = MemoryQueue(8)
        read_q = MemoryQueue(8)
        process_q = MemoryQueue(8)
        report_q = MemoryQueue(8)

        get_stage = GetStage(azure_input_q, get_q, report_q, logger=self.logger)
        read_stage = ReadStage(get_q, read_q, report_q, logger=self.logger)
        process_stage = ProcessStage(read_q, process_q, report_q, logger=self.logger)
        store_stage = StoreStage(process_q, report_q, report_q, logger=self.logger)
        report_stage = ReportStage(report_q, logger=self.logger)

        report_stage.start()
        store_stage.start()
        process_stage.start()
        read_stage.start()
        get_stage.start()


def main():
    message_processor = MessageProcessor()
    message_processor.run()


if __name__ == '__main__':
    # main()
    import logging
    # Create a logger instance
    logger = get_logger(__name__, 'DEBUG')
    logger.setLevel(logging.DEBUG)  # Set logging level (DEBUG, INFO, WARNING, ERROR)

    # Create a console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # This ensures all log messages are shown

    # Create a formatter for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    read_stage = ReadStage(logger=logger)

    # Sample message for testing
    wedding_test_message = Message(content={
        'photosIds': [
            9871358316,
            9871358323,
            9871358325,
            9871358324,
            9871358327,
            9871358335,
            9871358334,
            9871358332,
            9871357196,
            9871357195,
            9871357194,
            9871235647,
            9871357197,
            9871235654,
            9871357231,
            9871357237,
            9871357239,
            9871357241,
            9871235659,
            9871357247,
            9871357248,
            9871235697,
            9871235695,
            9871235701,
            9871235705,
            9871260660,
            9871260674,
            9871260678,
            9871260676,
            9871260682,
            9871260681,
            9871260683,
            9871260684,
            9871260685,
            9871260704,
            9871260710,
            9871260711,
            9871260717,
            9871260720,
            9871260723,
            9871260724,
            9871260738,
            9871260734,
            9871440524,
            9871440523,
            9871251425,
            9871251424,
            9871251433,
            9871251452,
            9871251455,
            9871251454,
            9871251453,
            9871251471,
            9871260186,
            9871260211,
            9871260195,
            9871268394,
            9871268418,
            9871268417,
            9871268410,
            9871269359,
            9871269374,
            9871269369,
            9871269376,
            9871269373,
            9871269386,
            9871272557,
            9871272567,
            9871272580,
            9871272576,
            9871272585,
            9871278314,
            9871278315,
            9871230080,
            9871230079,
            9871230099,
            9871230101,
            9871253523,
            9871253570,
            9871253571,
            9871253569,
            9871253597,
            9871253598,
            9871253599,
            9871359882,
            9871359886,
            9871359893,
            9871359894,
            9871359900,
            9871359903,
            9871359906,
            9871359922,
            9871359931,
            9871360599,
            9871360609,
            9871360605,
            9871369320,
            9871369321,
            9871369340,
            9871369352,
            9871372057,
            9871372091,
            9871373925,
            9871373959,
            9871380672,
            9871380661,
            9871380663,
            9871380667,
            9871380670,
            9871380695,
            9871380710,
            9871380711,
            9871388915,
            9871388926,
            9871388927,
            9871388941,
            9871388945,
            9871388946,
            9871388937,
            9871388955,
            9871388959,
            9871388973,
            9871388974,
            9871388990,
            9871389001],
        'projectURL': 'ptstorage_17://pictures/37/141/37141824/dmgb4onqc3hm',
        'storeId': 000,
        'layoutsCSV': r'C:\Users\karmel\Desktop\AlbumDesigner\files\layouts.csv',
        'sendTime': 'now'

    })

    non_wedding_test_message = Message(content={
        'photosIds': [10166866900,10166866901,10166866902,10166866903,10166866904,10166866905,10166866906,10166866907,10166866908,10166866909,10166866910,10166866911,10166866912,10166866913,10166866914,10166866915,10166866916,10166866917,10166866918,10166866919,10166866920],
        'projectURL': 'ptstorage_16://pictures/42/681/42681010/0y6hho9zt8y7dw5zmb',
        'storeId': 000,
        'layoutsCSV': r'C:\Users\karmel\Desktop\AlbumDesigner\files\layouts.csv',
        'sendTime': 'now'

    })

    # Call the read_messages method
    result = read_stage.read_messages([non_wedding_test_message])

    processing = ProcessStage(logger=logger)
    processing.process_message(result)
