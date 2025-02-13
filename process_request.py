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
        self.logger.info(f"READING Stage for {len(messages)} messages. Average time: {handling_time}")
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
        whole_messages_start = datetime.now()
        for message in messages:
            try:
                stage_start = datetime.now()
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
                        processed_df = sorted_df.merge(processed_content_df[['image_id', 'cluster_context']],
                                                       how='left', on='image_id')
                        df, cover_end_images_ids, cover_end_imgs_df = process_wedding_cover_end_image(processed_df,
                                                                                                      self.logger)
                    else:
                        df, cover_end_images_ids, cover_end_imgs_df = process_non_wedding_cover_image(sorted_df,
                                                                                                      self.logger)
                except Exception as e:
                    self.logger.error(f"Error in parallel content processing: {e}")
                    message.content['error'] = f"Error in parallel content processing: {e}"
                    continue

                cover_end_imgs_layouts = get_cover_end_layout(message.content['layouts_df'], logger)
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
                    album_result = start_processing_album(df, message.content['layouts_df'],
                                                          message.content['layout_id2data'],
                                                          message.content['is_wedding'], logger=self.logger)

                    if isinstance(album_result, str):  # Check if it's an error message, report it
                        message.content['error'] = album_result
                        continue

                    message.content['album'] = album_result
                    processing_time = datetime.now() - start

                    self.logger.debug('Lay-outing time: {}.For Processed album id: {}'.format(processing_time,
                                                                                              message.content.get(
                                                                                                  'projectURL', True)))
                    self.logger.debug(
                        'Processing Stage time: {}.For Processed album id: {}'.format(datetime.now() - stage_start,
                                                                                      message.content.get('projectURL',
                                                                                                          True)))

                except Exception as ex:
                    self.logger.error('Exception while processing messages: {}.'.format(ex))
                    message.content['error'] = ex
                    continue

            except Exception as e:
                self.logger.error(f"Unexpected error in message processing: {e}")
                message.content['error'] = f"Unexpected error in message processing: {e}"
                continue

        handling_time = (datetime.now() - whole_messages_start) / len(messages) if messages else 0
        self.logger.debug('Average Processing Stage time: {}. For : {} messages '.format(handling_time, len(messages)))
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
    wedding1_test_message = Message(content={
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

    wedding_2_test_message = Message(content={
        'photosIds': [10006861390,
                      10013216406,
                      10013216407,
                      10013216408,
                      10013216409,
                      10013216410,
                      10013216411,
                      10013216412,
                      10013216413,
                      10013216414,
                      10013216415,
                      10013216416,
                      10013216417,
                      10013216418,
                      10013216419,
                      10013216420,
                      10013216421,
                      10013216422,
                      10013216423,
                      10013216424,
                      10013216425,
                      10013216426,
                      10013216427,
                      10013216428,
                      10013216429,
                      10013216430,
                      10013216431,
                      10013216432,
                      10013216433,
                      10013216434,
                      10013216435,
                      10013216436,
                      10013216437,
                      10013216438,
                      10013216439,
                      10013216440,
                      10013216441,
                      10013216442,
                      10013216443,
                      10013216444,
                      10013216445,
                      10013216446,
                      10013216447,
                      10013216448,
                      10013216449,
                      10013216450,
                      10013216451,
                      10013216452,
                      10013216453,
                      10013216454,
                      10013216455,
                      10013216456,
                      10013216457,
                      10013216458,
                      10013216459,
                      10013216460,
                      10013216461,
                      10013216462,
                      10013216463,
                      10013216464,
                      10013216465,
                      10013216466,
                      10013216467,
                      10013216468,
                      10013216469,
                      10013216470,
                      10013216471,
                      10013216472,
                      10013216473,
                      10013216474,
                      10013216475,
                      10013216476,
                      10013216477,
                      10013216478,
                      10013216479,
                      10013216480,
                      10013216481,
                      10013216482,
                      10013216483,
                      10013216484,
                      10013216485,
                      10013216486,
                      10013216487,
                      10013216488,
                      10013216489,
                      10013216490,
                      10013216491,
                      10013216492,
                      10013216493,
                      10013216494,
                      10013216495,
                      10013216496,
                      10013216497,
                      10013216498,
                      10013216499,
                      10013216500,
                      10013216501,
                      10013216502,
                      10013216503,
                      10013216504,
                      10013216505,
                      10013216506,
                      10013216507,
                      10013216508,
                      10013216509,
                      10013216510,
                      10013216511,
                      10013216512,
                      10013216513,
                      10013216514,
                      10013216515,
                      10013216516,
                      10013216517,
                      10013216518,
                      10013227390,
                      10013227391,
                      10013227392,
                      10013227393,
                      10013227394,
                      10013227395,
                      10013227396,
                      10013227397,
                      10013227398,
                      10013227399,
                      10013227400,
                      10013227401,
                      10013227402,
                      10013227403,
                      10013227404,
                      10013227405],
        'projectURL': 'ptstorage_17://pictures/41/661/41661791/vjg4ekc180wegk0rq0',
        'storeId': 000,
        'layoutsCSV': r'C:\Users\karmel\Desktop\AlbumDesigner\files\layouts.csv',
        'sendTime': 'now'

    })

    wedding_2_test_message_250 = Message(content={
        'photosIds': [10006861390, 10006861391, 10006861392, 10006861393, 10006861394, 10006861395, 10006861396, 10013206010, 10013206011, 10013206012, 10013206013, 10013206014, 10013206015, 10013206016, 10013206017, 10013206018, 10013206019, 10013206020, 10013206021, 10013206022, 10013206023, 10013206024, 10013206025, 10013206026, 10013206027, 10013206028, 10013206029, 10013206030, 10013206031, 10013206032, 10013206033, 10013206034, 10013206035, 10013206036, 10013206037, 10013206038, 10013206039, 10013206040, 10013206041, 10013206042, 10013206043, 10013206044, 10013206045, 10013206046, 10013206047, 10013206048, 10013206049, 10013206050, 10013206051, 10013206052, 10013206053, 10013206054, 10013206055, 10013206056, 10013206057, 10013206058, 10013206059, 10013206060, 10013206061, 10013206062, 10013206063, 10013206064, 10013206065, 10013206066, 10013206067, 10013206068, 10013206069, 10013216405, 10013216406, 10013216407, 10013216408, 10013216409, 10013216410, 10013216411, 10013216412, 10013216413, 10013216414, 10013216415, 10013216416, 10013216417, 10013216418, 10013216419, 10013216420, 10013216421, 10013216422, 10013216423, 10013216424, 10013216425, 10013216426, 10013216427, 10013216428, 10013216429, 10013216430, 10013216431, 10013216432, 10013216433, 10013216434, 10013216435, 10013216436, 10013216437, 10013216438, 10013216439, 10013216440, 10013216441, 10013216442, 10013216443, 10013216444, 10013216445, 10013216446, 10013216447, 10013216448, 10013216449, 10013216450, 10013216451, 10013216452, 10013216453, 10013216454, 10013216455, 10013216456, 10013216457, 10013216458, 10013216459, 10013216460, 10013216461, 10013216462, 10013216463, 10013216464, 10013216465, 10013216466, 10013216467, 10013216468, 10013216469, 10013216470, 10013216471, 10013216472, 10013216473, 10013216474, 10013216475, 10013216476, 10013216477, 10013216478, 10013216479, 10013216480, 10013216481, 10013216482, 10013216483, 10013216484, 10013216485, 10013216486, 10013216487, 10013216488, 10013216489, 10013216490, 10013216491, 10013216492, 10013216493, 10013216494, 10013216495, 10013216496, 10013216497, 10013216498, 10013216499, 10013216500, 10013216501, 10013216502, 10013216503, 10013216504, 10013216505, 10013216506, 10013216507, 10013216508, 10013216509, 10013216510, 10013216511, 10013216512, 10013216513, 10013216514, 10013216515, 10013216516, 10013216517, 10013216518, 10013227390, 10013227391, 10013227392, 10013227393, 10013227394, 10013227395, 10013227396, 10013227397, 10013227398, 10013227399, 10013227400, 10013227401, 10013227402, 10013227403, 10013227404, 10013227405, 10013227406, 10013227407, 10013227408, 10013227409, 10013227410, 10013227411, 10013227412, 10013227413, 10013227414, 10013227415, 10013227416, 10013227417, 10013227418, 10013227419, 10013227420, 10013227421, 10013227422, 10013227423, 10013227424, 10013227425, 10013227426, 10013227427, 10013227428, 10013227429, 10013227430, 10013227431, 10013227432, 10013227433, 10013227434, 10013227435, 10013227436, 10013227437, 10013227438, 10013227439, 10013227440, 10013227441, 10013227442, 10013227443, 10013227444, 10013227445, 10013227446, 10013227447, 10013227448, 10013227449, 10013227450, 10013227451, 10013227452, 10013227453, 10013227454, 10013227455, 10013227456, 10013227457, 10013227458],
        'projectURL': 'ptstorage_17://pictures/41/661/41661791/vjg4ekc180wegk0rq0',
        'storeId': 000,
        'layoutsCSV': r'C:\Users\karmel\Desktop\AlbumDesigner\files\layouts.csv',
        'sendTime': 'now'

    })

    wedding_3_test_message = Message(content={
        'photosIds': [9468156508,
                      9468156509,
                      9468156510,
                      9468156511,
                      9468156512,
                      9468188529,
                      9468188530,
                      9468188531,
                      9468188532,
                      9468188533,
                      9468188534,
                      9468188535,
                      9468188536,
                      9468188537,
                      9468188538,
                      9468188539,
                      9468188540,
                      9468188541,
                      9468188542,
                      9468188543,
                      9468188544,
                      9468188545,
                      9468188546,
                      9468188547,
                      9468188548,
                      9468188549,
                      9468188550,
                      9468188551,
                      9468188552,
                      9468188553,
                      9468188554,
                      9468188555,
                      9468188556,
                      9468188557,
                      9468188558,
                      9468188559,
                      9468188560,
                      9468188561,
                      9468188562,
                      9468188563,
                      9468188564,
                      9468188565,
                      9468188566,
                      9468188567,
                      9468188568,
                      9468188569,
                      9468188570,
                      9468188571,
                      9468188572,
                      9468188573,
                      9468188574,
                      9468188575,
                      9468188576,
                      9468188577,
                      9468188578,
                      9468188579,
                      9468188580,
                      9468188581,
                      9468188582,
                      9468188583,
                      9468188584,
                      9468188585,
                      9468188586,
                      9468188587,
                      9468188588,
                      9468188589,
                      9468188590,
                      9468188591,
                      9468188592,
                      9468188593,
                      9468188594,
                      9468188595],
        'projectURL': 'ptstorage_18://pictures/38/978/38978635/ldvo72xf7pop',
        'storeId': 000,
        'layoutsCSV': r'C:\Users\karmel\Desktop\AlbumDesigner\files\layouts.csv',
        'sendTime': 'now'

    })

    wedding_4_test_message = Message(content={
        'photosIds': [9444433795,
                      9444433796,
                      9444433797,
                      9444433798,
                      9444433799,
                      9444433800,
                      9444433801,
                      9444433802,
                      9444433803,
                      9444433804,
                      9444433805,
                      9444433806,
                      9444433807,
                      9444433808,
                      9444433809,
                      9444433810,
                      9444433811,
                      9444433812,
                      9444433813,
                      9444433814,
                      9444433815,
                      9444433816,
                      9444433817,
                      9444433818,
                      9444433819,
                      9444433820,
                      9444433821,
                      9444433822,
                      9444433823,
                      9444433824,
                      9444433825,
                      9444433826,
                      9444433827,
                      9444433828,
                      9444433829,
                      9444433830,
                      9444433831,
                      9444433832,
                      9444433833,
                      9444433834,
                      9444433835,
                      9444433836,
                      9444433837,
                      9444433838,
                      9444433839,
                      9444433840,
                      9444433841,
                      9444433842,
                      9444433843,
                      9444433844,
                      9444433845,
                      9444433846,
                      9444433847,
                      9444433848,
                      9444433849,
                      9444433850,
                      9444433851,
                      9444433852,
                      9444433853,
                      9444433854,
                      9444433855,
                      9444433856,
                      9444433857,
                      9444433858,
                      9444433859,
                      9444433860,
                      9444433861,
                      9444433862,
                      9444433863,
                      9444433864,
                      9444433865,
                      9444433866,
                      9444433867,
                      9444433868,
                      9444433869,
                      9444433870,
                      9444433871,
                      9444433872,
                      9444433873,
                      9444433874,
                      9444433875,
                      9444433876,
                      9444433877,
                      9444433878,
                      ],
        'projectURL': 'ptstorage_18://pictures/38/122/38122574/jn4cl65tg2gf',
        'storeId': 000,
        'layoutsCSV': r'C:\Users\karmel\Desktop\AlbumDesigner\files\layouts.csv',
        'sendTime': 'now'

    })

    wedding_5_test_message = Message(content={
        'photosIds': [9238583750,
                      9238583751,
                      9238583752,
                      9238583753,
                      9238583754,
                      9238583755,
                      9238583756,
                      9238583757,
                      9238583758,
                      9238583759,
                      9238583760,
                      9238583761,
                      9238583762,
                      9238583763,
                      9238583764,
                      9238583765,
                      9238583766,
                      9238583767,
                      9238583768,
                      9238583769,
                      9238583770,
                      9238583771,
                      9238583772,
                      9238583773,
                      9238583774,
                      9238583775,
                      9238583776,
                      9238583777,
                      9238583778,
                      9238583779],
        'projectURL': 'ptstorage_18://pictures/37/36/37036946/0j7cgl13spuo',
        'storeId': 000,
        'layoutsCSV': r'C:\Users\karmel\Desktop\AlbumDesigner\files\layouts.csv',
        'sendTime': 'now'

    })

    non_wedding_test_message = Message(content={
        'photosIds': [10166866900, 10166866901, 10166866902, 10166866903, 10166866904, 10166866905, 10166866906,
                      10166866907, 10166866908, 10166866909, 10166866910, 10166866911, 10166866912, 10166866913,
                      10166866914, 10166866915, 10166866916, 10166866917, 10166866918, 10166866919, 10166866920],
        'projectURL': 'ptstorage_16://pictures/42/681/42681010/0y6hho9zt8y7dw5zmb',
        'storeId': 000,
        'layoutsCSV': r'C:\Users\karmel\Desktop\AlbumDesigner\files\layouts.csv',
        'sendTime': 'now'

    })

    wedding_2_test_message_setting = Message(content={
        'photosIds': [10013216405,
    10013216406,
    10013216407,
    10013216408,
    10013216409,
    10013216416,
    10013216418,
    10013216419,
    10013216417,
    10013216420,
    10013216421,
    10013216422,
    10013216423,
    10013216424,
    10013216425,
    10013216426,
    10013216427,
    10013216428,
    10013216432,
    10013216431,
    10013216434,
    10013216433,
    10013216436,
    10013216437,
    10013216438,
    10013216439,
    10013216440,
    10013216405,
    10013216406,
    10013216407,
    10013216408,
    10013216409,
    10013216416,
    10013216418,
    10013216419,
    10013216417,
    10013216420,
    10013216421,
    10013216422,
    10013216423,
    10013216424,
    10013216425,
    10013216426,
    10013216427,
    10013216428,
    10013216432,
    10013216431,
    10013216434,
    10013216433,
    10013216436,
    10013216437,
    10013216438,
    10013216439,
    10013216440],
        'projectURL': 'ptstorage_17://pictures/41/661/41661791/vjg4ekc180wegk0rq0',
        'storeId': 000,
        'layoutsCSV': r'C:\Users\karmel\Desktop\AlbumDesigner\files\layouts.csv',
        'sendTime': 'now'

    })



    # Call the read_messages method
    #result = read_stage.read_messages([wedding1_test_message,wedding_2_test_message,wedding_3_test_message,wedding_4_test_message,wedding_5_test_message])
    result = read_stage.read_messages([wedding_2_test_message_250])

    processing = ProcessStage(logger=logger)
    processing.process_message(result)
    print("Finished will all messages")
