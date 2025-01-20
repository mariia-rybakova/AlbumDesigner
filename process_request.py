import os

import pandas as pd
import torch
import warnings
import numpy as np
from multiprocessing import Pool
from typing import List, Union

from datetime import datetime
from pymongo import MongoClient

from ptinfra import intialize, get_logger, AbortRequested
from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, QWriter, MessageQueue, MemoryQueue, Message, RoundRobinReader
from ptinfra.config import get_variable

from utils.parser import CONFIGS
from utils.request_processing import read_messages, organize_one_message_results

from utils.clustering_time import  cluster_by_time
from utils.protobufs_processing import get_info_protobufs
from utils.load_layouts import load_layouts
from utils.cover_image import process_non_wedding_cover_image,process_wedding_cover_image,get_cover_layout
from utils.time_proessing import process_image_time
from utils.load_layouts import get_layouts_data
from src.album_processing import create_automatic_album

if os.environ.get('PTEnvironment') == 'dev' or os.environ.get('PTEnvironment') is None:
    os.environ['ConfigServiceURL'] = 'https://devqa.pic-time.com/config/'



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
        super().__init__('ReadStage', self.read_messages, in_q, out_q, err_q, batch_size=1, max_threads=2)
        self.logger = logger
        self.image_loading_timeout = CONFIGS['image_loading_timeout']
        self.queries_file = CONFIGS['queries_file']

    def read_messages(self, msgs: Union[Message, List[Message], AbortRequested]):
        if isinstance(msgs, AbortRequested):
            self.logger.info("Abort requested")
            return []

        messages = msgs if isinstance(msgs, list) else [msgs]

        start = datetime.now()

        #Read messages using a helper function
        try:
            messages = read_messages(messages, self.image_loading_timeout,self.logger)
        except Exception as e:
            self.logger.error(f"Error reading messages: {e}")
            return []

        enriched_messages = []
        for message in messages:
            try:
                # extract necessary fields from the message
                images = message.content.get('images', [])
                project_url = message.content.get('project_url', '')

                if not project_url or not images:
                    self.logger.warning(f"Incomplete message content: {message.content}")
                    continue

                df = pd.DataFrame(images, columns='image_id')
                # check if its wedding here! and added to the message
                gallery_info_df, is_wedding = get_info_protobufs(project_base_url=project_url,df=df,queries_file=queries_file,logger=self.logger )

                if not gallery_info_df.empty and is_wedding:
                        message.content['gallery_photos_info'] = gallery_info_df
                        message.content['is_wedding'] = is_wedding
                        enriched_messages.append(message)
                else:
                   self.logger.error(f"Failed to enrich image data for message: {message.content}")

            except Exception as e:
                self.logger.error(f"Error reading messages at reading stage: {e}")

        handling_time = (datetime.now() - start) / len(messages) if messages else 0
        self.logger.debug(f"Processed {len(enriched_messages)} messages. Average time: {handling_time}")

        return enriched_messages


class ProcessStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        super().__init__('ProcessingStage', self.process_message, in_q, out_q, err_q, batch_size=1, max_threads=2,
                         batch_wait_time=5)
        self.logger = logger

    def process_message(self, msgs: Union[Message, List[Message]]):
        # Load layout file (or data) as needed for each gallery
        layouts_df = load_layouts(msgs.content['layout_file'])
        layout_id2data = get_layouts_data(layouts_df)

        # Check if the message is a single message or a list of messages
        if isinstance(msgs, Message):
            images = [(msgs.image, -1)] if msgs.image is not None else []
        elif isinstance(msgs, list):
            images = [(one_msg.image, idx) for idx, one_msg in enumerate(msgs) if one_msg.image is not None]
        else:
            self.logger.error('Unrecognized type of messages: {}'.format(msgs))
            return msgs

        for message in msgs:
            df = message.content['gallery_photos_info'] if isinstance(message, Message) else [msg.content["photoId"] for
                                                                                              msg in msgs]
            # Sorting the DataFrame by "image_order" column
            sorted_df = df.sort_values(by="image_order", ascending=False)

            # Check if it's a wedding gallery or not and call appropriate method
            if message.content.get('is_wedding', False):  # Assuming there’s a key 'is_wedding'
                df,cover_img_id,cover_img_df = process_wedding_cover_image(sorted_df,self.logger)
            else:

                df,cover_img_id,cover_img_df = process_non_wedding_cover_image(sorted_df,self.logger)

            cover_img_layout = get_cover_layout(layouts_df)

            sorted_by_time_df, image_id2general_time = process_image_time(df)
            df_time = cluster_by_time(sorted_by_time_df)

            # Handle the processing time logging
            try:
                start = datetime.now()
                album_designer = create_automatic_album(df_time,layouts_df,layout_id2data,message.content['is_wedding'], logger=self.logger)
                album_result = album_designer.start_processing_album()
                # Format result in required way with cover image and end spread image with thier layouts
                message.content['album'] = album_result
                processing_time = (datetime.now() - start) / max(len(images), 1)
                self.logger.debug('Average processing time: {}. Processed images: {}'.format(processing_time,
                                                                                             [msg.content.get('photoId')
                                                                                              for msg in msgs]))
            except Exception as ex:
                self.logger.error('Exception while processing messages: {}.'.format(ex))

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
                                    avg_time(storing_time_list), avg_time(retorting_time_list), global_average))

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
            dev_queue = MessageQueue(prefix + input_queue, def_visibility=CONFIGS['visibility_timeout'], max_dequeue_allowed=100)
            test_queue = MessageQueue('test' + input_queue, def_visibility=CONFIGS['visibility_timeout'], max_dequeue_allowed=100)
            ep_queue = MessageQueue('ep' + input_queue, def_visibility=CONFIGS['visibility_timeout'], max_dequeue_allowed=100)
            azure_input_q = RoundRobinReader([dev_queue, test_queue, ep_queue])
        elif prefix == 'production':
            self.logger.info('PRODUCTION environment set, queue name: '+input_queue)
            azure_input_q = MessageQueue(input_queue, def_visibility=CONFIGS['visibility_timeout'], max_dequeue_allowed=100)
        else:
            self.logger.info(prefix + ' environment, queue name: ' + prefix + input_queue)
            azure_input_q = MessageQueue(prefix + input_queue, def_visibility=CONFIGS['visibility_timeout'], max_dequeue_allowed=100)


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
    main()