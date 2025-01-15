import os

import pandas as pd
import torch
import warnings
import numpy as np
from typing import List, Union

from datetime import datetime
from pymongo import MongoClient

from ptinfra import intialize, get_logger, AbortRequested
from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, QWriter, MessageQueue, MemoryQueue, Message, RoundRobinReader
from ptinfra.config import get_variable

from src.Remover import Remover
from utils.background_scores import get_background_scores
from utils.parser import CONFIGS
from utils.request_processing import read_messages, organize_one_message_results
from process_images import image_metrics
from utils.protobufs_processing import get_info_protobufs

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

                gallery_info_df = get_info_protobufs(project_base_url=project_url,df=df,logger=self.logger )

                if not gallery_info_df.empty:
                        message.content['gallery_photos_info'] = gallery_info_df
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
        bg_model_path = CONFIGS['bg_model_path']
        bg_model_name = CONFIGS['bg_model_name']
        input_size = CONFIGS['input_size']
        save_size = CONFIGS['save_size']
        self.batch_size = CONFIGS['batch_size']
        self.bg_remover = Remover(mode='fast', model_path=bg_model_path, model_name=bg_model_name,
                                  base_size=input_size, save_size=save_size)
        self.logger = logger

    def process_message(self, msgs: Union[Message, List[Message]]):
        if isinstance(msgs, Message):
            images = [(msgs.image, -1)] if msgs.image is not None else []
        elif isinstance(msgs, list):
            images = [(one_msg.image, idx) for idx, one_msg in enumerate(msgs) if one_msg.image is not None]
        else:
            self.logger.error('Unrecognized type of messages: {}'.format(msgs))  # what to do if msgs are not Message or list of Message?
            return msgs

        photo_ids = msgs.content['photoId'] if isinstance(msgs, Message) else [msg.content["photoId"] for msg in msgs]

        try:
            start = datetime.now()
            background_masks = self.bg_remover.process_all(images, batch_size=self.batch_size)
            for cur_bg_mask, cur_resized_image, general_msg_id in background_masks:
                if general_msg_id == -1:
                    msgs.bg_mask = cur_bg_mask
                    msgs.resized_image = cur_resized_image
                else:
                    msgs[general_msg_id].bg_mask = cur_bg_mask
                    msgs[general_msg_id].resized_image = cur_resized_image
            processing_time = (datetime.now() - start) / max(len(images), 1)
            processing_time_list.append(processing_time)
            # photo_ids = msgs.content['photoId'] if isinstance(msgs, Message) else [msg.content["photoId"] for msg in msgs]
            # self.logger.debug('Average processing time: {}. Processed photos: {}'.format(processing_time, photo_ids))
        except Exception as ex:
            self.logger.error('Exception while processing messages: {}. Photo ids: {}.'.format(ex, photo_ids))
        return msgs


class ProcessScoresStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        super().__init__('ProcessingScoresStage', self.process_message, in_q, out_q, err_q, batch_size=1, max_threads=1,
                         batch_wait_time=5)
        self.blob_input_size = CONFIGS['blob_input_size']
        self.logger = logger

    def process_message(self, msgs: Union[Message, List[Message]]):
        if isinstance(msgs, Message):
            images = [(msgs.image, -1)] if msgs.image is not None else []
            background_masks = [(msgs.bg_mask, msgs.resized_image, -1)] if msgs.image is not None else []
        elif isinstance(msgs, list):
            images = [(one_msg.image, idx) for idx, one_msg in enumerate(msgs) if one_msg.image is not None]
            background_masks = [(one_msg.bg_mask, one_msg.resized_image, idx) for idx, one_msg in enumerate(msgs) if one_msg.image is not None]
        else:
            self.logger.error('Unrecognized type of messages: {}'.format(msgs))  # what to do if msgs are not Message or list of Message?
            return msgs

        photo_ids = msgs.content['photoId'] if isinstance(msgs, Message) else [msg.content["photoId"] for msg in msgs]

        try:
            start = datetime.now()
            for image in images:
                image_id = image[1]
                image_obj = image[0]
                color, ar = image_metrics(image_obj)
                if image_id == -1:
                    msgs.ar = ar
                    msgs.color = color
                else:
                    msgs[image_id].ar = ar
                    msgs[image_id].color = color
            ar_color_time = (datetime.now() - start) / max(len(images), 1)

            scores_start = datetime.now()
            background_data = get_background_scores(background_masks, self.blob_input_size, self.logger)

            for cur_bg_data, general_msg_id in background_data:
                if general_msg_id == -1:
                    msgs.bg_data = cur_bg_data
                else:
                    msgs[general_msg_id].bg_data = cur_bg_data
            scores_processing_time = (datetime.now() - scores_start) / max(len(images), 1)
            processing_time = (datetime.now() - start) / max(len(images), 1)

            processing_scores_time_list.append(processing_time)
            # self.logger.debug('Average ar-color time/score processing time: {}/{}. Processed photos: {}'.format(ar_color_time, scores_processing_time, photo_ids))
        except Exception as ex:
            self.logger.error('Exception while scores processing: {}. Photo ids: {}'.format(ex, photo_ids))
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
        process_scores_q = MemoryQueue(8)
        report_q = MemoryQueue(8)

        get_stage = GetStage(azure_input_q, get_q, report_q, logger=self.logger)
        read_stage = ReadStage(get_q, read_q, report_q, logger=self.logger)
        process_stage = ProcessStage(read_q, process_q, report_q, logger=self.logger)
        process_scores_stage = ProcessScoresStage(process_q, process_scores_q, report_q, logger=self.logger)
        store_stage = StoreStage(process_scores_q, report_q, report_q, logger=self.logger)
        report_stage = ReportStage(report_q, logger=self.logger)

        report_stage.start()
        store_stage.start()
        process_scores_stage.start()
        process_stage.start()
        read_stage.start()
        get_stage.start()


def main():
    message_processor = MessageProcessor()
    message_processor.run()


if __name__ == '__main__':
    main()