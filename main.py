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
from utils.cover_image import process_non_wedding_cover_image, process_wedding_first_last_image, get_first_last_design_ids
from utils.time_proessing import process_image_time, get_time_clusters
from src.album_processing import start_processing_album
from utils.request_processing import read_messages, assembly_output
from utils.parser import CONFIGS
from utils.clusters_labels import map_cluster_label

if os.environ.get('PTEnvironment') == 'dev' or os.environ.get('PTEnvironment') is None:
    os.environ['ConfigServiceURL'] = 'https://devqa.pic-time.com/config/'

warnings.filterwarnings('ignore')
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

read_time_list = list()
processing_time_list = list()
reporting_time_list = list()


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
        super().__init__('ReadStage', self.read_messages, in_q, out_q, err_q, batch_size=1, max_threads=2)
        self.logger = logger


    def read_messages(self, msgs: Union[Message, List[Message], AbortRequested]):
        if isinstance(msgs, AbortRequested):
            self.logger.info("Abort requested")
            return []

        messages = msgs if isinstance(msgs, list) else [msgs]
        start = datetime.now()
        #Read messages using a helper function
        try:
            messages = read_messages(messages, self.logger)
        except Exception as e:
            self.logger.error(f"Error reading messages: {e}")
            return []

        handling_time = (datetime.now() - start) / max(len(messages), 1)
        read_time_list.append(handling_time)
        self.logger.info(f"READING Stage for {len(messages)} messages. Average time: {handling_time}")
        return messages

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

        Spread_score_threshold_params = [0.01,0.5,0.05,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        Partition_score_threshold_params = [100,100,100,200,150,100,100,100,100,100,100,100,100]
        Maxm_Combs_params = [1000,1000,1000,1000,1000,50,10,1000,1000,1000,1000,1000,1000]
        MaxCombsLargeGroups_params = [100,100,100,100,100,100,100,20,5,100,100,100,100]
        MaxOrientedCombs_params = [300,300,300,300,300,300,300,300,300,10,50,300,300]
        Max_photo_groups_params = [12,12,12,12,12,12,12,12,12,12,12,8,5]

        for i,message in enumerate(messages):
            # if i > 13:
            #     i = 0
            i=0

            params = [Spread_score_threshold_params[i], Partition_score_threshold_params[i], Maxm_Combs_params[i],MaxCombsLargeGroups_params[i],MaxOrientedCombs_params[i],Max_photo_groups_params[i]]
            print("Params for this Gallery are:", params)

            p = mp.Process(target=process_crop_images, args=(self.q, message.content.get('gallery_photos_info')))
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

                if message.content.get('is_wedding', True):
                    rows = sorted_df[['image_id', 'cluster_class']].to_dict('records')
                    #convert numeric ids to labels.
                    processed_rows = []
                    for row in rows:
                        cluster_class = row.get('cluster_class')
                        cluster_class_label = map_cluster_label(cluster_class)
                        row['cluster_context'] = cluster_class_label
                        processed_rows.append(row)

                    processed_content_df = pd.DataFrame(processed_rows)
                    processed_df = sorted_df.merge(processed_content_df[['image_id', 'cluster_context']],
                                                   how='left', on='image_id')
                    df, first_last_images_ids, first_last_imgs_df = process_wedding_first_last_image(processed_df,
                                                                                                     self.logger)
                else:
                    df, first_last_images_ids, first_last_imgs_df = process_non_wedding_cover_image(sorted_df,
                                                                                                      self.logger)

                first_last_design_ids = get_first_last_design_ids(message.designsInfo['anyPagelayouts_df'], self.logger)

                # Handle the processing time logging
                start = datetime.now()
                album_result = start_processing_album(df, message.designsInfo['anyPagelayouts_df'],
                                                      message.designsInfo['anyPagelayout_id2data'],
                                                      message.content['is_wedding'],params, logger=self.logger)

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

                df = df.merge(cropped_df, how='inner', on='image_id')
                first_last_imgs_df = first_last_imgs_df.merge(cropped_df, how='inner', on='image_id')

                self.logger.debug('waited for cropping process: {}'.format(datetime.now() - wait_start))



                final_response = assembly_output(album_result, message, df,
                                                 first_last_images_ids, first_last_imgs_df, first_last_design_ids)

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
                message.content['error'] = f"Unexpected error in message processing: {e}"
                continue

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
        report_q = MemoryQueue(2)

        read_stage = ReadStage(azure_input_q, read_q, report_q, logger=self.logger)
        process_stage = ProcessStage(read_q, report_q, report_q, logger=self.logger)
        report_stage = ReportStage(report_q, logger=self.logger)

        report_stage.start()
        process_stage.start()
        read_stage.start()


def main():
    message_processor = MessageProcessor()
    message_processor.run()


if __name__ == '__main__':
    main()
