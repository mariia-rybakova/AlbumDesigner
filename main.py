import os
import json
import copy
import torch
import warnings
import numpy as np

import pandas as pd
from typing import List, Union
from datetime import datetime

from sklearn.mixture import GaussianMixture

from ptinfra import intialize, get_logger
from ptinfra.pt_queue import  MessageQueue, MemoryQueue, RoundRobinReader
from ptinfra.config import get_variable
from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, QWriter, Message
from ptinfra import  AbortRequested
from ptinfra.azure.pt_file import PTFile


from utils.cover_image import process_non_wedding_cover_image, process_wedding_cover_end_image, get_cover_end_layout
from utils.time_proessing import process_image_time
from src.album_processing import start_processing_album
from utils.album_tools import assembly_output
from utils.request_processing import read_messages
from utils.parser import CONFIGS
from utils.clusters_labels import map_cluster_label

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

class ReadStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        super().__init__('ReadStage', self.read_messages, in_q, out_q, err_q, batch_size=1, max_threads=2)
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
        super().__init__('ProcessingStage', self.process_message, in_q, out_q, err_q, batch_size=1, max_threads=2,
        batch_wait_time=5)
        self.logger = logger

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
            if i > 13:
                i = 0

            params = [Spread_score_threshold_params[i], Partition_score_threshold_params[i], Maxm_Combs_params[i],MaxCombsLargeGroups_params[i],MaxOrientedCombs_params[i],Max_photo_groups_params[i]]
            print("Params for this Gallery are:", params)
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
                    df, cover_end_images_ids, cover_end_imgs_df = process_wedding_cover_end_image(processed_df,
                                                                                                  self.logger)
                else:
                    df, cover_end_images_ids, cover_end_imgs_df = process_non_wedding_cover_image(sorted_df,
                                                                                                      self.logger)
                cover_end_imgs_layouts = get_cover_end_layout(message.content['layouts_df'], self.logger)

                df, image_id2general_time = process_image_time(df)

                #Cluster by time
                X = df['general_time'].values.reshape(-1, 1)
                # Determine the optimal number of clusters using Bayesian Information Criterion (BIC)
                n_components = np.arange(1, 10)
                models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]
                bics = [m.bic(X) for m in models]
                # Select the model with the lowest BIC
                best_n = n_components[np.argmin(bics)]
                gmm = GaussianMixture(n_components=best_n, covariance_type='full', random_state=0)
                gmm.fit(X)
                clusters = gmm.predict(X)

                # Add cluster labels to the dataframe
                df['time_cluster'] = clusters

                # to ignore the read only memory
                df = pd.DataFrame(df.to_dict())

                # Handle the processing time logging
                start = datetime.now()
                album_result = start_processing_album(df, message.content['layouts_df'],
                                                      message.content['layout_id2data'],
                                                      message.content['is_wedding'],params, logger=self.logger)

                final_response = assembly_output(album_result, message, message.content['layouts_df'], df,
                                                 cover_end_images_ids, cover_end_imgs_df, cover_end_imgs_layouts)

                message.content['album'] = final_response
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

        handling_time = (datetime.now() - whole_messages_start) / len(messages) if messages else 0
        self.logger.debug('Average Processing Stage time: {}. For : {} messages '.format(handling_time, len(messages)))
        return msgs



class ReportStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        super().__init__('ReportMessage', self.report_message, in_q, out_q, err_q, batch_size=1, max_threads=1)
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
            test_queue = MessageQueue('test' + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                      max_dequeue_allowed=1000)
            ep_queue = MessageQueue('ep' + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                    max_dequeue_allowed=1000)
            azure_input_q = RoundRobinReader([dev_queue, test_queue, ep_queue])
        elif prefix == 'production':
            self.logger.info('PRODUCTION environment set, queue name: ' + input_queue)
            azure_input_q = MessageQueue(input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                         max_dequeue_allowed=1000)
        else:
            self.logger.info(prefix + ' environment, queue name: ' + prefix + input_queue)
            azure_input_q = MessageQueue(prefix + input_queue, def_visibility=CONFIGS['visibility_timeout'],
                                         max_dequeue_allowed=1000)

        read_q = MemoryQueue(2)
        process_q = MemoryQueue(2)
        report_q = MemoryQueue(2)

        read_stage = ReadStage(azure_input_q, read_q, report_q, logger=self.logger)
        process_stage = ProcessStage(read_q, process_q, report_q, logger=self.logger)
        report_stage = ReportStage(report_q, logger=self.logger)

        report_stage.start()
        process_stage.start()
        read_stage.start()


def main():
    message_processor = MessageProcessor()
    message_processor.run()


if __name__ == '__main__':
    main()
