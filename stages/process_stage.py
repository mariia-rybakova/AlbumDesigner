import pandas as pd
from typing import List, Union
from datetime import datetime

from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, QWriter, Message

from utils.clustering_time import cluster_by_time

from utils.cover_image import process_non_wedding_cover_image, process_wedding_cover_end_image, get_cover_end_layout
from utils.time_proessing import process_image_time
from src.album_processing import start_processing_album
from utils.parallel_methods import parallel_content_processing
from utils.album_tools import assembly_output
from utils.parser import CONFIGS



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

                cover_end_imgs_layouts = get_cover_end_layout(message.content['layouts_df'], self.logger)
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
                                                          message.content['is_wedding'],params, logger=self.logger)

                    final_response = assembly_output(album_result, message, message.content['layouts_df'], df,
                                                     cover_end_images_ids, cover_end_imgs_df, cover_end_imgs_layouts)

                    if isinstance(album_result, str):  # Check if it's an error message, report it
                        message.content['error'] = final_response
                        continue

                    message.content['album'] = final_response
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
