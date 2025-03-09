from typing import List, Union
from datetime import datetime

from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, QWriter, Message

read_time_list = list()
processing_time_list = list()
processing_scores_time_list = list()
storing_time_list = list()
retorting_time_list = list()
general_time_list = list()

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
