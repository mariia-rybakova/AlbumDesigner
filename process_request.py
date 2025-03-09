import os
import torch
import warnings
import numpy as np
import pandas as pd
from typing import List, Union

from datetime import datetime

from ptinfra import intialize, get_logger, AbortRequested
from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, QWriter, MessageQueue, MemoryQueue, Message, RoundRobinReader
from ptinfra.config import get_variable


from utils.parser import CONFIGS
from stages.read_stage import ReadStage
from stages.process_stage import ProcessStage
from stages.report_stage import ReportStage



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
