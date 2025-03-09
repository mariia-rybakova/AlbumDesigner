from typing import List, Union
from datetime import datetime

from ptinfra import  AbortRequested
from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, QWriter, Message

from utils.request_processing import read_messages
from utils.parser import CONFIGS

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
