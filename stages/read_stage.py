import json
from typing import List, Union
from datetime import datetime

from ptinfra import  AbortRequested
from ptinfra.stage import Stage
from ptinfra.pt_queue import QReader, QWriter, Message
from ptinfra.azure.pt_file import PTFile


from utils.request_processing import read_messages
from utils.parser import CONFIGS
import os


class ReadStage(Stage):
    def __init__(self, in_q: QReader = None, out_q: QWriter = None, err_q: QWriter = None,
                 logger=None):
        super().__init__('ReadStage', self.read_messages, in_q, out_q, err_q, batch_size=1, max_threads=2)
        self.logger = logger
        self.queries_file = CONFIGS['queries_file']
        self.products_json = PTFile(CONFIGS['products_json_location'])
        fileBytes = self.products_json.read_blob()
        self.products_json = json.loads(fileBytes.decode('utf-8'))

        self.architect_base = CONFIGS['architect_base']

        self.design_pack_base = CONFIGS['design_pack_base']


    def read_messages(self, msgs: Union[Message, List[Message], AbortRequested]):
        if isinstance(msgs, AbortRequested):
            self.logger.info("Abort requested")
            return []

        messages = msgs if isinstance(msgs, list) else [msgs]

        for msg in msgs:
            productId = msg.conten['productId']
            product_list = self.products_json['products']
            for product_dict in product_list:
                if product_dict['productId'] == productId:
                    break

            packageTypeId = product_dict['packageTypeId']
            productGroupId = product_dict['productGroupId']

            design_package = os.path.join(self.design_pack_base, f'pack_{packageTypeId}.json.txt')
            design_package = PTFile(design_package)
            fileBytes = design_package.read_blob()
            design_package = json.loads(fileBytes.decode('utf-8'))

            msg.designs = design_package

            architect_package = os.path.join(self.architect_base, f'{productGroupId}/architect2.json.en-us.txt')
            architect_package = PTFile(architect_package)
            fileBytes = architect_package.read_blob()
            architect_package = json.loads(fileBytes.decode('utf-8'))

            rules = architect_package['planningRules']
            msg.defaultPackageStyleId = rules['defaultPackageStyleId']

            compositions = rules['compositions']
            cover = None
            for composition in compositions:
                if composition['name']=='cover':
                    cover = composition
                    break

            msg.cover = False
            if cover is not None:
                msg.cover=True
                msg.cover_designs = cover['designIds']

            first = None
            for composition in compositions:
                if composition['name'] == 'first':
                    first = composition
                    break

            msg.first = False
            if first is not None:
                msg.first = True
                msg.first_designs = first['designIds']

            last = None
            for composition in compositions:
                if composition['name'] == 'last':
                    last = composition
                    break

            msg.last = False
            if last is not None:
                msg.last = True
                msg.last_designs = last['designIds']

            any_page = None
            for composition in compositions:
                if composition['name'] == 'any page':
                    any_page = composition
                    break
            if any_page is None:
                self.logger.error('no designs found for any page')
                msg.error = Exception('no designs found for any page')
            else:
                msg.designs = any_page['designIds']


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
