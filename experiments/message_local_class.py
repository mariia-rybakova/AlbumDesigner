from typing import List, Union
from datetime import datetime
import pandas as pd

# Mock implementations of the dependencies
class Message:
    def __init__(self, content):
        self.content = content

class AbortRequested:
    pass

class QReader:
    pass

class QWriter:
    pass

class Stage:
    def __init__(self, name, process_func, in_q=None, out_q=None, err_q=None, batch_size=1, max_threads=1):
        self.name = name
        self.process_func = process_func
        self.in_q = in_q
        self.out_q = out_q
        self.err_q = err_q
        self.batch_size = batch_size
        self.max_threads = max_threads

# Mock functions and configurations
CONFIGS = {
    'image_loading_timeout': 5,
    'queries_file': r'C:\Users\karmel\Desktop\AlbumDesigner\files\queries_features.pkl'
}



