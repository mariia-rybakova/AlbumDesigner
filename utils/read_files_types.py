import pickle
import json

def read_pkl_file(file):
    # Load the dictionary from the binary file
    with open(file, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def read_json_file(json_file_path):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data