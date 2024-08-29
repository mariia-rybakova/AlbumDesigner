import pickle


def read_pkl_file(file):
    # Load the dictionary from the binary file
    with open(file, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data