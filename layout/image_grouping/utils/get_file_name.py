import os


def get_file_names(file_dir: str) -> (list, list):
    '''process only one level of folders'''
    file_names = []
    item_names = os.listdir(file_dir)
    for item in item_names:
        if os.path.isfile(os.path.join(file_dir, item)):
            file_names.append(item)
        else:
            item_file_names = os.listdir(os.path.join(file_dir, item))
            item_full_file_names = [os.path.join(item, file_name) for file_name in item_file_names]
            file_names += item_full_file_names
    file_names.sort()
    file_paths = [os.path.join(file_dir, file_name) for file_name in file_names]
    return file_names, file_paths