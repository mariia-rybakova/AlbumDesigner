class Parameters:

    # clip model
    clip_model = 'ViT-B/32'

    # database
    db_creds = 'F:\\Projects\\pic_time\\credentials\\postgre_credentials\\local_postgre_credentials.json'
    db_engine_file = 'F:\\Projects\\pic_time\\credentials\\postgre_credentials\\db_engine_query.txt'

    # azure vision false positive tags
    azv_fp_tags = ['Food', 'Tables', 'Wedding Cake']

    # replacement for full cloud labels
    replace_dict = {'indoor': 'indoors',
                    'outdoor': 'outdoors',
                    'shoe': 'shoes',
                    'flower': 'flowers',
                    'table': 'tables'}
    azv_fp_labels = ['food', 'tables', 'wedding cake']