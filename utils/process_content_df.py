import copy
from utils.clusters_labels import map_cluster_label

def process_content(row_dict):
        row_dict = copy.deepcopy(row_dict)
        cluster_class = row_dict.get('cluster_class')
        cluster_class_label = map_cluster_label(cluster_class)
        row_dict['cluster_context'] = cluster_class_label
        return row_dict
