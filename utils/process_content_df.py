from utils.clusters_labels import map_cluster_label

def process_content(row):
        cluster_class = row['cluster_class']
        cluster_class_label = map_cluster_label(cluster_class)
        row['cluster_context'] = cluster_class_label
        return row
