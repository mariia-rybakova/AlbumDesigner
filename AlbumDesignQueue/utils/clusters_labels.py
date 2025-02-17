label_list = [
    'accessories',
    'bride',
    'bride and groom',
    'bride getting dressed',
    'bride party',
    'cake cutting',
    'ceremony',
    'couple',
    'dancing',
    'detail',
    'entertainment',
    'first dance',
    'food',
    'full party',
    'getting hair-makeup',
    'groom',
    'groom party',
    'inside vehicle',
    'invite',
    'kiss',
    'pet',
    'portrait',
    'rings',
    'settings',
    'speech',
    'suit',
    'vehicle',
    'very large group',
    'walking the aisle',
    'wedding dress',
    'other'
]



def map_cluster_label(cluster_label):
    if cluster_label == -1:
        return "None"
    elif cluster_label >= 0 and cluster_label < len(label_list):
        return label_list[cluster_label]
    else:
        return "Unknown"