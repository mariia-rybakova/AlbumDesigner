import pandas as pd
import networkx as nx

from datetime import timedelta


def identify_temporal_clusters(
        df: pd.DataFrame,
        time_col: str,
        threshold_minutes: int,
        min_group_size: int,
        logger,
) -> pd.DataFrame:
    """
    Identifies temporal clusters of images using a graph-based approach.

    Images are nodes, and an edge exists if their timestamps are within the threshold.
    Filters out any cluster smaller than min_group_size.

    Args:
        df: DataFrame containing image data with a timestamp column.
        time_col: The name of the timestamp column.
        threshold_minutes: The maximum time difference in minutes to connect two images.
        min_group_size: The minimum number of images required for a cluster to be kept.

    Returns:
        A DataFrame containing only the images from valid temporal clusters,
        with a new 'temporal_group_id' column.
    """
    try:
        if df.empty or time_col not in df.columns:
            return pd.DataFrame()

        df[time_col] = pd.to_datetime(df[time_col])

        df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
        threshold = timedelta(minutes=threshold_minutes)

        G = nx.Graph()
        # Add all images as nodes
        G.add_nodes_from(df_sorted.index)

        # Efficiently add edges between images within the time threshold
        for i in range(len(df_sorted)):
            j = i + 1
            # This subtraction now works correctly (datetime - datetime = timedelta)
            while j < len(df_sorted) and (df_sorted.at[j, time_col] - df_sorted.at[i, time_col]) <= threshold:
                G.add_edge(i, j)
                j += 1

        # Find connected components (the actual temporal clusters)
        connected_components = list(nx.connected_components(G))

        # Create a mapping from image index to a component ID
        node_to_component = {node: i for i, component in enumerate(connected_components) for node in component}
        df_sorted['temporal_group_id'] = df_sorted.index.map(node_to_component)

        # Filter out small, isolated groups (orphans)
        group_sizes = df_sorted['temporal_group_id'].value_counts()
        valid_groups = group_sizes[group_sizes >= min_group_size].index

    except Exception as e:
        logger.error(f"Error in identify_temporal_clusters {e}")
        return pd.DataFrame()

    return df_sorted[df_sorted['temporal_group_id'].isin(valid_groups)].copy()
