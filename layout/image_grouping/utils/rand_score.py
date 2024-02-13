def rand_index(ground_truth, predicted):
    """
    Compute the Rand Index between two clusterings.

    Parameters:
    ground_truth (dict): Dictionary representing the ground truth clustering.
    predicted (dict): Dictionary representing the predicted clustering.

    Returns:
    float: Rand Index value.
    """
    # Initialize variables to count true positives, true negatives, false positives, and false negatives
    tp = tn = fp = fn = 0

    for true_group in ground_truth.values():
        for pred_group in predicted.values():
            # Count pairs in the same group in both ground truth and predicted clustering
            tp += len(set(true_group).intersection(pred_group)) * (
                        len(set(true_group).intersection(pred_group)) - 1) / 2

            # Count pairs in different groups in both ground truth and predicted clustering
            fn += len(set(true_group)) * (len(set(true_group)) - 1) / 2 - tp
            fp += len(set(pred_group)) * (len(set(pred_group)) - 1) / 2 - tp

            # Count pairs in the same group in ground truth but different groups in predicted clustering
            fp += len(set(pred_group) - set(true_group)) * len(true_group)

            # Count pairs in different groups in ground truth but same group in predicted clustering
            fn += len(set(true_group) - set(pred_group)) * len(pred_group)

    # Calculate Rand Index
    rand_index = (tp + tn) / (tp + tn + fp + fn)
    return rand_index


# # Example usage:
# ground_truth = {"Spread_3": ["O25", "O26", "O27"], "Spread_4": ["O28", "O29"]}
# predicted = {"Spread_3": ["O25", "O27"], "Spread_4": ["O28", "O29", "O26"]}
# print("Rand Index:", rand_index(ground_truth, predicted))
