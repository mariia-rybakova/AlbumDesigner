import random

from collections import defaultdict,Counter
from itertools import combinations_with_replacement

from utils.parser import CONFIGS,spreads_required_per_category,min_images_per_category,priority_categories
from utils.person_clustering import modified_run_person_clustering_experiment
from utils.time_orientation_clustering import select_by_new_clustering_experiment

def get_possible_image_sums():
    # Precompute all possible image sums for 1 to max required spreads
    max_spreads = max(spreads_required_per_category.values())
    spread_image_sums = defaultdict(set)  # {spread_count: set of total image sums}

    for spread_count in range(1, max_spreads + 1):
        for combo in combinations_with_replacement(CONFIGS['layouts'], spread_count):
            spread_image_sums[spread_count].add(sum(combo))

    # Now calculate results for each category
    result = {}
    for category, spreads_needed in spreads_required_per_category.items():
        result[category] = spread_image_sums[spreads_needed]

    return result

def allocate_images_to_categories(
        available_images_per_category,
        possible_image_sums,
):
    result = {}

    # 1. Initial assignment from min_images_per_category
    initial_total = 0
    for category in available_images_per_category.keys():
        min_images = min_images_per_category.get(category, 0)
        result[category] = min_images
        initial_total += min_images

    # 2. Remaining budget
    remaining = CONFIGS['total_target_images'] - initial_total

    if remaining <= 0:
        return result

        # 3. Calculate allocation scores
    scores = {}
    max_limits = {}
    for i, category in enumerate(priority_categories):
        spreads = spreads_required_per_category.get(category, 0)
        if spreads == 0:
            continue

        max_possible_combo = max(possible_image_sums.get(category, []), default=0)
        max_available = available_images_per_category.get(category, float('inf'))
        current = result.get(category, 0)

        max_limit = min(max_possible_combo, max_available)
        max_limits[category] = max_limit

        allocation_capacity = max(0, max_limit - current)

        # Higher priority → higher weight (reverse index)
        priority_weight = len(priority_categories) - i
        score = priority_weight * allocation_capacity
        if score > 0:
            scores[category] = score

    # 4. Normalize scores to percentages
    total_score = sum(scores.values())
    if total_score == 0:
        return result  # No capacity left to allocate

    # 5. Distribute remaining budget by percentage
    for category, score in scores.items():
        share = (score / total_score) * remaining
        allocated = int(round(share))  # You can fine-tune rounding logic if needed
        max_limit = max_limits[category]
        current = result.get(category, 0)

        to_add = min(max_limit - current, allocated)
        if category not in result:
            continue
        result[category] += to_add

    return result


def get_clusters(df):
    clusters_ids = {}
    # image class
    for index, data in df.iterrows():
        class_id = data['cluster_label']
        image_id = data['image_id']
        if class_id not in clusters_ids:
            clusters_ids[class_id] = []

        clusters_ids[class_id].append(image_id)
    return clusters_ids



def select_non_similar_images(event,clusters_ids,image_order_dict,needed_count):
    not_people_events = ['vehicle', 'settings', 'rings', 'entertainment', 'accessories', 'food',
                         'wedding dress', 'suit', 'pet','speech', 'cake cutting']
    result = []
    generators = {k: iter(v) for k, v in clusters_ids.items()}  # Create generators for each list

    while needed_count > 0:
        for key in generators:
            if needed_count <= 0:
                break
            if needed_count < len(clusters_ids):
                items_to_take = 1
            # Check list size to determine how many to take
            elif len(clusters_ids[key]) <= 5 or event in not_people_events:  # Small list: take at most 1
                items_to_take = min(1, needed_count)
            else:  # Large list: take up to the remaining needed count
                list_size = len(clusters_ids[key])
                items_to_take = round(list_size/ needed_count)

            images_ranked = sorted(clusters_ids[key], key=lambda img: image_order_dict.get(img, float('inf')), reverse=True)

            selected_items = images_ranked[:items_to_take]
            clusters_ids[key] = [item for item in clusters_ids[key] if item not in selected_items]

            # Add the selected items to the result and update remaining needed count
            result.extend(selected_items)
            needed_count -= len(selected_items)

    return result



def remove_similar_images(category,needed_count, selected_images, df,final_selected_images):
    persons_categories = ['portrait', 'very large group']
    #'portrait', 'very large group'
    orientation_time_categories = ['bride', 'groom', 'bride and groom', 'bride party', 'groom party','speech', 'full party','walking the aisle', 'bride getting dress', 'getting hair-makeup', 'first dance', 'cake cutting', 'ceremony', 'dancing']
    # 'bride', 'groom', 'bride and groom', 'bride party', 'groom party'

    if len(selected_images) == 1:
        return selected_images

    clusters_ids = get_clusters(df)

    # Identify grayscale images in the filtered list
    grayscale_images = [img for img in selected_images if
                        df.set_index('image_id').loc[img, 'image_color'] == 0]
    num_grayscale = len(grayscale_images)

    if num_grayscale > CONFIGS['grays_scale_limit']: # get high score of the grayscale image instead of random one
        selected_gray_image = random.choice(grayscale_images)
        selected_images = [image for image in selected_images if image not in grayscale_images]
        # Remove other grayscale images except the selected one
        final_selected_images.append(selected_gray_image)
        needed_count = needed_count - 1

    # needed_count = calculate_selection(category, len(selected_images), relations[user_relation])

    filtered_df = df[df['image_id'].isin(selected_images)]

    if needed_count == len(selected_images):
        chosen_images = selected_images
    elif category in persons_categories:
        perf_data = {"category": category,
                     "initial_image_count": len(selected_images)}

        # Select images using persons clustering
        final_selected_images_list = modified_run_person_clustering_experiment(
            input_images_for_category=selected_images,
            df_all_data=df,  # Assuming df_all_data is comprehensive
            image_cluster_dict_for_fallback_logic=perf_data
        )
        return final_selected_images_list

    elif category in orientation_time_categories:
        # Select based on time clustering, then cluster label
        final_selected_images_list = select_by_new_clustering_experiment(
            needed_count=needed_count,
            df_all_data=filtered_df,
        )
        return final_selected_images_list
    else:
        # Select by Clusters label
        chosen_images = select_non_similar_images(category, clusters_ids, df, needed_count)

    final_selected_images.extend(chosen_images)

    return final_selected_images