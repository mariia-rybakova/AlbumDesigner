
class_order_mapping = {
"bride getting dressed":1,
"getting hair-makeup": 1 ,
"groom getting dress":1,
"bride":2,
"groom":2,
"bride and groom":3,
"walking the aisle":4,
"ceremony":5,
"settings":6,
"portrait" :7,
'speech':8,
"dancing":9
}


def handle_edited_time(df):
    # Step 2: Map classes to their order
    df['class_order'] = df['cluster_context'].map(class_order_mapping)

    # Step 3: Sort the dataframe by class order and general_time
    df = df.sort_values(['class_order', 'general_time']).reset_index(drop=True)

    # Step 4: Assign corrected time values
    delta_time = 1.0  # Adjust the increment as needed
    starting_time = df['general_time'].min() - delta_time
    current_time = starting_time
    edited_general_times = []

    for idx, row in df.iterrows():
        current_time += delta_time
        edited_general_times.append(current_time)

    df['edited_general_time'] = edited_general_times

    return df
