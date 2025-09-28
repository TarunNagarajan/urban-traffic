import pandas as pd

try:
    df = pd.read_csv("simulation_data.csv")
    # The same fillna logic from server.py
    df['current_intersection_id'] = df['current_intersection_id'].fillna('')
    
    # Filter out empty strings before finding unique intersections
    unique_intersections = df[df['current_intersection_id'] != '']['current_intersection_id'].unique()
    
    num_unique = len(unique_intersections)
    
    if num_unique > 0:
        print(f"Found {num_unique} unique intersection(s):")
        for intersection_id in unique_intersections:
            print(f"- {intersection_id}")
    else:
        print("No intersections found in the data.")
        
except FileNotFoundError:
    print("Error: simulation_data.csv not found.")
except Exception as e:
    print(f"An error occurred: {e}")
