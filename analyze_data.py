import pandas as pd

try:
    df = pd.read_csv("simulation_data.csv")
    
    # Analyze intersections
    df['current_intersection_id'] = df['current_intersection_id'].fillna('')
    unique_intersections = df[df['current_intersection_id'] != '']['current_intersection_id'].unique()
    num_unique_intersections = len(unique_intersections)
    
    # Analyze vehicles
    unique_vehicles = df['vehicle_id'].unique()
    num_unique_vehicles = len(unique_vehicles)
    
    print("--- Data Analysis ---")
    print(f"Unique intersections: {num_unique_intersections}")
    if num_unique_intersections > 0:
        print("Intersection IDs:", ", ".join(unique_intersections))
    
    print(f"\nUnique vehicles: {num_unique_vehicles}")
    
except FileNotFoundError:
    print("Error: simulation_data.csv not found.")
except Exception as e:
    print(f"An error occurred: {e}")
