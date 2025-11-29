
import sumolib
import json
import os

# Path to the network file
SUMO_NET_FILE = os.path.join(os.path.dirname(__file__), 'roughsumocfgfiles', 'aksshayt.net.xml')
# Output path for the map data
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'frontend', 'public', 'map_data.json')

def extract_map_data():
    """
    Reads a SUMO network file and extracts the shape of all roads,
    saving them to a JSON file.
    """
    print(f"Reading network from: {SUMO_NET_FILE}")
    net = sumolib.net.readNet(SUMO_NET_FILE)
    
    map_data = []
    
    for edge in net.getEdges():
        # We only care about roads vehicles can drive on
        if edge.getFunction() != 'internal':
            shape = edge.getShape()
            # Convert shape points to a list of [x, y] coordinates
            shape_coords = [[point[0], point[1]] for point in shape]
            map_data.append({
                'id': edge.getID(),
                'shape': shape_coords
            })
            
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    print(f"Writing map data to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(map_data, f, indent=2)
        
    print("Map data extraction complete.")

if __name__ == "__main__":
    extract_map_data()
