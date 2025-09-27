
import xml.etree.ElementTree as ET
import json
import os
import argparse

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>SUMO Network Visualization</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <style>
        html, body, #map { margin: 0; padding: 0; width: 100%; height: 100%; background: #333; }
    </style>
</head>
<body>

<div id="map"></div>

<script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
<script src="https://unpkg.com/deck.gl@8.4.17/dist.min.js"></script>
<script src="https://unpkg.com/@deck.gl/leaflet@8.4.17/dist.min.js"></script>

<script type="text/javascript">
    const roadData = {road_data_json};
    const initialViewState = {view_state_json};

    const map = L.map('map').setView([initialViewState.latitude, initialViewState.longitude], initialViewState.zoom);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    const deckLayer = new deck.LeafletLayer({
        views: [
            new deck.MapView({id: 'deckgl-view', controller: true})
        ],
        layers: [
            new deck.PathLayer({
                id: 'sumo-network-layer',
                data: roadData,
                getPath: d => d.path,
                getColor: [255, 255, 255, 80], // White with some transparency
                getWidth: 3,
                widthMinPixels: 1,
                widthMaxPixels: 10,
                pickable: true,
                onHover: info => {
                    if (info.object) {
                        info.object.properties.id = info.object.id;
                        console.log('Hovered on edge:', info.object.id);
                    }
                }
            })
        ]
    });
    map.addLayer(deckLayer);

</script>
</body>
</html>
"""

def parse_net_xml(net_file):
    """Parses a SUMO .net.xml file and extracts road geometries."""
    print(f"Parsing network file: {net_file}")
    tree = ET.parse(net_file)
    root = tree.getroot()

    all_coords = []
    road_data = []

    for edge in root.findall('edge'):
        # Ignore internal/junction edges
        if not edge.get('function') == 'internal':
            shape = edge.get('shape')
            if shape:
                path_coords = []
                coords = shape.split(' ')
                for coord in coords:
                    try:
                        x, y = map(float, coord.split(','))
                        path_coords.append([x, y])
                        all_coords.append([x, y])
                    except ValueError:
                        print(f"Could not parse coordinate: '{coord}' in edge {edge.get('id')}")
                
                road_data.append({
                    'id': edge.get('id'),
                    'path': path_coords
                })
    
    print(f"Found {len(road_data)} road segments.")
    return road_data, all_coords

def get_initial_view_state(coords):
    """Calculates the center and zoom level for the map."""
    if not coords:
        return {'latitude': 0, 'longitude': 0, 'zoom': 2}

    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # A simple heuristic for zoom level based on the bounding box size
    # This might need adjustment for different coordinate systems
    span = max(max_x - min_x, max_y - min_y)
    zoom = 12
    if span > 1000:
        zoom = 14
    if span < 500:
        zoom = 16

    # IMPORTANT: SUMO uses a Cartesian coordinate system.
    # Leaflet/Mapbox use Latitude/Longitude.
    # For this visualization, we are treating the SUMO coordinates as if they are
    # a form of projected coordinates and centering the map on them.
    # A true geo-referenced visualization would require coordinate transformation.
    return {
        'longitude': center_x,
        'latitude': center_y,
        'zoom': zoom
    }

def main():
    parser = argparse.ArgumentParser(description='Generate a map visualization from a SUMO network file.')
    parser.add_argument('--net_file', type=str, default='nets/demo.net.xml', help='Path to the SUMO .net.xml file.')
    parser.add_argument('--output_file', type=str, default='map_visualization.html', help='Name of the output HTML file.')
    args = parser.parse_args()

    if not os.path.exists(args.net_file):
        print(f"Error: Network file not found at {args.net_file}")
        return

    road_data, all_coords = parse_net_xml(args.net_file)
    initial_view_state = get_initial_view_state(all_coords)

    # Embed the data into the HTML template
    final_html = HTML_TEMPLATE.replace('{road_data_json}', json.dumps(road_data))
    final_html = final_html.replace('{view_state_json}', json.dumps(initial_view_state))

    with open(args.output_file, 'w') as f:
        f.write(final_html)
    
    print(f"Successfully generated map visualization: {args.output_file}")
    print("Open this file in your web browser to see the map.")

if __name__ == "__main__":
    main()
