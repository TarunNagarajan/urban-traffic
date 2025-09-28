#!/usr/bin/env python3
"""
Generate route file for SUMO simulation
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import random
import math

def read_network_edges(net_file):
    """Read edges from the network file"""
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    edges = []
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        from_node = edge.get('from')
        to_node = edge.get('to')
        edge_type = edge.get('type', 'highway.primary')
        
        # Get lanes
        lanes = []
        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            speed = float(lane.get('speed', '13.89'))
            lanes.append({'id': lane_id, 'speed': speed})
        
        edges.append({
            'id': edge_id,
            'from': from_node,
            'to': to_node,
            'type': edge_type,
            'lanes': lanes
        })
    
    return edges

def find_fringe_edges(edges):
    """Find edges that are suitable for trip starts/ends"""
    # Count how many times each node appears as 'from' or 'to'
    node_counts = {}
    
    for edge in edges:
        from_node = edge['from']
        to_node = edge['to']
        
        node_counts[from_node] = node_counts.get(from_node, 0) + 1
        node_counts[to_node] = node_counts.get(to_node, 0) + 1
    
    # Fringe edges are those connected to nodes with low connectivity
    fringe_edges = []
    for edge in edges:
        from_count = node_counts.get(edge['from'], 0)
        to_count = node_counts.get(edge['to'], 0)
        
        # Consider edges connected to nodes with low connectivity as fringe
        if from_count <= 2 or to_count <= 2:
            fringe_edges.append(edge)
    
    return fringe_edges

def generate_trips(edges, num_trips=100, duration=3600):
    """Generate random trips"""
    random.seed(42)
    
    # Find suitable edges for trips
    fringe_edges = find_fringe_edges(edges)
    all_edges = [e for e in edges if e['type'] in ['highway.primary', 'highway.secondary', 'highway.tertiary']]
    
    trips = []
    
    for i in range(num_trips):
        # Select random from and to edges
        from_edge = random.choice(fringe_edges + all_edges)
        to_edge = random.choice(fringe_edges + all_edges)
        
        # Avoid same edge
        if from_edge['id'] == to_edge['id']:
            continue
        
        # Random departure time
        depart_time = random.uniform(0, duration)
        
        trip = {
            'id': f'veh{i}',
            'from': from_edge['id'],
            'to': to_edge['id'],
            'depart': depart_time,
            'departLane': 'best',
            'departSpeed': 'max' if from_edge in fringe_edges else '0'
        }
        
        trips.append(trip)
    
    return trips

def create_route_file(trips, output_file):
    """Create SUMO route file"""
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    
    # Add vehicle type
    vtype = ET.SubElement(root, "vType")
    vtype.set("id", "veh_passenger")
    vtype.set("vClass", "passenger")
    
    # Add trips
    for trip in trips:
        trip_elem = ET.SubElement(root, "trip")
        trip_elem.set("id", trip['id'])
        trip_elem.set("depart", f"{trip['depart']:.2f}")
        trip_elem.set("from", trip['from'])
        trip_elem.set("to", trip['to'])
        trip_elem.set("departLane", trip['departLane'])
        trip_elem.set("type", "veh_passenger")
        
        if trip['departSpeed'] == 'max':
            trip_elem.set("departSpeed", "max")
    
    # Pretty print
    rough_string = ET.tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Remove empty lines
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

def main():
    print("Reading network file...")
    edges = read_network_edges('aksshayt.net.xml')
    print(f"Found {len(edges)} edges")
    
    print("Generating trips...")
    trips = generate_trips(edges, num_trips=150, duration=3600)
    print(f"Generated {len(trips)} trips")
    
    print("Creating route file...")
    create_route_file(trips, 'aksshayt.rou.xml')
    print("Route file created: aksshayt.rou.xml")
    
    print("\nRoute file contains:")
    print(f"- {len(trips)} vehicle trips")
    print("- 1 hour simulation duration (3600 seconds)")
    print("- Random departure times")
    print("- Passenger vehicles only")
    print("- Fringe factor applied for realistic traffic")

if __name__ == "__main__":
    main()
