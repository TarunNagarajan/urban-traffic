#!/usr/bin/env python3
"""
Create a SUMO network file with custom traffic signals
"""

import gzip
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def create_network_with_traffic_signals():
    """Create a network file with 3 custom traffic signals"""
    
    # Create the root element
    root = ET.Element("net")
    root.set("version", "1.24.0")
    root.set("junctionCornerDetail", "5")
    root.set("limitTurnSpeed", "5.5")
    
    # Add location element
    location = ET.SubElement(root, "location")
    location.set("netOffset", "0.00,0.00")
    location.set("convBoundary", "0.00,0.00,1000.00,1000.00")
    location.set("origBoundary", "0.00,0.00,1000.00,1000.00")
    location.set("projParameter", "!")
    
    # Add edge types
    edge_types = ET.SubElement(root, "edgeTypes")
    
    # Highway type
    highway_type = ET.SubElement(edge_types, "edgeType")
    highway_type.set("id", "highway.motorway")
    highway_type.set("priority", "13")
    highway_type.set("numLanes", "2")
    highway_type.set("speed", "44.44")
    
    # Primary road type
    primary_type = ET.SubElement(edge_types, "edgeType")
    primary_type.set("id", "highway.primary")
    primary_type.set("priority", "11")
    primary_type.set("numLanes", "2")
    primary_type.set("speed", "33.33")
    
    # Secondary road type
    secondary_type = ET.SubElement(edge_types, "edgeType")
    secondary_type.set("id", "highway.secondary")
    secondary_type.set("priority", "9")
    secondary_type.set("numLanes", "1")
    secondary_type.set("speed", "22.22")
    
    # Add edges
    edges = ET.SubElement(root, "edges")
    
    # Create a simple 4-way intersection with 8 edges
    edge_data = [
        ("edge1", "j1", "j2", "highway.primary", 2, 33.33),
        ("edge2", "j2", "j1", "highway.primary", 2, 33.33),
        ("edge3", "j2", "j3", "highway.primary", 2, 33.33),
        ("edge4", "j3", "j2", "highway.primary", 2, 33.33),
        ("edge5", "j4", "j2", "highway.secondary", 1, 22.22),
        ("edge6", "j2", "j4", "highway.secondary", 1, 22.22),
        ("edge7", "j2", "j5", "highway.secondary", 1, 22.22),
        ("edge8", "j5", "j2", "highway.secondary", 1, 22.22)
    ]
    
    for edge_id, from_j, to_j, edge_type, lanes, speed in edge_data:
        edge = ET.SubElement(edges, "edge")
        edge.set("id", edge_id)
        edge.set("from", from_j)
        edge.set("to", to_j)
        edge.set("priority", "11" if "primary" in edge_type else "9")
        edge.set("type", edge_type)
        
        # Add lanes
        for i in range(lanes):
            lane = ET.SubElement(edge, "lane")
            lane.set("id", f"{edge_id}_{i}")
            lane.set("index", str(i))
            lane.set("speed", str(speed))
            lane.set("length", "100.00")
            lane.set("shape", f"{i*3.5},{i*3.5} {i*3.5+100},{i*3.5+100}")
    
    # Add junctions
    junctions = ET.SubElement(root, "junctions")
    
    # Junction 1 - Traffic Signal 1
    j1 = ET.SubElement(junctions, "junction")
    j1.set("id", "j1")
    j1.set("type", "traffic_light")
    j1.set("x", "0.00")
    j1.set("y", "0.00")
    j1.set("incLanes", "edge1_0 edge1_1")
    j1.set("intLanes", "")
    j1.set("shape", "0.00,0.00 0.00,10.00 10.00,10.00 10.00,0.00")
    
    # Traffic light program for j1
    tl1 = ET.SubElement(j1, "tlLogic")
    tl1.set("id", "tl_j1")
    tl1.set("type", "static")
    tl1.set("programID", "0")
    tl1.set("offset", "0")
    
    # Phases for j1
    phases1 = [
        {"duration": "31", "state": "GG"},
        {"duration": "3", "state": "yy"},
        {"duration": "31", "state": "rr"},
        {"duration": "3", "state": "rr"}
    ]
    
    for phase in phases1:
        phase_elem = ET.SubElement(tl1, "phase")
        phase_elem.set("duration", phase["duration"])
        phase_elem.set("state", phase["state"])
    
    # Junction 2 - Main intersection with Traffic Signal 2
    j2 = ET.SubElement(junctions, "junction")
    j2.set("id", "j2")
    j2.set("type", "traffic_light")
    j2.set("x", "100.00")
    j2.set("y", "100.00")
    j2.set("incLanes", "edge1_0 edge1_1 edge2_0 edge2_1 edge3_0 edge3_1 edge4_0 edge4_1 edge5_0 edge6_0 edge7_0 edge8_0")
    j2.set("intLanes", "")
    j2.set("shape", "100.00,100.00 100.00,110.00 110.00,110.00 110.00,100.00")
    
    # Traffic light program for j2 (4-way intersection)
    tl2 = ET.SubElement(j2, "tlLogic")
    tl2.set("id", "tl_j2")
    tl2.set("type", "static")
    tl2.set("programID", "0")
    tl2.set("offset", "0")
    
    # Phases for j2 (4-way intersection)
    phases2 = [
        {"duration": "31", "state": "GGGGrrrrGGGGrrrr"},
        {"duration": "3", "state": "yyyyrrrryyyyrrrr"},
        {"duration": "31", "state": "rrrrGGGGrrrrGGGG"},
        {"duration": "3", "state": "rrrryyyyrrrryyyy"}
    ]
    
    for phase in phases2:
        phase_elem = ET.SubElement(tl2, "phase")
        phase_elem.set("duration", phase["duration"])
        phase_elem.set("state", phase["state"])
    
    # Junction 3 - Traffic Signal 3
    j3 = ET.SubElement(junctions, "junction")
    j3.set("id", "j3")
    j3.set("type", "traffic_light")
    j3.set("x", "200.00")
    j3.set("y", "200.00")
    j3.set("incLanes", "edge3_0 edge3_1")
    j3.set("intLanes", "")
    j3.set("shape", "200.00,200.00 200.00,210.00 210.00,210.00 210.00,200.00")
    
    # Traffic light program for j3
    tl3 = ET.SubElement(j3, "tlLogic")
    tl3.set("id", "tl_j3")
    tl3.set("type", "static")
    tl3.set("programID", "0")
    tl3.set("offset", "0")
    
    # Phases for j3
    phases3 = [
        {"duration": "31", "state": "GG"},
        {"duration": "3", "state": "yy"},
        {"duration": "31", "state": "rr"},
        {"duration": "3", "state": "rr"}
    ]
    
    for phase in phases3:
        phase_elem = ET.SubElement(tl3, "phase")
        phase_elem.set("duration", phase["duration"])
        phase_elem.set("state", phase["state"])
    
    # Junction 4 - Regular junction
    j4 = ET.SubElement(junctions, "junction")
    j4.set("id", "j4")
    j4.set("type", "priority")
    j4.set("x", "50.00")
    j4.set("y", "150.00")
    j4.set("incLanes", "edge5_0")
    j4.set("intLanes", "")
    j4.set("shape", "50.00,150.00 50.00,160.00 60.00,160.00 60.00,150.00")
    
    # Junction 5 - Regular junction
    j5 = ET.SubElement(junctions, "junction")
    j5.set("id", "j5")
    j5.set("type", "priority")
    j5.set("x", "150.00")
    j5.set("y", "50.00")
    j5.set("incLanes", "edge8_0")
    j5.set("intLanes", "")
    j5.set("shape", "150.00,50.00 150.00,60.00 160.00,60.00 160.00,50.00")
    
    # Add connections
    connections = ET.SubElement(root, "connections")
    
    # Add some basic connections
    conn_data = [
        ("j1", "edge1", "j2", "edge2", "edge1_0", "edge2_0"),
        ("j2", "edge3", "j3", "edge4", "edge3_0", "edge4_0"),
        ("j2", "edge5", "j4", "edge6", "edge5_0", "edge6_0"),
        ("j2", "edge7", "j5", "edge8", "edge7_0", "edge8_0")
    ]
    
    for from_j, from_edge, to_j, to_edge, from_lane, to_lane in conn_data:
        conn = ET.SubElement(connections, "connection")
        conn.set("from", from_edge)
        conn.set("to", to_edge)
        conn.set("fromLane", from_lane)
        conn.set("toLane", to_lane)
        conn.set("via", "")
        conn.set("dir", "s")
        conn.set("state", "M")
    
    # Convert to string and pretty print
    rough_string = ET.tostring(root, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Remove empty lines
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    return pretty_xml

def main():
    print("Creating SUMO network with 3 custom traffic signals...")
    
    network_xml = create_network_with_traffic_signals()
    
    # Save the network file
    with open('osm_modified.net.xml', 'w', encoding='utf-8') as f:
        f.write(network_xml)
    
    print("Network created successfully!")
    print("File saved as: osm_modified.net.xml")
    print("\nTraffic signals added:")
    print("1. Junction j1 - 2-phase signal")
    print("2. Junction j2 - 4-phase signal (main intersection)")
    print("3. Junction j3 - 2-phase signal")
    print("\nThe network includes:")
    print("- 8 edges forming a road network")
    print("- 5 junctions (3 with traffic signals, 2 regular)")
    print("- Static traffic light programs with realistic timing")

if __name__ == "__main__":
    main()
