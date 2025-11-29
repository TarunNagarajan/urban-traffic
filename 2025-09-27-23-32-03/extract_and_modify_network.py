#!/usr/bin/env python3
"""
Script to extract the SUMO network file and add custom traffic signals
"""

import gzip
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re

def extract_network():
    """Extract the compressed network file"""
    with gzip.open('osm.net.xml.gz', 'rt', encoding='utf-8') as f:
        content = f.read()
    
    with open('osm.net.xml', 'w', encoding='utf-8') as f:
        f.write(content)
    
    return content

def find_suitable_junctions(network_content):
    """Find junctions that could have traffic signals"""
    # Look for junction elements in the network
    junctions = re.findall(r'<junction id="([^"]+)"[^>]*type="([^"]+)"[^>]*>', network_content)
    
    # Filter for priority junctions (suitable for traffic lights)
    priority_junctions = [j for j in junctions if j[1] == 'priority']
    
    print(f"Found {len(junctions)} total junctions")
    print(f"Found {len(priority_junctions)} priority junctions suitable for traffic signals")
    
    return priority_junctions[:3]  # Return first 3 suitable junctions

def add_traffic_signals(network_content, junction_ids):
    """Add traffic signal definitions to the network"""
    
    # Parse the XML
    root = ET.fromstring(network_content)
    
    # Find the junctions section
    junctions_elem = root.find('junctions')
    if junctions_elem is None:
        print("No junctions section found!")
        return network_content
    
    # Add traffic signals for the selected junctions
    for i, junction_id in enumerate(junction_ids):
        # Find the junction element
        junction = None
        for j in junctions_elem.findall('junction'):
            if j.get('id') == junction_id:
                junction = j
                break
        
        if junction is not None:
            # Add traffic light type
            junction.set('type', 'traffic_light')
            
            # Add traffic light program
            tl_program = ET.SubElement(junction, 'tlLogic')
            tl_program.set('id', f'tl_{junction_id}')
            tl_program.set('type', 'static')
            tl_program.set('programID', '0')
            tl_program.set('offset', '0')
            
            # Add phases (red, yellow, green cycle)
            phases = [
                {'state': 'rrrrGGGGrrrrGGGG', 'duration': '31'},  # Green for main roads
                {'state': 'rrrryyyyrrrryyyy', 'duration': '3'},   # Yellow
                {'state': 'GGGGrrrrGGGGrrrr', 'duration': '31'},  # Green for side roads
                {'state': 'yyyyrrrryyyyrrrr', 'duration': '3'}    # Yellow
            ]
            
            for phase in phases:
                phase_elem = ET.SubElement(tl_program, 'phase')
                phase_elem.set('duration', phase['duration'])
                phase_elem.set('state', phase['state'])
            
            print(f"Added traffic signal to junction {junction_id}")
    
    # Convert back to string
    rough_string = ET.tostring(root, encoding='unicode')
    
    # Pretty print
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Remove empty lines
    pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
    
    return pretty_xml

def main():
    print("Extracting SUMO network file...")
    network_content = extract_network()
    
    print("Finding suitable junctions for traffic signals...")
    suitable_junctions = find_suitable_junctions(network_content)
    
    if len(suitable_junctions) < 3:
        print(f"Warning: Only found {len(suitable_junctions)} suitable junctions")
        junction_ids = [j[0] for j in suitable_junctions]
    else:
        junction_ids = [j[0] for j in suitable_junctions[:3]]
    
    print(f"Adding traffic signals to junctions: {junction_ids}")
    
    modified_content = add_traffic_signals(network_content, junction_ids)
    
    # Save the modified network
    with open('osm_modified.net.xml', 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print("Modified network saved as 'osm_modified.net.xml'")
    print("Traffic signals added successfully!")

if __name__ == "__main__":
    main()
