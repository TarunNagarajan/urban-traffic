
import React, { useRef, useEffect, useMemo } from 'react';
import * as d3 from 'd3';

type VehicleData = {
  simulation_time_s: number;
  vehicle_id: string;
  x_coord_m: number;
  y_coord_m: number;
  current_intersection_id: string | null;
  intersection_queue_length_veh: number | null;
  waiting_time_s: number;
};

type SumoMapProps = {
  vehicles: VehicleData[];
  width?: number;
  height?: number;
};

export function SumoMap({
  vehicles,
  width = 800,
  height = 600,
}: SumoMapProps): JSX.Element {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous render

    const g = svg.append('g');

    // Fetch and parse the network XML
    d3.xml('/aksshayt.net.xml').then(netXml => {
      const edges = Array.from(netXml.getElementsByTagName('edge'));
      const junctions = Array.from(netXml.getElementsByTagName('junction'));

      // A simple scaling function
      const scaleX = d3.scaleLinear().domain([0, 1000]).range([0, width]);
      const scaleY = d3.scaleLinear().domain([0, 1200]).range([0, height]);

      // Draw edges
      g.selectAll('.edge')
        .data(edges)
        .enter()
        .append('line')
        .attr('class', 'edge')
        .attr('x1', d => {
          const fromNode = junctions.find(j => j.getAttribute('id') === d.getAttribute('from'));
          return fromNode ? scaleX(parseFloat(fromNode.getAttribute('x') ?? '0')) : 0;
        })
        .attr('y1', d => {
          const fromNode = junctions.find(j => j.getAttribute('id') === d.getAttribute('from'));
          return fromNode ? scaleY(parseFloat(fromNode.getAttribute('y') ?? '0')) : 0;
        })
        .attr('x2', d => {
          const toNode = junctions.find(j => j.getAttribute('id') === d.getAttribute('to'));
          return toNode ? scaleX(parseFloat(toNode.getAttribute('x') ?? '0')) : 0;
        })
        .attr('y2', d => {
          const toNode = junctions.find(j => j.getAttribute('id') === d.getAttribute('to'));
          return toNode ? scaleY(parseFloat(toNode.getAttribute('y') ?? '0')) : 0;
        })
        .style('stroke', '#ccc')
        .style('stroke-width', 2);

      // Draw junctions
      g.selectAll('.junction')
        .data(junctions)
        .enter()
        .append('circle')
        .attr('class', 'junction')
        .attr('cx', d => scaleX(parseFloat(d.getAttribute('x') ?? '0')))
        .attr('cy', d => scaleY(parseFloat(d.getAttribute('y') ?? '0')))
        .attr('r', 3)
        .style('fill', 'blue');
    });
  }, [width, height]);

  useEffect(() => {
    const svg = d3.select(svgRef.current);
    const g = svg.select('g');

    const scaleX = d3.scaleLinear().domain([0, 1000]).range([0, width]);
    const scaleY = d3.scaleLinear().domain([0, 1200]).range([0, height]);

    const vehicleUpdate = g.selectAll('.vehicle').data(vehicles, (d: any) => d.vehicle_id);

    vehicleUpdate.enter()
      .append('circle')
      .attr('class', 'vehicle')
      .attr('r', 5)
      .style('fill', 'red')
      .merge(vehicleUpdate as any)
      .attr('cx', d => scaleX(d.x_coord_m))
      .attr('cy', d => scaleY(d.y_coord_m));

    vehicleUpdate.exit().remove();

  }, [vehicles, width, height]);

  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 8, overflow: 'hidden' }}>
      <svg ref={svgRef} width={width} height={height}></svg>
    </div>
  );
}
