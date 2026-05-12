import { useEffect, useRef, useCallback } from 'react';
import cytoscape, { Core, EventObject } from 'cytoscape';
import { useStore } from '../store';
import { fetchModel } from '../api';
import type { GraphNode, GraphEdge } from '../types';

const SUBSYSTEM_COLORS: Record<string, string> = {
  profit: '#d62728',
  energy_atmosphere: '#2ca02c',
  energy_politics: '#ff7f0e',
  revolt: '#9467bd',
  knowledge: '#1f77b4',
};

function buildCyElements(model: any, showAgents: boolean) {
  const elements: cytoscape.ElementDefinition[] = [];

  // SD stocks
  for (const [name, spec] of Object.entries(model.stocks || {}) as [string, any][]) {
    const color = SUBSYSTEM_COLORS[spec.subsystem] || '#888';
    elements.push({
      data: { id: name, label: name.replace(/_/g, '\n'), kind: 'stock', subsystem: spec.subsystem },
      style: { 'background-color': color, shape: 'rectangle', width: 80, height: 40 },
    });
  }

  // SD auxiliaries
  for (const [name, spec] of Object.entries(model.auxiliaries || {}) as [string, any][]) {
    const color = SUBSYSTEM_COLORS[spec.subsystem] || '#888';
    elements.push({
      data: { id: name, label: name.replace(/_/g, '\n'), kind: 'auxiliary', subsystem: spec.subsystem, operator: spec.operator },
      style: { 'background-color': color, shape: 'ellipse', width: 70, height: 35 },
    });
    // Edges from inputs
    for (const [src, pol] of Object.entries(spec.inputs || {}) as [string, string][]) {
      elements.push({
        data: {
          id: `${src}->${name}`,
          source: src,
          target: name,
          polarity: pol === 'negative' ? 'negative' : 'positive',
          role: 'input',
        },
      });
    }
  }

  // Stock flows
  for (const [name, spec] of Object.entries(model.stocks || {}) as [string, any][]) {
    for (const inflow of spec.inflows || []) {
      elements.push({
        data: { id: `${inflow}->${name}`, source: inflow, target: name, polarity: 'positive', role: 'inflow' },
      });
    }
    for (const outflow of spec.outflows || []) {
      elements.push({
        data: { id: `${name}->${outflow}`, source: outflow, target: name, polarity: 'negative', role: 'outflow' },
      });
    }
  }

  // Agent templates (bridge view)
  if (showAgents && model.agent_templates) {
    for (const [name, tmpl] of Object.entries(model.agent_templates) as [string, any][]) {
      elements.push({
        data: { id: `agent:${name}`, label: name, kind: 'agent_template', subsystem: '' },
        style: { 'background-color': '#17becf', shape: 'hexagon', width: 80, height: 46 },
        classes: 'agent-node',
      });
      // Reads: SD var -> agent
      for (const readVar of tmpl.reads || []) {
        elements.push({
          data: { id: `${readVar}->agent:${name}`, source: readVar, target: `agent:${name}`, polarity: 'positive', role: 'read' },
          classes: 'agent-edge',
        });
      }
      // Writes: agent -> SD aux
      for (const [, writeSpec] of Object.entries(tmpl.writes || {}) as [string, any][]) {
        elements.push({
          data: { id: `agent:${name}->${writeSpec.target}`, source: `agent:${name}`, target: writeSpec.target, polarity: 'positive', role: 'write' },
          classes: 'agent-edge',
        });
      }
    }
  }

  return elements;
}

export default function GraphEditor() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const layout = useStore((s) => s.layout);
  const setLayout = useStore((s) => s.setLayout);
  const model = useStore((s) => s.model);
  const showAgentBridge = useStore((s) => s.showAgentBridge);

  useEffect(() => {
    if (!containerRef.current || cyRef.current) return;

    const cy = cytoscape({
      container: containerRef.current,
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '8px',
            color: '#fff',
            'text-wrap': 'wrap',
          },
        },
        {
          selector: 'edge[role="input"]',
          style: { 'line-color': '#2a7', 'target-arrow-color': '#2a7', 'target-arrow-shape': 'triangle', width: 1.5 },
        },
        {
          selector: 'edge[role="input"][polarity="negative"]',
          style: { 'line-color': '#c33', 'target-arrow-color': '#c33', 'line-style': 'dashed' },
        },
        {
          selector: 'edge[role="inflow"]',
          style: { 'line-color': '#2a7', 'target-arrow-color': '#2a7', 'target-arrow-shape': 'triangle', width: 2 },
        },
        {
          selector: 'edge[role="outflow"]',
          style: { 'line-color': '#c33', 'target-arrow-color': '#c33', 'target-arrow-shape': 'triangle', 'line-style': 'dashed', width: 2 },
        },
        {
          selector: '.agent-node',
          style: { 'border-width': 2, 'border-color': '#17becf' },
        },
        {
          selector: '.agent-edge',
          style: { 'line-color': '#1f77b4', 'target-arrow-color': '#1f77b4', 'target-arrow-shape': 'triangle', 'line-style': 'dotted', width: 1 },
        },
      ],
      layout: { name: 'cose', animate: false, nodeRepulsion: 8000 },
      wheelSensitivity: 0.3,
    });

    cyRef.current = cy;
    return () => { cy.destroy(); cyRef.current = null; };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !model) return;

    const elements = buildCyElements(model, showAgentBridge);
    cy.elements().remove();
    cy.add(elements);

    // Restore saved positions or auto-layout
    if (Object.keys(layout).length > 0) {
      cy.nodes().forEach((node) => {
        const pos = layout[node.id()];
        if (pos) node.position(pos);
      });
    } else {
      cy.layout({ name: 'cose', animate: true, nodeRepulsion: 8000 }).run();
    }

    // Save positions on drag end
    cy.on('dragfree', () => {
      const positions: Record<string, { x: number; y: number }> = {};
      cy.nodes().forEach((node) => {
        positions[node.id()] = { ...node.position() };
      });
      setLayout(positions);
    });
  }, [model, showAgentBridge]);

  const handleReload = useCallback(async () => {
    const m = await fetchModel();
    useStore.getState().setModel(m);
  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div style={{ position: 'absolute', top: 8, right: 8, zIndex: 10, display: 'flex', gap: 8 }}>
        <button onClick={handleReload} style={{ padding: '4px 12px', background: '#333', color: '#ccc', border: '1px solid #555', borderRadius: 4, cursor: 'pointer' }}>
          Reload
        </button>
      </div>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
