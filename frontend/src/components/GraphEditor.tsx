import { useEffect, useRef, useCallback } from 'react';
import cytoscape, { Core } from 'cytoscape';
import { useStore } from '../store';
import { fetchModel } from '../api';

const SUBSYSTEM_COLORS: Record<string, string> = {
  profit: '#d62728',
  energy_atmosphere: '#2ca02c',
  energy_politics: '#ff7f0e',
  revolt: '#9467bd',
  knowledge: '#1f77b4',
};

function buildCyElements(model: any, showAgents: boolean) {
  const elements: cytoscape.ElementDefinition[] = [];

  for (const [name, spec] of Object.entries(model.stocks || {}) as [string, any][]) {
    const color = SUBSYSTEM_COLORS[spec.subsystem] || '#888';
    elements.push({
      data: { id: name, label: name.replace(/_/g, '\n'), kind: 'stock', subsystem: spec.subsystem },
      style: { 'background-color': color, shape: 'rectangle', width: 80, height: 40 },
    });
  }

  for (const [name, spec] of Object.entries(model.auxiliaries || {}) as [string, any][]) {
    const color = SUBSYSTEM_COLORS[spec.subsystem] || '#888';
    elements.push({
      data: { id: name, label: name.replace(/_/g, '\n'), kind: 'auxiliary', subsystem: spec.subsystem, operator: spec.operator },
      style: { 'background-color': color, shape: 'ellipse', width: 70, height: 35 },
    });
    for (const [src, pol] of Object.entries(spec.inputs || {}) as [string, string][]) {
      elements.push({
        data: {
          id: `${src}->${name}`, source: src, target: name,
          polarity: pol === 'negative' ? 'negative' : 'positive',
          role: 'input',
        },
      });
    }
  }

  for (const [name, spec] of Object.entries(model.stocks || {}) as [string, any][]) {
    for (const inflow of spec.inflows || []) {
      elements.push({
        data: { id: `${inflow}->${name}`, source: inflow, target: name, polarity: 'positive', role: 'inflow' },
      });
    }
    for (const outflow of spec.outflows || []) {
      elements.push({
        data: { id: `${outflow}->${name}`, source: outflow, target: name, polarity: 'negative', role: 'outflow' },
      });
    }
  }

  if (showAgents && model.agent_templates) {
    for (const [name, tmpl] of Object.entries(model.agent_templates) as [string, any][]) {
      elements.push({
        data: { id: `agent:${name}`, label: name, kind: 'agent_template', subsystem: '' },
        style: { 'background-color': '#17becf', shape: 'hexagon', width: 80, height: 46 },
        classes: 'agent-node',
      });
      for (const readVar of tmpl.reads || []) {
        elements.push({
          data: { id: `${readVar}->agent:${name}`, source: readVar, target: `agent:${name}`, polarity: 'positive', role: 'read' },
          classes: 'agent-edge',
        });
      }
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
            'font-size': '7px',
            'font-family': '"JetBrains Mono", monospace',
            color: '#e0e0e0',
            'text-wrap': 'wrap',
            'text-outline-width': 1,
            'text-outline-color': '#0c0c14',
            'border-width': 1.2,
            'border-color': '#1a1a26',
          },
        },
        {
          selector: 'node:selected',
          style: { 'border-color': '#e8a830', 'border-width': 2 },
        },
        {
          selector: 'edge[role="input"]',
          style: {
            'line-color': '#2a6a3c',
            'target-arrow-color': '#2a6a3c',
            'target-arrow-shape': 'triangle',
            width: 1.2,
            'curve-style': 'bezier',
          },
        },
        {
          selector: 'edge[role="input"][polarity="negative"]',
          style: {
            'line-color': '#8a2a2a',
            'target-arrow-color': '#8a2a2a',
            'line-style': 'dashed',
          },
        },
        {
          selector: 'edge[role="inflow"]',
          style: {
            'line-color': '#3dd68c',
            'target-arrow-color': '#3dd68c',
            'target-arrow-shape': 'triangle',
            width: 2,
          },
        },
        {
          selector: 'edge[role="outflow"]',
          style: {
            'line-color': '#e84840',
            'target-arrow-color': '#e84840',
            'target-arrow-shape': 'triangle',
            'line-style': 'dashed',
            width: 2,
          },
        },
        {
          selector: '.agent-node',
          style: {
            'border-width': 1.5,
            'border-color': '#17becf',
            'border-opacity': 0.7,
            'background-opacity': 0.9,
          },
        },
        {
          selector: '.agent-edge',
          style: {
            'line-color': '#4da8da',
            'target-arrow-color': '#4da8da',
            'target-arrow-shape': 'triangle',
            'line-style': 'dotted',
            width: 1,
            'line-opacity': 0.6,
          },
        },
      ],
      layout: { name: 'cose', animate: false, nodeRepulsion: 10000, idealEdgeLength: 120 },
      wheelSensitivity: 0.3,
      minZoom: 0.15,
      maxZoom: 3,
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

    if (Object.keys(layout).length > 0) {
      cy.nodes().forEach((node) => {
        const pos = layout[node.id()];
        if (pos) node.position(pos);
      });
    } else {
      cy.layout({ name: 'cose', animate: true, nodeRepulsion: 10000, idealEdgeLength: 120 }).run();
    }

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

  const handleFit = useCallback(() => {
    cyRef.current?.fit(undefined, 40);
  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div className="graph-toolbar">
        <button className="btn-tool" onClick={handleReload} title="Reload model from disk">
          RELOAD
        </button>
        <button className="btn-tool" onClick={handleFit} title="Fit graph to view">
          FIT
        </button>
      </div>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
