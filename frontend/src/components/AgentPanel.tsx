import { useState, useCallback } from 'react';
import { useStore } from '../store';
import { fetchAgentPopulation, initializeAgents, fetchAgentTemplates, createAgentTemplate, deleteAgentTemplate } from '../api';
import type { AgentTemplate, AgentInstance } from '../types';

export default function AgentPanel() {
  const model = useStore((s) => s.model);
  const agentPopulation = useStore((s) => s.agentPopulation);
  const setAgentPopulation = useStore((s) => s.setAgentPopulation);
  const [activeTab, setActiveTab] = useState<'templates' | 'population'>('templates');

  const handleInitialize = useCallback(async () => {
    const result = await initializeAgents();
    const agents = await fetchAgentPopulation();
    setAgentPopulation(agents);
  }, [setAgentPopulation]);

  const templates = model?.agent_templates || {};

  return (
    <div style={{ padding: 12, borderTop: '1px solid #333' }}>
      <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Agents</h3>

      <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
        <button onClick={() => setActiveTab('templates')}
          style={tabStyle(activeTab === 'templates')}>Templates</button>
        <button onClick={() => setActiveTab('population')}
          style={tabStyle(activeTab === 'population')}>Population</button>
        <button onClick={handleInitialize}
          style={{ padding: '4px 10px', background: '#1f77b4', color: '#fff', border: 'none', borderRadius: 3, cursor: 'pointer', fontSize: 11 }}>
          Init
        </button>
      </div>

      {activeTab === 'templates' && (
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {Object.entries(templates).length === 0 && (
            <div style={{ color: '#666', fontSize: 11 }}>No agent templates defined in the model.</div>
          )}
          {Object.entries(templates).map(([name, tmpl]) => (
            <div key={name} style={{ marginBottom: 8, padding: 8, background: '#2a2a3e', borderRadius: 4 }}>
              <div style={{ fontWeight: 600, fontSize: 12, color: '#17becf' }}>{name}</div>
              <div style={{ fontSize: 10, color: '#999', marginTop: 4 }}>
                Count: {tmpl.count} | Topology: {tmpl.topology}
              </div>
              <div style={{ fontSize: 10, color: '#999' }}>
                Reads: [{tmpl.reads?.join(', ')}]
              </div>
              <div style={{ fontSize: 10, color: '#999' }}>
                Rules: {tmpl.decision_rules?.length || 0}
              </div>
              {tmpl.internal_stocks && (
                <div style={{ fontSize: 10, color: '#999' }}>
                  Internal: {Object.keys(tmpl.internal_stocks).join(', ')}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {activeTab === 'population' && (
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {agentPopulation.length === 0 && (
            <div style={{ color: '#666', fontSize: 11 }}>Click "Init" to create agent instances.</div>
          )}
          {agentPopulation.map((agent, i) => (
            <div key={i} style={{ marginBottom: 4, padding: 4, background: '#2a2a3e', borderRadius: 3, fontSize: 10 }}>
              <span style={{ color: '#17becf', fontWeight: 600 }}>{agent.template}</span>
              {' '}
              {Object.entries(agent.state).map(([k, v]) => (
                <span key={k} style={{ marginLeft: 6, color: '#aaa' }}>{k}: {Number(v).toFixed(3)}</span>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function tabStyle(active: boolean): React.CSSProperties {
  return {
    padding: '4px 12px',
    background: active ? '#2a2a3e' : 'transparent',
    color: active ? '#fff' : '#888',
    border: active ? '1px solid #555' : '1px solid transparent',
    borderRadius: 3,
    cursor: 'pointer',
    fontSize: 11,
  };
}
