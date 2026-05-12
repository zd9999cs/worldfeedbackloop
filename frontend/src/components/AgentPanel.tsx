import { useState, useCallback } from 'react';
import { useStore } from '../store';
import { fetchAgentPopulation, initializeAgents } from '../api';

export default function AgentPanel() {
  const model = useStore((s) => s.model);
  const agentPopulation = useStore((s) => s.agentPopulation);
  const setAgentPopulation = useStore((s) => s.setAgentPopulation);
  const [activeTab, setActiveTab] = useState<'templates' | 'population'>('templates');

  const handleInitialize = useCallback(async () => {
    await initializeAgents();
    const agents = await fetchAgentPopulation();
    setAgentPopulation(agents);
  }, [setAgentPopulation]);

  const templates = model?.agent_templates || {};

  return (
    <div>
      <div className="section-header">
        <h3>Agents</h3>
        <button className="btn-icon" onClick={handleInitialize}>
          INIT
        </button>
      </div>

      <div className="tab-row">
        <button className={`tab-btn ${activeTab === 'templates' ? 'active' : ''}`} onClick={() => setActiveTab('templates')}>
          Templates
        </button>
        <button className={`tab-btn ${activeTab === 'population' ? 'active' : ''}`} onClick={() => setActiveTab('population')}>
          Population
        </button>
      </div>

      {activeTab === 'templates' && (
        <div className="section-body" style={{ maxHeight: 240, overflowY: 'auto' }}>
          {Object.entries(templates).length === 0 && (
            <div className="empty-state" style={{ padding: 24 }}>No agent templates in model</div>
          )}
          {Object.entries(templates).map(([name, tmpl]: [string, any]) => (
            <div key={name} className="agent-card">
              <div className="agent-card-title">{name}</div>
              <div className="agent-card-meta">
                Count: {tmpl.count} · Topology: {tmpl.topology}<br />
                Reads: [{tmpl.reads?.join(', ') || 'none'}]<br />
                Rules: {tmpl.decision_rules?.length || 0}
                {tmpl.internal_stocks && <> · Internal: {Object.keys(tmpl.internal_stocks).join(', ')}</>}
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'population' && (
        <div className="section-body" style={{ maxHeight: 240, overflowY: 'auto' }}>
          {agentPopulation.length === 0 && (
            <div className="empty-state" style={{ padding: 24 }}>Click INIT to create agents</div>
          )}
          {agentPopulation.map((agent, i) => (
            <div key={i} className="agent-card" style={{ marginBottom: 2, padding: '4px 8px' }}>
              <span className="agent-card-title" style={{ marginRight: 8 }}>{agent.template}</span>
              {Object.entries(agent.state).map(([k, v]) => (
                <span key={k} style={{ fontSize: 10, color: '#78788a', marginLeft: 6 }}>
                  {k}: {Number(v).toFixed(3)}
                </span>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
