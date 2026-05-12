import { useEffect, useState } from 'react';
import { useStore } from './store';
import { fetchModel, fetchLoops } from './api';
import GraphEditor from './components/GraphEditor';
import ControlPanel from './components/ControlPanel';
import Charts from './components/Charts';
import AgentPanel from './components/AgentPanel';
import ScenarioManager from './components/ScenarioManager';
import './App.css';

export default function App() {
  const setModel = useStore((s) => s.setModel);
  const showAgentBridge = useStore((s) => s.showAgentBridge);
  const setShowAgentBridge = useStore((s) => s.setShowAgentBridge);
  const simResult = useStore((s) => s.simResult);
  const simRunning = useStore((s) => s.simRunning);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    fetchModel()
      .then(setModel)
      .catch(() => setLoadError(true));
    fetchLoops().then((loops) => console.log('Loops:', loops.length));
  }, []);

  const statusLabel = simRunning
    ? 'SIMULATING'
    : simResult
      ? `READY · ${simResult.mode.toUpperCase()}`
      : 'IDLE';

  return (
    <div className="app">
      <div className="top-bar">
        <div className="top-bar-logo">
          <div className="logo-mark" />
          <h1>World Feedback<span>Loop</span></h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <button
            className={`btn-tool ${showAgentBridge ? 'active' : ''}`}
            onClick={() => setShowAgentBridge(!showAgentBridge)}
            title="Toggle agent bridge visibility"
          >
            AGENTS
          </button>
          <ScenarioManager />
        </div>
      </div>
      <div className="main-area">
        <div className="left-panel">
          <GraphEditor />
          <div className="graph-overlay" />
        </div>
        <div className="right-panel">
          {loadError && (
            <div className="empty-state">Backend unreachable — start the API server</div>
          )}
          <ControlPanel />
          <Charts />
          {showAgentBridge && <AgentPanel />}
        </div>
      </div>
      <div className="status-bar">
        <div className="status-indicator">
          <div className={`status-dot ${simRunning ? 'running' : simResult ? 'ready' : ''}`} />
          <span>{statusLabel}</span>
        </div>
        <div>
          {simResult
            ? `${(simResult as any).t?.length || simResult.ensemble?.[0]?.t?.length || 0} steps`
            : 'NO DATA'}
        </div>
        <div>
          {loadError ? 'API OFFLINE' : 'API ONLINE'}
        </div>
      </div>
    </div>
  );
}
