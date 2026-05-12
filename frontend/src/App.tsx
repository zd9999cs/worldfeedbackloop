import { useEffect } from 'react';
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
  const setGraph = useStore((s) => s.setGraph);
  const showAgentBridge = useStore((s) => s.showAgentBridge);

  useEffect(() => {
    fetchModel().then(setModel);
    fetchLoops().then((loops) => console.log('Loops:', loops.length));
  }, []);

  return (
    <div className="app">
      <div className="top-bar">
        <h1>World Feedback Loop</h1>
        <ScenarioManager />
      </div>
      <div className="main-area">
        <div className="left-panel">
          <GraphEditor />
        </div>
        <div className="right-panel">
          <ControlPanel />
          <Charts />
          {showAgentBridge && <AgentPanel />}
        </div>
      </div>
    </div>
  );
}
