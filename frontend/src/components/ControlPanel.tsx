import { useState, useCallback } from 'react';
import { useStore } from '../store';
import { runSimulation, updateParameter } from '../api';

export default function ControlPanel() {
  const model = useStore((s) => s.model);
  const setSimResult = useStore((s) => s.setSimResult);
  const simRunning = useStore((s) => s.simRunning);
  const setSimRunning = useStore((s) => s.setSimRunning);

  const [mode, setMode] = useState<'deterministic' | 'stochastic'>('deterministic');
  const [nSteps, setNSteps] = useState(201);
  const [nEnsemble, setNEnsemble] = useState(10);
  const [seed, setSeed] = useState(42);
  const [activeTab, setActiveTab] = useState<'params' | 'run'>('params');

  const handleRun = useCallback(async () => {
    setSimRunning(true);
    try {
      const result = await runSimulation({ mode, n_steps: nSteps, n_ensemble: nEnsemble, seed });
      setSimResult(result);
    } finally {
      setSimRunning(false);
    }
  }, [mode, nSteps, nEnsemble, seed, setSimResult, setSimRunning]);

  const handleParamChange = useCallback(async (name: string, value: number) => {
    if (!model) return;
    await updateParameter(name, value);
    useStore.getState().setModel({
      ...model,
      parameters: { ...model.parameters, [name]: value },
    });
  }, [model]);

  if (!model) return <div className="empty-state">Loading model...</div>;

  const params = model.parameters || {};

  return (
    <div>
      <div className="section-header">
        <h3>Controls</h3>
      </div>

      <div className="tab-row">
        <button className={`tab-btn ${activeTab === 'params' ? 'active' : ''}`} onClick={() => setActiveTab('params')}>
          Parameters
        </button>
        <button className={`tab-btn ${activeTab === 'run' ? 'active' : ''}`} onClick={() => setActiveTab('run')}>
          Run
        </button>
      </div>

      {activeTab === 'params' && (
        <div className="section-body" style={{ maxHeight: 340, overflowY: 'auto' }}>
          {Object.entries(params).map(([name, value]) => {
            const v = Number(value);
            const min = v * 0.1;
            const max = v * 10;
            const step = (max - min) / 200;
            return (
              <div key={name} className="param-row">
                <div className="param-label">
                  <span className="param-name">{name}</span>
                  <span className="param-value">{v.toFixed(4)}</span>
                </div>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={v}
                  onChange={(e) => handleParamChange(name, parseFloat(e.target.value))}
                />
              </div>
            );
          })}
        </div>
      )}

      {activeTab === 'run' && (
        <div className="section-body">
          <div className="field-row">
            <label className="field-label">Simulation Mode</label>
            <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
              <option value="deterministic">Deterministic</option>
              <option value="stochastic">Stochastic</option>
            </select>
          </div>
          <div className="field-row">
            <label className="field-label">Time Steps</label>
            <input type="number" value={nSteps} onChange={(e) => setNSteps(parseInt(e.target.value) || 201)} />
          </div>
          {mode === 'stochastic' && (
            <>
              <div className="field-row">
                <label className="field-label">Ensemble Size</label>
                <input type="number" value={nEnsemble} onChange={(e) => setNEnsemble(parseInt(e.target.value) || 10)} />
              </div>
              <div className="field-row">
                <label className="field-label">Seed</label>
                <input type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value) || 42)} />
              </div>
            </>
          )}
          <button className="btn btn-primary" onClick={handleRun} disabled={simRunning}>
            {simRunning ? 'SIMULATING…' : 'RUN SIMULATION'}
          </button>
        </div>
      )}
    </div>
  );
}
