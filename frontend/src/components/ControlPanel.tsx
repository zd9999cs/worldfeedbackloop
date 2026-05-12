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

  if (!model) return <div style={{ padding: 16 }}>Loading model...</div>;

  const params = model.parameters || {};

  return (
    <div style={{ padding: 12, borderBottom: '1px solid #333' }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <button
          onClick={() => setActiveTab('params')}
          style={tabStyle(activeTab === 'params')}
        >
          Parameters
        </button>
        <button
          onClick={() => setActiveTab('run')}
          style={tabStyle(activeTab === 'run')}
        >
          Run
        </button>
      </div>

      {activeTab === 'params' && (
        <div style={{ maxHeight: 400, overflowY: 'auto' }}>
          {Object.entries(params).map(([name, value]) => {
            const v = Number(value);
            const min = v * 0.1;
            const max = v * 10;
            const step = (max - min) / 200;
            return (
              <div key={name} style={{ marginBottom: 8 }}>
                <label style={{ display: 'block', fontSize: 11, color: '#aaa', marginBottom: 2 }}>
                  {name}: {v.toFixed(4)}
                </label>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={v}
                  onChange={(e) => handleParamChange(name, parseFloat(e.target.value))}
                  style={{ width: '100%' }}
                />
              </div>
            );
          })}
        </div>
      )}

      {activeTab === 'run' && (
        <div>
          <div style={{ marginBottom: 8 }}>
            <label style={{ display: 'block', fontSize: 11, color: '#aaa', marginBottom: 2 }}>Mode</label>
            <select value={mode} onChange={(e) => setMode(e.target.value as any)}
              style={{ width: '100%', padding: 4, background: '#2a2a3e', color: '#ccc', border: '1px solid #555', borderRadius: 3 }}>
              <option value="deterministic">Deterministic</option>
              <option value="stochastic">Stochastic</option>
            </select>
          </div>
          <div style={{ marginBottom: 8 }}>
            <label style={{ display: 'block', fontSize: 11, color: '#aaa', marginBottom: 2 }}>Time steps</label>
            <input type="number" value={nSteps} onChange={(e) => setNSteps(parseInt(e.target.value) || 201)}
              style={{ width: '100%', padding: 4, background: '#2a2a3e', color: '#ccc', border: '1px solid #555', borderRadius: 3 }} />
          </div>
          {mode === 'stochastic' && (
            <>
              <div style={{ marginBottom: 8 }}>
                <label style={{ display: 'block', fontSize: 11, color: '#aaa', marginBottom: 2 }}>Ensemble size</label>
                <input type="number" value={nEnsemble} onChange={(e) => setNEnsemble(parseInt(e.target.value) || 10)}
                  style={{ width: '100%', padding: 4, background: '#2a2a3e', color: '#ccc', border: '1px solid #555', borderRadius: 3 }} />
              </div>
              <div style={{ marginBottom: 8 }}>
                <label style={{ display: 'block', fontSize: 11, color: '#aaa', marginBottom: 2 }}>Seed</label>
                <input type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
                  style={{ width: '100%', padding: 4, background: '#2a2a3e', color: '#ccc', border: '1px solid #555', borderRadius: 3 }} />
              </div>
            </>
          )}
          <button
            onClick={handleRun}
            disabled={simRunning}
            style={{
              width: '100%', padding: '8px 0', marginTop: 8,
              background: simRunning ? '#555' : '#2ca02c', color: '#fff',
              border: 'none', borderRadius: 4, cursor: simRunning ? 'default' : 'pointer',
              fontWeight: 600,
            }}
          >
            {simRunning ? 'Running...' : 'Run Simulation'}
          </button>
        </div>
      )}
    </div>
  );
}

function tabStyle(active: boolean): React.CSSProperties {
  return {
    flex: 1,
    padding: '6px 0',
    background: active ? '#2a2a3e' : 'transparent',
    color: active ? '#fff' : '#888',
    border: active ? '1px solid #555' : '1px solid transparent',
    borderRadius: 4,
    cursor: 'pointer',
    fontSize: 12,
    fontWeight: active ? 600 : 400,
  };
}
