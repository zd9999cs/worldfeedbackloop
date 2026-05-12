import { useState, useEffect, useCallback } from 'react';
import { useStore } from '../store';
import { fetchModel } from '../api';

export default function ScenarioManager() {
  const setModel = useStore((s) => s.setModel);
  const [scenarios, setScenarios] = useState<string[]>([]);

  const loadScenarios = useCallback(async () => {
    try {
      const resp = await fetch('/api/model/scenarios');
      if (resp.ok) {
        const data = await resp.json();
        setScenarios(data.scenarios || []);
      }
    } catch {
      // scenarios list endpoint may not exist yet — ignore
    }
  }, []);

  useEffect(() => { loadScenarios(); }, [loadScenarios]);

  const handleLoadScenario = useCallback(async (name: string) => {
    try {
      const resp = await fetch(`/api/model/scenarios/${encodeURIComponent(name)}`);
      if (resp.ok) {
        const model = await resp.json();
        setModel(model);
        await fetch(`/api/model/scenarios/${encodeURIComponent(name)}/activate`, { method: 'POST' });
      }
    } catch (err) {
      console.error('Failed to load scenario:', err);
    }
  }, [setModel]);

  const handleReload = useCallback(async () => {
    const model = await fetchModel();
    setModel(model);
    loadScenarios();
  }, [setModel, loadScenarios]);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <select
        onChange={(e) => { if (e.target.value) handleLoadScenario(e.target.value); }}
        style={{ padding: '2px 8px', background: '#2a2a3e', color: '#ccc', border: '1px solid #555', borderRadius: 3, fontSize: 12, minWidth: 160 }}
        defaultValue=""
      >
        <option value="" disabled>Scenarios...</option>
        {scenarios.map(name => (
          <option key={name} value={name}>{name}</option>
        ))}
      </select>
      <button onClick={handleReload}
        style={{ padding: '2px 10px', background: '#333', color: '#ccc', border: '1px solid #555', borderRadius: 3, cursor: 'pointer', fontSize: 11 }}>
        Refresh
      </button>
    </div>
  );
}
