import { useMemo, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { useStore } from '../store';

const DEFAULT_VARS = [
  'population', 'capital_stock', 'rate_of_profit',
  'oil_stock', 'oil_price', 'co2',
  'food_prices', 'biofuel_use',
  'world_tension', 'youth_unemployment',
  'graduate_pop', 'rate_of_revolutions',
];

function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = (p / 100) * (sorted.length - 1);
  return sorted[Math.round(idx)];
}

export default function Charts() {
  const simResult = useStore((s) => s.simResult);
  const [selectedVars, setSelectedVars] = useState<string[]>(DEFAULT_VARS.slice(0, 6));
  const [showVarPicker, setShowVarPicker] = useState(false);

  const chartData = useMemo(() => {
    if (!simResult) return null;

    if (simResult.mode === 'deterministic' && simResult.t) {
      const t = simResult.t!;
      const stocks = simResult.stocks || {};
      const auxs = simResult.auxiliaries || {};
      return t.map((time, i) => {
        const point: Record<string, number> = { t: time };
        for (const v of selectedVars) {
          point[v] = stocks[v]?.[i] ?? auxs[v]?.[i] ?? 0;
        }
        return point;
      });
    }

    if (simResult.mode === 'stochastic' && simResult.ensemble) {
      const member = simResult.ensemble[0];
      const t = member.t;
      const stocks = member.stocks || {};
      const auxs = member.auxiliaries || {};
      return t.map((time, i) => {
        const point: Record<string, number> = { t: time };
        for (const v of selectedVars) {
          point[v] = stocks[v]?.[i] ?? auxs[v]?.[i] ?? 0;
        }

        if (selectedVars.length > 0 && simResult.ensemble) {
          const firstVar = selectedVars[0];
          const allVals = simResult.ensemble.map(m => {
            return m.stocks?.[firstVar]?.[i] ?? m.auxiliaries?.[firstVar]?.[i] ?? 0;
          });
          point[`${firstVar}_p10`] = percentile(allVals, 10);
          point[`${firstVar}_p50`] = percentile(allVals, 50);
          point[`${firstVar}_p90`] = percentile(allVals, 90);
        }
        return point;
      });
    }

    return null;
  }, [simResult, selectedVars]);

  const allVarNames = useMemo(() => {
    if (!simResult) return [];
    const names = new Set<string>();
    if (simResult.stocks) Object.keys(simResult.stocks).forEach(n => names.add(n));
    if (simResult.auxiliaries) Object.keys(simResult.auxiliaries).forEach(n => names.add(n));
    if (simResult.ensemble?.[0]) {
      Object.keys(simResult.ensemble[0].stocks || {}).forEach(n => names.add(n));
      Object.keys(simResult.ensemble[0].auxiliaries || {}).forEach(n => names.add(n));
    }
    return [...names].sort();
  }, [simResult]);

  const COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'];

  return (
    <div style={{ padding: 12, flex: 1, minHeight: 300 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, fontWeight: 600 }}>Trajectories</h3>
        <button
          onClick={() => setShowVarPicker(!showVarPicker)}
          style={{ padding: '2px 8px', background: '#333', color: '#ccc', border: '1px solid #555', borderRadius: 3, cursor: 'pointer', fontSize: 11 }}
        >
          Variables
        </button>
      </div>

      {showVarPicker && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 8, maxHeight: 150, overflowY: 'auto' }}>
          {allVarNames.map(name => (
            <label key={name} style={{ fontSize: 10, display: 'flex', alignItems: 'center', gap: 2, padding: '2px 6px', background: selectedVars.includes(name) ? '#2a2a4e' : '#1a1a2e', borderRadius: 3, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={selectedVars.includes(name)}
                onChange={() => {
                  setSelectedVars(prev =>
                    prev.includes(name) ? prev.filter(v => v !== name) : [...prev, name]
                  );
                }}
                style={{ margin: 0 }}
              />
              {name}
            </label>
          ))}
        </div>
      )}

      {chartData && chartData.length > 0 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {selectedVars.map((varName, idx) => (
            <div key={varName} style={{ width: '100%', height: 120 }}>
              <div style={{ fontSize: 10, color: '#aaa', marginBottom: 2 }}>{varName}</div>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="t" tick={{ fontSize: 9, fill: '#888' }} />
                  <YAxis tick={{ fontSize: 9, fill: '#888' }} width={60} />
                  <Tooltip
                    contentStyle={{ background: '#1a1a2e', border: '1px solid #555', fontSize: 11 }}
                    labelFormatter={(v) => `t=${Number(v).toFixed(0)}`}
                  />
                  <Line type="monotone" dataKey={varName} stroke={COLORS[idx % COLORS.length]} dot={false} strokeWidth={1.5} />
                  {simResult?.mode === 'stochastic' && idx === 0 && (
                    <>
                      <Line type="monotone" dataKey={`${varName}_p10`} stroke={COLORS[0]} strokeWidth={0.5} dot={false} strokeDasharray="2 2" opacity={0.4} />
                      <Line type="monotone" dataKey={`${varName}_p90`} stroke={COLORS[0]} strokeWidth={0.5} dot={false} strokeDasharray="2 2" opacity={0.4} />
                    </>
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ color: '#666', fontSize: 12, textAlign: 'center', paddingTop: 60 }}>
          Run a simulation to see trajectories
        </div>
      )}
    </div>
  );
}
