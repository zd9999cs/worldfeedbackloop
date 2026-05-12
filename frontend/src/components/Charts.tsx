import { useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useStore } from '../store';

const DEFAULT_VARS = [
  'population', 'capital_stock', 'rate_of_profit',
  'oil_stock', 'oil_price', 'co2',
  'food_prices', 'biofuel_use',
  'world_tension', 'youth_unemployment',
  'graduate_pop', 'rate_of_revolutions',
];

const CHART_COLORS = ['#3dd68c', '#e8a830', '#4da8da', '#e84840', '#9467bd', '#17becf'];

function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.round((p / 100) * (sorted.length - 1))];
}

export default function Charts() {
  const simResult = useStore((s) => s.simResult);
  const [selectedVars, setSelectedVars] = useState<string[]>(DEFAULT_VARS.slice(0, 6));
  const [showVarPicker, setShowVarPicker] = useState(false);

  const chartData = useMemo(() => {
    if (!simResult) return null;

    if (simResult.mode === 'deterministic' && simResult.t) {
      const t = simResult.t;
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
        if (simResult.ensemble) {
          for (const v of selectedVars) {
            const allVals = simResult.ensemble.map((m) => m.stocks?.[v]?.[i] ?? m.auxiliaries?.[v]?.[i] ?? 0);
            point[`${v}_p10`] = percentile(allVals, 10);
            point[`${v}_p50`] = percentile(allVals, 50);
            point[`${v}_p90`] = percentile(allVals, 90);
          }
        }
        return point;
      });
    }

    return null;
  }, [simResult, selectedVars]);

  const allVarNames = useMemo(() => {
    if (!simResult) return [];
    const names = new Set<string>();
    if (simResult.stocks) Object.keys(simResult.stocks).forEach((n) => names.add(n));
    if (simResult.auxiliaries) Object.keys(simResult.auxiliaries).forEach((n) => names.add(n));
    if (simResult.ensemble?.[0]) {
      Object.keys(simResult.ensemble[0].stocks || {}).forEach((n) => names.add(n));
      Object.keys(simResult.ensemble[0].auxiliaries || {}).forEach((n) => names.add(n));
    }
    return [...names].sort();
  }, [simResult]);

  return (
    <div>
      <div className="section-header">
        <h3>Trajectories</h3>
        <button className="btn-icon" onClick={() => setShowVarPicker(!showVarPicker)}>
          {showVarPicker ? 'CLOSE' : 'VARIABLES'}
        </button>
      </div>

      {showVarPicker && (
        <div className="var-picker">
          {allVarNames.map((name) => {
            const sel = selectedVars.includes(name);
            return (
              <label key={name} className={`var-chip ${sel ? 'selected' : ''}`} onClick={() => {
                setSelectedVars((prev) =>
                  prev.includes(name) ? prev.filter((v) => v !== name) : [...prev, name]
                );
              }}>
                {name}
              </label>
            );
          })}
        </div>
      )}

      <div className="section-body">
        {chartData && chartData.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {selectedVars.map((varName, idx) => (
              <div key={varName} className="mini-chart">
                <div className="mini-chart-label">
                  {varName.replace(/_/g, ' ')}
                </div>
                <ResponsiveContainer width="100%" height={110}>
                  <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a1a26" />
                    <XAxis dataKey="t" tick={{ fontSize: 9, fill: '#4a4a58', fontFamily: '"JetBrains Mono", monospace' }} axisLine={{ stroke: '#1a1a26' }} tickLine={false} />
                    <YAxis tick={{ fontSize: 9, fill: '#4a4a58', fontFamily: '"JetBrains Mono", monospace' }} width={56} axisLine={false} tickLine={false} />
                    <Tooltip
                      contentStyle={{
                        background: '#111118',
                        border: '1px solid #3a3420',
                        borderRadius: 2,
                        fontFamily: '"JetBrains Mono", monospace',
                        fontSize: 10,
                        color: '#d4d4dc',
                      }}
                      labelFormatter={(v) => `t = ${Number(v).toFixed(0)}`}
                    />
                    <Line
                      type="monotone"
                      dataKey={varName}
                      stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                      dot={false}
                      strokeWidth={1.5}
                    />
                    {simResult?.mode === 'stochastic' && (
                      <>
                        <Line
                          type="monotone"
                          dataKey={`${varName}_p10`}
                          stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                          strokeWidth={0.5}
                          dot={false}
                          strokeDasharray="2 3"
                          opacity={0.35}
                        />
                        <Line
                          type="monotone"
                          dataKey={`${varName}_p90`}
                          stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                          strokeWidth={0.5}
                          dot={false}
                          strokeDasharray="2 3"
                          opacity={0.35}
                        />
                      </>
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">Run a simulation to see trajectories</div>
        )}
      </div>
    </div>
  );
}
