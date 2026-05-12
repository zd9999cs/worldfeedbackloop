# World System Feedback Model

Executable extraction of the causal-loop diagram in
**Cockshott — "Main feedback loops in world system"**, packaged so you can
simulate the baseline scenario and extend the model with new loops.

## Files

| file | role |
|---|---|
| `model.yaml`              | Canonical model: stocks, auxiliaries, equations, parameters, polarities, operators |
| `simulator.py`            | Load + integrate + analyze. `WorldSystem` class is the public API |
| `run_baseline.py`         | Runs the baseline scenario, plots trajectories, draws the loop diagram |
| `extend_example.py`       | Adds an AI/automation feedback loop to demonstrate the extension pattern |
| `model_extended.yaml`     | The extended model serialized back to YAML |
| `baseline_trajectories.png`, `loop_diagram.png`, `extension_comparison.png` | Output figures |

## Quick start

```bash
pip install pyyaml scipy matplotlib networkx
python run_baseline.py
python extend_example.py
```

## Model summary

13 stocks, 31 auxiliaries, 21 feedback loops detected automatically
(7 reinforcing, 14 balancing). Subsystems mirror the five frames in the
original slide deck: **profit**, **energy/atmosphere**, **energy/politics**,
**revolt**, and **knowledge/technology**.

Baseline scenario (2020–2120) reproduces the central qualitative findings
of Cockshott's slides:

- **Falling rate of profit** — `rate_of_profit` declines from ≈0.28 to ≈0.14
  as capital accumulates faster than the labour pool grows, matching
  Cockshott's eq. (7), `R* = (n+g+δ)/λ`
- **Oil depletion** — `oil_stock` falls steeply through ≈2050
- **Climate–food coupling** — CO2 peaks ≈440 ppm, food prices triple
- **Biofuel saturation** — the "dramatic new element" displaces fossil fuel
  use as oil prices rise
- **Demographic peak** — population peaks ≈8 B around 2035, declines under
  food/climate stress

## Schema

### `stocks`
Integrated state variables. `dX/dt = Σ inflows − Σ outflows`.
```yaml
stocks:
  population:
    initial: 7.8e9
    units: people
    inflows:  [rate_of_birth]
    outflows: [rate_of_death]
    subsystem: profit
```

### `auxiliaries`
Computed each timestep. `operator` records which combiner node from the
diagram (× / + / ÷ / identity) the variable represents; `inputs` records
the polarity of each influence (the +/− labels in the original).
```yaml
rate_of_industrial_activity:
  equation: "(population / 7.8e9) * industrialisation"
  operator: multiply
  inputs:   {population: positive, industrialisation: positive}
  subsystem: profit
```

The graph extracted from `inputs` + stock `inflows`/`outflows` is exactly
the causal loop diagram. NetworkX cycle enumeration recovers the feedback
loops with their net polarity.

### Operators
| in diagram | yaml value | semantics |
|---|---|---|
| ×    | `multiply` | rate is a product of inputs |
| +    | `add`      | rate is a sum |
| ÷    | `divide`   | rate is a ratio |
| (none) | `identity` | passes a single input through |

## Extending the model

Two equivalent paths.

**1. Edit `model.yaml`** (canonical). Add new stocks/auxiliaries and
re-run; loops are re-detected automatically.

**2. Python API** (programmatic):
```python
from simulator import WorldSystem
ws = WorldSystem("model.yaml")

ws.set_parameter("ai_growth_from_knowledge", 0.03)
ws.add_stock("ai_capability", initial=0.05,
             inflows=["ai_growth"], outflows=["ai_attrition"],
             subsystem="knowledge")
ws.add_auxiliary("ai_growth",
    equation="ai_growth_from_knowledge * scientific_knowledge "
             "* max(0, rate_of_accumulation - 0.05) "
             "* max(0, 1 - ai_capability)",
    operator="multiply",
    inputs={"scientific_knowledge": "positive",
            "rate_of_accumulation":  "positive",
            "ai_capability":         "negative"},
    subsystem="knowledge")

ws.recompile()
res = ws.simulate()
ws.save_yaml("my_extended.yaml")
```

To couple a new variable into an existing one, edit the existing
auxiliary's equation in `ws.raw["auxiliaries"][name]["equation"]` and add
the new input to its `inputs` dict, then `ws.recompile()`. See
`extend_example.py` for a worked case.

## Equations — what's calibrated vs. illustrative

The **profit subsystem** follows Cockshott's derivation directly:
- `S = (1 − w) L`              (eq. 1)
- `dK/dt = λS − (g + δ)K`      (eq. 2)
- `R  = S / K`                  (eq. 4)

Other subsystems use plausible default functional forms parameterized for
adjustment. Particular tunable knobs:

| parameter | controls | currently |
|---|---|---|
| `co2_emission_factor`    | climate forcing per barrel    | calibrated to ≈2 ppm/yr at current fuel use |
| `oil_price_speed`        | how fast price tracks scarcity | mean-reverting toward `80 · stock_ref/stock` |
| `revolt_threshold`       | when revolutions trigger      | 0.4 — never hit in baseline; lower to study revolt dynamics |
| `youth_unemp_damping`    | how fast unemployment decays  | 0.05 — strong; lowering reveals the AI-displacement signal |
| `wage_share`             | w in profit eqs               | 0.60 — try 0.55 or 0.65 |
| `investment_share`       | λ                             | 0.50 |

## Analysis

```python
G = ws.graph()                    # NetworkX DiGraph (kind/subsystem/operator/polarity attrs)
loops = ws.find_loops(max_len=8)  # [{nodes, edges, polarity: reinforcing|balancing}]
```

## Known limitations & TODO

- Revolutions don't trigger in baseline — calibration choice, not a bug.
  Lower `revolt_threshold` or raise food-price coupling to see revolt
  dynamics.
- AI loop's unemployment effect is small under default damping; tune
  `youth_unemp_damping` and `ai_unemployment_pressure` in
  `extend_example.py` to amplify.
- The oil-stock floor and `oil_stock_ref / oil_stock` cap (currently 8)
  prevent runaway prices; review if simulating to extreme depletion.
- No stochastic forcing — wars/revolts are deterministic functions of
  state. Adding noise on `rate_of_big_wars` or `rate_of_revolutions`
  would change the qualitative picture.
