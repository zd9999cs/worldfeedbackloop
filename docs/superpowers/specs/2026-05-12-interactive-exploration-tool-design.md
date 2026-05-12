# Interactive Exploration Tool — Design Spec

## Purpose

Turn the World System Feedback Model into an interactive, game-like sandbox where the user can:
- Visualize the causal-loop diagram and trajectory charts
- Tweak parameters and the network structure itself in real time
- Run stochastic ensembles to reason under uncertainty
- Edit the model via both CLI (Claude Code, vim) and visual UI, with a shared canonical YAML source of truth
- Expand the model freely with new subsystems (e.g., monetary)
- Model heterogeneous interacting agents (banks, rebel factions, nations) that coexist with the system-dynamics environment

Audience: the user, for personal exploration, decision-aiding, and understanding structural dynamics behind events.

---

## Architecture: Hybrid SD + Agent-Based

```
Browser
  ├── Graph Editor    (Cytoscape.js — SD nodes/edges, agent read/write bridges)
  ├── Agent Panel     (template editor, population list/map, per-agent state)
  ├── Charts          (D3 or Recharts — SD trajectories + agent aggregates + histograms)
  ├── Control Panel   (React — parameter sliders, run controls, agent controls)
  └── State Store     (Zustand)
          │
          │ REST + WebSocket
          ▼
FastAPI Server
  ├── /api/model/...    CRUD on model YAML (SD layer)
  ├── /api/agents/...   CRUD on agent templates, initialize/reset populations
  ├── /api/sim/...      Run simulation, stream timesteps
  └── /api/loops/...    Feedback loop analysis (SD graph only)
          │
          ▼
Simulation Engine
  ├── WorldSystem      (SD ODE integrator — simulator.py, unchanged core)
  └── AgentScheduler   (new — instantiates templates, steps agents each timestep)
          │
          ▼
Files on disk: model.yaml, scenarios/*.yaml
```

**Key principles:**
- The YAML file on disk is the single source of truth — no database
- The SD layer holds the shared environment (climate, oil, population, food prices, etc.)
- Agents read from SD variables and write into designated SD auxiliaries (aggregates)
- Agents never directly modify SD stocks
- Agent templates and their SD bridges are defined in the same YAML

### How the layers interact

```
                    reads SD vars
  AGENTS  ───────────────────────────►  SD LAYER
  (heterogeneous,                      (shared environment:
   N instances each,                     stocks, auxiliaries,
   internal state,                       parameters)
   decision rules)

  AGENTS  ───────────────────────────►  SD LAYER
           write into designated
           aggregate auxiliaries
```

Every timestep: (1) agents read current SD state, (2) each agent evaluates its decision rules, updating its internal state, (3) agent outputs are aggregated and fed into designated SD auxiliaries, (4) the SD ODE integrator advances one step.

### Interaction topologies (per template)

- **All-to-all** (default): every agent instance interacts with every other of the same template (e.g., banks competing in one lending pool)
- **Network**: agents connected by a generated graph (small-world, scale-free, or explicit edge list) — e.g., interbank lending network
- **Spatial**: agents placed on a 1D/2D grid, interacting with neighbors — e.g., rebel factions controlling adjacent territory

---

## YAML Schema Extension

### Agent templates block (new)

```yaml
agent_templates:
  Bank:
    count: 50
    topology: all_to_all
    internal_stocks:
      capital:
        initial: {distribution: lognormal, mean: 1.0, sigma: 0.5}
      risk_appetite:
        initial: {distribution: uniform, min: 0.1, max: 0.9}
    decision_rules:
      - condition: "oil_price > 100 and risk_appetite < 0.5"
        action: "reduce_lending: capital * 0.2"
      - condition: "rate_of_profit > 0.15"
        action: "expand_lending: capital * 0.3"
    reads:  [oil_price, rate_of_profit, world_tension]
    writes:
      bank_lending:    {target: aggregate_bank_lending, method: sum}
      bank_fragility:  {target: aggregate_bank_fragility, method: weighted_average, weight: capital}
    interactions:
      - type: competition
        with: Bank
        field: capital
        strength: 0.01

  RebelFaction:
    count: 20
    topology: spatial_1d
    internal_stocks:
      fighters:     {initial: {distribution: uniform, min: 100, max: 5000}}
      support:      {initial: {distribution: uniform, min: 0.0, max: 0.3}}
    decision_rules:
      - condition: "food_prices > 2.0 and support > 0.2"
        action: "mobilize: fighters * 0.1"
      - condition: "youth_unemployment > 0.3"
        action: "recruit: population * 0.001"
    reads:  [food_prices, youth_unemployment, revolt_pressure]
    writes:
      faction_revolt_pressure: {target: rate_of_revolutions, method: sum}
    interactions:
      - type: competition
        with: RebelFaction
        field: support
        strength: 0.05
```

- `count`: number of agent instances to create
- `topology`: `all_to_all`, `network`, `spatial_1d`, `spatial_2d`
- `internal_stocks`: per-agent state variables with initial distributions
- `decision_rules`: if-then rules evaluated at each timestep by each agent
- `reads`: SD variables this agent type can see
- `writes`: how agent outputs aggregate into SD auxiliaries (`sum`, `weighted_average`, `max`, `min`)
- `interactions`: agent-agent influence (competition, cooperation, etc.) with configurable strength

### Stochastic block (extends existing)

```yaml
stochastic:
  sd_noise:                       # noise on SD auxiliaries
    rate_of_big_wars:      {noise_scale: 0.1}
    rate_of_revolutions:   {noise_scale: 0.1}
    rate_of_birth:         {noise_scale: 0.05}
    rate_of_death:         {noise_scale: 0.05}
  agent_noise:                     # noise on agent decision thresholds
    scale: 0.05                    # global agent noise multiplier
```

---

## Graph Editor (updated for hybrid)

### SD Layer (unchanged from previous)

- Stocks: squares; auxiliaries: circles; color-coded by subsystem
- Edges: green solid (+), red dashed (−)
- Full editing: add/remove nodes and edges, edit equations, reassign subsystems

### Agent Bridge visualization (new)

- Agent templates rendered as a distinct shape (hexagon or double-circle) on the graph, placed near the SD variables they read/write
- Dotted edges from SD variable → agent template (read), and agent template → SD auxiliary (write), in a distinct color (e.g., blue)
- Clicking an agent template opens the Agent Panel
- The bridge view toggles on/off via a checkbox — agent nodes can be hidden to show only the clean SD causal graph

---

## Agent Panel (new)

A dedicated panel in the UI for working with agents:

### Template editor
- List of defined agent templates, with add/delete
- For each template: edit count, topology, internal stocks with distribution params, decision rules (condition + action), reads/writes mappings
- Form-based editing — no raw YAML editing required (though YAML sync is maintained)

### Population view
- Tabular list of all agent instances, filterable by template
- Each row shows the agent's current internal state (capital, fighters, support, etc.)
- Color-coded by a selected internal variable (e.g., color banks by risk_appetite)
- Click an agent → highlight it on the spatial map or network view

### Agent charts
- Histograms of agent internal states at the current timestep (e.g., distribution of bank capital)
- Trace of aggregate variables over time overlaid on SD trajectory charts

---

## Control Panel (updated)

### Parameters tab
- Every SD `parameters:` key gets a slider with current numeric value
- Agent template parameters (count, interaction strength, rule thresholds) also get sliders
- Slider range: ~0.1× to 10× the default

### Run controls
- **Run** — single deterministic trajectory (SD + agents)
- **Stochastic run** — N ensemble members with noise on SD auxiliaries AND agent decisions
- **Pause / Resume / Reset** to initial conditions
- **Speed slider** for time playback (1× to 100×)
- **Ensemble size** (N) and random seed inputs
- **Reinitialize agents** button — re-rolls agent initial values from their distributions

---

## Charts (updated)

### SD trajectories
- Stocks and auxiliaries over time (2020–2120 default)
- In stochastic mode: median line + 50%/90% confidence bands

### Agent aggregates
- Aggregate agent outputs over time (e.g., total bank lending, total faction revolt pressure)
- Overlaid on the same time axis as SD trajectories

### Agent distributions (new)
- Histograms of agent internal states at user-selected timesteps
- Toggle between "current timestep" and "end state" view

---

## API Design (FastAPI, updated)

### /api/model (SD layer)

| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/model | Get full model as JSON |
| PUT | /api/model | Update full model |
| GET | /api/model/nodes | List all stocks + auxiliaries |
| POST | /api/model/stocks | Add a stock |
| PUT | /api/model/stocks/{name} | Update a stock |
| DELETE | /api/model/stocks/{name} | Delete a stock |
| POST | /api/model/auxiliaries | Add an auxiliary |
| PUT | /api/model/auxiliaries/{name} | Update an auxiliary |
| DELETE | /api/model/auxiliaries/{name} | Delete an auxiliary |
| PUT | /api/model/parameters/{name} | Update a parameter |
| POST | /api/model/reload | Re-read YAML from disk |

### /api/agents (new)

| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/agents/templates | List all agent templates |
| POST | /api/agents/templates | Create a new agent template |
| PUT | /api/agents/templates/{name} | Update an agent template |
| DELETE | /api/agents/templates/{name} | Delete an agent template |
| POST | /api/agents/initialize | Initialize agent population from templates |
| GET | /api/agents/population | Get current state of all agent instances |
| GET | /api/agents/population/{template} | Get state of agents of a specific template |

### /api/sim

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/sim/run | Run simulation (SD + agents, deterministic or ensemble) |
| WS | /api/sim/stream | WebSocket for step-by-step streaming (includes agent states per step) |
| GET | /api/sim/results/{id} | Get cached result set |

### /api/loops

| Method | Path | Purpose |
|--------|------|---------|
| GET | /api/loops | Get feedback loop analysis (SD graph only) |
| GET | /api/loops/{variable} | Loops that include a specific variable |

---

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Frontend framework | React (Vite) | Ecosystem, component model |
| Graph visualization | Cytoscape.js | Purpose-built for editable node/edge graphs |
| Charts | D3 or Recharts | D3 for flexibility, Recharts for React-native simplicity |
| State management | Zustand | Lightweight, no boilerplate |
| Backend | FastAPI + uvicorn | Fast, WebSocket support, Python-native |
| SD engine | simulator.py | Unchanged; imported as library |
| Agent engine | new `agent_scheduler.py` | Minimal — ~200 lines, template instantiation + rule evaluation + aggregation |
| File format | YAML | Already canonical; human-editable |

---

## Simulation Engine Changes

### New: `agent_scheduler.py`

A lightweight companion to `simulator.py`:
- Reads `agent_templates` from the model YAML
- Instantiates N agents per template (sampling from specified initial distributions)
- Each timestep: evaluates each agent's decision rules against current SD state, updates agent internal state, computes agent-agent interactions, aggregates outputs into designated SD auxiliaries
- Provides the aggregated values to the SD ODE integrator before it advances

No changes to `simulator.py` itself — the agent layer is an independent module invoked between ODE steps.

---

## Extensibility

New SD subsystems (e.g., monetary, trade, disease) are added by editing the YAML — new stocks, auxiliaries, parameters, and a `subsystems:` entry. The graph editor auto-renders any additions.

New agent templates are added similarly — define the template in YAML, specify what it reads/writes, and the UI surfaces it in the agent panel and graph bridge view.

The architecture places no limit on the number of subsystems or agent templates.

---

## Out of Scope (for now)

- Multi-user / collaborative editing
- Authentication / authorization
- Real-time data feeds (news, market data)
- Model calibration against empirical data
- Mobile support
- Agent learning / adaptation over time (decision rules are static per simulation)
- Hierarchical agents (agents containing sub-agents)
