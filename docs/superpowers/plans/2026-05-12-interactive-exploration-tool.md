# Interactive Exploration Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a web-based interactive sandbox for the World System Feedback Model — editable causal-loop graph, real-time simulation with stochastic ensembles, and hybrid SD+agent modeling.

**Architecture:** FastAPI backend wraps the existing WorldSystem engine + new AgentScheduler. React/Vite frontend with Cytoscape.js graph editor, Recharts trajectory plots, and Zustand state store. All model state lives in YAML files on disk — no database.

**Tech Stack:** Python + FastAPI + uvicorn (backend), TypeScript + React + Vite + Cytoscape.js + Recharts + Zustand (frontend)

---

## File Structure

```
worldfeedbackloop/
├── backend/
│   ├── server.py              # FastAPI app — all routes, CORS, WebSocket
│   ├── agent_scheduler.py     # Agent engine — template instantiation, rule eval, aggregation
│   ├── simulator.py           # Existing SD engine — NO CHANGES
│   ├── model.yaml             # Existing canonical model
│   └── scenarios/             # User scenario YAML files
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── index.html
│   └── src/
│       ├── main.tsx           # React entry point
│       ├── App.tsx            # Application shell + layout
│       ├── App.css            # Layout styles
│       ├── types.ts           # Shared TypeScript types
│       ├── store.ts           # Zustand state store
│       ├── api.ts             # REST + WebSocket API client
│       └── components/
│           ├── GraphEditor.tsx    # Cytoscape.js causal-loop diagram
│           ├── ControlPanel.tsx   # Parameter sliders, run controls, stochastic
│           ├── Charts.tsx         # Trajectory plots, agent aggregates, histograms
│           ├── AgentPanel.tsx     # Template editor, population view, agent charts
│           └── ScenarioManager.tsx # File browser for scenario YAMLs
├── tests/
│   ├── test_server.py         # FastAPI endpoint tests
│   └── test_agent_scheduler.py # Agent engine unit tests
├── environment.yml
└── docs/
```

**Boundaries:**
- `simulator.py` is not modified — the agent layer is independent
- `agent_scheduler.py` only depends on the model dict (not on `WorldSystem`)
- Frontend components share types via `types.ts` and state via `store.ts`, but each has its own rendering logic
- The API client (`api.ts`) is the only module that talks to the backend

---

### Task 1: Backend scaffolding — FastAPI server skeleton + model CRUD

**Files:**
- Create: `backend/server.py`
- Create: `tests/test_server.py`
- Modify: `environment.yml` (add fastapi, uvicorn, httpx)

- [ ] **Step 1: Add backend dependencies to environment.yml**

```yaml
name: worldfeedbackloop
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.10
  - pyyaml
  - numpy
  - scipy
  - matplotlib
  - networkx
  - fastapi
  - uvicorn[standard]
  - httpx
  - pip
```

Run: `conda env update -f environment.yml`

- [ ] **Step 2: Write the failing test for GET /api/model**

```python
# tests/test_server.py
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import shutil
import tempfile
import yaml

# We'll define the app fixture after creating server.py
# For now, test the expected shape

def test_dummy():
    """Placeholder — will be replaced when server.py exists."""
    pass
```

- [ ] **Step 3: Create backend/server.py with FastAPI app, /api/model endpoints**

```python
"""FastAPI server for the World System Feedback Model."""
from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from simulator import WorldSystem

MODEL_DIR = Path(__file__).resolve().parent
SCENARIOS_DIR = MODEL_DIR / "scenarios"
SCENARIOS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="World Feedback Loop API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- Pydantic models --------------------------------------------------

class StockSpec(BaseModel):
    initial: float
    units: str = ""
    inflows: list[str] = []
    outflows: list[str] = []
    subsystem: str = ""

class AuxiliarySpec(BaseModel):
    equation: str
    operator: str = "identity"
    inputs: dict[str, str] = {}
    subsystem: str = ""
    units: str = ""

class ModelOut(BaseModel):
    metadata: dict[str, Any]
    parameters: dict[str, Any]
    stocks: dict[str, Any]
    auxiliaries: dict[str, Any]
    subsystems: dict[str, Any]
    agent_templates: dict[str, Any] = {}
    stochastic: dict[str, Any] = {}

# -- Helpers ----------------------------------------------------------

def _load_raw(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def _save_raw(path: str, data: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

def _current_model_path() -> str:
    """Return path to the active model YAML on disk."""
    return str(MODEL_DIR / "model.yaml")

# -- Model CRUD -------------------------------------------------------

@app.get("/api/model")
def get_model() -> ModelOut:
    raw = _load_raw(_current_model_path())
    return ModelOut(
        metadata=raw.get("metadata", {}),
        parameters=raw.get("parameters", {}),
        stocks=raw.get("stocks", {}),
        auxiliaries=raw.get("auxiliaries", {}),
        subsystems=raw.get("subsystems", {}),
        agent_templates=raw.get("agent_templates", {}),
        stochastic=raw.get("stochastic", {}),
    )


@app.put("/api/model")
def update_model(model: ModelOut) -> ModelOut:
    raw = {
        "metadata": model.metadata,
        "parameters": model.parameters,
        "stocks": model.stocks,
        "auxiliaries": model.auxiliaries,
        "subsystems": model.subsystems,
        "agent_templates": model.agent_templates,
        "stochastic": model.stochastic,
    }
    _save_raw(_current_model_path(), raw)
    return model


@app.get("/api/model/nodes")
def list_nodes():
    raw = _load_raw(_current_model_path())
    stocks = [
        {"name": k, "kind": "stock", "subsystem": v.get("subsystem", "")}
        for k, v in raw.get("stocks", {}).items()
    ]
    auxiliaries = [
        {
            "name": k,
            "kind": "auxiliary",
            "subsystem": v.get("subsystem", ""),
            "operator": v.get("operator", "identity"),
            "inputs": v.get("inputs", {}),
        }
        for k, v in raw.get("auxiliaries", {}).items()
    ]
    return {"nodes": stocks + auxiliaries}


@app.post("/api/model/stocks")
def add_stock(name: str, spec: StockSpec):
    raw = _load_raw(_current_model_path())
    raw.setdefault("stocks", {})[name] = spec.model_dump()
    _save_raw(_current_model_path(), raw)
    return {"status": "ok", "name": name}


@app.put("/api/model/stocks/{name}")
def update_stock(name: str, spec: StockSpec):
    raw = _load_raw(_current_model_path())
    if name not in raw.get("stocks", {}):
        raise HTTPException(404, f"Stock '{name}' not found")
    raw["stocks"][name] = spec.model_dump()
    _save_raw(_current_model_path(), raw)
    return {"status": "ok", "name": name}


@app.delete("/api/model/stocks/{name}")
def delete_stock(name: str):
    raw = _load_raw(_current_model_path())
    if name not in raw.get("stocks", {}):
        raise HTTPException(404, f"Stock '{name}' not found")
    del raw["stocks"][name]
    _save_raw(_current_model_path(), raw)
    return {"status": "ok"}


@app.post("/api/model/auxiliaries")
def add_auxiliary(name: str, spec: AuxiliarySpec):
    raw = _load_raw(_current_model_path())
    raw.setdefault("auxiliaries", {})[name] = spec.model_dump()
    _save_raw(_current_model_path(), raw)
    return {"status": "ok", "name": name}


@app.put("/api/model/auxiliaries/{name}")
def update_auxiliary(name: str, spec: AuxiliarySpec):
    raw = _load_raw(_current_model_path())
    if name not in raw.get("auxiliaries", {}):
        raise HTTPException(404, f"Auxiliary '{name}' not found")
    raw["auxiliaries"][name] = spec.model_dump()
    _save_raw(_current_model_path(), raw)
    return {"status": "ok", "name": name}


@app.delete("/api/model/auxiliaries/{name}")
def delete_auxiliary(name: str):
    raw = _load_raw(_current_model_path())
    if name not in raw.get("auxiliaries", {}):
        raise HTTPException(404, f"Auxiliary '{name}' not found")
    del raw["auxiliaries"][name]
    _save_raw(_current_model_path(), raw)
    return {"status": "ok"}


@app.put("/api/model/parameters/{name}")
def update_parameter(name: str, value: float):
    raw = _load_raw(_current_model_path())
    raw.setdefault("parameters", {})[name] = value
    _save_raw(_current_model_path(), raw)
    return {"status": "ok", "name": name, "value": value}


@app.post("/api/model/reload")
def reload_model():
    return {"status": "ok", "message": "Model re-read from disk on next request"}
```

- [ ] **Step 4: Write the actual test for GET /api/model**

```python
# tests/test_server.py
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import shutil
import tempfile
import yaml
import sys

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

# We'll monkeypatch MODEL_DIR to use a temp dir
import server


@pytest.fixture
def client(tmp_path):
    """Create a test client pointed at a temp model directory."""
    model_yaml = tmp_path / "model.yaml"
    # Write a minimal valid model
    model_yaml.write_text(yaml.safe_dump({
        "metadata": {"name": "test"},
        "parameters": {"alpha": 1.0},
        "stocks": {"x": {"initial": 10, "inflows": [], "outflows": []}},
        "auxiliaries": {"rate": {"equation": "alpha * x", "operator": "multiply", "inputs": {"x": "positive"}}},
        "subsystems": {"default": {"label": "Default", "color": "#888"}},
    }))
    # Patch the module globals
    original_model_dir = server.MODEL_DIR
    original_scenarios = server.SCENARIOS_DIR
    server.MODEL_DIR = tmp_path
    server.SCENARIOS_DIR = tmp_path / "scenarios"
    server.SCENARIOS_DIR.mkdir(exist_ok=True)
    yield TestClient(server.app)
    server.MODEL_DIR = original_model_dir
    server.SCENARIOS_DIR = original_scenarios


def test_get_model(client):
    resp = client.get("/api/model")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["name"] == "test"
    assert data["parameters"]["alpha"] == 1.0
    assert "x" in data["stocks"]


def test_list_nodes(client):
    resp = client.get("/api/model/nodes")
    assert resp.status_code == 200
    nodes = resp.json()["nodes"]
    assert any(n["name"] == "x" and n["kind"] == "stock" for n in nodes)
    assert any(n["name"] == "rate" and n["kind"] == "auxiliary" for n in nodes)


def test_add_and_delete_stock(client):
    resp = client.post("/api/model/stocks?name=test_stock", json={
        "initial": 5.0, "inflows": [], "outflows": [], "subsystem": "default"
    })
    assert resp.status_code == 200

    resp = client.get("/api/model")
    assert "test_stock" in resp.json()["stocks"]

    resp = client.delete("/api/model/stocks/test_stock")
    assert resp.status_code == 200

    resp = client.get("/api/model")
    assert "test_stock" not in resp.json()["stocks"]


def test_update_parameter(client):
    resp = client.put("/api/model/parameters/alpha?value=99.0")
    assert resp.status_code == 200
    resp = client.get("/api/model")
    assert resp.json()["parameters"]["alpha"] == 99.0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python -m pytest ../tests/test_server.py -v`
Expected: 4 tests pass

- [ ] **Step 6: Commit**

```bash
git add backend/server.py tests/test_server.py environment.yml
git commit -m "feat: add FastAPI server with model CRUD endpoints"
```

---

### Task 2: Agent scheduler engine

**Files:**
- Create: `backend/agent_scheduler.py`
- Create: `tests/test_agent_scheduler.py`

- [ ] **Step 1: Write failing tests for AgentScheduler**

```python
# tests/test_agent_scheduler.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import pytest
import numpy as np
from agent_scheduler import AgentScheduler


@pytest.fixture
def model_with_agents():
    return {
        "parameters": {"alpha": 1.0, "beta": 0.5},
        "stocks": {
            "capital": {"initial": 100.0, "inflows": ["invest"], "outflows": ["deprec"], "subsystem": "econ"},
        },
        "auxiliaries": {
            "invest": {"equation": "alpha * capital", "operator": "multiply",
                        "inputs": {"capital": "positive"}, "subsystem": "econ"},
            "deprec": {"equation": "0.1 * capital", "operator": "multiply",
                        "inputs": {"capital": "positive"}, "subsystem": "econ"},
            "aggregate_lending": {"equation": "0.0", "operator": "identity",
                                   "inputs": {}, "subsystem": "econ"},
        },
        "agent_templates": {
            "Bank": {
                "count": 3,
                "topology": "all_to_all",
                "internal_stocks": {
                    "capital": {"initial": {"distribution": "uniform", "min": 0.5, "max": 1.5}},
                    "risk": {"initial": {"distribution": "uniform", "min": 0.1, "max": 0.5}},
                },
                "decision_rules": [
                    {"condition": "rate_of_profit > 0.1", "action": "lend: capital * 0.1"},
                ],
                "reads": ["capital", "invest"],
                "writes": {
                    "bank_lending": {"target": "aggregate_lending", "method": "sum"},
                },
                "interactions": [
                    {"type": "competition", "with": "Bank", "field": "capital", "strength": 0.01},
                ],
            }
        },
    }


def test_instantiate_agents(model_with_agents):
    sched = AgentScheduler(model_with_agents)
    sched.initialize(np.random.default_rng(42))
    assert len(sched.agents) == 3
    for agent in sched.agents:
        assert agent.template_name == "Bank"
        assert 0.5 <= agent.state["capital"] <= 1.5
        assert 0.1 <= agent.state["risk"] <= 0.5


def test_evaluate_rules(model_with_agents):
    sched = AgentScheduler(model_with_agents)
    sched.initialize(np.random.default_rng(42))
    sd_state = {"capital": 100.0, "invest": 10.0, "rate_of_profit": 0.15}
    sched.step(sd_state, np.random.default_rng(42))
    # Agents with profit > 0.1 should have lent
    for agent in sched.agents:
        assert "bank_lending" in agent.outputs


def test_aggregate_writes(model_with_agents):
    sched = AgentScheduler(model_with_agents)
    sched.initialize(np.random.default_rng(42))
    sd_state = {"capital": 100.0, "invest": 10.0, "rate_of_profit": 0.15}
    aggregates = sched.step(sd_state, np.random.default_rng(42))
    assert "aggregate_lending" in aggregates
    assert aggregates["aggregate_lending"] > 0


def test_competition_interaction(model_with_agents):
    sched = AgentScheduler(model_with_agents)
    sched.initialize(np.random.default_rng(42))
    capitals_before = [a.state["capital"] for a in sched.agents]
    sd_state = {"capital": 100.0, "invest": 10.0, "rate_of_profit": 0.15}
    sched.step(sd_state, np.random.default_rng(42))
    capitals_after = [a.state["capital"] for a in sched.agents]
    # Competition should have shifted values (not all identical to before)
    assert capitals_after != capitals_before


def test_spatial_topology():
    model = {
        "parameters": {},
        "stocks": {},
        "auxiliaries": {},
        "agent_templates": {
            "Faction": {
                "count": 10,
                "topology": "spatial_1d",
                "internal_stocks": {
                    "support": {"initial": {"distribution": "uniform", "min": 0.0, "max": 1.0}},
                },
                "decision_rules": [],
                "reads": [],
                "writes": {},
                "interactions": [
                    {"type": "competition", "with": "Faction", "field": "support", "strength": 0.1},
                ],
            }
        },
    }
    sched = AgentScheduler(model)
    sched.initialize(np.random.default_rng(42))
    # 1D spatial: neighbors are adjacent indices
    neighbors = sched._topology["Faction"]
    for i in range(10):
        assert (i - 1) % 10 in neighbors[i] or (i + 1) % 10 in neighbors[i]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest ../tests/test_agent_scheduler.py -v`
Expected: all FAIL with ModuleNotFoundError or similar

- [ ] **Step 3: Implement agent_scheduler.py**

```python
"""Agent scheduler for hybrid SD + agent-based simulation.

Instantiates agent templates from the model dict, evaluates decision rules
each timestep, computes agent-agent interactions, and aggregates outputs
into SD auxiliaries.
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.random import Generator

# Safe eval namespace for decision rules (same pattern as simulator.py)
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "pow": pow,
}


@dataclass
class AgentInstance:
    template_name: str
    state: dict[str, float]
    outputs: dict[str, float] = field(default_factory=dict)


class AgentScheduler:
    """Manages a population of agents defined by YAML `agent_templates`."""

    def __init__(self, model: dict):
        self.templates = model.get("agent_templates", {})
        self.agents: list[AgentInstance] = []
        self._topology: dict[str, dict[int, list[int]]] = {}
        self._rng: Generator | None = None

    def initialize(self, rng: Generator) -> None:
        """Create agent instances by sampling initial distributions."""
        self._rng = rng
        self.agents = []
        for name, tmpl in self.templates.items():
            count = int(tmpl.get("count", 1))
            for _ in range(count):
                state = {}
                for var, spec in tmpl.get("internal_stocks", {}).items():
                    init = spec["initial"]
                    dist = init["distribution"]
                    if dist == "uniform":
                        state[var] = rng.uniform(init["min"], init["max"])
                    elif dist == "lognormal":
                        state[var] = rng.lognormal(init["mean"], init["sigma"])
                    else:
                        state[var] = float(init.get("mean", 1.0))
                self.agents.append(AgentInstance(template_name=name, state=state))
        self._build_topologies()

    def _build_topologies(self) -> None:
        """Precompute neighbor sets for each template's interaction topology."""
        self._topology = {}
        for name, tmpl in self.templates.items():
            topo = tmpl.get("topology", "all_to_all")
            indices = [i for i, a in enumerate(self.agents) if a.template_name == name]
            n = len(indices)
            if topo == "all_to_all" or n <= 1:
                neighbors = {idx: [j for j in indices if j != idx] for idx in indices}
            elif topo == "spatial_1d":
                neighbors = {}
                for pos, idx in enumerate(indices):
                    nbrs = []
                    if n > 1:
                        nbrs.append(indices[(pos - 1) % n])
                        nbrs.append(indices[(pos + 1) % n])
                    neighbors[idx] = nbrs
            elif topo == "spatial_2d":
                side = int(math.ceil(math.sqrt(n)))
                neighbors = {}
                for pos, idx in enumerate(indices):
                    row, col = divmod(pos, side)
                    nbrs = []
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = (row + dr) % side, (col + dc) % side
                        npos = nr * side + nc
                        if npos < n:
                            nbrs.append(indices[npos])
                    neighbors[idx] = nbrs
            else:
                neighbors = {idx: [j for j in indices if j != idx] for idx in indices}
            self._topology[name] = neighbors

    def step(self, sd_state: dict[str, float], rng: Generator | None = None) -> dict[str, float]:
        """Evaluate decision rules, apply interactions, aggregate outputs.

        Returns a dict of SD auxiliary variable names -> aggregated values.
        """
        rng = rng or self._rng
        # Clear previous outputs
        for agent in self.agents:
            agent.outputs = {}

        # Evaluate decision rules for each agent
        for agent in self.agents:
            tmpl = self.templates[agent.template_name]
            ns = {**sd_state, **agent.state}
            for rule in tmpl.get("decision_rules", []):
                try:
                    cond = _safe_eval(rule["condition"], ns)
                except Exception:
                    cond = False
                if cond:
                    # Parse action: "name: expression"
                    action = rule["action"]
                    if ":" in action:
                        out_name, expr = action.split(":", 1)
                        out_name = out_name.strip()
                        try:
                            agent.outputs[out_name] = float(_safe_eval(expr.strip(), ns))
                        except Exception:
                            pass

        # Apply agent-agent interactions
        for name, tmpl in self.templates.items():
            for inter in tmpl.get("interactions", []):
                other_template = inter["with"]
                field = inter["field"]
                strength = inter["strength"]
                topo = self._topology.get(name, {})
                for i, agent in enumerate(self.agents):
                    if agent.template_name != name:
                        continue
                    neighbors = topo.get(i, [])
                    for j in neighbors:
                        other = self.agents[j]
                        if other.template_name != other_template:
                            continue
                        # Competition: stronger neighbor reduces this agent's field
                        if inter["type"] == "competition":
                            delta = strength * (other.state[field] - agent.state[field])
                            agent.state[field] -= delta * 0.01
                        elif inter["type"] == "cooperation":
                            delta = strength * (other.state[field] - agent.state[field])
                            agent.state[field] += delta * 0.01

        # Aggregate agent outputs into SD auxiliaries
        aggregates: dict[str, list[float]] = {}
        for agent in self.agents:
            tmpl = self.templates[agent.template_name]
            for out_name, out_val in agent.outputs.items():
                write_spec = tmpl.get("writes", {}).get(out_name, {})
                target = write_spec.get("target", out_name)
                method = write_spec.get("method", "sum")
                weight_key = write_spec.get("weight", None)

                if method == "weighted_average" and weight_key:
                    weight = agent.state.get(weight_key, 1.0)
                else:
                    weight = 1.0

                aggregates.setdefault(target, ([], []))
                aggregates[target][0].append(out_val * weight)
                aggregates[target][1].append(weight)

        result: dict[str, float] = {}
        for target, (vals, weights) in aggregates.items():
            # Determine method from any agent's write spec
            method = "sum"
            for agent in self.agents:
                tmpl = self.templates.get(agent.template_name, {})
                for oname, wspec in tmpl.get("writes", {}).items():
                    if wspec.get("target") == target:
                        method = wspec.get("method", "sum")
                        break
            if method == "weighted_average":
                result[target] = sum(vals) / sum(weights) if sum(weights) > 0 else 0.0
            elif method == "max":
                result[target] = max(vals) if vals else 0.0
            elif method == "min":
                result[target] = min(vals) if vals else 0.0
            else:  # sum
                result[target] = sum(vals)
        return result


def _safe_eval(expr: str, namespace: dict[str, float]) -> float:
    return eval(expr, {"__builtins__": _SAFE_BUILTINS}, namespace)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest ../tests/test_agent_scheduler.py -v`
Expected: 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add backend/agent_scheduler.py tests/test_agent_scheduler.py
git commit -m "feat: add agent scheduler engine with template instantiation, rule evaluation, and aggregation"
```

---

### Task 3: Simulation endpoints + WebSocket streaming

**Files:**
- Modify: `backend/server.py` (add sim and loops routes)
- Create: `tests/test_sim.py`

- [ ] **Step 1: Write failing test for simulation endpoint**

```python
# tests/test_sim.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
import yaml
import server

def test_run_sim_deterministic(tmp_path, monkeypatch):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(yaml.safe_dump({
        "metadata": {"t_start": 2020, "t_end": 2040},
        "parameters": {"alpha": 1.0},
        "stocks": {"x": {"initial": 100, "inflows": ["inflow"], "outflows": [], "subsystem": "test"}},
        "auxiliaries": {"inflow": {"equation": "alpha * x", "operator": "multiply", "inputs": {"x": "positive"}, "subsystem": "test"}},
        "subsystems": {"test": {"label": "Test", "color": "#888"}},
    }))
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir(exist_ok=True)
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(server, "SCENARIOS_DIR", scenarios)

    from fastapi.testclient import TestClient
    client = TestClient(server.app)

    resp = client.post("/api/sim/run", json={"mode": "deterministic", "n_steps": 21})
    assert resp.status_code == 200
    data = resp.json()
    assert "t" in data
    assert len(data["t"]) == 21
    assert "x" in data["stocks"]
    assert "inflow" in data["auxiliaries"]


def test_run_sim_stochastic(tmp_path, monkeypatch):
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(yaml.safe_dump({
        "metadata": {"t_start": 2020, "t_end": 2040},
        "parameters": {"alpha": 1.0},
        "stocks": {"x": {"initial": 100, "inflows": ["inflow"], "outflows": [], "subsystem": "test"}},
        "auxiliaries": {"inflow": {"equation": "alpha * x", "operator": "multiply", "inputs": {"x": "positive"}, "subsystem": "test"}},
        "subsystems": {"test": {"label": "Test", "color": "#888"}},
        "stochastic": {"sd_noise": {"inflow": {"noise_scale": 0.1}}},
    }))
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir(exist_ok=True)
    monkeypatch.setattr(server, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(server, "SCENARIOS_DIR", scenarios)

    from fastapi.testclient import TestClient
    client = TestClient(server.app)

    resp = client.post("/api/sim/run", json={
        "mode": "stochastic", "n_ensemble": 5, "n_steps": 21, "seed": 42
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "stochastic"
    assert len(data["ensemble"]) == 5
```

- [ ] **Step 2: Add simulation routes to server.py**

Insert after the existing routes in `backend/server.py`:

```python
# -- Simulation --------------------------------------------------------
from agent_scheduler import AgentScheduler
import numpy as np

# In-memory result cache (simple dict, no persistence across restarts)
_result_cache: dict[str, dict] = {}


class SimRequest(BaseModel):
    mode: str = "deterministic"  # deterministic | stochastic
    n_steps: int = 401
    n_ensemble: int = 10
    seed: int | None = None


@app.post("/api/sim/run")
def run_simulation(req: SimRequest):
    path = _current_model_path()
    raw = _load_raw(path)
    ws = WorldSystem(model=raw)

    if req.mode == "deterministic":
        res = ws.simulate(n_points=req.n_steps)
        result = {
            "mode": "deterministic",
            "t": res["t"].tolist(),
            "stocks": {k: v.tolist() for k, v in res["stocks"].items()},
            "auxiliaries": {k: v.tolist() for k, v in res["auxiliaries"].items()},
        }
    elif req.mode == "stochastic":
        stochastic_cfg = raw.get("stochastic", {})
        sd_noise = stochastic_cfg.get("sd_noise", {})
        agent_noise = stochastic_cfg.get("agent_noise", {})
        ensemble = []
        base_rng = np.random.default_rng(req.seed or 42)
        seeds = base_rng.integers(0, 2**31, size=req.n_ensemble)

        for ens_i in range(req.n_ensemble):
            rng = np.random.default_rng(int(seeds[ens_i]))
            # Run with noise injected into designated auxiliaries
            ws_i = WorldSystem(model=raw)
            # If agent templates exist, initialize them
            sched = None
            has_agents = bool(raw.get("agent_templates"))
            if has_agents:
                sched = AgentScheduler(raw)
                sched.initialize(rng)

            # We need a custom simulate that injects noise + agents.
            # For now, simulate deterministically and add noise post-hoc
            # (full integration with agent step comes in Task 8)
            res = ws_i.simulate(n_points=req.n_steps)
            # Apply log-normal noise to noised auxiliaries
            for aux_name, cfg in sd_noise.items():
                if aux_name in res["auxiliaries"]:
                    noise = rng.lognormal(0, cfg["noise_scale"], size=len(res["t"]))
                    res["auxiliaries"][aux_name] = res["auxiliaries"][aux_name] * noise
            ensemble.append({
                "t": res["t"].tolist(),
                "stocks": {k: v.tolist() for k, v in res["stocks"].items()},
                "auxiliaries": {k: v.tolist() for k, v in res["auxiliaries"].items()},
            })
        result = {"mode": "stochastic", "ensemble": ensemble}
    else:
        raise HTTPException(400, f"Unknown mode: {req.mode}")

    result_id = str(uuid.uuid4())[:8]
    _result_cache[result_id] = result
    result["id"] = result_id
    return result


@app.get("/api/sim/results/{result_id}")
def get_results(result_id: str):
    if result_id not in _result_cache:
        raise HTTPException(404, "Result not found")
    return _result_cache[result_id]


# -- Loop analysis ------------------------------------------------------

@app.get("/api/loops")
def get_loops(max_len: int = 8):
    path = _current_model_path()
    ws = WorldSystem(path)
    loops = ws.find_loops(max_len=max_len)
    return {"loops": loops}


@app.get("/api/loops/{variable}")
def get_loops_for_variable(variable: str, max_len: int = 8):
    path = _current_model_path()
    ws = WorldSystem(path)
    all_loops = ws.find_loops(max_len=max_len)
    matching = [L for L in all_loops if variable in L["nodes"]]
    return {"variable": variable, "loops": matching}
```

- [ ] **Step 3: Run the sim tests**

Run: `cd backend && python -m pytest ../tests/test_sim.py -v`
Expected: 2 tests pass

- [ ] **Step 4: Commit**

```bash
git add backend/server.py tests/test_sim.py
git commit -m "feat: add simulation endpoints (deterministic + stochastic) and loop analysis"
```

---

### Task 4: Agent API endpoints

**Files:**
- Modify: `backend/server.py` (add /api/agents routes)
- Modify: `tests/test_server.py` (add agent endpoint tests)

- [ ] **Step 1: Add agent endpoint tests**

Append to `tests/test_server.py`:

```python
def test_list_agent_templates(client, tmp_path):
    """Write a model with agent templates, then list them."""
    import yaml
    model_yaml = tmp_path / "model.yaml"  # Note: the fixture creates tmp_path/model.yaml already, we overwrite
    model_yaml.write_text(yaml.safe_dump({
        "metadata": {"name": "test"},
        "parameters": {},
        "stocks": {},
        "auxiliaries": {},
        "subsystems": {},
        "agent_templates": {
            "TestAgent": {
                "count": 5,
                "topology": "all_to_all",
                "internal_stocks": {"x": {"initial": {"distribution": "uniform", "min": 0, "max": 1}}},
                "decision_rules": [],
                "reads": [],
                "writes": {},
                "interactions": [],
            }
        },
    }))
    resp = client.post("/api/model/reload")
    resp = client.get("/api/agents/templates")
    assert resp.status_code == 200
    data = resp.json()
    assert "TestAgent" in data["templates"]


def test_initialize_agents(client, tmp_path):
    import yaml
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(yaml.safe_dump({
        "metadata": {"name": "test"},
        "parameters": {},
        "stocks": {"pop": {"initial": 100, "inflows": [], "outflows": [], "subsystem": "test"}},
        "auxiliaries": {},
        "subsystems": {"test": {"label": "Test", "color": "#888"}},
        "agent_templates": {
            "Bank": {
                "count": 10,
                "topology": "all_to_all",
                "internal_stocks": {
                    "capital": {"initial": {"distribution": "uniform", "min": 0.5, "max": 1.5}},
                },
                "decision_rules": [],
                "reads": [],
                "writes": {},
                "interactions": [],
            }
        },
    }))
    resp = client.post("/api/agents/initialize", json={"seed": 42})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 10

    resp = client.get("/api/agents/population")
    assert resp.status_code == 200
    agents = resp.json()["agents"]
    assert len(agents) == 10


def test_create_and_delete_template(client, tmp_path):
    import yaml
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(yaml.safe_dump({
        "metadata": {"name": "test"},
        "parameters": {},
        "stocks": {},
        "auxiliaries": {},
        "subsystems": {},
    }))
    resp = client.post("/api/model/reload")
    resp = client.post("/api/agents/templates", json={
        "name": "NewAgent",
        "template": {
            "count": 3,
            "topology": "all_to_all",
            "internal_stocks": {},
            "decision_rules": [],
            "reads": [],
            "writes": {},
            "interactions": [],
        },
    })
    assert resp.status_code == 200

    resp = client.get("/api/agents/templates")
    assert "NewAgent" in resp.json()["templates"]

    resp = client.delete("/api/agents/templates/NewAgent")
    assert resp.status_code == 200
    resp = client.get("/api/agents/templates")
    assert "NewAgent" not in resp.json()["templates"]
```

- [ ] **Step 2: Add agent routes to server.py**

Insert before the `# -- Simulation` block in `backend/server.py`:

```python
# -- Agent CRUD --------------------------------------------------------

class AgentTemplateSpec(BaseModel):
    name: str
    template: dict[str, Any]


@app.get("/api/agents/templates")
def list_agent_templates():
    raw = _load_raw(_current_model_path())
    return {"templates": raw.get("agent_templates", {})}


@app.post("/api/agents/templates")
def create_agent_template(spec: AgentTemplateSpec):
    raw = _load_raw(_current_model_path())
    raw.setdefault("agent_templates", {})[spec.name] = spec.template
    _save_raw(_current_model_path(), raw)
    return {"status": "ok", "name": spec.name}


@app.put("/api/agents/templates/{name}")
def update_agent_template(name: str, template: dict[str, Any]):
    raw = _load_raw(_current_model_path())
    raw.setdefault("agent_templates", {})[name] = template
    _save_raw(_current_model_path(), raw)
    return {"status": "ok", "name": name}


@app.delete("/api/agents/templates/{name}")
def delete_agent_template(name: str):
    raw = _load_raw(_current_model_path())
    if name not in raw.get("agent_templates", {}):
        raise HTTPException(404, f"Agent template '{name}' not found")
    del raw["agent_templates"][name]
    _save_raw(_current_model_path(), raw)
    return {"status": "ok"}


# Agent population (in-memory, lives for duration of a simulation)
_agent_scheduler: AgentScheduler | None = None
_agent_sd_state: dict[str, float] = {}


class InitAgentsRequest(BaseModel):
    seed: int | None = None


@app.post("/api/agents/initialize")
def initialize_agents(req: InitAgentsRequest):
    global _agent_scheduler
    raw = _load_raw(_current_model_path())
    _agent_scheduler = AgentScheduler(raw)
    rng = np.random.default_rng(req.seed or 42)
    _agent_scheduler.initialize(rng)
    return {"status": "ok", "count": len(_agent_scheduler.agents)}


@app.get("/api/agents/population")
def get_agent_population():
    if _agent_scheduler is None:
        return {"agents": []}
    agents = [
        {
            "template": a.template_name,
            "state": a.state,
            "outputs": a.outputs,
        }
        for a in _agent_scheduler.agents
    ]
    return {"agents": agents}


@app.get("/api/agents/population/{template}")
def get_agent_population_by_template(template: str):
    if _agent_scheduler is None:
        return {"agents": []}
    agents = [
        {
            "template": a.template_name,
            "state": a.state,
            "outputs": a.outputs,
        }
        for a in _agent_scheduler.agents
        if a.template_name == template
    ]
    return {"agents": agents}
```

- [ ] **Step 3: Run the agent endpoint tests**

Run: `cd backend && python -m pytest ../tests/test_server.py -v -k "agent"`
Expected: 3 tests pass (`test_list_agent_templates`, `test_initialize_agents`, `test_create_and_delete_template`)

- [ ] **Step 4: Run all backend tests to verify nothing broke**

Run: `cd backend && python -m pytest ../tests/ -v`
Expected: all 9 tests pass

- [ ] **Step 5: Commit**

```bash
git add backend/server.py tests/test_server.py
git commit -m "feat: add agent CRUD and population endpoints"
```

---

### Task 5: Frontend scaffolding — React + Vite + Zustand + types

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/App.css`
- Create: `frontend/src/types.ts`
- Create: `frontend/src/store.ts`

- [ ] **Step 1: Create package.json**

```json
{
  "name": "worldfeedbackloop-ui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "cytoscape": "^3.30.4",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "recharts": "^2.15.0",
    "zustand": "^5.0.3"
  },
  "devDependencies": {
    "@types/cytoscape": "^3.21.8",
    "@types/react": "^18.3.18",
    "@types/react-dom": "^18.3.5",
    "@vitejs/plugin-react": "^4.3.4",
    "typescript": "^5.7.3",
    "vite": "^6.0.11"
  }
}
```

- [ ] **Step 2: Create vite.config.ts**

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
});
```

- [ ] **Step 3: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": false,
    "noUnusedParameters": false,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"]
}
```

- [ ] **Step 4: Create tsconfig.node.json**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "strict": true,
    "noUnusedLocals": false,
    "noUnusedParameters": false,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 5: Create index.html**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>World Feedback Loop</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 6: Create types.ts**

```typescript
// ---- SD Model types ----

export interface ModelData {
  metadata: Record<string, unknown>;
  parameters: Record<string, number>;
  stocks: Record<string, StockSpec>;
  auxiliaries: Record<string, AuxiliarySpec>;
  subsystems: Record<string, SubsystemSpec>;
  agent_templates: Record<string, AgentTemplate>;
  stochastic: StochasticConfig;
}

export interface StockSpec {
  initial: number;
  units: string;
  inflows: string[];
  outflows: string[];
  subsystem: string;
}

export interface AuxiliarySpec {
  equation: string;
  operator: string;
  inputs: Record<string, string>;
  subsystem: string;
  units: string;
}

export interface SubsystemSpec {
  label: string;
  color: string;
}

// ---- Graph node types (for Cytoscape) ----

export interface GraphNode {
  name: string;
  kind: 'stock' | 'auxiliary' | 'agent_template';
  subsystem: string;
  operator?: string;
  inputs?: Record<string, string>;
  position?: { x: number; y: number };
}

export interface GraphEdge {
  source: string;
  target: string;
  polarity: 'positive' | 'negative';
  role?: 'inflow' | 'outflow' | 'read' | 'write';
}

// ---- Agent types ----

export interface AgentTemplate {
  count: number;
  topology: 'all_to_all' | 'network' | 'spatial_1d' | 'spatial_2d';
  internal_stocks: Record<string, {
    initial: { distribution: string; mean?: number; sigma?: number; min?: number; max?: number };
  }>;
  decision_rules: DecisionRule[];
  reads: string[];
  writes: Record<string, { target: string; method: string; weight?: string }>;
  interactions: AgentInteraction[];
}

export interface DecisionRule {
  condition: string;
  action: string;
}

export interface AgentInteraction {
  type: 'competition' | 'cooperation';
  with: string;
  field: string;
  strength: number;
}

export interface AgentInstance {
  template: string;
  state: Record<string, number>;
  outputs: Record<string, number>;
}

// ---- Simulation types ----

export interface SimResult {
  id?: string;
  mode: 'deterministic' | 'stochastic';
  t?: number[];
  stocks?: Record<string, number[]>;
  auxiliaries?: Record<string, number[]>;
  ensemble?: {
    t: number[];
    stocks: Record<string, number[]>;
    auxiliaries: Record<string, number[]>;
  }[];
}

export interface StochasticConfig {
  sd_noise: Record<string, { noise_scale: number }>;
  agent_noise: { scale: number };
}

// ---- Loop analysis ----

export interface FeedbackLoop {
  nodes: string[];
  edges: [string, string, string][];  // [source, target, polarity]
  polarity: 'reinforcing' | 'balancing';
}
```

- [ ] **Step 7: Create store.ts**

```typescript
import { create } from 'zustand';
import type { ModelData, SimResult, GraphNode, GraphEdge, AgentInstance } from './types';

interface AppState {
  // Model
  model: ModelData | null;
  setModel: (model: ModelData) => void;

  // Graph (derived from model, but positions are UI-only)
  nodes: GraphNode[];
  edges: GraphEdge[];
  setGraph: (nodes: GraphNode[], edges: GraphEdge[]) => void;

  // Simulation
  simResult: SimResult | null;
  setSimResult: (result: SimResult | null) => void;
  simRunning: boolean;
  setSimRunning: (running: boolean) => void;

  // Agents
  agentPopulation: AgentInstance[];
  setAgentPopulation: (agents: AgentInstance[]) => void;

  // UI state
  showAgentBridge: boolean;
  setShowAgentBridge: (show: boolean) => void;
  selectedTimestep: number;
  setSelectedTimestep: (t: number) => void;

  // Derived: is there a result to display
  hasResult: () => boolean;
}

export const useStore = create<AppState>((set, get) => ({
  model: null,
  setModel: (model) => set({ model }),

  nodes: [],
  edges: [],
  setGraph: (nodes, edges) => set({ nodes, edges }),

  simResult: null,
  setSimResult: (result) => set({ simResult: result }),
  simRunning: false,
  setSimRunning: (running) => set({ simRunning: running }),

  agentPopulation: [],
  setAgentPopulation: (agents) => set({ agentPopulation: agents }),

  showAgentBridge: true,
  setShowAgentBridge: (show) => set({ showAgentBridge: show }),
  selectedTimestep: 0,
  setSelectedTimestep: (t) => set({ selectedTimestep: t }),

  hasResult: () => get().simResult !== null,
}));
```

- [ ] **Step 8: Create main.tsx**

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

- [ ] **Step 9: Create App.tsx (shell layout)**

```tsx
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
```

- [ ] **Step 10: Create App.css**

```css
* { box-sizing: border-box; margin: 0; padding: 0; }

.app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 16px;
  background: #1a1a2e;
  color: #eee;
  height: 44px;
  flex-shrink: 0;
}
.top-bar h1 { font-size: 16px; font-weight: 600; }

.main-area {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.left-panel {
  flex: 1;
  min-width: 0;
  border-right: 1px solid #333;
}

.right-panel {
  width: 420px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  background: #1e1e2e;
  color: #ccc;
}
```

- [ ] **Step 11: Install dependencies and verify build**

```bash
cd frontend && npm install && npx tsc --noEmit
```
Expected: tsc reports errors for missing components (GraphEditor, ControlPanel, etc.) — this is expected and will be resolved in subsequent tasks.

- [ ] **Step 12: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold frontend with React+Vite+Zustand, types, and layout shell"
```

---

### Task 6: API client

**Files:**
- Create: `frontend/src/api.ts`

- [ ] **Step 1: Create api.ts**

```typescript
import type { ModelData, SimResult, FeedbackLoop, AgentTemplate, AgentInstance } from './types';

const BASE = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!resp.ok) {
    throw new Error(`API ${options?.method || 'GET'} ${url}: ${resp.status}`);
  }
  return resp.json();
}

// ---- Model ----

export function fetchModel(): Promise<ModelData> {
  return request<ModelData>('/model');
}

export function updateModel(model: ModelData): Promise<ModelData> {
  return request<ModelData>('/model', {
    method: 'PUT',
    body: JSON.stringify(model),
  });
}

export function fetchNodes(): Promise<{ nodes: { name: string; kind: string; subsystem: string; operator?: string; inputs?: Record<string, string> }[] }> {
  return request('/model/nodes');
}

export function addStock(name: string, spec: Record<string, unknown>): Promise<void> {
  return request(`/model/stocks?name=${encodeURIComponent(name)}`, {
    method: 'POST',
    body: JSON.stringify(spec),
  });
}

export function deleteStock(name: string): Promise<void> {
  return request(`/model/stocks/${encodeURIComponent(name)}`, { method: 'DELETE' });
}

export function addAuxiliary(name: string, spec: Record<string, unknown>): Promise<void> {
  return request(`/model/auxiliaries?name=${encodeURIComponent(name)}`, {
    method: 'POST',
    body: JSON.stringify(spec),
  });
}

export function deleteAuxiliary(name: string): Promise<void> {
  return request(`/model/auxiliaries/${encodeURIComponent(name)}`, { method: 'DELETE' });
}

export function updateParameter(name: string, value: number): Promise<void> {
  return request(`/model/parameters/${encodeURIComponent(name)}?value=${value}`, { method: 'PUT' });
}

export function reloadModel(): Promise<void> {
  return request('/model/reload', { method: 'POST' });
}

// ---- Simulation ----

export function runSimulation(params: {
  mode: 'deterministic' | 'stochastic';
  n_steps?: number;
  n_ensemble?: number;
  seed?: number;
}): Promise<SimResult> {
  return request<SimResult>('/sim/run', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

// ---- Loops ----

export function fetchLoops(maxLen = 8): Promise<FeedbackLoop[]> {
  return request<{ loops: FeedbackLoop[] }>(`/loops?max_len=${maxLen}`).then(r => r.loops);
}

export function fetchLoopsForVariable(variable: string, maxLen = 8): Promise<FeedbackLoop[]> {
  return request<{ loops: FeedbackLoop[] }>(`/loops/${encodeURIComponent(variable)}?max_len=${maxLen}`).then(r => r.loops);
}

// ---- Agents ----

export function fetchAgentTemplates(): Promise<Record<string, AgentTemplate>> {
  return request<{ templates: Record<string, AgentTemplate> }>('/agents/templates').then(r => r.templates);
}

export function createAgentTemplate(name: string, template: AgentTemplate): Promise<void> {
  return request('/agents/templates', {
    method: 'POST',
    body: JSON.stringify({ name, template }),
  });
}

export function deleteAgentTemplate(name: string): Promise<void> {
  return request(`/agents/templates/${encodeURIComponent(name)}`, { method: 'DELETE' });
}

export function initializeAgents(seed?: number): Promise<{ count: number }> {
  return request('/agents/initialize', {
    method: 'POST',
    body: JSON.stringify({ seed: seed ?? null }),
  });
}

export function fetchAgentPopulation(): Promise<AgentInstance[]> {
  return request<{ agents: AgentInstance[] }>('/agents/population').then(r => r.agents);
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: still errors on missing components, but `api.ts` should have no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat: add API client layer for all backend endpoints"
```

---

### Task 7: Graph Editor component (Cytoscape.js)

**Files:**
- Create: `frontend/src/components/GraphEditor.tsx`

- [ ] **Step 1: Create GraphEditor.tsx**

```tsx
import { useEffect, useRef, useCallback } from 'react';
import cytoscape, { Core, EventObject } from 'cytoscape';
import { useStore } from '../store';
import { fetchModel } from '../api';
import type { GraphNode, GraphEdge } from '../types';

const SUBSYSTEM_COLORS: Record<string, string> = {
  profit: '#d62728',
  energy_atmosphere: '#2ca02c',
  energy_politics: '#ff7f0e',
  revolt: '#9467bd',
  knowledge: '#1f77b4',
};

function buildCyElements(model: any, showAgents: boolean) {
  const elements: cytoscape.ElementDefinition[] = [];

  // SD stocks
  for (const [name, spec] of Object.entries(model.stocks || {}) as [string, any][]) {
    const color = SUBSYSTEM_COLORS[spec.subsystem] || '#888';
    elements.push({
      data: { id: name, label: name.replace(/_/g, '\n'), kind: 'stock', subsystem: spec.subsystem },
      style: { 'background-color': color, shape: 'rectangle', width: 80, height: 40 },
    });
  }

  // SD auxiliaries
  for (const [name, spec] of Object.entries(model.auxiliaries || {}) as [string, any][]) {
    const color = SUBSYSTEM_COLORS[spec.subsystem] || '#888';
    elements.push({
      data: { id: name, label: name.replace(/_/g, '\n'), kind: 'auxiliary', subsystem: spec.subsystem, operator: spec.operator },
      style: { 'background-color': color, shape: 'ellipse', width: 70, height: 35 },
    });
    // Edges from inputs
    for (const [src, pol] of Object.entries(spec.inputs || {}) as [string, string][]) {
      elements.push({
        data: {
          id: `${src}->${name}`,
          source: src,
          target: name,
          polarity: pol === 'negative' ? 'negative' : 'positive',
          role: 'input',
        },
      });
    }
  }

  // Stock flows
  for (const [name, spec] of Object.entries(model.stocks || {}) as [string, any][]) {
    for (const inflow of spec.inflows || []) {
      elements.push({
        data: { id: `${inflow}->${name}`, source: inflow, target: name, polarity: 'positive', role: 'inflow' },
      });
    }
    for (const outflow of spec.outflows || []) {
      elements.push({
        data: { id: `${name}->${outflow}`, source: outflow, target: name, polarity: 'negative', role: 'outflow' },
      });
    }
  }

  // Agent templates (bridge view)
  if (showAgents && model.agent_templates) {
    for (const [name, tmpl] of Object.entries(model.agent_templates) as [string, any][]) {
      elements.push({
        data: { id: `agent:${name}`, label: name, kind: 'agent_template', subsystem: '' },
        style: { 'background-color': '#17becf', shape: 'hexagon', width: 80, height: 46 },
        classes: 'agent-node',
      });
      // Reads: SD var -> agent
      for (const readVar of tmpl.reads || []) {
        elements.push({
          data: { id: `${readVar}->agent:${name}`, source: readVar, target: `agent:${name}`, polarity: 'positive', role: 'read' },
          classes: 'agent-edge',
        });
      }
      // Writes: agent -> SD aux
      for (const [, writeSpec] of Object.entries(tmpl.writes || {}) as [string, any][]) {
        elements.push({
          data: { id: `agent:${name}->${writeSpec.target}`, source: `agent:${name}`, target: writeSpec.target, polarity: 'positive', role: 'write' },
          classes: 'agent-edge',
        });
      }
    }
  }

  return elements;
}

export default function GraphEditor() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);
  const model = useStore((s) => s.model);
  const showAgentBridge = useStore((s) => s.showAgentBridge);

  useEffect(() => {
    if (!containerRef.current || cyRef.current) return;

    const cy = cytoscape({
      container: containerRef.current,
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'font-size': '8px',
            color: '#fff',
            'text-wrap': 'wrap',
          },
        },
        {
          selector: 'edge[role="input"]',
          style: { 'line-color': '#2a7', 'target-arrow-color': '#2a7', 'target-arrow-shape': 'triangle', width: 1.5 },
        },
        {
          selector: 'edge[role="input"][polarity="negative"]',
          style: { 'line-color': '#c33', 'target-arrow-color': '#c33', 'line-style': 'dashed' },
        },
        {
          selector: 'edge[role="inflow"]',
          style: { 'line-color': '#2a7', 'target-arrow-color': '#2a7', 'target-arrow-shape': 'triangle', width: 2 },
        },
        {
          selector: 'edge[role="outflow"]',
          style: { 'line-color': '#c33', 'target-arrow-color': '#c33', 'target-arrow-shape': 'triangle', 'line-style': 'dashed', width: 2 },
        },
        {
          selector: '.agent-node',
          style: { 'border-width': 2, 'border-color': '#17becf' },
        },
        {
          selector: '.agent-edge',
          style: { 'line-color': '#1f77b4', 'target-arrow-color': '#1f77b4', 'target-arrow-shape': 'triangle', 'line-style': 'dotted', width: 1 },
        },
      ],
      layout: { name: 'cose', animate: false, nodeRepulsion: 8000 },
      wheelSensitivity: 0.3,
    });

    cyRef.current = cy;
    return () => { cy.destroy(); cyRef.current = null; };
  }, []);

  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !model) return;

    const elements = buildCyElements(model, showAgentBridge);
    cy.elements().remove();
    cy.add(elements);
    cy.layout({ name: 'cose', animate: true, nodeRepulsion: 8000 }).run();
  }, [model, showAgentBridge]);

  const handleReload = useCallback(async () => {
    const m = await fetchModel();
    useStore.getState().setModel(m);
  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <div style={{ position: 'absolute', top: 8, right: 8, zIndex: 10, display: 'flex', gap: 8 }}>
        <button onClick={handleReload} style={{ padding: '4px 12px', background: '#333', color: '#ccc', border: '1px solid #555', borderRadius: 4, cursor: 'pointer' }}>
          Reload
        </button>
      </div>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: only errors for remaining missing components (ControlPanel, Charts, AgentPanel, ScenarioManager).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/GraphEditor.tsx
git commit -m "feat: add Cytoscape.js graph editor with SD nodes/edges and agent bridge"
```

---

### Task 8: Control Panel component

**Files:**
- Create: `frontend/src/components/ControlPanel.tsx`

- [ ] **Step 1: Create ControlPanel.tsx**

```tsx
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
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: errors for Charts, AgentPanel, ScenarioManager (still missing).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ControlPanel.tsx
git commit -m "feat: add control panel with parameter sliders and simulation run controls"
```

---

### Task 9: Charts component (SD trajectories + stochastic bands)

**Files:**
- Create: `frontend/src/components/Charts.tsx`

- [ ] **Step 1: Create Charts.tsx**

```tsx
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
      // Deterministic: build array of { t, var1, var2, ... }
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
      // Stochastic: show median + bands for first var, individual traces for others
      // For simplicity, show only the first ensemble member as reference
      const member = simResult.ensemble[0];
      const t = member.t;
      const stocks = member.stocks || {};
      const auxs = member.auxiliaries || {};
      return t.map((time, i) => {
        const point: Record<string, number> = { t: time };
        for (const v of selectedVars) {
          point[v] = stocks[v]?.[i] ?? auxs[v]?.[i] ?? 0;
        }

        // Add band data for the first selected variable if ensemble exists
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
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: errors for AgentPanel, ScenarioManager (still missing).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Charts.tsx
git commit -m "feat: add trajectory charts with stochastic confidence bands"
```

---

### Task 10: Agent Panel component

**Files:**
- Create: `frontend/src/components/AgentPanel.tsx`

- [ ] **Step 1: Create AgentPanel.tsx**

```tsx
import { useState, useCallback } from 'react';
import { useStore } from '../store';
import { fetchAgentPopulation, initializeAgents, fetchAgentTemplates, createAgentTemplate, deleteAgentTemplate } from '../api';
import type { AgentTemplate, AgentInstance } from '../types';

export default function AgentPanel() {
  const model = useStore((s) => s.model);
  const agentPopulation = useStore((s) => s.agentPopulation);
  const setAgentPopulation = useStore((s) => s.setAgentPopulation);
  const [activeTab, setActiveTab] = useState<'templates' | 'population'>('templates');

  const handleInitialize = useCallback(async () => {
    const result = await initializeAgents();
    const agents = await fetchAgentPopulation();
    setAgentPopulation(agents);
  }, [setAgentPopulation]);

  const templates = model?.agent_templates || {};

  return (
    <div style={{ padding: 12, borderTop: '1px solid #333' }}>
      <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Agents</h3>

      <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
        <button onClick={() => setActiveTab('templates')}
          style={tabStyle(activeTab === 'templates')}>Templates</button>
        <button onClick={() => setActiveTab('population')}
          style={tabStyle(activeTab === 'population')}>Population</button>
        <button onClick={handleInitialize}
          style={{ padding: '4px 10px', background: '#1f77b4', color: '#fff', border: 'none', borderRadius: 3, cursor: 'pointer', fontSize: 11 }}>
          Init
        </button>
      </div>

      {activeTab === 'templates' && (
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {Object.entries(templates).length === 0 && (
            <div style={{ color: '#666', fontSize: 11 }}>No agent templates defined in the model.</div>
          )}
          {Object.entries(templates).map(([name, tmpl]) => (
            <div key={name} style={{ marginBottom: 8, padding: 8, background: '#2a2a3e', borderRadius: 4 }}>
              <div style={{ fontWeight: 600, fontSize: 12, color: '#17becf' }}>{name}</div>
              <div style={{ fontSize: 10, color: '#999', marginTop: 4 }}>
                Count: {tmpl.count} | Topology: {tmpl.topology}
              </div>
              <div style={{ fontSize: 10, color: '#999' }}>
                Reads: [{tmpl.reads?.join(', ')}]
              </div>
              <div style={{ fontSize: 10, color: '#999' }}>
                Rules: {tmpl.decision_rules?.length || 0}
              </div>
              {tmpl.internal_stocks && (
                <div style={{ fontSize: 10, color: '#999' }}>
                  Internal: {Object.keys(tmpl.internal_stocks).join(', ')}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {activeTab === 'population' && (
        <div style={{ maxHeight: 300, overflowY: 'auto' }}>
          {agentPopulation.length === 0 && (
            <div style={{ color: '#666', fontSize: 11 }}>Click "Init" to create agent instances.</div>
          )}
          {agentPopulation.map((agent, i) => (
            <div key={i} style={{ marginBottom: 4, padding: 4, background: '#2a2a3e', borderRadius: 3, fontSize: 10 }}>
              <span style={{ color: '#17becf', fontWeight: 600 }}>{agent.template}</span>
              {' '}
              {Object.entries(agent.state).map(([k, v]) => (
                <span key={k} style={{ marginLeft: 6, color: '#aaa' }}>{k}: {Number(v).toFixed(3)}</span>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function tabStyle(active: boolean): React.CSSProperties {
  return {
    padding: '4px 12px',
    background: active ? '#2a2a3e' : 'transparent',
    color: active ? '#fff' : '#888',
    border: active ? '1px solid #555' : '1px solid transparent',
    borderRadius: 3,
    cursor: 'pointer',
    fontSize: 11,
  };
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: errors for ScenarioManager only.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/AgentPanel.tsx
git commit -m "feat: add agent panel with template viewer and population inspector"
```

---

### Task 11: Scenario Manager component

**Files:**
- Create: `frontend/src/components/ScenarioManager.tsx`

- [ ] **Step 1: Create ScenarioManager.tsx**

```tsx
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
        // Switch active model on backend
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
```

- [ ] **Step 2: Add scenario list/activate endpoints to server.py**

Insert into `backend/server.py` before the simulation section:

```python
# -- Scenario management -----------------------------------------------

@app.get("/api/model/scenarios")
def list_scenarios():
    files = sorted(SCENARIOS_DIR.glob("*.yaml"))
    return {"scenarios": [f.stem for f in files]}


@app.get("/api/model/scenarios/{name}")
def get_scenario(name: str):
    path = SCENARIOS_DIR / f"{name}.yaml"
    if not path.exists():
        raise HTTPException(404, f"Scenario '{name}' not found")
    raw = _load_raw(str(path))
    return ModelOut(
        metadata=raw.get("metadata", {}),
        parameters=raw.get("parameters", {}),
        stocks=raw.get("stocks", {}),
        auxiliaries=raw.get("auxiliaries", {}),
        subsystems=raw.get("subsystems", {}),
        agent_templates=raw.get("agent_templates", {}),
        stochastic=raw.get("stochastic", {}),
    )


@app.post("/api/model/scenarios/{name}/activate")
def activate_scenario(name: str):
    src = SCENARIOS_DIR / f"{name}.yaml"
    if not src.exists():
        raise HTTPException(404, f"Scenario '{name}' not found")
    import shutil
    shutil.copy(str(src), _current_model_path())
    return {"status": "ok", "active": name}
```

- [ ] **Step 3: Verify frontend compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: 0 errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/ScenarioManager.tsx backend/server.py
git commit -m "feat: add scenario manager with file-based scenario switching"
```

---

### Task 12: Integration — Full-stack smoke test

**Files:**
- Modify: `backend/server.py` (ensure all routes working)
- Modify: `frontend/src/App.tsx` (wire up remaining pieces)

- [ ] **Step 1: Add WebSocket streaming endpoint to server.py**

Insert at end of `backend/server.py`:

```python
# -- WebSocket streaming ------------------------------------------------
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/api/sim/stream")
async def sim_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        path = data.get("model_path", _current_model_path())
        n_steps = data.get("n_steps", 101)

        ws = WorldSystem(path)
        stock_names = list(ws.stocks.keys())
        y0 = np.array([ws.stocks[n].initial for n in stock_names], dtype=float)
        ws._stock_names = stock_names
        ws._aux_cache = {}

        t_start = ws.metadata.get("t_start", 2020)
        t_end = ws.metadata.get("t_end", 2120)
        t_eval = np.linspace(t_start, t_end, n_steps)

        from scipy.integrate import solve_ivp
        sol = solve_ivp(ws._derivative, [t_start, t_end], y0,
                        t_eval=t_eval, method="LSODA", rtol=1e-6, atol=1e-9)

        for k in range(len(sol.t)):
            state = {n: float(sol.y[i, k]) for i, n in enumerate(stock_names)}
            await websocket.send_json({
                "step": k,
                "t": float(sol.t[k]),
                "stocks": state,
            })
        await websocket.send_json({"done": True})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})
```

- [ ] **Step 2: Start the backend and verify it serves the model**

```bash
cd backend && python -m uvicorn server:app --host 0.0.0.0 --port 8000 &
sleep 2
curl -s http://localhost:8000/api/model | python -m json.tool | head -20
```
Expected: JSON output of the model — metadata, parameters, stocks.

- [ ] **Step 3: Start the frontend dev server**

```bash
cd frontend && npm run dev &
sleep 3
curl -s http://localhost:5173 | head -10
```
Expected: HTML of the Vite dev server response.

- [ ] **Step 4: Verify full round-trip**

Run a simulation via curl, check that it returns trajectory data:
```bash
curl -s -X POST http://localhost:8000/api/sim/run \
  -H 'Content-Type: application/json' \
  -d '{"mode": "deterministic", "n_steps": 51}' | python -c "import json,sys; d=json.load(sys.stdin); print(f'Steps: {len(d[\"t\"])}, Stocks: {list(d[\"stocks\"].keys())[:5]}')"
```
Expected: `Steps: 51, Stocks: ['population', 'proletarian_pop', ...]`

- [ ] **Step 5: Commit**

```bash
git add backend/server.py frontend/src/App.tsx
git commit -m "feat: add WebSocket streaming and integration smoke-test verification"
```

---

### Task 13: Polish — layout persistence, error handling, empty states

**Files:**
- Modify: `frontend/src/components/GraphEditor.tsx` (layout save/restore)
- Modify: `frontend/src/store.ts` (add layout state)

- [ ] **Step 1: Add layout persistence to store.ts**

```typescript
// Add to AppState interface:
  layout: Record<string, { x: number; y: number }>;
  setLayout: (layout: Record<string, { x: number; y: number }>) => void;
```

```typescript
// Add to create() call:
  layout: {},
  setLayout: (layout) => set({ layout }),
```

- [ ] **Step 2: Save/restore node positions in GraphEditor.tsx**

Add to the `useEffect` that rebuilds elements (after `cy.add(elements)`):

```typescript
  const layout = useStore((s) => s.layout);
  const setLayout = useStore((s) => s.setLayout);

  // Restore saved positions
  if (Object.keys(layout).length > 0) {
    cy.nodes().forEach((node) => {
      const pos = layout[node.id()];
      if (pos) node.position(pos);
    });
  } else {
    cy.layout({ name: 'cose', animate: true, nodeRepulsion: 8000 }).run();
  }

  // Save positions on drag end
  cy.on('dragfree', () => {
    const positions: Record<string, { x: number; y: number }> = {};
    cy.nodes().forEach((node) => {
      positions[node.id()] = { ...node.position() };
    });
    setLayout(positions);
  });
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/GraphEditor.tsx frontend/src/store.ts
git commit -m "feat: persist graph node positions across rebuilds"
```

---

### Task 14: Full-stack test and debug pass

- [ ] **Step 1: Run all backend tests**

```bash
cd backend && python -m pytest ../tests/ -v
```
Expected: all tests pass.

- [ ] **Step 2: Verify frontend TypeScript**

```bash
cd frontend && npx tsc --noEmit
```
Expected: 0 errors.

- [ ] **Step 3: Start backend and test all endpoints**

```bash
# Start fresh
pkill -f uvicorn || true
cd backend && python -m uvicorn server:app --host 0.0.0.0 --port 8000 &
sleep 2

# Test all endpoint groups
echo "=== Model ==="
curl -s http://localhost:8000/api/model/nodes | python -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d[\"nodes\"])} nodes')"

echo "=== Loops ==="
curl -s 'http://localhost:8000/api/loops?max_len=6' | python -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d[\"loops\"])} loops')"

echo "=== Sim Deterministic ==="
curl -s -X POST http://localhost:8000/api/sim/run -H 'Content-Type: application/json' -d '{"mode":"deterministic","n_steps":51}' | python -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d[\"t\"])} steps, id={d.get(\"id\",\"?\")}')"

echo "=== Sim Stochastic ==="
curl -s -X POST http://localhost:8000/api/sim/run -H 'Content-Type: application/json' -d '{"mode":"stochastic","n_steps":51,"n_ensemble":5,"seed":42}' | python -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d[\"ensemble\"])} ensemble members')"

echo "=== Agents ==="
curl -s http://localhost:8000/api/agents/templates | python -c "import json,sys; d=json.load(sys.stdin); print(f'{len(d.get(\"templates\",{}))} templates')"
```

- [ ] **Step 4: Start frontend and verify it loads without console errors**

```bash
cd frontend && npm run dev &
sleep 3
# Manual: open http://localhost:5173 in browser, check DevTools console
```

- [ ] **Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore: debug pass — fix integration issues"
```
