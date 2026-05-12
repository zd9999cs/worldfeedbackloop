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
from agent_scheduler import AgentScheduler
import numpy as np

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


class SimRequest(BaseModel):
    mode: str = "deterministic"  # deterministic | stochastic
    n_steps: int = 401
    n_ensemble: int = 10
    seed: int | None = None


class AgentTemplateSpec(BaseModel):
    name: str
    template: dict[str, Any]


class InitAgentsRequest(BaseModel):
    seed: int | None = None

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


# -- Agent CRUD --------------------------------------------------------

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


# -- Simulation --------------------------------------------------------

# In-memory result cache (simple dict, no persistence across restarts)
_result_cache: dict[str, dict] = {}


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
        ensemble = []
        base_rng = np.random.default_rng(req.seed or 42)
        seeds = base_rng.integers(0, 2**31, size=req.n_ensemble)

        for ens_i in range(req.n_ensemble):
            rng = np.random.default_rng(int(seeds[ens_i]))
            ws_i = WorldSystem(model=raw)
            # If agent templates exist, initialize them
            has_agents = bool(raw.get("agent_templates"))
            if has_agents:
                sched = AgentScheduler(raw)
                sched.initialize(rng)

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
