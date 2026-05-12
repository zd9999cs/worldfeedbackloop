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
    if name not in raw.get("parameters", {}):
        raise HTTPException(404, f"Parameter '{name}' not found")
    raw["parameters"][name] = value
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
    if name not in raw.get("agent_templates", {}):
        raise HTTPException(404, f"Agent template '{name}' not found")
    raw["agent_templates"][name] = template
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


# -- Simulation --------------------------------------------------------

# In-memory result cache (simple dict, no persistence across restarts)
_result_cache: dict[str, dict] = {}


@app.post("/api/sim/run")
def run_simulation(req: SimRequest):
    path = _current_model_path()
    raw = _load_raw(path)

    stochastic_cfg = raw.get("stochastic", {})
    sd_noise = stochastic_cfg.get("sd_noise", {})
    has_agents = bool(raw.get("agent_templates"))

    if req.mode == "deterministic":
        rng = np.random.default_rng(req.seed or 42)
        res = _simulate_core(raw, req.n_steps, sd_noise if req.mode == "stochastic" else {}, rng)
        result = {
            "mode": "deterministic",
            "t": res["t"].tolist(),
            "stocks": {k: v.tolist() for k, v in res["stocks"].items()},
            "auxiliaries": {k: v.tolist() for k, v in res["auxiliaries"].items()},
        }
    elif req.mode == "stochastic":
        ensemble = []
        base_rng = np.random.default_rng(req.seed or 42)
        seeds = base_rng.integers(0, 2**31, size=req.n_ensemble)

        for ens_i in range(req.n_ensemble):
            rng = np.random.default_rng(int(seeds[ens_i]))
            res = _simulate_core(raw, req.n_steps, sd_noise, rng)
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


def _simulate_core(raw: dict, n_points: int, sd_noise: dict, rng: np.random.Generator) -> dict:
    """Run a single trajectory with optional agent coupling and per-step noise.

    Uses step-by-step solve_ivp so that agent decisions and stochastic
    shocks are injected BETWEEN integrator steps, not post-hoc.
    """
    from scipy.integrate import solve_ivp

    ws = WorldSystem(model=raw)
    meta = raw.get("metadata", {})
    t_start = float(meta.get("t_start", 2020))
    t_end = float(meta.get("t_end", 2120))

    # Set up agent scheduler if templates exist
    sched = None
    if raw.get("agent_templates"):
        sched = AgentScheduler(raw)
        sched.initialize(rng)

    stock_names = list(ws.stocks.keys())
    y = np.array([ws.stocks[n].initial for n in stock_names], dtype=float)
    ws._stock_names = stock_names
    ws._aux_cache = {}

    t_eval = np.linspace(t_start, t_end, n_points)
    y_traj = np.zeros((len(stock_names), n_points))
    y_traj[:, 0] = y
    aux_traj = {n: np.zeros(n_points) for n in ws._aux_order}

    for i in range(n_points):
        t_curr = t_eval[i]
        state = {n: float(y[j]) for j, n in enumerate(stock_names)}
        ns = ws._evaluate_auxiliaries(state, ws._aux_cache)

        # --- Agent step: agents read SD state, produce aggregated outputs ---
        if sched:
            agent_outputs = sched.step(ns, rng)
            for aux_name, val in agent_outputs.items():
                ns[aux_name] = val
                ws._aux_cache[aux_name] = val

        # --- Inject noise into designated auxiliaries (per-step) ---
        for aux_name, cfg in sd_noise.items():
            if aux_name in ns:
                shock = rng.lognormal(0, cfg["noise_scale"])
                ns[aux_name] = ns[aux_name] * shock
                ws._aux_cache[aux_name] = ns[aux_name]

        # Record auxiliaries at this step
        for n in ws._aux_order:
            aux_traj[n][i] = ns.get(n, 0.0)
            ws._aux_cache[n] = ns.get(n, 0.0)

        # Integrate to next output point (or record last point)
        if i < n_points - 1:
            t_next = t_eval[i + 1]
            sol = solve_ivp(
                ws._derivative, [t_curr, t_next], y,
                method="LSODA", rtol=1e-6, atol=1e-9,
                t_eval=[t_next],
            )
            if sol.success:
                y = sol.y[:, -1]
                y_traj[:, i + 1] = y
            else:
                # Fallback: Euler step
                dy = ws._derivative(t_curr, y)
                y = y + dy * (t_next - t_curr)
                y_traj[:, i + 1] = y

    return {
        "t": t_eval,
        "stocks": {n: y_traj[j] for j, n in enumerate(stock_names)},
        "auxiliaries": aux_traj,
    }


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
