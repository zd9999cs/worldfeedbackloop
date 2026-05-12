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
