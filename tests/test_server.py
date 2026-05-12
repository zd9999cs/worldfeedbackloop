# tests/test_server.py
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import yaml
import sys

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

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
