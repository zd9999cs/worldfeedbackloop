# tests/test_sim.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
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
