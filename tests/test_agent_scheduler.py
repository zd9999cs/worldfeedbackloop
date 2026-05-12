# tests/test_agent_scheduler.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

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
                    {"condition": "rate_of_profit > 0.1", "action": "bank_lending: capital * 0.1"},
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
