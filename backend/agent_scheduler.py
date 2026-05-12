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
