"""
World System simulator
======================
Loads the canonical YAML model, builds an ODE system, integrates it,
and exposes a clean Python API for extension and analysis.

Quick start
-----------
    from simulator import WorldSystem
    ws = WorldSystem("model.yaml")
    res = ws.simulate()
    ws.plot(res)

Extension pattern
-----------------
    ws.add_stock("ai_capability", initial=0.1, inflows=["ai_growth"],
                 outflows=[], subsystem="knowledge")
    ws.add_auxiliary("ai_growth",
                     equation="0.08 * ai_capability * (graduate_pop / 2e8)",
                     inputs={"ai_capability": "positive",
                             "graduate_pop": "positive"},
                     operator="multiply", subsystem="knowledge")
    ws.recompile()
    res2 = ws.simulate()

Graph analysis
--------------
    G = ws.graph()              # NetworkX DiGraph
    loops = ws.find_loops()     # list of cycles with polarities
"""

from __future__ import annotations

import math
import re
import copy
from dataclasses import dataclass, field
from typing import Any

import yaml
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------
# Safe expression evaluation
# ---------------------------------------------------------------------
# We allow a fixed set of math functions plus model variables/parameters.
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round,
    "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pow": pow,
}


def _safe_eval(expr: str, namespace: dict[str, float]) -> float:
    return eval(expr, {"__builtins__": _SAFE_BUILTINS}, namespace)


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------
@dataclass
class Stock:
    name: str
    initial: float
    units: str = ""
    inflows: list[str] = field(default_factory=list)
    outflows: list[str] = field(default_factory=list)
    subsystem: str = ""


@dataclass
class Auxiliary:
    name: str
    equation: str
    operator: str = "identity"   # multiply | add | divide | identity
    inputs: dict[str, str] = field(default_factory=dict)  # {var: polarity}
    subsystem: str = ""
    units: str = ""


# ---------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------
class WorldSystem:

    def __init__(self, yaml_path: str | None = None, *, model: dict | None = None):
        if yaml_path is not None:
            with open(yaml_path) as f:
                model = yaml.safe_load(f)
        elif model is None:
            raise ValueError("Provide either yaml_path or model dict.")
        self.raw = copy.deepcopy(model)
        self._build()

    # -- internal: parse YAML into dataclasses --------------------------------
    def _build(self) -> None:
        m = self.raw
        self.metadata = m.get("metadata", {})
        # Coerce to float — YAML 1.1 reads "3.0e10" as a string.
        self.parameters = {k: float(v) for k, v in m.get("parameters", {}).items()}
        self.subsystems = m.get("subsystems", {})

        self.stocks: dict[str, Stock] = {}
        for name, spec in m.get("stocks", {}).items():
            self.stocks[name] = Stock(
                name=name,
                initial=float(spec["initial"]),
                units=spec.get("units", ""),
                inflows=list(spec.get("inflows", [])),
                outflows=list(spec.get("outflows", [])),
                subsystem=spec.get("subsystem", ""),
            )

        self.auxiliaries: dict[str, Auxiliary] = {}
        for name, spec in m.get("auxiliaries", {}).items():
            self.auxiliaries[name] = Auxiliary(
                name=name,
                equation=spec["equation"],
                operator=spec.get("operator", "identity"),
                inputs=dict(spec.get("inputs", {})),
                subsystem=spec.get("subsystem", ""),
                units=spec.get("units", ""),
            )

        self._resolve_order()

    # -- topological sort over auxiliaries ------------------------------------
    def _resolve_order(self) -> None:
        """Order auxiliaries so each is evaluated after its inputs.
        Cycles among auxiliaries are broken by skipping unresolved deps
        and trusting the previous timestep's value (held in self._cache)."""
        remaining = dict(self.auxiliaries)
        ordered: list[str] = []
        known = set(self.stocks) | set(self.parameters)
        guard = 0
        while remaining and guard < len(self.auxiliaries) + 5:
            progress = False
            for name, aux in list(remaining.items()):
                deps = set(aux.inputs.keys())
                # also pull any identifiers from the equation
                deps |= self._idents(aux.equation)
                deps -= set(_SAFE_BUILTINS)
                unresolved = deps - known - {name}
                if not unresolved:
                    ordered.append(name)
                    known.add(name)
                    remaining.pop(name)
                    progress = True
            if not progress:
                # Break a cycle: take any remaining aux and assume its
                # cyclic dependencies use last-step values.
                stuck = next(iter(remaining))
                ordered.append(stuck)
                known.add(stuck)
                remaining.pop(stuck)
            guard += 1
        self._aux_order = ordered

    @staticmethod
    def _idents(expr: str) -> set[str]:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))

    def recompile(self) -> None:
        """Re-parse stocks/auxiliaries after programmatic edits."""
        self._build()

    # -- extension API --------------------------------------------------------
    def add_stock(self, name: str, *, initial: float,
                  inflows: list[str] | None = None,
                  outflows: list[str] | None = None,
                  units: str = "", subsystem: str = "") -> None:
        self.raw.setdefault("stocks", {})[name] = {
            "initial": initial,
            "units": units,
            "inflows": inflows or [],
            "outflows": outflows or [],
            "subsystem": subsystem,
        }

    def add_auxiliary(self, name: str, *, equation: str,
                      operator: str = "identity",
                      inputs: dict[str, str] | None = None,
                      subsystem: str = "", units: str = "") -> None:
        self.raw.setdefault("auxiliaries", {})[name] = {
            "equation": equation,
            "operator": operator,
            "inputs": inputs or {},
            "subsystem": subsystem,
            "units": units,
        }

    def set_parameter(self, name: str, value: float) -> None:
        self.raw.setdefault("parameters", {})[name] = value
        self.parameters[name] = value

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False, default_flow_style=False)

    # -- evaluation -----------------------------------------------------------
    def _evaluate_auxiliaries(self, state: dict[str, float],
                              cache: dict[str, float]) -> dict[str, float]:
        ns: dict[str, Any] = {}
        ns.update(self.parameters)
        ns.update(state)
        # seed cyclic vars with previous values
        for n in self._aux_order:
            if n in cache:
                ns[n] = cache[n]
        for name in self._aux_order:
            try:
                ns[name] = _safe_eval(self.auxiliaries[name].equation, ns)
            except Exception as e:
                raise RuntimeError(
                    f"Error evaluating auxiliary '{name}': {e}\n"
                    f"  equation: {self.auxiliaries[name].equation}"
                ) from e
        return ns

    def _derivative(self, t: float, y: np.ndarray) -> np.ndarray:
        state = {name: float(y[i]) for i, name in enumerate(self._stock_names)}
        ns = self._evaluate_auxiliaries(state, self._aux_cache)
        # update cache for next call (used to break aux cycles)
        for name in self._aux_order:
            self._aux_cache[name] = ns[name]

        dy = np.zeros_like(y)
        for i, name in enumerate(self._stock_names):
            stock = self.stocks[name]
            inflow  = sum(ns[fl] for fl in stock.inflows)
            outflow = sum(ns[fl] for fl in stock.outflows)
            dy[i] = inflow - outflow
        return dy

    # -- public simulation ----------------------------------------------------
    def simulate(self, t_start: float | None = None,
                 t_end:   float | None = None,
                 n_points: int = 401,
                 method: str = "LSODA") -> dict:
        if t_start is None:
            t_start = self.metadata.get("t_start", 0.0)
        if t_end is None:
            t_end = self.metadata.get("t_end", 100.0)

        self._stock_names = list(self.stocks.keys())
        y0 = np.array([self.stocks[n].initial for n in self._stock_names], dtype=float)
        self._aux_cache: dict[str, float] = {}

        t_eval = np.linspace(t_start, t_end, n_points)
        sol = solve_ivp(self._derivative, [t_start, t_end], y0,
                        t_eval=t_eval, method=method,
                        rtol=1e-6, atol=1e-9)
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")

        # Recompute auxiliaries along the trajectory for plotting
        aux_traj = {n: np.empty_like(sol.t) for n in self._aux_order}
        self._aux_cache = {}
        for k, t in enumerate(sol.t):
            state = {n: float(sol.y[i, k]) for i, n in enumerate(self._stock_names)}
            ns = self._evaluate_auxiliaries(state, self._aux_cache)
            for n in self._aux_order:
                aux_traj[n][k] = ns[n]
                self._aux_cache[n] = ns[n]

        return {
            "t": sol.t,
            "stocks": {n: sol.y[i] for i, n in enumerate(self._stock_names)},
            "auxiliaries": aux_traj,
        }

    # -- plotting -------------------------------------------------------------
    def plot(self, res: dict, variables: list[str] | None = None,
             out_path: str | None = None) -> None:
        import matplotlib.pyplot as plt
        if variables is None:
            variables = ["population", "capital_stock", "rate_of_profit",
                         "oil_stock", "oil_price", "co2",
                         "food_prices", "biofuel_use",
                         "world_tension", "youth_unemployment",
                         "graduate_pop", "rate_of_revolutions"]
        n = len(variables)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.7 * nrows))
        axes = axes.ravel()
        t = res["t"]
        for ax, var in zip(axes, variables):
            if var in res["stocks"]:
                ax.plot(t, res["stocks"][var], lw=1.8)
            elif var in res["auxiliaries"]:
                ax.plot(t, res["auxiliaries"][var], lw=1.8, color="C1")
            else:
                ax.text(0.5, 0.5, f"unknown: {var}", ha="center",
                        transform=ax.transAxes)
            ax.set_title(var, fontsize=10)
            ax.grid(alpha=0.3)
        for ax in axes[len(variables):]:
            ax.axis("off")
        fig.suptitle(self.metadata.get("name", "World System"), fontsize=12)
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"saved -> {out_path}")
        return fig

    # -- graph / loop analysis ------------------------------------------------
    def graph(self):
        import networkx as nx
        G = nx.DiGraph()
        for name, st in self.stocks.items():
            G.add_node(name, kind="stock", subsystem=st.subsystem)
        for name, ax in self.auxiliaries.items():
            G.add_node(name, kind="auxiliary",
                       subsystem=ax.subsystem, operator=ax.operator)
        # auxiliary inputs -> auxiliary
        for name, ax in self.auxiliaries.items():
            for src, pol in ax.inputs.items():
                G.add_edge(src, name, polarity=pol)
        # stock flows: rate -> stock (inflow +), rate -> stock (outflow -)
        for name, st in self.stocks.items():
            for fl in st.inflows:
                G.add_edge(fl, name, polarity="positive", role="inflow")
            for fl in st.outflows:
                G.add_edge(fl, name, polarity="negative", role="outflow")
        return G

    def find_loops(self, max_len: int = 8) -> list[dict]:
        """Enumerate simple cycles with their net polarity."""
        import networkx as nx
        G = self.graph()
        loops = []
        for cyc in nx.simple_cycles(G):
            if len(cyc) > max_len:
                continue
            sign = 1
            edges = []
            for a, b in zip(cyc, cyc[1:] + [cyc[0]]):
                pol = G[a][b].get("polarity", "positive")
                sign *= -1 if pol == "negative" else 1
                edges.append((a, b, pol))
            loops.append({
                "nodes": cyc,
                "edges": edges,
                "polarity": "reinforcing" if sign > 0 else "balancing",
            })
        return loops


# ---------------------------------------------------------------------
# CLI: python simulator.py model.yaml
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "model.yaml"
    ws = WorldSystem(path)
    print(f"Loaded model: {ws.metadata.get('name')}")
    print(f"  stocks:      {len(ws.stocks)}")
    print(f"  auxiliaries: {len(ws.auxiliaries)}")
    print(f"  parameters:  {len(ws.parameters)}")
    res = ws.simulate()
    print(f"Simulated {len(res['t'])} timesteps "
          f"from {res['t'][0]:.0f} to {res['t'][-1]:.0f}")
    loops = ws.find_loops()
    rein = sum(1 for l in loops if l["polarity"] == "reinforcing")
    bal  = sum(1 for l in loops if l["polarity"] == "balancing")
    print(f"Found {len(loops)} feedback loops ({rein} reinforcing, {bal} balancing)")
