"""
Extension example: add an AI/automation feedback loop.

Demonstrates the pattern for extending the model with a new subsystem that
interacts with several existing variables. The new loop:

    ai_capability  (new stock)
        ↑
        ├── grows from scientific_knowledge   (+)
        ├── grows from rate_of_accumulation   (+)   [needs investment]
        ├── boosts rate_of_industrial_activity (+)  [labour productivity]
        └── raises youth_unemployment          (+)  [job displacement]

This is the "rising organic composition via AI" loop: AI raises productivity
(boosting profit short-term) but displaces labour (raising unemployment and
potentially feeding revolt), while also requiring continued investment to
grow. It interacts with three existing subsystems.

Two ways to do this:
  1. Edit model.yaml directly (canonical, version-controllable)
  2. Use the Python API to mutate the loaded model in memory (this file)
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from simulator import WorldSystem


def add_ai_loop(ws: WorldSystem) -> None:
    """Add an AI/automation subsystem to a loaded model in place."""

    # New parameters for the AI subsystem
    ws.set_parameter("ai_growth_from_knowledge", 0.030)
    ws.set_parameter("ai_investment_threshold", 0.05)    # min accumulation rate to grow
    ws.set_parameter("ai_decay",                0.005)
    ws.set_parameter("ai_productivity_boost",   0.40)    # multiplier saturation
    ws.set_parameter("ai_unemployment_pressure", 0.06)   # how strongly AI raises unemp

    # New stock
    ws.add_stock(
        "ai_capability",
        initial=0.05,                       # ~early-2020s baseline
        inflows=["ai_growth"],
        outflows=["ai_attrition"],
        units="index",
        subsystem="knowledge",
    )

    # Growth: depends on knowledge AND on having investment available
    ws.add_auxiliary(
        "ai_growth",
        equation="ai_growth_from_knowledge * scientific_knowledge * "
                 "max(0.0, rate_of_accumulation - ai_investment_threshold) * "
                 "max(0.0, 1.0 - ai_capability)",                    # saturates at 1
        operator="multiply",
        inputs={"scientific_knowledge": "positive",
                "rate_of_accumulation":  "positive",
                "ai_capability":         "negative"},
        subsystem="knowledge",
    )
    ws.add_auxiliary(
        "ai_attrition",
        equation="ai_decay * ai_capability",
        operator="multiply",
        inputs={"ai_capability": "positive"},
        subsystem="knowledge",
    )

    # Couple AI -> industrial activity (productivity boost).
    # We REPLACE the existing rate_of_industrial_activity equation so it
    # incorporates the AI multiplier.
    ws.raw["auxiliaries"]["rate_of_industrial_activity"]["equation"] = (
        "(population / 7.8e9) * industrialisation * "
        "(1.0 + ai_productivity_boost * ai_capability)"
    )
    ws.raw["auxiliaries"]["rate_of_industrial_activity"]["inputs"]["ai_capability"] = "positive"

    # Couple AI -> youth unemployment (displacement).
    # Add an extra term to unemployment_change.
    ws.raw["auxiliaries"]["unemployment_change"]["equation"] = (
        "youth_unemp_growth_coupling * max(0.0, "
        "  (graduate_inflow / max(graduate_pop, 1.0e6)) "
        "  - (rate_of_accumulation / max(capital_stock, 0.01)) "
        "  + ai_unemployment_pressure * ai_capability"
        ") * max(0.0, 1.0 - youth_unemployment) "
        "- youth_unemp_damping * youth_unemployment"
    )
    ws.raw["auxiliaries"]["unemployment_change"]["inputs"]["ai_capability"] = "positive"

    ws.recompile()


def compare_runs() -> None:
    # Baseline
    baseline = WorldSystem("model.yaml")
    res_b = baseline.simulate()

    # With AI loop
    extended = WorldSystem("model.yaml")
    add_ai_loop(extended)
    res_e = extended.simulate()

    print(f"Baseline: {len(baseline.stocks)} stocks, "
          f"{len(baseline.find_loops())} loops")
    print(f"Extended: {len(extended.stocks)} stocks, "
          f"{len(extended.find_loops())} loops")

    # Plot side-by-side
    variables = ["rate_of_industrial_activity", "rate_of_profit",
                 "ai_capability", "youth_unemployment",
                 "capital_stock", "revolt_pressure"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    axes = axes.ravel()

    def get(res, name):
        return res["stocks"].get(name, res["auxiliaries"].get(name))

    for ax, var in zip(axes, variables):
        b = get(res_b, var)
        e = get(res_e, var)
        if b is not None:
            ax.plot(res_b["t"], b, lw=1.8, label="baseline", color="#888")
        if e is not None:
            ax.plot(res_e["t"], e, lw=2.0, label="with AI loop", color="C3")
        ax.set_title(var, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Baseline vs Extended (AI/automation loop)", fontsize=12)
    fig.tight_layout()
    fig.savefig("extension_comparison.png", dpi=130, bbox_inches="tight")
    print("saved -> extension_comparison.png")

    # Save the extended model to YAML for record
    extended.save_yaml("model_extended.yaml")
    print("saved -> model_extended.yaml")


if __name__ == "__main__":
    compare_runs()
