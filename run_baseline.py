"""Run the baseline World System simulation and produce plots + diagram."""
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from simulator import WorldSystem


def plot_trajectories(ws: WorldSystem, res: dict, out_path: str) -> None:
    ws.plot(res, out_path=out_path)


def plot_loop_diagram(ws: WorldSystem, out_path: str) -> None:
    """Draw the causal loop diagram with subsystem coloring,
    operator markers (×, +, ÷), and signed edges."""
    G = ws.graph()
    fig, ax = plt.subplots(figsize=(16, 11))

    subs = ws.subsystems
    color_for = lambda n: subs.get(G.nodes[n].get("subsystem", ""), {}).get("color", "#888")

    # Try a layered layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="neato")
    except Exception:
        pos = nx.spring_layout(G, seed=2, k=1.8, iterations=200)

    # Edges by polarity
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("polarity") == "positive"]
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("polarity") == "negative"]

    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color="#2a7", width=0.9,
                           arrows=True, arrowsize=10, alpha=0.7, ax=ax,
                           connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color="#c33", width=0.9,
                           arrows=True, arrowsize=10, alpha=0.7, style="dashed", ax=ax,
                           connectionstyle="arc3,rad=0.05")

    # Nodes: rectangles for stocks, circles for auxiliaries
    stock_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "stock"]
    aux_nodes   = [n for n, d in G.nodes(data=True) if d["kind"] == "auxiliary"]

    nx.draw_networkx_nodes(G, pos, nodelist=stock_nodes,
                           node_color=[color_for(n) for n in stock_nodes],
                           node_shape="s", node_size=1100,
                           edgecolors="black", linewidths=1.4, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=aux_nodes,
                           node_color=[color_for(n) for n in aux_nodes],
                           node_shape="o", node_size=700, alpha=0.85,
                           edgecolors="black", linewidths=0.8, ax=ax)

    # Labels
    labels = {n: n.replace("rate_of_", "").replace("_", "\n") for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6.5, ax=ax)

    # Legend
    patches = [mpatches.Patch(color=v["color"], label=v["label"]) for v in subs.values()]
    patches.append(mpatches.Patch(color="#2a7", label="+ influence"))
    patches.append(mpatches.Patch(color="#c33", label="− influence"))
    ax.legend(handles=patches, loc="lower left", fontsize=8, frameon=True)
    ax.set_title("World System — causal loop diagram (rect = stock, circle = rate/aux)",
                 fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"saved -> {out_path}")


def print_loops(ws: WorldSystem, top: int = 12) -> None:
    loops = ws.find_loops()
    loops.sort(key=lambda l: len(l["nodes"]))
    print(f"\n{len(loops)} feedback loops detected. Showing shortest {top}:\n")
    for L in loops[:top]:
        kind = "R+" if L["polarity"] == "reinforcing" else "B−"
        path = " → ".join(L["nodes"] + [L["nodes"][0]])
        print(f"  [{kind}]  {path}")


if __name__ == "__main__":
    ws = WorldSystem("model.yaml")
    print(f"Model: {ws.metadata['name']}")
    print(f"  {len(ws.stocks)} stocks, {len(ws.auxiliaries)} auxiliaries, "
          f"{len(ws.parameters)} parameters")

    res = ws.simulate()
    plot_trajectories(ws, res, "baseline_trajectories.png")
    plot_loop_diagram(ws, "loop_diagram.png")
    print_loops(ws, top=15)
