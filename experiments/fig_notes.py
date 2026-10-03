# This code is part of qcr.
#
# (C) Copyright Arnav Kapoor 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Explanatory figures for the study notes (not in the paper)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from common import FIGS

from qcr.compute_graph import build
from qcr.gates import make_pool

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C = ["#1f4e79", "#2e86c1", "#76b7e5", "#f39c12", "#c0392b"]

# 1) the complete 1-qubit compute graph (= Cayley graph of the 24-element Clifford group)
pool = make_pool("ion", 1)
cg = build(1, pool, 10)
G = nx.DiGraph()
for i in range(cg.num_nodes):
    G.add_node(i)
lab = {"rx": "X", "ry": "Y", "rz": "Z"}
from qcr.gates import gate_matrix
from qcr.unitary import phase_key

edges = {}
for i in range(cg.num_nodes):
    for k, g in enumerate(pool):
        j = cg.index[phase_key(gate_matrix(g, 1) @ cg.unitaries[i])]
        if g.theta > 0:
            edges.setdefault(lab[g.name], []).append((i, j))
fig, ax = plt.subplots(figsize=(6.2, 5.2))
shells = [list(np.flatnonzero(cg.dist == d)) for d in range(cg.dist.max() + 1)]
pos = nx.shell_layout(G, nlist=shells)
cols = {"X": C[4], "Y": C[3], "Z": C[1]}
for name, es in edges.items():
    und = {tuple(sorted(e)) for e in es}
    nx.draw_networkx_edges(
        G.to_undirected(),
        pos,
        edgelist=list(und),
        edge_color=cols[name],
        width=1.4,
        alpha=0.75,
        ax=ax,
        label=f"R{name}(±π/2)",
    )
nx.draw_networkx_nodes(
    G,
    pos,
    node_color=[plt.cm.Blues(0.35 + 0.2 * d) for d in cg.dist],
    node_size=260,
    edgecolors="k",
    linewidths=0.6,
    ax=ax,
)
nx.draw_networkx_labels(G, pos, {i: str(cg.dist[i]) for i in G}, font_size=8, ax=ax, font_color="white")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_title(
    "Complete compute graph of one qubit: 24 nodes = the single-qubit Clifford group\n"
    "(node label = shortest-circuit length; shells = depth 0,1,2,3)",
    fontsize=9.5,
)
ax.axis("off")
fig.tight_layout()
fig.savefig(FIGS / "notes_1q_graph.png", dpi=200)

# 2) growth of the 2-qubit compute graph until it saturates
pool2 = make_pool("ion_pm", 2)
cg2 = build(2, pool2, 20)
lv = np.array(cg2.level_sizes)
cum = np.cumsum(lv)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
d = np.arange(len(lv))
ax.bar(d, lv, color=C[1], label="new nodes at exactly depth d")
ax.plot(d, cum, "o-", color=C[0], label="nodes in graph of depth d (Table I)")
ax.axhline(11520, ls="--", color=C[4], lw=1)
ax.text(0.2, 11800, "group order |C₂/U(1)| = 11,520 (graph stops growing)", color=C[4], fontsize=8.5)
for k in range(1, 7):
    ax.annotate(f"{cum[k]:,}", (k, cum[k]), textcoords="offset points", xytext=(-14, 6), fontsize=7.5)
ax.axvspan(0.5, 6.5, color="0.92", zorder=0)
ax.text(1.0, 6800, "depths reported\nin the paper", fontsize=8, color="0.35")
ax.set_xlabel("depth d (circuit length)")
ax.set_ylabel("number of unitaries")
ax.set_xticks(d)
ax.legend(frameon=False, fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(FIGS / "notes_2q_growth.png", dpi=200)
print("levels", lv.tolist())
