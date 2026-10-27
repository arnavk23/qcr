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

"""Every figure of the paper and the talks (manuscripts/paper/figs/, copies in manuscripts/talk/figs/).

  python make_paper_figs.py [compute_graph growth window_grid examples paper_flow paper_bench cliffordt
                             hw_flow tvd layout hw_real]

Every figure is drawn from results/ (and from small computations repeated here, e.g. the compute graphs). The look
(sizes, colours, markers, legend placement, circuit scale) comes from figstyle.py. The example circuits are computed
once and cached in results/fig_examples.json, so the paper and the talks show the same circuits.
"""

import json
import sys

import numpy as np
from common import RESULTS
from figstyle import (
    FAINT,
    GRID,
    INK,
    MUTED,
    SHADE,
    W,
    circuit,
    col,
    fig_legend,
    handle,
    lab,
    legend_above,
    light_grid,
    mk,
    panel,
    PAPER_FIGS,
    plt,
    save,
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

BLUES = LinearSegmentedColormap.from_list("blues", ["#f2f6fc", "#9cc3f0", "#2a78d6", "#0b3c6e"])
DIST = LinearSegmentedColormap.from_list("dist", ["#0b3c6e", "#2a78d6", "#9cc3f0"])  # dark = close to identity


def J(name):
    p = RESULTS / name
    if not p.exists() and p.with_suffix(".jsonl").exists():  # unfinished run: use the rows so far
        return [json.loads(line) for line in open(p.with_suffix(".jsonl"))]
    return json.load(open(p))


def tot(c):
    return sum(c.values()) if isinstance(c, dict) else c


def gmean(v):
    return float(np.exp(np.mean(np.log(np.asarray(v, float)))))


def endlabel(ax, x, y, text, color, dy=0, ha="left"):
    ax.annotate(text, (x, y), xytext=(3 if ha == "left" else -3, dy), textcoords="offset points", va="center",
                ha=ha, fontsize=6.3, color=INK)  # fmt: skip


# ------------------------------------------------------------------ compute graphs
def compute_graph():
    """Two compute graphs of the same size: the 2-qubit Clifford ball closes up, the Clifford+T ball is a tree."""
    import networkx as nx

    from qcr.cliffordt import CliffordTBall, canonical, keys_of, matmul
    from qcr.cliffordt import exact_gate as ct_gate
    from qcr.compute_graph import build
    from qcr.gates import gate_matrix, make_pool
    from qcr.unitary import phase_key

    pool = make_pool("ion_pm", 2)
    cg = build(2, pool, 3)
    G = nx.Graph()
    for i in range(cg.num_nodes):
        G.add_node(i, d=int(cg.dist[i]))
    mats = [gate_matrix(g, 2) for g in pool]
    for i in range(cg.num_nodes):
        for M in mats:
            j = cg.index.get(phase_key(M @ cg.unitaries[i]))
            if j is not None and j != i:
                G.add_edge(i, j)
    ball = CliffordTBall(2).build(keep_matrices_up_to=2)
    H = nx.Graph()
    idx = np.concatenate([s[2] for s in ball.spheres])
    Ms = np.concatenate([s[0] for s in ball.spheres])
    ks = np.concatenate([s[1] for s in ball.spheres])
    for i in idx:
        H.add_node(int(i), d=int(ball.dist[i]))
    for g in ball.pool:
        Gm, kg = ct_gate(g)
        C, kc = matmul(Gm, kg, Ms, ks)
        P, kp = canonical(C, kc)
        hit = ball._lookup(keys_of(P, kp))
        for a, b in zip(idx, hit):
            if b >= 0 and a != b:
                H.add_edge(int(a), int(b))
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.75))
    for ax, Gr, letter, title in (
        (axs[0], G, "a", "Clifford, depth $\\leq3$"),
        (axs[1], H, "b", "Clifford+T, depth $\\leq2$"),
    ):
        pos = nx.kamada_kawai_layout(Gr) if Gr.number_of_nodes() < 700 else nx.spring_layout(Gr, seed=1)
        d = np.array([Gr.nodes[v]["d"] for v in Gr])
        nx.draw_networkx_edges(Gr, pos, ax=ax, width=0.15, alpha=0.4, edge_color=MUTED)
        sc = ax.scatter([pos[v][0] for v in Gr], [pos[v][1] for v in Gr], c=d, cmap=DIST, vmin=0, vmax=3,
                        s=[22 if x == 0 else 5 for x in d], zorder=3, linewidths=0)  # fmt: skip
        panel(ax, letter, f"{title}: {Gr.number_of_nodes()} nodes, {Gr.number_of_edges()} edges")
        ax.set_axis_off()
    cb = fig.colorbar(sc, ax=axs, orientation="horizontal", fraction=0.05, pad=0.03, aspect=40, ticks=[0, 1, 2, 3])
    cb.set_label("optimal gate count (distance from the identity)")
    cb.outline.set_linewidth(0.4)
    save(fig, "fig_compute_graph")


# ------------------------------------------------------------------ group growth and databases
def _spheres(ax):
    cay3 = J("cayley3_ion_pm.json")["levels"]
    cay3n = J("cayley3_nisq.json")["levels"]
    from qcr.gates import make_pool
    from qcr.tableau import cayley_bfs

    cay2 = cayley_bfs(2, make_pool("ion_pm", 2), verbose=False)[2]
    ct = J("ext_cliffordt.json")["ball"]["levels"] if (RESULTS / "ext_cliffordt.json").exists() else None
    ax.axvspan(-0.4, 5.4, color=SHADE, zorder=0, lw=0)
    ax.text(2.5, 0.97, "original\ndatabase", ha="center", va="top", fontsize=6, color=MUTED,
            transform=ax.get_xaxis_transform())  # fmt: skip
    series = [
        (cay2, "o", "-", "2q Clifford, ion"),
        (cay3, "s", "-", "3q Clifford, ion"),
        (cay3n, "^", "--", "3q Clifford, IBM-style"),
    ]
    if ct:
        series.append((ct, "D", ":", "2q Clifford+T (infinite)"))
    for lv, m, ls, name in series:
        ax.semilogy(range(len(lv)), lv, marker=m, ls=ls, ms=3, color=INK, lw=1.0, mfc="white", mew=0.8, label=name)
    ax.set_xlabel("optimal gate count $d$")
    ax.set_ylabel("elements $|S(d)|$")
    ax.set_xlim(-0.6, 20)
    ax.set_xticks(range(0, 21, 4))
    light_grid(ax)


def growth():
    """(a) exact sphere sizes of every Cayley graph used as ground truth, (b) level sizes of our databases."""
    gdb = J("gdb_info.json")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw=dict(wspace=0.32))
    ax = axs[0]
    _spheres(ax)
    legend_above(ax, ncol=2, pad=0.12)
    panel(ax, "a")
    ax = axs[1]
    names = {"heron/line": "Heron", "ion/tri": "ion", "ion_paper/tri": "ion, $\\pm\\pi/2$", "ibm_paper/tri": "IBM, $\\pm\\pi/2$", "nam/tri": "Nam"}
    mks = {"heron/line": "o", "ion/tri": "s", "ion_paper/tri": "^", "ibm_paper/tri": "v", "nam/tri": "D"}
    for key, info in gdb.items():
        dev, topo, kind = key.split("/")
        if kind != "phase":
            continue  # input- and output-aware variants are listed in the database table
        lv = info["levels"]
        ax.semilogy(range(len(lv)), lv, marker=mks[f"{dev}/{topo}"], ms=3, color=col("general"), mfc="white", mew=0.8,
                    lw=1.0, label=f"{names[f'{dev}/{topo}']} ({info['size'] / 1e6:.1f}M)")  # fmt: skip
    ax.axvspan(-0.4, 5.4, color=SHADE, zorder=0, lw=0)
    ax.set_xlim(-0.6, 11.5)
    ax.set_xticks(range(0, 12, 2))
    ax.set_xlabel("database depth")
    ax.set_ylabel("new classes per depth")
    light_grid(ax)
    legend_above(ax, ncol=3, pad=0.12)
    panel(ax, "b")
    save(fig, "fig_growth")
    # talk: the 3-qubit spheres alone
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    _spheres(ax)
    legend_above(ax, ncol=2, pad=0.04)
    save(fig, "ext_growth", talk=True)


# ------------------------------------------------------------------ window and database radius
def _window(ax):
    ws = [2, 3, 4, 6, 8, 12, 16, 24, 32]
    for k, name, ls, m in (("ion", "ion trap", "-", "o"), ("nisq", "IBM-style", "--", "s")):
        rs = J(f"ext_gap3_{k}.json")
        mean = [np.mean([r[f"ew{w}"] for r in rs]) for w in ws]
        ax.plot(ws, mean, ls=ls, marker=m, ms=3.5, color=col("cliff"), mfc="white", mew=0.9, label=f"exact window, {name}")
        ax.axhline(np.mean([r["opt"] for r in rs]), color=col("optimum"), ls=ls, lw=1.0)
        ax.plot([8], [np.mean([r["faithful"] for r in rs])], marker="^", ms=6, ls="none", color=col("original"),
                mec="white", mew=0.6, zorder=4)  # fmt: skip
    ax.set_xscale("log", base=2)
    ax.set_xticks(ws, ws)
    ax.minorticks_off()
    ax.set_xlabel("window length $w$ (gates)")
    ax.set_ylabel("output gates (mean of 20)")
    light_grid(ax)
    hs = [
        handle("cliff", label="exact window, ion", marker="o", mfc="white"),
        handle("cliff", label="exact window, IBM-style", ls="--", marker="s", mfc="white"),
        handle("original", line=False, label="original method ($w=8$)"),
        handle("optimum", label="proven optimum", marker=None),
    ]
    return hs


def _grid(ax, fig):
    grid = J("ext_grid_ion.json")
    Wl, R = [4, 6, 8, 10, 12, 16], [3, 4, 5, 6, 8, 99]
    A = np.array([[np.mean([g[f"w{w}_r{r}"] - g["opt"] for g in grid]) for w in Wl] for r in R])
    im = ax.imshow(A, cmap=BLUES, origin="lower", aspect="auto", vmin=0)
    for i in range(len(R)):
        for j in range(len(Wl)):
            ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if A[i, j] > 0.55 * A.max() else INK)  # fmt: skip
    ax.set_xticks(range(len(Wl)), Wl)
    ax.set_yticks(range(len(R)), [str(r) if r < 99 else "$\\infty$" for r in R])
    ax.set_xlabel("window length $w$")
    ax.set_ylabel("database radius $r$")
    ax.add_patch(plt.Rectangle((Wl.index(8) - 0.5, R.index(5) - 0.5), 1, 1, fill=False, ec=col("original"), lw=1.6))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("mean excess over the optimum")
    cb.outline.set_linewidth(0.4)


def window_grid():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.35))
    hs = _window(axs[0])
    legend_above(axs[0], hs, ncol=2, pad=0.12)
    panel(axs[0], "a")
    _grid(axs[1], fig)
    panel(axs[1], "b", "excess over the optimum, ion")
    save(fig, "fig_window_grid")
    fig, ax = plt.subplots(figsize=(3.6, 2.4))  # talk copies, one panel each
    legend_above(ax, _window(ax), ncol=2, pad=0.04)
    save(fig, "ext_window", talk=True)
    fig, ax = plt.subplots(figsize=(2.9, 2.4))
    _grid(ax, fig)
    save(fig, "ext_grid", talk=True)


# ------------------------------------------------------------------ example circuits
def _example_data():
    """The example circuits, computed once (the original method is randomised and time-budgeted)."""
    from qcr.gates import Gate

    cache = RESULTS / "fig_examples.json"
    if cache.exists():
        D = json.load(open(cache))
    else:
        D = {}
        talk = RESULTS / "talk_example.json"  # the circuit shown in the advisor talk
        if talk.exists():
            D["paper"] = json.load(open(talk))
        D.update(_compute_examples(skip_paper="paper" in D))
        json.dump(D, open(cache, "w"), indent=0)
    out = {}
    for key, rows in D.items():
        out[key] = {k: [Gate(n, tuple(q), t) for n, q, t in v] if isinstance(v, list) else v for k, v in rows.items()}
    return out


def _compute_examples(skip_paper):
    import gen_bench as gb
    from ext_gap import oracle
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    from qcr.compute_graph import cached_build
    from qcr.gates import make_pool, random_circuit
    from qcr.general import Config, Optimizer
    from qcr.hardware import coupling_list, heron, qiskit_basis
    from qcr.reducer import DBEngine, reduce_circuit

    def rows(gs):
        return [(g.name, list(g.qubits), g.theta) for g in gs]

    D = {}
    if not skip_paper:
        pool = make_pool("ion_pm", 3)
        cg = cached_build(3, pool, 5, "ion_pm", cache_dir=RESULTS.parent / "cache")
        O = oracle("ion_pm")
        best = None
        for seed in range(8):
            circ = random_circuit(pool, 80, np.random.default_rng(100 + seed))
            orig, _ = reduce_circuit(circ, DBEngine(cg, "hash"), pool, np.random.default_rng(seed), time_budget=40,
                                     stall_attempts=20000)  # fmt: skip
            opt = O.optimal_word(O.code_of(circ))
            if best is None or len(orig) - len(opt) > len(best[1]) - len(best[2]):
                best = (circ, orig, opt)
            if len(orig) - len(opt) >= 8:
                break
        D["paper"] = dict(zip(("input", "orig", "opt"), map(rows, best)))
    dev = heron()
    suite = gb.suite_circuits()
    dbs = gb.load_dbs("heron")
    for name in ("qasmbench/toffoli_n3", "qasmbench/hs4_n4"):
        l3 = gb.qk_to_gates(transpile(suite[name], basis_gates=qiskit_basis(dev), seed_transpiler=0,
                                      coupling_map=CouplingMap(coupling_list(dev)), optimization_level=3))  # fmt: skip
        ours = Optimizer(dev, dbs, Config(mode="exact", cost="calibrated", patience=4)).run(l3).gates
        a, n, m = gb.compact(l3)
        D[name.split("/")[1]] = {"l3": rows(a), "ours": rows([g.remap(m) for g in ours]), "n": n}
    if (RESULTS / "ext_cliffordt.json").exists():
        from qcr.cliffordt import CliffordTBall, alphabet, planted_circuit
        from qcr.general import GRIDS, Database
        from qcr.hardware import Device

        ball = CliffordTBall(4).build()
        r = np.random.default_rng(3)
        pl = alphabet()
        while True:
            core = [pl[i] for i in r.integers(len(pl), size=5)]
            d, w, ex = ball.distance(core)
            if ex and d == 5:
                break
        circ = planted_circuit(core, 36, ball, r)
        devt = Device("ion_cliffordT", 2, None, ("rx", "ry", "rz"), "rxx", (), False,
                      angle_grid=(-np.pi / 2, -np.pi / 4, np.pi / 4, np.pi / 2, np.pi))  # fmt: skip
        db = {("phase", "tri"): Database.load(RESULTS.parent / "cache" / "gdb_ion_tri_phase.npz", GRIDS["ion"])}
        out = Optimizer(devt, db, Config(mode="phase", cost="count", patience=8, restarts=4)).run(circ).gates
        D["cliffordt"] = {"input": rows(circ), "out": rows(out), "opt": d}
    return D


def examples():
    """One PDF per circuit at print scale. The LaTeX figure places them, so every panel has the same gate size."""
    D = _example_data()
    p = D["paper"]
    circuit(p["input"], 3, "fig_ex_paper_input", fold=28)  # talk only
    circuit(p["orig"], 3, "fig_ex_paper_orig")
    circuit(p["opt"], 3, "fig_ex_paper_opt")
    for name in ("toffoli_n3", "hs4_n4"):
        circuit(D[name]["l3"], D[name]["n"], f"fig_ex_{name}_l3")
        circuit(D[name]["ours"], D[name]["n"], f"fig_ex_{name}_ours")
    if "cliffordt" in D:
        circuit(D["cliffordt"]["input"], 2, "fig_ex_ct_input")
        circuit(D["cliffordt"]["out"], 2, "fig_ex_ct_out")
    # gate and two-qubit-gate counts for the captions
    twoq = lambda gs: sum(len(g.qubits) == 2 for g in gs)
    M = {"ExPaperIn": len(p["input"]), "ExPaperOrig": len(p["orig"]), "ExPaperOpt": len(p["opt"])}
    for name, tag in (("toffoli_n3", "Tof"), ("hs4_n4", "Hs")):
        for k, nm in (("l3", "LThree"), ("ours", "Ours")):
            M[f"Ex{tag}{nm}"] = len(D[name][k])
            M[f"Ex{tag}{nm}Cz"] = twoq(D[name][k])
    if "cliffordt" in D:
        M |= {"ExCtIn": len(D["cliffordt"]["input"]), "ExCtOut": len(D["cliffordt"]["out"]), "ExCtOpt": D["cliffordt"]["opt"]}
    with open(PAPER_FIGS.parent / "fig_numbers.tex", "w") as fh:
        fh.write("% generated by experiments/make_paper_figs.py examples -- do not edit\n")
        for k, v in M.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(M)


# ------------------------------------------------------------------ original benchmarks
STAGES = {
    "input": "input",
    "qiskit": "Qiskit L3",
    "original": "original",
    "cliff": "Clifford CW",
    "general": "ours",
    "optimum": "optimum",
}


def paper_flow():
    """From the input to the output of every method, per benchmark setting (grey: individual circuits)."""
    q = {(r["cfg"], r["seed"]): r for r in J("quality_all.json")}
    g = {(r["cfg"], r["seed"]): r for r in J("general_paper_bench_T3_T4_NJP_ion_NJP_nisq_T5.json")}
    e = {(r["cfg"], r["seed"]): r for r in J("ext_bench_cliff_T5_NJP_ion_NJP_nisq.json")}
    gap = {("T3", r["seed"]): r for r in J("ext_gap3_ion.json")} | {("T4", r["seed"]): r for r in J("ext_gap3_nisq.json")}
    QM = {"T3": "T3_ion_3q", "T4": "T4_nisq_3q", "NJP_ion": "NJP_ion_4q", "NJP_nisq": "NJP_nisq_4q", "T5": "T5_nisq_15q"}
    LB = {"NJP_ion": 11.1, "NJP_nisq": 12.9}
    titles = {"T3": "ion 3q", "T4": "IBM 3q", "NJP_ion": "ion 4q", "NJP_nisq": "IBM 4q", "T5": "IBM 15q"}
    fig, axs = plt.subplots(1, 5, figsize=(W, 2.2), gridspec_kw=dict(wspace=0.5))
    for i, (ax, cfg) in enumerate(zip(axs, titles)):
        seeds = sorted(s for (c, s) in q if c == QM[cfg] and (cfg, s) in g)
        keys = ["input", "qiskit", "original"] + (["cliff"] if cfg in ("NJP_ion", "NJP_nisq", "T5") else []) + ["general"]
        keys += ["optimum"] if cfg in ("T3", "T4") else []
        A = []
        for s in seeds:
            r = q[(QM[cfg], s)]
            v = {"input": tot(r["original"]), "qiskit": tot(r["qiskit_L3"]), "original": tot(r["ours"]),
                 "general": tot(g[(cfg, s)]["general"])}  # fmt: skip
            if "cliff" in keys:
                v["cliff"] = tot(e[(cfg, s)]["ours"]) if (cfg, s) in e else np.nan
            if "optimum" in keys:
                v["optimum"] = gap[(cfg, s)]["opt"]
            A.append([v[k] for k in keys])
        A = np.array(A, float)
        x = np.arange(len(keys))
        for row in A:
            ax.plot(x, row, color=FAINT, lw=0.35, zorder=1)
        mean = np.nanmean(A, axis=0)
        ax.plot(x, mean, color=INK, lw=0.9, zorder=2)
        for j, k in enumerate(keys):
            ax.plot([j], [mean[j]], marker=mk(k), ms=4.5, color=col(k), mec="white", mew=0.5, zorder=3, ls="none")
        last = len(keys) - 1
        ax.annotate(f"{mean[last]:.1f}", (last, mean[last]), xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=6, color=INK)  # fmt: skip
        if cfg in LB:
            ax.axhline(LB[cfg], color=col("optimum"), ls=(0, (3, 2)), lw=0.9)
            ax.annotate(f"bound {LB[cfg]}", (0, LB[cfg]), xytext=(0, 2), textcoords="offset points", ha="left", va="bottom",
                        fontsize=5.5, color=INK)  # fmt: skip
        ax.set_xticks(x, [STAGES[k] for k in keys], rotation=60, ha="right", rotation_mode="anchor", fontsize=6)
        ax.set_ylim(0, None)
        light_grid(ax)
        panel(ax, "abcde"[i], titles[cfg])
    axs[0].set_ylabel("gates")
    fig_legend(fig, [handle(k, line=False, label=STAGES[k]) for k in STAGES], ncol=6, y=1.04)
    save(fig, "fig_paper_flow")


def paper_bench():
    """Talk figure: output length relative to the original method, per benchmark setting."""
    q = {(r["cfg"], r["seed"]): r for r in J("quality_all.json")}
    g = J("general_paper_bench_T3_T4_NJP_ion_NJP_nisq_T5.json")
    e = {(r["cfg"], r["seed"]): r for r in J("ext_bench_cliff_T5_NJP_ion_NJP_nisq.json")}
    gap = {("T3", r["seed"]): r for r in J("ext_gap3_ion.json")} | {("T4", r["seed"]): r for r in J("ext_gap3_nisq.json")}
    QM = {"T3": "T3_ion_3q", "T4": "T4_nisq_3q", "NJP_ion": "NJP_ion_4q", "NJP_nisq": "NJP_nisq_4q", "T5": "T5_nisq_15q"}
    names = {"T3": "ion, 3q", "T4": "IBM-style, 3q", "NJP_ion": "ion, 4q", "NJP_nisq": "IBM-style, 4q", "T5": "IBM-style, 15q"}
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    h = 0.26
    for i, cfg in enumerate(names):
        P = [r for r in g if r["cfg"] == cfg and (QM[cfg], r["seed"]) in q]
        base = lambda r: tot(q[(QM[cfg], r["seed"])]["ours"])
        vals = {"general": np.mean([tot(r["general"]) / base(r) for r in P])}
        cw = [tot(e[(cfg, r["seed"])]["ours"]) / base(r) for r in P if (cfg, r["seed"]) in e]
        op = [gap[(cfg, r["seed"])]["opt"] / base(r) for r in P if (cfg, r["seed"]) in gap]
        if cw:
            vals["cliff"] = np.mean(cw)
        if op:
            vals["optimum"] = np.mean(op)
        for k, (key, v) in enumerate(vals.items()):
            y = i + (k - (len(vals) - 1) / 2) * h
            ax.barh(y, v, height=h - 0.05, color=col(key), edgecolor="white", lw=0.5)
            ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=6)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(1.0, len(names) - 0.45, "original method", ha="right", va="bottom", fontsize=6, color=MUTED)
    ax.set_yticks(range(len(names)), list(names.values()))
    ax.set_xlim(0, 1.12)
    ax.invert_yaxis()
    ax.set_xlabel("output length relative to the original method")
    legend_above(ax, [Patch(color=col(k), label=lab(k)) for k in ("general", "cliff", "optimum")], ncol=3)
    save(fig, "gen_paper_bench", talk=True)


def cliffordt():
    """Clifford+T: certified optimum vs. methods (random words and planted circuits)."""
    rows = J("ext_cliffordt.json")["rows"]
    keys = ("optimum", "general", "original", "qiskit")
    src = {"optimum": "optimum", "general": "general", "original": "original", "qiskit": "qiskit_L3"}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    Ls = sorted({r["size"] for r in rows if r["kind"] == "random"})
    ax.plot(Ls, Ls, color=FAINT, ls=(0, (2, 2)), lw=0.9)
    ax.text(Ls[-1], Ls[-1], "input ", ha="right", va="bottom", fontsize=6, color=MUTED)
    for k in keys:
        m = []
        for L in Ls:
            rs = [r for r in rows if r["kind"] == "random" and r["size"] == L and r["optimum_exact"]]
            m.append(np.mean([r[src[k]] for r in rs]) if rs else np.nan)
        ax.plot(Ls, m, marker=mk(k), ms=3.5, color=col(k), mec="white", mew=0.4)
    ax.set_xticks(Ls)
    ax.set_xlabel("input length (random words, 2 qubits)")
    ax.set_ylabel("output gates (certified circuits)")
    light_grid(ax)
    panel(ax, "a", "random words")
    ax = axs[1]
    P = [r for r in rows if r["kind"] == "planted"]
    cfgs = sorted({tuple(r["size"]) for r in P})
    x = np.arange(len(cfgs))
    w = 0.2
    for i, k in enumerate(keys):
        vals = [np.mean([r[src[k]] for r in P if tuple(r["size"]) == cf]) for cf in cfgs]
        ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=col(k), edgecolor="white", lw=0.4)
    ax.set_xticks(x, [f"{c} in {L}" for c, L in cfgs], fontsize=6)
    ax.set_xlabel("planted core length in input length")
    ax.set_ylabel("output gates (mean)")
    light_grid(ax)
    panel(ax, "b", "planted circuits")
    fig_legend(fig, [handle(k, label={"qiskit": "Qiskit L3 (continuous angles)"}.get(k, lab(k))) for k in keys],
               ncol=4, y=1.02)  # fmt: skip
    save(fig, "fig_cliffordt")


# ------------------------------------------------------------------ hardware suite
def _suite():
    B = J("general/bench_v2.json")
    for fn, src, dst in (("bqskit_v2.jsonl", "bqskit_L3", "bqskit_L3"), ("bqskit_stack_v1.jsonl", "bqskit_L3+ours", "bq_ours")):
        p = RESULTS / "general" / fn
        rows = {(r["name"], r["device"]): r for r in map(json.loads, open(p))} if p.exists() else {}
        for r in B:
            if src in rows.get((r["name"], r["device"]), {}):
                r[dst] = rows[(r["name"], r["device"])][src]
    return [r for r in B if r["name"] not in {"nam/pf1_10", "nam/pf2_10"}]


def hw_flow():
    """Per-circuit total-gate ratio to Qiskit L3 for every pipeline (dots), and the summed counts (bars)."""
    B = _suite()
    pipes = [
        ("qiskit_L1", "Qiskit L1", "qiskit_L1"),
        ("tket", "TKET", "tket"),
        ("bqskit_L3", "BQSKit L3", "bqskit"),
        ("ours_exact_cal", "ours from L1", "general"),
        ("ours_exact_cal_fromL3", "ours from L3", "general"),
        ("bq_ours", "ours on BQSKit L3", "bq_ours"),
    ]
    pipes = [p for p in pipes if all(p[0] in r for r in B)]
    fig, axs = plt.subplots(2, 2, figsize=(W, 3.6), gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.55, wspace=0.22))
    for c, (dev, title) in enumerate((("heron", "IBM Heron"), ("ion", "trapped ion"))):
        R = [r for r in B if r["device"] == dev]
        ax = axs[0, c]
        rng = np.random.default_rng(0)
        for i, (k, name, ent) in enumerate(pipes):
            v = np.array([r[k]["total"] / r["qiskit_L3"]["total"] for r in R])
            ax.scatter(v, i + rng.uniform(-0.22, 0.22, len(v)), s=4, color=col(ent), alpha=0.55, linewidths=0,
                       marker="o" if k != "ours_exact_cal" else "o")  # fmt: skip
            gm = gmean(v)
            ax.plot([gm, gm], [i - 0.34, i + 0.34], color=INK, lw=1.2)
            ax.text(1.01, i, f"$\\times${gm:.2f}", va="center", ha="left", fontsize=6,
                    transform=ax.get_yaxis_transform())  # fmt: skip
        ax.axvline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.set_xscale("log")
        ax.set_xlim(0.16, 4.8)
        ax.set_xticks([0.25, 0.5, 1, 2, 4], ["0.25", "0.5", "1", "2", "4"])
        ax.minorticks_off()
        ax.set_yticks(range(len(pipes)), [p[1] for p in pipes] if c == 0 else [])
        ax.invert_yaxis()
        ax.set_xlabel("total gates / Qiskit L3 (per circuit)")
        light_grid(ax, "x")
        panel(ax, "ab"[c], f"{title}, {len(R)} circuits")
        bx = axs[1, c]
        allp = [("qiskit_L3", "Qiskit L3", "qiskit")] + pipes
        t2 = np.array([sum(r[k]["twoq"] for r in R) for k, _, _ in allp])
        t1 = np.array([sum(r[k]["total"] - r[k]["twoq"] for r in R) for k, _, _ in allp])
        yy = np.arange(len(allp))
        bx.barh(yy, t2, color=[col(e) for _, _, e in allp], edgecolor="white", lw=0.4, height=0.72)
        bx.barh(yy, t1, left=t2, color=[col(e) for _, _, e in allp], alpha=0.3, edgecolor="white", lw=0.4, height=0.72)
        for i, (a, b) in enumerate(zip(t2, t1)):
            bx.text(a + b, i, f" {a + b:,}", va="center", fontsize=5.8)
        bx.set_yticks(yy, [p[1] for p in allp] if c == 0 else [])
        bx.invert_yaxis()
        bx.set_xlim(0, (t1 + t2).max() * 1.32)
        bx.set_xlabel("summed gates (dark: two-qubit)")
        bx.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k" if v else "0")
        light_grid(bx, "x")
        panel(bx, "cd"[c])
    save(fig, "fig_hw_flow")


def tvd():
    """Talk figure: noisy-simulation TVD of ours (calibrated) against Qiskit L3, per circuit."""
    B = _suite()
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.3), gridspec_kw=dict(wspace=0.38))
    for ax, dev, title in zip(axs, ("heron", "ion"), ("IBM Heron (FakeTorino noise)", "trapped ion (model noise)")):
        T = [r for r in B if r["device"] == dev and "tvd" in r]
        L3 = np.array([r["tvd"]["qiskit_L3"] for r in T])
        CA = np.array([r["tvd"]["ours_exact_cal"] for r in T])
        lim = max(L3.max(), CA.max()) * 1.05
        ax.fill_between([0, lim], [0, 0], [0, lim], color="#eaf2fb", lw=0)
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.7, ls=(0, (3, 2)))
        ax.scatter(L3, CA, s=9, color=col("general"), edgecolor="white", linewidths=0.3, zorder=3)
        ax.text(0.97 * lim, 0.04 * lim, f"ours better\n{int((CA < L3).sum())}/{len(T)}", ha="right", va="bottom", fontsize=6)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("TVD, Qiskit L3")
        ax.set_ylabel("TVD, ours")
        panel(ax, "ab"[dev == "ion"], title)
    save(fig, "gen_tvd", talk=True)


def layout():
    """Layout-aware rewriting: ESP (with readout) and noisy TVD per pipeline, pipelines as rows."""
    R = J("general/bench_layout.json")
    keys = [
        ("L3", "Qiskit L3", "qiskit"),
        ("L3+ours", "Qiskit L3 + ours", "general"),
        ("L3na", "noise-aware L3", "qiskit_L1"),
        ("L3na+ours", "noise-aware L3 + ours", "general"),
        ("L3na+ours+rel", "+ relocation", "general"),
        ("joint", "joint layout selection", "optimum"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    base = np.array([r["L3na"]["success_readout"] for r in R])
    for i, (k, name, ent) in enumerate(keys):
        v = np.array([r[k]["success_readout"] for r in R]) / base
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), s=7,
                   color=col(ent), alpha=0.75, linewidths=0)  # fmt: skip
        gm = gmean(v)
        ax.plot([gm, gm], [i - 0.32, i + 0.32], color=INK, lw=1.2)
        ax.text(1.01, i, f"$\\times${gm:.2f}", va="center", ha="left", fontsize=6, transform=ax.get_yaxis_transform())
    ax.axvline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xlim(0.15, 2.3)
    ax.set_xticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_yticks(range(len(keys)), [n for _, n, _ in keys])
    ax.invert_yaxis()
    ax.set_xlabel("ESP / noise-aware Qiskit L3")
    light_grid(ax, "x")
    panel(ax, "a", f"ESP with readout, {len(R)} circuits")
    ax = axs[1]
    T = [r for r in R if "tvd_readout" in r["L3na"]]
    for i, (k, name, ent) in enumerate(keys):
        v = np.array([r[k]["tvd_readout"] for r in T])
        ax.plot([v.min(), v.max()], [i, i], color=FAINT, lw=1.0, zorder=1)
        ax.scatter(v, np.full(len(v), i), s=7, color=col(ent), alpha=0.75, linewidths=0, zorder=2)
        ax.plot([v.mean()], [i], marker="|", ms=9, mew=1.4, color=INK, zorder=3)
        ax.text(1.01, i, f"{v.mean():.3f}", va="center", ha="left", fontsize=6, transform=ax.get_yaxis_transform())
    ax.set_xlim(0.1, 0.62)
    ax.set_yticks(range(len(keys)), [])
    ax.invert_yaxis()
    ax.set_xlabel("noisy TVD with readout (lower is better)", loc="right")
    light_grid(ax, "x")
    panel(ax, "b", f"FakeTorino simulation, {len(T)} circuits")
    save(fig, "fig_layout")


def hw_real():
    """Real device: success probability relative to Qiskit L3, mean and sd of the paired per-job ratio."""
    a = J("hardware/ibm_marrakesh_v1/analysis.json")
    rows = sorted(a["rows"], key=lambda r: r["pipes"]["L3"]["total"])
    fig, ax = plt.subplots(figsize=(W, 2.0))
    x = np.arange(len(rows))
    for off, p, m, mfc, name in ((-0.15, "L3_ours", "o", None, "Qiskit L3, then ours"),
                                 (0.15, "L1_ours", "s", "white", "Qiskit L1, then ours")):  # fmt: skip
        ratio = [np.array(r["pipes"][p]["values"]) / np.array(r["pipes"]["L3"]["values"]) for r in rows]
        ax.errorbar(x + off, [v.mean() for v in ratio], yerr=[v.std(ddof=1) for v in ratio], fmt=m, ms=3.8,
                    mfc=mfc or col("general"), color=col("general"), elinewidth=0.9, capsize=0, mew=0.9, label=name)  # fmt: skip
    ax.axhline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(x, [r["name"].split("/")[1] for r in rows], rotation=40, ha="right",
                  rotation_mode="anchor", fontsize=6)  # fmt: skip
    ax.set_ylabel("$\\hat p(x^\\star)$ / Qiskit L3")
    light_grid(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, borderaxespad=0)
    save(fig, "fig_hw_real")


if __name__ == "__main__":
    which = sys.argv[1:] or ["compute_graph", "growth", "window_grid", "examples", "paper_flow", "paper_bench",
                             "cliffordt", "hw_flow", "tvd", "layout", "hw_real"]  # fmt: skip
    for w in which:
        globals()[w]()
