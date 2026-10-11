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

"""Figures for the journal version of the extension paper (manuscripts/extension/figs/fig_*.pdf).

  python make_paper_figs.py [compute_graph growth examples paper_flow hw_flow cliffordt layout]

Every figure is drawn from results/ (and from small computations repeated here, e.g. the compute graphs).
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from common import RESULTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parents[1] / "manuscripts" / "extension" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
W = 5.2  # Springer sn-jnl text width is 372pt = 5.15in: draw at final size
# one categorical palette for methods, used consistently in every figure
C = {
    "input": "#9aa0a6",
    "qiskit": "#6b6a63",
    "tket": "#a07cc5",
    "original": "#eb6834",
    "cliff": "#d4a020",
    "general": "#2a78d6",
    "optimum": "#1baf7a",
    "bqskit": "#c2417a",
}
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 8,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    }
)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def J(name):
    p = RESULTS / name
    if not p.exists() and p.with_suffix(".jsonl").exists():  # unfinished run: use the rows so far
        return [json.loads(line) for line in open(p.with_suffix(".jsonl"))]
    return json.load(open(p))


def tot(c):
    return sum(c.values()) if isinstance(c, dict) else c


# ------------------------------------------------------------------ compute graphs
def compute_graph():
    """Two compute graphs of the same size: the 2-qubit Clifford ball closes up, the Clifford+T ball is a tree."""
    import networkx as nx

    from qcr.cliffordt import CliffordTBall, canonical, keys_of, matmul
    from qcr.cliffordt import exact_gate as ct_gate
    from qcr.compute_graph import build
    from qcr.gates import gate_matrix, make_pool
    from qcr.unitary import phase_key

    fig, axs = plt.subplots(1, 2, figsize=(W, 2.9))
    # (a) Clifford, ion alphabet, 2 qubits, depth <= 3
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
    # (b) Clifford+T, same alphabet plus pi/4 angles, depth <= 2 (a ball of similar size)
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
    cmap = plt.get_cmap("viridis")
    for ax, Gr, title in (
        (
            axs[0],
            G,
            f"(a) Clifford: ion alphabet, 2 qubits, depth $\\leq$3\n{G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges",
        ),
        (
            axs[1],
            H,
            f"(b) Clifford+T: ion alphabet with $\\pm\\pi/4$ and $\\pi$ angles, depth $\\leq$2\n{H.number_of_nodes()} nodes, "
            f"{H.number_of_edges()} edges",
        ),
    ):
        pos = nx.kamada_kawai_layout(Gr) if Gr.number_of_nodes() < 700 else nx.spring_layout(Gr, seed=1)
        d = np.array([Gr.nodes[v]["d"] for v in Gr])
        nx.draw_networkx_edges(Gr, pos, ax=ax, width=0.15, alpha=0.35, edge_color="#6b6a63")
        sc = ax.scatter(
            [pos[v][0] for v in Gr],
            [pos[v][1] for v in Gr],
            c=d,
            cmap=cmap,
            s=[18 if x == 0 else 4 for x in d],
            vmin=0,
            vmax=3,
            zorder=3,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_axis_off()
    cb = fig.colorbar(sc, ax=axs, fraction=0.025, pad=0.01, ticks=[0, 1, 2, 3])
    cb.set_label("distance from identity (optimal gate count)")
    save(fig, "fig_compute_graph")


def growth():
    """Sphere sizes of every exact ball and database used in the paper."""
    cay3 = J("cayley3_ion_pm.json")["levels"]
    cay3n = J("cayley3_nisq.json")["levels"]
    from qcr.gates import make_pool
    from qcr.tableau import cayley_bfs

    cay2 = cayley_bfs(2, make_pool("ion_pm", 2), verbose=False)[2]
    ct = J("ext_cliffordt.json")["ball"]["levels"] if (RESULTS / "ext_cliffordt.json").exists() else None
    gdb = J("gdb_info.json")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
    ax = axs[0]
    ax.semilogy(range(len(cay2)), cay2, "o-", ms=3, color=C["cliff"], label="Clifford, 2q, ion (11,520)")
    ax.semilogy(range(len(cay3)), cay3, "o-", ms=3, color=C["original"], label="Clifford, 3q, ion (92,897,280)")
    ax.semilogy(range(len(cay3n)), cay3n, "s-", ms=3, color=C["qiskit"], label="Clifford, 3q, IBM-style")
    if ct:
        ax.semilogy(range(len(ct)), ct, "^-", ms=3, color=C["general"], label="Clifford+T, 2q, ion (infinite)")
    ax.axvspan(-0.3, 5.3, color="#e4e3dc", zorder=0)
    ax.text(0.2, 2e7, "radius of the\noriginal database", fontsize=7, color=C["qiskit"])
    ax.set_xlabel("distance $d$ (optimal gate count)")
    ax.set_ylabel("elements at distance $d$")
    ax.set_title("(a) exact Cayley-graph spheres")
    ax.legend(fontsize=6.5, loc="lower right")
    ax = axs[1]
    style = {
        "heron/line": ("o", C["general"]),
        "ion/tri": ("s", C["original"]),
        "ion_paper/tri": ("^", C["cliff"]),
        "ibm_paper/tri": ("v", C["qiskit"]),
    }
    for key, info in gdb.items():
        dev, topo, kind = key.split("/")
        mk, col = style[f"{dev}/{topo}"]
        ls = {"phase": "-", "out": "--", "state": ":"}[kind]
        ax.semilogy(
            range(len(info["levels"])),
            info["levels"],
            marker=mk,
            ls=ls,
            ms=3,
            color=col,
            label=f"{dev} {topo} {kind} ({info['size'] / 1e6:.1f}M)",
        )
    ax.set_xlabel("depth")
    ax.set_ylabel("new classes at depth")
    ax.set_title("(b) databases of the general optimiser")
    ax.legend(fontsize=5.5, ncol=2, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_growth")


# ------------------------------------------------------------------ example circuits
def _draw(gates, n, ax, title):
    from qiskit.visualization import circuit_drawer

    from qcr.verify import to_qiskit

    qc = to_qiskit(gates, n)
    circuit_drawer(qc, output="mpl", ax=ax, fold=-1, style={"name": "bw", "fontsize": 9, "subfontsize": 6})
    ax.set_title(title, fontsize=9, loc="left")


def _figure_rows(rows, name, width=W):
    """rows: list of (gates, n, title); one circuit per row, widths proportional to gate count."""
    from qiskit.visualization import circuit_drawer

    from qcr.verify import to_qiskit

    imgs = []
    for row in rows:
        gates, n, title = row[:3]
        fold = row[3] if len(row) > 3 else -1
        f = circuit_drawer(
            to_qiskit(gates, n), output="mpl", fold=fold, style={"name": "bw", "fontsize": 11, "subfontsize": 7}
        )
        f.canvas.draw()
        imgs.append((np.asarray(f.canvas.buffer_rgba()), title))
        plt.close(f)
    maxw = max(im.shape[1] for im, _ in imgs)
    heights = [im.shape[0] / maxw * width for im, _ in imgs]
    fig, axs = plt.subplots(
        len(imgs), 1, figsize=(width, sum(heights) + 0.3 * len(imgs)), gridspec_kw=dict(height_ratios=heights)
    )
    for ax, (im, title) in zip(np.atleast_1d(axs), imgs):
        ax.imshow(im, extent=(0, im.shape[1] / maxw, 0, im.shape[0] / maxw))
        ax.set_xlim(0, 1)
        ax.set_axis_off()
        ax.set_title(title, fontsize=8, loc="left")
    save(fig, name)


def examples():
    """Input and output circuits: a paper benchmark circuit, a real algorithm on Heron, a planted Clifford+T one."""
    import gen_bench as gb
    from ext_gap import oracle
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    from qcr.compute_graph import cached_build
    from qcr.gates import make_pool, random_circuit
    from qcr.general import Config, Optimizer
    from qcr.hardware import coupling_list, heron, qiskit_basis
    from qcr.reducer import DBEngine, reduce_circuit

    # (1) paper setting (ion alphabet, 3 qubits): input -> original method -> proven optimum
    pool = make_pool("ion_pm", 3)
    cg = cached_build(3, pool, 5, "ion_pm", cache_dir=RESULTS.parent / "cache")
    O = oracle("ion_pm")
    best = None
    for seed in range(8):  # an input on which the original method stops clearly above the optimum
        circ = random_circuit(pool, 80, np.random.default_rng(100 + seed))
        orig, _ = reduce_circuit(
            circ, DBEngine(cg, "hash"), pool, np.random.default_rng(seed), time_budget=40, stall_attempts=20000
        )
        opt = O.optimal_word(O.code_of(circ))
        if best is None or len(orig) - len(opt) > len(best[1]) - len(best[2]):
            best = (circ, orig, opt)
        if len(orig) - len(opt) >= 8:
            break
    circ, orig, opt = best
    _figure_rows(
        [
            (circ, 3, f"(a) input: random 3-qubit word over the ion alphabet, {len(circ)} gates", 41),
            (orig, 3, f"(b) original method (database radius 5): {len(orig)} gates"),
            (opt, 3, f"(c) proven optimum (exact Cayley table): {len(opt)} gates"),
        ],
        "fig_example_paper",
    )
    # (2) a real algorithm on IBM Heron: Qiskit level 3 -> ours
    dev = heron()
    suite = gb.suite_circuits()
    dbs = gb.load_dbs("heron")
    rows = []
    for name, (la, lb) in (("qasmbench/toffoli_n3", "ab"), ("qasmbench/hs4_n4", "cd")):
        l3 = gb.qk_to_gates(
            transpile(
                suite[name],
                basis_gates=qiskit_basis(dev),
                coupling_map=CouplingMap(coupling_list(dev)),
                optimization_level=3,
                seed_transpiler=0,
            )
        )
        ours = Optimizer(dev, dbs, Config(mode="exact", cost="calibrated", patience=4)).run(l3).gates
        a, n, m = gb.compact(l3)
        b = [g.remap(m) for g in ours]
        nq = sum(len(x.qubits) == 2 for x in a), sum(len(x.qubits) == 2 for x in b)
        short = name.split("/")[1]
        rows += [
            (a, n, f"({la}) {short} after Qiskit level 3 on IBM Heron: {len(a)} gates, {nq[0]} CZ"),
            (b, n, f"({lb}) {short} after qcr (exact incl. global phase): {len(b)} gates, {nq[1]} CZ"),
        ]
    _figure_rows(rows, "fig_example_heron")
    # (3) planted Clifford+T circuit (2 qubits): input -> general optimiser = certified optimum
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
        devt = Device(
            "ion_cliffordT",
            2,
            None,
            ("rx", "ry", "rz"),
            "rxx",
            (),
            False,
            angle_grid=(-np.pi / 2, -np.pi / 4, np.pi / 4, np.pi / 2, np.pi),
        )
        db = {("phase", "tri"): Database.load(RESULTS.parent / "cache" / "gdb_ion_tri_phase.npz", GRIDS["ion"])}
        out = Optimizer(devt, db, Config(mode="phase", cost="count", patience=8, restarts=4)).run(circ).gates
        _figure_rows(
            [
                (circ, 2, f"(a) planted Clifford+T circuit: {len(circ)} gates hiding a 5-gate optimum"),
                (out, 2, f"(b) general optimiser: {len(out)} gates (certified optimum: {d})"),
            ],
            "fig_example_cliffordt",
        )


# ------------------------------------------------------------------ start -> end benchmark figures
def paper_flow():
    """From 300/500 input gates to the output of every method, per benchmark setting (paired circuits)."""
    q = {(r["cfg"], r["seed"]): r for r in J("quality_all.json")}
    g = {(r["cfg"], r["seed"]): r for r in J("general_paper_bench_T3_T4_NJP_ion_NJP_nisq_T5.json")}
    e = {(r["cfg"], r["seed"]): r for r in J("ext_bench_cliff_T5_NJP_ion_NJP_nisq.json")}
    gap = {("T3", r["seed"]): r for r in J("ext_gap3_ion.json")} | {
        ("T4", r["seed"]): r for r in J("ext_gap3_nisq.json")
    }
    QM = {
        "T3": "T3_ion_3q",
        "T4": "T4_nisq_3q",
        "NJP_ion": "NJP_ion_4q",
        "NJP_nisq": "NJP_nisq_4q",
        "T5": "T5_nisq_15q",
    }
    LB = {"NJP_ion": 11.1, "NJP_nisq": 12.9}
    titles = {
        "T3": "T3: ion, 3 qubits",
        "T4": "T4: IBM-style, 3 qubits",
        "NJP_ion": "journal: ion, 4 qubits",
        "NJP_nisq": "journal: IBM-style, 4 qubits",
        "T5": "T5: IBM-style, 15 qubits",
    }
    fig = plt.figure(figsize=(W, 4.4))
    gs = fig.add_gridspec(2, 6, hspace=0.95, wspace=1.4)
    axs = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    for ax, cfg in zip(axs, titles):
        seeds = sorted(s for (c, s) in q if c == QM[cfg] and (cfg, s) in g)
        stages = [("input", C["input"]), ("Qiskit L3", C["qiskit"]), ("original", C["original"])]
        if cfg in ("NJP_ion", "NJP_nisq", "T5"):
            stages.append(("Clifford CW", C["cliff"]))
        stages.append(("general", C["general"]))
        if cfg in ("T3", "T4"):
            stages.append(("optimum", C["optimum"]))
        rows = []
        for s in seeds:
            r = q[(QM[cfg], s)]
            v = [tot(r["original"]), tot(r["qiskit_L3"]), tot(r["ours"])]
            if cfg in ("NJP_ion", "NJP_nisq", "T5"):
                v.append(tot(e[(cfg, s)]["ours"]) if (cfg, s) in e else np.nan)
            v.append(tot(g[(cfg, s)]["general"]))
            if cfg in ("T3", "T4"):
                v.append(gap[(cfg, s)]["opt"])
            rows.append(v)
        A = np.array(rows, float)
        x = np.arange(A.shape[1])
        for row in A:
            ax.plot(x, row, color="#c9c8c1", lw=0.4, zorder=1)
        mean = np.nanmean(A, axis=0)
        ax.plot(x, mean, color="#1f1f1e", lw=1.0, zorder=2)
        for i, (lab, col) in enumerate(stages):
            ax.scatter([i], [mean[i]], color=col, s=22, zorder=3)
            ax.annotate(
                f"{mean[i]:.0f}" if mean[i] >= 20 else f"{mean[i]:.1f}",
                (i, mean[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=6,
            )
        if cfg in LB:
            ax.axhline(LB[cfg], color=C["optimum"], ls=(0, (3, 2)), lw=0.8)
            ax.text(len(stages) - 1, LB[cfg], "lower bound", ha="right", va="bottom", fontsize=5.5, color=C["optimum"])
        ax.set_xticks(x, [s for s, _ in stages], rotation=55, ha="right", fontsize=6.5)
        ax.set_ylim(0, None)
        ax.set_title(f"{titles[cfg]}\n({len(seeds)} circuits)", fontsize=7)
    axs[0].set_ylabel("gates")
    axs[3].set_ylabel("gates")
    save(fig, "fig_paper_flow")


def hw_flow():
    """Hardware suite: start (Qiskit level 1) to end, per circuit and summed, for every pipeline."""
    B = J("general/bench_v2.json")
    EX = {"nam/pf1_10", "nam/pf2_10"}
    # BQSKit (ext_bqskit.py) on the same level-1 inputs, merged by (name, device)
    for fn, src, dst in (
        ("bqskit_v2.jsonl", "bqskit_L3", "bqskit_L3"),
        ("bqskit_stack_v1.jsonl", "bqskit_L3+ours", "bq_ours"),
    ):
        p = RESULTS / "general" / fn
        rows = {(r["name"], r["device"]): r for r in map(json.loads, open(p))} if p.exists() else {}
        for r in B:
            if src in rows.get((r["name"], r["device"]), {}):
                r[dst] = rows[(r["name"], r["device"])][src]
    pipes = [
        ("qiskit_L1", "Qiskit L1", C["input"]),
        ("tket", "TKET", C["tket"]),
        ("qiskit_L3", "Qiskit L3", C["qiskit"]),
        ("bqskit_L3", "BQSKit L3", C["bqskit"]),
        ("ours_exact_cal", "qcr from L1", "#8ab4e8"),
        ("ours_exact_cal_fromL3", "qcr from L3", C["general"]),
        ("bq_ours", "qcr on BQSKit L3", "#7a1f4b"),
    ]
    pipes = [p for p in pipes if all(p[0] in r for r in B)]
    fig = plt.figure(figsize=(W, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[4.6, 0.2 * len(pipes)], hspace=0.22, wspace=0.75)
    for col, (dev, title) in enumerate((("heron", "IBM Heron (FakeTorino)"), ("ion", "trapped ion (model)"))):
        R = sorted([r for r in B if r["device"] == dev and r["name"] not in EX], key=lambda r: r["qiskit_L3"]["total"])
        ax = fig.add_subplot(gs[0, col])
        y = np.arange(len(R))
        for r, yy in zip(R, y):
            vals = [r[k]["total"] for k, _, _ in pipes]
            ax.plot([min(vals), max(vals)], [yy, yy], color="#e4e3dc", lw=1.2, zorder=1)
        for k, lab, c in pipes:
            ax.scatter([r[k]["total"] for r in R], y, s=7, color=c, label=lab, zorder=3, linewidths=0)
        ax.set_xscale("log")
        ax.set_yticks(y, [r["name"].split("/")[1] for r in R], fontsize=4.8)
        ax.set_ylim(-1, len(R))
        ax.set_xlabel("total gates (log)")
        ax.set_title(title)
        if col == 0:
            ax.legend(fontsize=6, loc="lower right", markerscale=1.6)
        bx = fig.add_subplot(gs[1, col])
        tot2 = [sum(r[k]["twoq"] for r in R) for k, _, _ in pipes]
        tot1 = [sum(r[k]["total"] - r[k]["twoq"] for r in R) for k, _, _ in pipes]
        yy = np.arange(len(pipes))
        bx.barh(yy, tot2, color=[c for _, _, c in pipes], edgecolor="white")
        bx.barh(yy, tot1, left=tot2, color=[c for _, _, c in pipes], alpha=0.35, edgecolor="white")
        for i, (a, b) in enumerate(zip(tot2, tot1)):
            bx.text(a + b, i, f" {a:,} / {a + b:,}", va="center", fontsize=5.2)
        bx.set_yticks(yy, [lab for _, lab, _ in pipes], fontsize=6)
        bx.invert_yaxis()
        bx.set_xlabel("summed gates, 58 circuits\n(solid bars are two-qubit gates)", fontsize=6.5)
        bx.set_xlim(0, max(a + b for a, b in zip(tot2, tot1)) * 1.6)
        bx.tick_params(axis="x", labelsize=6)
    save(fig, "fig_hw_flow")


# ------------------------------------------------------------------ new experiments
def cliffordt():
    """Clifford+T: certified optimum vs. methods (random words and planted circuits)."""
    D = J("ext_cliffordt.json")
    rows = D["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
    ax = axs[0]
    Ls = sorted({r["size"] for r in rows if r["kind"] == "random"})
    for key, lab, c, mk in (
        ("optimum", "certified optimum", C["optimum"], "o"),
        ("general", "general", C["general"], "s"),
        ("original", "original", C["original"], "^"),
        ("qiskit_L3", "Qiskit L3 (continuous)", C["qiskit"], "v"),
    ):
        m = []
        for L in Ls:
            rs = [r for r in rows if r["kind"] == "random" and r["size"] == L and r["optimum_exact"]]
            m.append(np.mean([r[key] for r in rs]) if rs else np.nan)
        ax.plot(Ls, m, marker=mk, ms=3.5, color=c, label=lab)
    ax.plot(Ls, Ls, color="#c9c8c1", ls=(0, (2, 2)), lw=0.8, label="input")
    frac = [np.mean([r["optimum_exact"] for r in rows if r["kind"] == "random" and r["size"] == L]) for L in Ls]
    for L, f in zip(Ls, frac):
        if f < 1:
            ax.annotate(
                f"{f:.0%} certified", (L, L), textcoords="offset points", xytext=(0, 4), fontsize=5.5, ha="center"
            )
    ax.set_xlabel("input length (random Clifford+T words, 2 qubits)")
    ax.set_ylabel("gates (mean over certified circuits)")
    ax.set_title("(a) random words")
    ax.legend(fontsize=6)
    ax = axs[1]
    P = [r for r in rows if r["kind"] == "planted"]
    cfgs = sorted({tuple(r["size"]) for r in P})
    x = np.arange(len(cfgs))
    w = 0.2
    for i, (key, lab, c) in enumerate(
        (
            ("optimum", "certified optimum", C["optimum"]),
            ("general", "general", C["general"]),
            ("original", "original", C["original"]),
            ("qiskit_L3", "Qiskit L3", C["qiskit"]),
        )
    ):
        vals = [np.mean([r[key] for r in P if tuple(r["size"]) == cf]) for cf in cfgs]
        ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=lab)
    ax.set_xticks(x, [f"core {c}\nin {L}" for c, L in cfgs], fontsize=6)
    ax.set_ylabel("output gates (mean)")
    ax.set_title("(b) planted circuits (optimum = core length)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, "fig_cliffordt")


def layout():
    """Layout-aware rewriting: ESP (with readout) and noisy TVD per pipeline."""
    R = J("general/bench_layout.json")
    keys = [
        ("L3", "Qiskit L3", C["qiskit"]),
        ("L3+ours", "L3 + qcr", "#8ab4e8"),
        ("L3na", "Qiskit L3\nnoise-aware", C["tket"]),
        ("L3na+ours", "noise-aware\n+ qcr", "#7aa6dd"),
        ("L3na+ours+rel", "+ relocation", C["general"]),
        ("joint", "joint layout\n(qcr)", C["optimum"]),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.7))
    ax = axs[0]
    base = np.array([r["L3na"]["success_readout"] for r in R])
    for i, (k, lab, c) in enumerate(keys):
        v = np.array([r[k]["success_readout"] for r in R]) / base
        ax.scatter(
            np.full(len(v), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(v)), v, s=4, color=c, alpha=0.7
        )
        gm = float(np.exp(np.mean(np.log(v))))
        ax.plot([i - 0.3, i + 0.3], [gm, gm], color="#1f1f1e", lw=1.2)
        ax.annotate(f"×{gm:.2f}", (i, gm), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)
    ax.axhline(1, color="#c9c8c1", lw=0.8, ls=(0, (2, 2)))
    ax.set_yscale("log")
    ax.set_xticks(range(len(keys)), [lab for _, lab, _ in keys], fontsize=5.8)
    ax.set_ylabel("ESP incl. readout / noise-aware Qiskit L3")
    ax.set_title(f"(a) estimated success probability ({len(R)} circuits)")
    ax = axs[1]
    T = [r for r in R if "tvd_readout" in r["L3na"]]
    for i, (k, lab, c) in enumerate(keys):
        v = [r[k]["tvd_readout"] for r in T]
        ax.boxplot(
            [v],
            positions=[i],
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor=c, alpha=0.6, lw=0.5),
            medianprops=dict(color="#1f1f1e"),
        )
        ax.annotate(
            f"{np.mean(v):.3f}",
            (i, np.percentile(v, 75)),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=6,
        )
    ax.set_xticks(range(len(keys)), [lab for _, lab, _ in keys], fontsize=5.8)
    ax.set_ylabel("noisy TVD incl. readout (lower is better)")
    ax.set_title(f"(b) noisy simulation, FakeTorino ({len(T)} circuits $\\leq$ 8 qubits)")
    fig.tight_layout()
    save(fig, "fig_layout")


if __name__ == "__main__":
    which = sys.argv[1:] or ["compute_graph", "growth", "examples", "paper_flow", "hw_flow", "cliffordt", "layout"]
    for w in which:
        globals()[w]()
