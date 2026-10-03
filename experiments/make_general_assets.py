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


"""Numbers, tables and figures for the general-engine part of the extension paper.

Inputs:  results/general/bench_v2.json (hardware suite, run 2),
         results/general_paper_bench_*.json (paper benchmarks, general engine),
         results/quality_all.json, ext_bench_cliff_*.json, ext_gap3_*.json (reference numbers).
Outputs: extension/gen_numbers.tex, extension/gen_hw_table.tex, extension/figs/gen_paper_bench.pdf,
         extension/figs/gen_tvd.pdf.

Conventions (same as manuscripts/talk/talk.tex): count ratios are ratios of summed counts; success-probability
ratios are geometric means of per-circuit ratios; Nam pf1_10 / pf2_10 are excluded from the comparison
with Qiskit L3 because L3 drops their ~1e-7 rad rotations (approximate, not exact).
"""

import json
import math
import statistics as st
from pathlib import Path

import matplotlib
from common import RESULTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXT = Path(__file__).resolve().parents[1] / "manuscripts" / "extension"
FIG = EXT / "figs"
EXCLUDE = {"nam/pf1_10", "nam/pf2_10"}
M = {}
tot = lambda c: sum(c.values()) if isinstance(c, dict) else c


def cmd(name, val):
    M[name] = val


def sumratio(R, key, m, base="qiskit_L3"):
    return sum(r[key][m] for r in R) / sum(r[base][m] for r in R)


def gmratio(R, key, m, base="qiskit_L3"):
    P = [(r[base][m], r[key][m]) for r in R if r[base][m] > 0 and r[key][m] > 0]
    return math.exp(st.mean(math.log(b / a) for a, b in P))


def sci(v, nd=1):
    m, e = f"{v:.{nd}e}".split("e")
    return f"{m}\\cdot10^{{{int(e)}}}"


def x(v, nd=2):
    return f"\\(\\times\\){v:.{nd}f}"


# ---------------------------------------------------------------- hardware suite (run 2)
B = json.load(open(RESULTS / "general" / "bench_v2.json"))
DEV = {"heron": "Heron", "ion": "Ion"}
cmd("HwCircuits", len({r["name"] for r in B}))
cmd("HwCompared", len({r["name"] for r in B} - EXCLUDE))
PIPES = [
    ("tket", "TKET (FullPeepholeOptimise on L1)"),
    ("ours_exact_cal", "ours, exact, from L1"),
    ("ours_exact_cal_fromL3", "ours, exact, from L3"),
    ("ours_approx_cal", "ours, approximate ($\\varepsilon=10^{-3}$), from L1"),
    ("ours_state_cal", "ours, state/measurement-aware, from L1"),
]
rows = {}
for dev, D in DEV.items():
    R = [r for r in B if r["device"] == dev and r["name"] not in EXCLUDE]
    rows[dev] = {k: (sumratio(R, k, "twoq"), sumratio(R, k, "total"), gmratio(R, k, "success")) for k, _ in PIPES}
    A = [r for r in B if r["device"] == dev]
    # stacked on L3: never worse in 2q count; total may rise when 1q pulses are traded (calibrated cost)
    f = [(r["qiskit_L3"]["twoq"], r["ours_exact_cal_fromL3"]["twoq"]) for r in R]
    cmd(f"{D}FromLThreeBetter", sum(b < a for a, b in f))
    cmd(f"{D}FromLThreeEqual", sum(b == a for a, b in f))
    cmd(f"{D}FromLThreeWorse", sum(b > a for a, b in f))
    cmd(f"{D}FromLThreeTotalUp", sum(r["ours_exact_cal_fromL3"]["total"] > r["qiskit_L3"]["total"] for r in R))
    cmd(
        f"{D}FromLThreeSuccDown",
        sum(r["ours_exact_cal_fromL3"]["success"] < r["qiskit_L3"]["success"] - 1e-12 for r in R),
    )
    # where the ESP gain from L1 comes from: L1's own layout vs our rewriting
    cmd(f"{D}EspLOneOverLThree", f"{gmratio(R, 'qiskit_L1', 'success'):.2f}")
    cmd(f"{D}EspOursOverLOne", f"{gmratio(R, 'ours_exact_cal', 'success', base='qiskit_L1'):.2f}")
    # which gates the L3-stacked run removes
    for k, nm in (("qiskit_L3", "LThree"), ("ours_exact_cal_fromL3", "FromLThree")):
        cmd(f"{D}{nm}Rz", sum(r[k]["rz"] for r in R))
        cmd(f"{D}{nm}Pulse", sum(r[k]["oneq"] - r[k]["rz"] for r in R))
        cmd(f"{D}{nm}Twoq", sum(r[k]["twoq"] for r in R))
    cmd(f"{D}MaxQubits", max(r["n"] for r in A))
    cmd(f"{D}MaxInput", max(r["input_size"] for r in A))
    # verification
    runs = [r[k] for r in A for k in r if k.startswith("ours_")]
    cmd(f"{D}Runs", len(runs))
    cmd(f"{D}Unverified", sum(v["verify"] is None for v in runs))
    ex = [
        r[k]["verify"]
        for r in A
        for k in ("ours_exact_count", "ours_exact_w2", "ours_exact_cal", "ours_exact_cal_fromL3")
        if r[k]["verify"] is not None
    ]
    cmd(f"{D}MaxExactErr", sci(max(ex)))
    cmd(
        f"{D}MaxStateErr",
        sci(max(r["ours_state_cal"]["verify"] for r in A if r["ours_state_cal"]["verify"] is not None), 0),
    )
    ap = [
        (r["ours_approx_cal"]["verify"], r["ours_approx_cal"]["err_bound"])
        for r in A
        if r["ours_approx_cal"]["verify"] is not None
    ]
    cmd(f"{D}MaxApproxErr", sci(max(a for a, _ in ap)))
    cmd(f"{D}ApproxWithinBound", sum(a <= b + 1e-9 for a, b in ap))
    cmd(f"{D}AllLegal", sum(all(r[k].get("legal", True) for k in r if k.startswith("ours_")) for r in A))
    cmd(f"{D}StateEmpty", sum(r["ours_state_cal"]["total"] == 0 for r in A))
    cmd(f"{D}PhaseGadgetGates", sum(r["ours_exact_cal"]["stats"].get("phase_gadget_gates", 0) for r in A))
    cmd(f"{D}MedianSec", f"{st.median(r['ours_exact_cal']['seconds'] for r in A):.1f}")
    cmd(f"{D}MaxSec", f"{max(r['ours_exact_cal']['seconds'] for r in A):.0f}")
    # objective selection: noisy-simulation TVD
    T = [r for r in A if "tvd" in r and r["name"] not in EXCLUDE]
    cmd(f"{D}TvdCircuits", len(T))
    for tag, nm in (
        ("qiskit_L3", "LThree"),
        ("L1", "LOne"),
        ("ours_exact_count", "Count"),
        ("ours_exact_w2", "WTwo"),
        ("ours_exact_cal", "Cal"),
    ):
        cmd(
            f"{D}Tvd{nm}",
            f"{st.mean(r['tvd'][tag] for r in T):.4f}" if dev == "ion" else f"{st.mean(r['tvd'][tag] for r in T):.3f}",
        )
    cmd(f"{D}CalBeatsLThree", sum(r["tvd"]["ours_exact_cal"] < r["tvd"]["qiskit_L3"] for r in T))
    cmd(f"{D}CalBeatsCount", sum(r["tvd"]["ours_exact_cal"] < r["tvd"]["ours_exact_count"] for r in T))

with open(EXT / "gen_hw_table.tex", "w") as fh:
    fh.write("% generated by experiments/make_general_assets.py -- do not edit\n\\newcommand{\\HwRows}{%\n")
    for k, label in PIPES:
        h, i = rows["heron"][k], rows["ion"][k]
        fh.write(f"{label} & {x(h[0])} & {x(h[1])} & {x(h[2])} && {x(i[0])} & {x(i[1])} & {x(i[2], 3)}\\\\\n")
        if k == "tket":
            fh.write("\\midrule\n")
    fh.write("}\n")

# pf1_10 / pf2_10 separately
for name in sorted(EXCLUDE):
    r = next(r for r in B if r["name"] == name and r["device"] == "heron")
    key = name.split("/")[1].replace("_10", "").replace("pf", "Pf").replace("1", "One").replace("2", "Two")
    cmd(f"{key}LThreeTwoq", r["qiskit_L3"]["twoq"])
    cmd(f"{key}OursTwoq", r["ours_exact_cal"]["twoq"])
    cmd(f"{key}LOneTwoq", r["qiskit_L1"]["twoq"])

# ---------------------------------------------------------------- paper benchmarks, general engine
g = json.load(open(RESULTS / "general_paper_bench_T3_T4_NJP_ion_NJP_nisq_T5.json"))
q = {(r["cfg"], r["seed"]): r for r in json.load(open(RESULTS / "quality_all.json"))}
e = {(r["cfg"], r["seed"]): r for r in json.load(open(RESULTS / "ext_bench_cliff_T5_NJP_ion_NJP_nisq.json"))}
gap = {("T3", r["seed"]): r for r in json.load(open(RESULTS / "ext_gap3_ion.json"))} | {
    ("T4", r["seed"]): r for r in json.load(open(RESULTS / "ext_gap3_nisq.json"))
}
QM = {"T3": "T3_ion_3q", "T4": "T4_nisq_3q", "NJP_ion": "NJP_ion_4q", "NJP_nisq": "NJP_nisq_4q", "T5": "T5_nisq_15q"}
NM = {"T3": "TThree", "T4": "TFour", "NJP_ion": "NjpIon", "NJP_nisq": "NjpIbm", "T5": "TFive"}
cmd("PaperBenchN", len(g))
cmd("PaperBenchOk", sum(r["ok"] and r["legal"] for r in g))
bars = []
for cfg in ("T3", "T4", "NJP_ion", "NJP_nisq", "T5"):
    G = [r for r in g if r["cfg"] == cfg]
    nm = NM[cfg]
    cmd(f"{nm}Gen", f"{st.mean(tot(r['general']) for r in G):.1f}")
    cmd(f"{nm}GenSec", f"{st.mean(r['seconds'] for r in G):.1f}")
    cmd(f"{nm}N", len(G))
    P = [r for r in G if (QM[cfg], r["seed"]) in q]
    paper = lambda r: tot(q[(QM[cfg], r["seed"])]["ours"])
    cmd(f"{nm}Paper", f"{st.mean(paper(r) for r in P):.1f}")
    cmd(f"{nm}PaperN", len(P))
    gen = st.mean(tot(r["general"]) / paper(r) for r in P)
    cmd(f"{nm}GenOverPaper", f"{gen:.2f}")
    l3 = [tot(q[(QM[cfg], r["seed"])]["qiskit_L3"]) for r in P if "qiskit_L3" in q[(QM[cfg], r["seed"])]]
    cmd(f"{nm}LThree", f"{st.mean(l3):.1f}" if l3 else "--")
    C = [r for r in G if (cfg, r["seed"]) in e]
    cw = [tot(e[(cfg, r["seed"])]["ours"]) / paper(r) for r in P if (cfg, r["seed"]) in e]
    cmd(f"{nm}Cliff", f"{st.mean(tot(e[(cfg, r['seed'])]['ours']) for r in C):.1f}" if C else "--")
    O = [r for r in G if (cfg, r["seed"]) in gap]
    if O:
        cmd(f"{nm}Opt", f"{st.mean(gap[(cfg, r['seed'])]['opt'] for r in O):.1f}")
        cmd(f"{nm}AtOpt", sum(tot(r["general"]) == gap[(cfg, r["seed"])]["opt"] for r in O))
    else:
        cmd(f"{nm}Opt", "--")
    op = [gap[(cfg, r["seed"])]["opt"] / paper(r) for r in P if (cfg, r["seed"]) in gap]
    bars.append((cfg, gen, st.mean(cw) if cw else None, st.mean(op) if op else None))

with open(EXT / "gen_numbers.tex", "w") as fh:
    fh.write("% generated by experiments/make_general_assets.py -- do not edit\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a63"
LBL = {
    "T3": "T3  ion, 3q",
    "T4": "T4  IBM-style, 3q",
    "NJP_ion": "NJP  ion, 4q",
    "NJP_nisq": "NJP  IBM-style, 4q",
    "T5": "T5  IBM-style, 15q",
}
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=200)
h = 0.28
series = [
    ("general engine (no Clifford knowledge)", BLUE, 1),
    ("Clifford-specific CW", ORANGE, 2),
    ("proven optimum", AQUA, 3),
]
for i, row in enumerate(bars):
    present = [(col, row[idx]) for _, col, idx in series if row[idx] is not None]
    for k, (col, v) in enumerate(present):
        y = i + (k - (len(present) - 1) / 2) * h
        ax.barh(y, v, height=h - 0.04, color=col)
        ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=7)
ax.axvline(1.0, color=MUTED, lw=1, ls=(0, (3, 2)))
ax.set_yticks(range(len(bars)), [LBL[r[0]] for r in bars])
ax.set_xlim(0, 1.12)
ax.invert_yaxis()
ax.set_xlabel("output length / original method (same circuits, mean ratio)")
from matplotlib.patches import Patch

ax.legend(
    handles=[Patch(color=c, label=n) for n, c, _ in series],
    loc="lower center",
    bbox_to_anchor=(0.45, 1.0),
    ncol=3,
    fontsize=7,
    frameon=False,
)
fig.tight_layout()
fig.savefig(FIG / "gen_paper_bench.pdf")
fig.savefig(FIG / "gen_paper_bench.png")  # README
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), dpi=200)
for ax, dev in zip(axs, ("heron", "ion")):
    T = [r for r in B if r["device"] == dev and "tvd" in r and r["name"] not in EXCLUDE]
    L3 = [r["tvd"]["qiskit_L3"] for r in T]
    CA = [r["tvd"]["ours_exact_cal"] for r in T]
    lim = max(L3 + CA) * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.scatter(L3, CA, s=9, color=BLUE)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("TVD, Qiskit L3")
    ax.set_ylabel("TVD, ours (calibrated)")
    ax.set_title({"heron": "IBM Heron (FakeTorino noise)", "ion": "trapped ion (model noise)"}[dev], fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "gen_tvd.pdf")
fig.savefig(FIG / "gen_tvd.png")  # README
print(json.dumps(M, indent=1))
print(json.dumps(rows, indent=1))
