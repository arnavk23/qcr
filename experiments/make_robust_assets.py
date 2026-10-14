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


"""Numbers and tables for the robustness and baseline sections of the extension paper.

Inputs:  results/general/robust_seeds_v1.jsonl, robust_ablation_v1.jsonl (gen_robust.py),
         logical_v1.jsonl (gen_logical.py), large_bqskit_v1.jsonl and large_v1.jsonl (gen_large.py).
Outputs: extension/robust_numbers.tex.

Same conventions as the other asset scripts: count columns are ratios of summed counts relative to Qiskit level 3,
ESP is the geometric mean of per-circuit ratios, and Nam pf* circuits are excluded from the hardware-suite and
large-suite comparisons (Qiskit level 3 drops their ~1e-7 rad rotations approximately).
"""

import json
import math
import statistics as st
from pathlib import Path

from common import RESULTS

EXT = Path(__file__).resolve().parents[1] / "manuscripts" / "extension"
G = RESULTS / "general"
DEV = {"heron": "Heron", "ion": "Ion"}
M = {}
TABLES = {}


def cmd(name, val):
    M[name] = val


def jsonl(name):
    return [json.loads(line) for line in open(G / name)]


def pf(r):
    return r["name"].split("/")[-1].startswith(("pf1_", "pf2_", "pf4_"))


def sr(R, k, m, base="qiskit_L3"):
    return sum(r[k][m] for r in R) / sum(r[base][m] for r in R)


def gm(R, k, base="qiskit_L3"):
    return math.exp(st.mean(math.log(r[k]["success"] / r[base]["success"]) for r in R))


def x(v, nd=2):
    return f"\\(\\times\\){v:.{nd}f}"


def pct(v, nd=0):
    s = f"{100 * v:.{nd}f}"
    return ("0" if s in ("-0", "-0.0") else s) + "\\%"


def thousands(v):
    return f"{v:,}".replace(",", "{,}")


# ---------------------------------------------------------------- seeds (60-circuit hardware suite)
S = [r for r in jsonl("robust_seeds_v1.jsonl") if not r["name"].endswith(("pf1_10", "pf2_10"))]
seeds = sorted({r["transpile_seed"] for r in S})
cmd("SeedN", len(seeds))
cmd("SeedCircuits", len({r["name"] for r in S}))
for dev, D in DEV.items():
    per = {t: [r for r in S if r["device"] == dev and r["transpile_seed"] == t] for t in seeds}
    for key, tag in (("ours_fromL3", "LThree"), ("ours_fromL1", "LOne")):
        for m, mt in (("twoq", "Twoq"), ("total", "All")):
            v = [sr(R, key, m) for R in per.values()]
            cmd(f"Seed{D}{tag}{mt}Mean", f"{st.mean(v):.3f}")
            cmd(f"Seed{D}{tag}{mt}Sd", f"{st.stdev(v):.3f}")
        v = [gm(R, key) for R in per.values()]
        cmd(f"Seed{D}{tag}EspMean", f"{st.mean(v):.2f}")
        cmd(f"Seed{D}{tag}EspSd", f"{st.stdev(v):.2f}")
        cmd(f"Seed{D}{tag}EspMin", f"{min(v):.2f}")
        cmd(f"Seed{D}{tag}EspMax", f"{max(v):.2f}")
    R0 = per[0]
    tot = [sum(r["ours_fromL3"]["total"] for r in R0)]
    tot += [sum(r[f"ours_fromL3_seed{s}"]["total"] for r in R0) for s in range(1, len(seeds))]
    cmd(f"Seed{D}OptSpread", pct((max(tot) - min(tot)) / st.mean(tot), 1))
    bad = sum(
        1
        for r in S
        if r["device"] == dev
        for v in r.values()
        if isinstance(v, dict) and "verify" in v and ((v["verify"] or 0) > 1e-8 or not v["legal"])
    )
    cmd(f"Seed{D}Failed", bad)
cmd("SeedOutputs", thousands(sum(1 for r in S for v in r.values() if isinstance(v, dict) and "verify" in v)))

# ---------------------------------------------------------------- ablation
A = [r for r in jsonl("robust_ablation_v1.jsonl") if not r["name"].endswith(("pf1_10", "pf2_10"))]
VARIANTS = [
    ("full", "full optimiser"),
    ("contiguous", "contiguous windows"),
    ("w8", "window \\(w=8\\)"),
    ("w4", "window \\(w=4\\)"),
    ("no_shuffle", "no shuffles"),
    ("no_db", "no database"),
    ("radius3", "database radius \\(\\le3\\)"),
    ("no_mitm", "no meet in the middle"),
    ("no_resynth", "no resynthesis"),
    ("cost_count", "gate-count cost"),
    ("cost_w2", "2q-weighted cost"),
]
H = [r for r in A if r["device"] == "heron"]
I = [r for r in A if r["device"] == "ion"]  # noqa: E741
rows = []
for v, lab in VARIANTS:
    k3, k1 = f"{v}_fromL3", f"{v}_fromL1"
    rows.append(
        f"{lab} & {pct(1 - sr(H, k3, 'total'), 1)} & {x(sr(H, k3, 'twoq'))} & {x(gm(H, k3))} && "
        f"{x(sr(H, k1, 'total'))} & {x(sr(H, k1, 'twoq'))} && {x(sr(I, k1, 'total'))} & {x(sr(I, k1, 'twoq'))}\\\\"
    )
    if v == "full":
        rows.append("\\midrule")
TABLES["AblationRows"] = rows
cmd("AblationCircuits", len(H))
cmd("AblationFullSaved", pct(1 - sr(H, "full_fromL3", "total"), 1))
for v, tag in (("w4", "WFour"), ("w8", "WEight"), ("no_db", "NoDb"), ("radius3", "RadiusThree"),
               ("no_resynth", "NoResynth"), ("no_mitm", "NoMitm"), ("no_shuffle", "NoShuffle"),
               ("contiguous", "Contiguous")):  # fmt: skip
    cmd(f"Ablation{tag}Saved", pct(1 - sr(H, f"{v}_fromL3", "total"), 1))
cmd("AblationIonNoResynthTwoq", x(sr(I, "no_resynth_fromL1", "twoq")))
cmd("AblationIonFullTwoq", x(sr(I, "full_fromL1", "twoq")))
cmd("AblationHeronCountEsp", x(gm(H, "cost_count_fromL3"), 3))
cmd("AblationHeronCalEsp", x(gm(H, "full_fromL3"), 3))

# ---------------------------------------------------------------- logical level (Nam gate set)
Lg = jsonl("logical_v1.jsonl")
by = {}
for r in Lg:
    by.setdefault(r["name"], {})[r["tool"]] = r
TOOLS = [
    ("qiskit", "Qiskit level 3"),
    ("tket", "TKET (FullPeepholeOptimise)"),
    ("queso5", "QUESO (5 min per circuit)"),
    ("voqc", "VOQC"),
    ("ours", "ours (exact up to phase, gate count)"),
    ("voqc_ours", "VOQC, then ours"),
]
names = sorted(n for n, d in by.items() if all(t in d for t, _ in TOOLS))
inp = sum(by[n]["qiskit"]["input_total"] for n in names)
inp_cx = sum(by[n]["qiskit"]["input_cx"] for n in names)
rows = [f"input & {thousands(inp)} & -- & {thousands(inp_cx)} & -- & --\\\\", "\\midrule"]
for t, lab in TOOLS:
    tot = sum(by[n][t]["total"] for n in names)
    cx = sum(by[n][t]["cx"] for n in names)
    singles = [u for u, _ in TOOLS if u != "voqc_ours"]
    best = (
        "--" if t == "voqc_ours" else sum(by[n][t]["total"] == min(by[n][u]["total"] for u in singles) for n in names)
    )
    if t == "voqc_ours":
        rows.append("\\midrule")
    rows.append(
        f"{lab} & {thousands(tot)} & {pct(tot / inp - 1, 1)} & {thousands(cx)} & {pct(cx / inp_cx - 1, 1)} & {best}\\\\"
    )
    tag = t.title().replace("_", "").replace("5", "Five")  # LaTeX macro names cannot contain digits
    M[f"Logical{tag}Total"] = thousands(tot)
    M[f"Logical{tag}Saved"] = pct(1 - tot / inp, 1)
    M[f"Logical{tag}CxSaved"] = pct(1 - cx / inp_cx, 1)
TABLES["LogicalRows"] = rows
cmd("LogicalCircuits", len(names))
cmd("LogicalOutputs", len(Lg))
cmd("LogicalFailed", sum(not r["check"]["ok"] or not r["native"] for r in Lg))
cmd("LogicalVoqcOursBeatsVoqc", sum(by[n]["voqc_ours"]["total"] < by[n]["voqc"]["total"] for n in names))
cmd("LogicalOursBeatsVoqc", sum(by[n]["ours"]["total"] < by[n]["voqc"]["total"] for n in names))
cmd("LogicalOursBeatsQueso", sum(by[n]["ours"]["total"] < by[n]["queso5"]["total"] for n in names))
cmd("LogicalModFiveOurs", by["mod5_4"]["ours"]["total"])
cmd("LogicalModFiveVoqc", by["mod5_4"]["voqc"]["total"])
q1 = [n for n in names if "queso" in by[n]]  # the circuits QUESO also ran for 1 h
cmd("LogicalQuesoHourN", len(q1))
cmd("LogicalQuesoHourTotal", sum(by[n]["queso"]["total"] for n in q1))
cmd("LogicalQuesoFiveOnHourTotal", sum(by[n]["queso5"]["total"] for n in q1))
cmd("LogicalOursMedSec", f"{st.median(by[n]['ours']['seconds'] for n in names):.0f}")
cmd("LogicalVoqcMedSec", f"{st.median(by[n]['voqc']['seconds'] for n in names):.2f}")

# ---------------------------------------------------------------- BQSKit on the large suite
LB = jsonl("large_bqskit_v1.jsonl")
LL = {(r["name"], r["device"]): r for r in jsonl("large_v1.jsonl")}
for dev, D in DEV.items():
    R = []
    for b in LB:
        a = LL.get((b["name"], b["device"]))
        if b["device"] != dev or a is None or pf(b) or "twoq" not in b.get("bqskit_L3", {}):
            continue
        R.append(a | {"bqskit_L2": b["bqskit_L2"], "bqskit_L3": b["bqskit_L3"]})
    cmd(f"LargeBq{D}N", len(R))
    cmd(f"LargeBq{D}MaxQubits", max(r["n"] for r in R))
    for k, tag in (("bqskit_L3", "BqLThree"), ("ours_exact_cal_fromL3", "OursLThree"), ("ours_exact_cal", "OursLOne")):
        cmd(f"Large{D}{tag}Twoq", x(sr(R, k, "twoq")))
        cmd(f"Large{D}{tag}All", x(sr(R, k, "total")))
    cmd(
        f"Large{D}BqFewerTwoqThanOursLThree",
        sum(r["bqskit_L3"]["twoq"] < r["ours_exact_cal_fromL3"]["twoq"] for r in R),
    )
ver = [b[k]["verify"] for b in LB for k in ("bqskit_L2", "bqskit_L3") if isinstance(b.get(k, {}).get("verify"), dict)]
sv = [v["err"] for v in ver if v.get("method") == "statevector"]
cmd("LargeBqRows", len(LB))
cmd("LargeBqCircuits", len({b["name"] for b in LB}))
cmd("LargeBqChecked", len(sv))
cmd("LargeBqUnchecked", len(ver) - len(sv))
e = f"{max(sv):.0e}".split("e")
cmd("LargeBqMaxErr", f"{e[0]}\\cdot10^{{{int(e[1])}}}")

with open(EXT / "robust_numbers.tex", "w") as fh:
    fh.write("% generated by experiments/make_robust_assets.py -- do not edit\n")
    for name, rows in TABLES.items():
        fh.write(f"\\newcommand{{\\{name}}}{{%\n" + "\n".join(rows) + "\n}\n")
    for name, val in M.items():
        fh.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
print(f"wrote {EXT / 'robust_numbers.tex'} ({len(M)} macros)")
