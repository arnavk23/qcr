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


"""ext_numbers.tex for the paper (its figures are drawn by make_paper_figs.py), and numbers.tex for the
reproduction report.  Missing inputs are rendered as '??'."""

import json
from pathlib import Path

import numpy as np
from common import RESULTS

EXT = Path(__file__).resolve().parents[1] / "manuscripts" / "paper"
REP = Path(__file__).resolve().parents[1] / "manuscripts" / "reproduction"
M = {}


def J(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def fmt(x, nd=1):
    return "??" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


# ---------------------------------------------------------------- group data
ci, cn = J("cayley3_ion_pm.json"), J("cayley3_nisq.json")
M["DiamIonThree"] = fmt(ci and ci["diameter"])
M["DiamIonThreePlusOne"] = fmt(ci and ci["diameter"] + 1)
M["MeanIonThree"] = fmt(ci and ci["mean_dist"], 2)
M["DiamIbmThree"] = fmt(cn and cn["diameter"])
M["MeanIbmThree"] = fmt(cn and cn["mean_dist"], 2)
M["BFSMinutes"] = fmt(ci and round(ci["bfs_seconds"] / 60))
par = {}
for f in RESULTS.glob("ext_parity_*.json"):
    par.update(json.load(open(f)))
rho = par.get("ion_pm", {}).get("shortest_odd_relator")
M["OddRelIon"] = fmt(rho)
M["OddRelIonMinusOne"] = fmt(rho and rho - 1)
M["OddRelIonHalf"] = fmt(rho and (rho + 1) // 2)
M["OddRelIonHalfMinusOne"] = fmt(rho and (rho + 1) // 2 - 1)
M["OddRelIbm"] = fmt(par.get("nisq", {}).get("shortest_odd_relator"))

# ---------------------------------------------------------------- 2q counts
tq = J("ext_twoq3.json")
if tq:
    lv = tq["layers"]
    M["TwoQThreeDist"] = (
        "/".join(f"{x:,}" for x in lv).replace(",", "{,}") + f" for $0/\\dots/{len(lv) - 1}$ entangling gates"
    )
    M["TwoQThreeMean"] = fmt(tq["mean"], 2)
    M["TwoQThreeMax"] = fmt(len(lv) - 1)
else:
    M["TwoQThreeDist"] = M["TwoQThreeMean"] = M["TwoQThreeMax"] = "??"

# ---------------------------------------------------------------- gap (3 qubits)
ws = [2, 3, 4, 6, 8, 12, 16, 24, 32]
gaps = {}
for k in ("ion", "nisq"):
    rs = J(f"ext_gap3_{k}.json")
    if not rs:
        continue
    gaps[k] = rs
    tag = "Ion" if k == "ion" else "Ibm"
    mean = lambda key: float(np.mean([r[key] for r in rs]))
    M[f"Opt{tag}"] = fmt(mean("opt"), 1)
    M[f"Faith{tag}"] = fmt(mean("faithful"), 1)
    M[f"EWeight{tag}"] = fmt(mean("ew8"), 1)
    M[f"EWsixteen{tag}"] = fmt(mean("ew16"), 1)
    M[f"Faith{tag}Sec"] = fmt(mean("faithful_s"), 0)
    M[f"EWeight{tag}Sec"] = fmt(mean("ew8_s"), 2)
    M[f"EWsixteen{tag}Sec"] = fmt(mean("ew16_s"), 2)
    M[f"GapFactor{tag}"] = fmt(mean("faithful") / mean("opt"), 1)
    M[f"Speedup{tag}"] = fmt(round(mean("faithful_s") / mean("ew16_s"), -1), 0)
    wstar = next((w for w in ws if all(r[f"ew{w}"] == r["opt"] for r in rs)), None)
    M[f"WStar{tag}"] = fmt(wstar)
for key in ("OptIbm", "FaithIbm", "EWeightIbm", "EWsixteenIbm", "GapFactorIbm", "WStarIbm", "SpeedupIbm"):
    M.setdefault(key, "??")
grid = J("ext_grid_ion.json")
if grid:
    W, R = [4, 6, 8, 10, 12, 16], [3, 4, 5, 6, 8, 99]  # figure: make_paper_figs.window_grid
    A = np.array([[np.mean([g[f"w{w}_r{r}"] - g["opt"] for g in grid]) for w in W] for r in R])
    M["GridEightFive"] = fmt(float(A[R.index(5), W.index(8)]), 1)
    M["GridEightInf"] = fmt(float(A[R.index(99), W.index(8)]), 1)
else:
    M["GridEightFive"] = M["GridEightInf"] = "??"

# ---------------------------------------------------------------- mixing
mx = J("ext_mixing.json")
if mx:
    u = mx["ion_pm_2"]["uniform"]
    M["MeanIonTwoEven"], M["MeanIonTwoOdd"] = fmt(u["mean_even"], 2), fmt(u["mean_odd"], 2)
    eig = [abs(x) for x in mx["spectral_ion2"]["top_abs_eigs"]]
    beta = max(e for e in eig if e < 1 - 1e-9)
    M["BetaTwo"] = fmt(beta, 3)
    M["LMixTwo"] = str(int(np.ceil(np.log(0.01 / (0.5 * np.sqrt(11520))) / np.log(beta))))
    if "ion_pm_3" in mx:
        target = mx["ion_pm_3"]["uniform"]["mean"]
        Lmix = next((c["L"] for c in mx["ion_pm_3"]["curve"] if abs(c["mean"] - target) < 0.15), None)
        M["MixLThree"] = fmt(Lmix)
for k in ("MeanIonTwoEven", "MeanIonTwoOdd", "BetaTwo", "LMixTwo", "MixLThree"):
    M.setdefault(k, "??")

# ---------------------------------------------------------------- four qubits
sp = J("ext_four_spheres.json")
M["LBFourIon"] = fmt(sp and sp["ion_pm"]["mean_lb"], 1)
M["LBFourIbm"] = fmt(sp and sp["nisq"]["mean_lb"], 1)
from ext_four import lower_bounds  # noqa: E402

if ci:
    M["LBThreeCheck"] = fmt(lower_bounds(ci["levels"][:7], 92897280)["mean_lb"], 1)
qs = J("quality_summary.json") or {}
for k, tag, g2 in (("ion", "Ion", "rxx"), ("nisq", "Ibm", "cz")):
    q = qs.get(f"NJP_{k}_4q")
    ewall = J("ext_four_ew_ion_nisq.json") or []
    ew = [r for r in ewall if r["kind"] == k] or None
    cwall = J("ext_four_cw_ion_nisq.json") or []
    cwk = [r for r in cwall if r["kind"] == k] or None
    M[f"CWFour{tag}"] = fmt(cwk and float(np.mean([r["total"] for r in cwk])), 1)
    M[f"CWFour{tag}TwoQ"] = fmt(cwk and float(np.mean([r["counts"].get(g2, 0) for r in cwk])), 1)
    if k == "ion":
        M["CWFourSec"] = fmt(cwk and float(np.mean([r["seconds"] for r in cwk])), 1)
        M["EWFourSec"] = fmt(ew and float(np.mean([r["seconds"] for r in ew])), 1)
        M["FaithFourSec"] = fmt(q and q["ours_s"], 0)
    M[f"EWFour{tag}"] = fmt(ew and float(np.mean([r["total"] for r in ew])), 1)
    M[f"EWFour{tag}TwoQ"] = fmt(ew and float(np.mean([r["counts"].get(g2, 0) for r in ew])), 1)
    M[f"FaithFour{tag}"] = fmt(q and q["totals"]["ours"], 1)
    M[f"FaithFour{tag}TwoQ"] = fmt(q and q["twoq"]["ours"], 1)
    M[f"QFour{tag}"] = fmt(q and q["totals"]["qiskit_L3"], 1)
    M[f"QFour{tag}TwoQ"] = fmt(q and q["twoq"]["qiskit_L3"], 1)
    pub = 111 if k == "ion" else 107
    M[f"FourImprovement{tag}"] = "??" if not cwk else f"{100 * (1 - np.mean([r['total'] for r in cwk]) / pub):.0f}\\%"
    t3 = qs.get("T3_ion_3q" if k == "ion" else "T4_nisq_3q")
    M[f"QThree{tag}"] = fmt(t3 and t3["totals"]["qiskit_L3"], 1)

# CW-24 against our BQSKit level-4 run on the same leading seeds (bqskit_njp_v1.jsonl, ext_bqskit.py)
bq4p = Path(RESULTS) / "general" / "bqskit_njp_v1.jsonl"
bq4 = {(r["kind"], r["seed"]): r for r in map(json.loads, open(bq4p))} if bq4p.exists() else {}
cwall = J("ext_four_cw_ion_nisq.json") or []
for k, tag, g2 in (("ion", "Ion", "rxx"), ("nisq", "Ibm", "cz")):
    S = [r for r in cwall if r["kind"] == k and (k, r["seed"]) in bq4]
    if len(S) < 30:  # BQSKit level-4 run not finished: placeholders keep the paper compiling
        for nm in ("N", "TwoQ", "Fewer", "More", "TotalRatio"):
            M[f"CWBq{tag}{nm}"] = "??"
        continue
    b2 = [bq4[(k, r["seed"])] for r in S]
    c2 = [r["counts"].get(g2, 0) for r in S]
    M[f"CWBq{tag}N"] = fmt(len(S))
    M[f"CWBq{tag}TwoQ"] = fmt(float(np.mean(c2)), 1)
    M[f"CWBq{tag}Fewer"] = fmt(sum(c < b["twoq"] for c, b in zip(c2, b2)))
    M[f"CWBq{tag}More"] = fmt(sum(c > b["twoq"] for c, b in zip(c2, b2)))
    M[f"CWBq{tag}TotalRatio"] = fmt(float(np.mean([r["total"] for r in S]) / np.mean([b["total"] for b in b2])), 2)

for k in ("GapFindings", "FourFindings"):
    M.setdefault(k, "")
M["GapFindings"] = (EXT / "findings_gap.tex").read_text() if (EXT / "findings_gap.tex").exists() else ""
M["FourFindings"] = (EXT / "findings_four.tex").read_text() if (EXT / "findings_four.tex").exists() else ""

with open(EXT / "ext_numbers.tex", "w") as f:
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(json.dumps({k: v for k, v in M.items() if "Findings" not in k}, indent=0))
