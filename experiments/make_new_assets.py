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

"""new_numbers.tex for the journal version: Clifford+T ground truth and layout-aware rewriting.

Inputs: results/ext_cliffordt.json (ext_cliffordt.py), results/general/bench_layout.json (gen_layout.py).
Missing inputs render as '??' so the manuscript always compiles.
"""

import json
import math
import statistics as st
from pathlib import Path

from common import RESULTS

EXT = Path(__file__).resolve().parents[1] / "manuscripts" / "extension"
M = {}


def gm(xs):
    return math.exp(st.mean(math.log(x) for x in xs))


# ---------------------------------------------------------------- Clifford+T
p = RESULTS / "ext_cliffordt.json"
if p.exists():
    D = json.load(open(p))
    b, rows = D["ball"], D["rows"]
    M["CTRadius"] = b["radius"]
    M["CTBallSize"] = f"{sum(b['levels']):,}".replace(",", "{,}")
    M["CTBallLevels"] = ", ".join(f"{x:,}".replace(",", "{,}") for x in b["levels"])
    M["CTBallMinutes"] = f"{b['seconds'] / 60:.0f}"
    M["CTCertified"] = b["certified_up_to"]
    assert b["collision_free"], "Clifford+T ball has a hash collision"
    R = [r for r in rows if r["kind"] == "random"]
    Rc = [r for r in R if r["optimum_exact"]]
    P = [r for r in rows if r["kind"] == "planted"]
    M["CTRandN"] = len(R)
    M["CTRandCert"] = len(Rc)
    M["CTRandGenAtOpt"] = f"{sum(r['general'] == r['optimum'] for r in Rc)}/{len(Rc)}"
    M["CTRandOrigAtOpt"] = f"{sum(r['original'] == r['optimum'] for r in Rc)}/{len(Rc)}"
    M["CTPlantN"] = len(P)
    M["CTPlantGenAtOpt"] = sum(r["general"] == r["optimum"] for r in P)
    M["CTPlantOrigAtOpt"] = sum(r["original"] == r["optimum"] for r in P)
    for key, name in (
        ("input", "Input"),
        ("optimum", "Opt"),
        ("general", "Gen"),
        ("original", "Orig"),
        ("qiskit_L3", "Qiskit"),
    ):
        M[f"CTPlant{name}Mean"] = f"{st.mean(r[key] for r in P):.1f}"
    ok = all(r["general_ok"] and r["original_ok"] and r["general_legal"] for r in rows)
    M["CTAllExact"] = "yes" if ok else "NO"
    assert ok, "a Clifford+T output failed exact verification"
    assert all(r["optimum_exact"] for r in P), "planted optimum not certified"

# ---------------------------------------------------------------- layout-aware rewriting
p = RESULTS / "general" / "bench_layout.json"
part = p.with_suffix(".jsonl")  # unfinished run: use the rows so far
if p.exists() or part.exists():
    R = json.load(open(p)) if p.exists() else [json.loads(line) for line in open(part)]
    M["LayN"] = len(R)
    base = [r["L3na"]["success_readout"] for r in R]
    for key, name in (("L3", "LThree"), ("L3na+ours", "NAOurs"), ("L3na+ours+rel", "Rel"), ("joint", "Joint")):
        M[f"LayEsp{name}"] = f"{gm([r[key]['success_readout'] / b for r, b in zip(R, base)]):.2f}"
    for key, name in (
        ("L3", "LThree"),
        ("L3na", "NA"),
        ("L3na+ours", "NAOurs"),
        ("L3na+ours+rel", "Rel"),
        ("joint", "Joint"),
    ):
        M[f"LayTwoq{name}"] = f"{sum(r[key]['twoq'] for r in R):,}".replace(",", "{,}")
    T = [r for r in R if "tvd_readout" in r["L3na"]]
    M["LayTvdN"] = len(T)
    for key, name in (
        ("L3", "LThree"),
        ("L3na", "NA"),
        ("L3na+ours", "NAOurs"),
        ("L3na+ours+rel", "Rel"),
        ("joint", "Joint"),
    ):
        M[f"LayTvd{name}"] = f"{st.mean(r[key]['tvd_readout'] for r in T):.3f}"
    M["LayJointBeatsNA"] = sum(r["joint"]["tvd_readout"] < r["L3na"]["tvd_readout"] for r in T)
    share = st.mean(r["joint"]["choice"].startswith("noise-aware") for r in R)
    M["LayJointNAShare"] = f"{100 * share:.0f}\\%"
    errs = [
        r[k]["verify"]
        for r in R
        for k in ("L3+ours", "L3na+ours", "L3na+ours+rel", "joint")
        if r[k].get("verify") is not None
    ]
    errs += [r["joint"]["verify_optimised"] for r in R if r["joint"].get("verify_optimised") is not None]
    M["LayMaxErr"] = f"{max(errs):.1e}"
    assert max(errs) < 1e-8, "a layout output failed verification"
    assert all(r["relocation_preserves_gates"] and r["joint"]["relocation_preserves_gates"] for r in R)

NAMES = [
    "CTRadius",
    "CTBallSize",
    "CTBallLevels",
    "CTBallMinutes",
    "CTCertified",
    "CTRandN",
    "CTRandCert",
    "CTRandGenAtOpt",
    "CTRandOrigAtOpt",
    "CTPlantN",
    "CTPlantGenAtOpt",
    "CTPlantOrigAtOpt",
    "CTPlantInputMean",
    "CTPlantOptMean",
    "CTPlantGenMean",
    "CTPlantOrigMean",
    "CTPlantQiskitMean",
    "LayN",
    "LayEspLThree",
    "LayEspNAOurs",
    "LayEspRel",
    "LayEspJoint",
    "LayTwoqLThree",
    "LayTwoqNA",
    "LayTwoqNAOurs",
    "LayTwoqRel",
    "LayTwoqJoint",
    "LayTvdN",
    "LayTvdLThree",
    "LayTvdNA",
    "LayTvdNAOurs",
    "LayTvdRel",
    "LayTvdJoint",
    "LayJointBeatsNA",
    "LayJointNAShare",
    "LayMaxErr",
]
with open(EXT / "new_numbers.tex", "w") as fh:
    fh.write("% generated by experiments/make_new_assets.py -- do not edit\n")
    for k in NAMES:
        fh.write(f"\\newcommand{{\\{k}}}{{{M.get(k, '??')}}}\n")
print(json.dumps(M, indent=1))
