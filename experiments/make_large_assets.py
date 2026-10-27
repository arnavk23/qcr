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


"""Numbers and tables for the large-circuit suite and the run-time comparison (paper).

Inputs:  results/general/large_v1.jsonl (gen_large.py ours), large_recert_v1.jsonl (gen_large.py recert),
         results/general/bench_v2.json and bqskit_v2.jsonl (60-circuit suite, for the run-time table).
Outputs: paper/large_numbers.tex.

The run may be partial (gen_large.py is resumable): every number is computed from the rows present.
Nam pf1_* / pf2_* are excluded from the comparison because Qiskit level 3 drops their ~1e-7 rad rotations
(approximately, not exactly), as for pf1_10 / pf2_10 in the 60-circuit suite.
"""

import json
import math
import statistics as st
from pathlib import Path

from common import RESULTS

EXT = Path(__file__).resolve().parents[1] / "manuscripts" / "paper"
DEV = {"heron": "Heron", "ion": "Ion"}
M = {}


def cmd(name, val):
    M[name] = val


def jsonl(name):
    p = RESULTS / "general" / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def sumratio(R, key, m):
    return sum(r[key][m] for r in R) / sum(r["qiskit_L3"][m] for r in R)


def gmratio(R, key, m, base="qiskit_L3"):
    return math.exp(st.mean(math.log(r[key][m] / r[base][m]) for r in R))


def x(v, nd=2):
    return f"\\(\\times\\){v:.{nd}f}"


def sec(v):
    return f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 100 else f"{v:,.0f}".replace(",", "{,}")


# ESP below this is noise-dominated: at ~1e-10 a ratio of two such numbers swings by orders of magnitude and a
# geometric mean over the suite flips with one circuit, so the ESP column only uses circuits Qiskit L3 brings above it
ESP_MIN = 1e-3

L = jsonl("large_v1.jsonl")
RC = {(r["name"], r["device"], r["key"]): r for r in jsonl("large_recert_v1.jsonl")}
OURS = ("ours_exact_cal", "ours_exact_cal_fromL3")
pf = lambda r: r["name"].split("/")[-1].startswith(("pf1_", "pf2_", "pf4_"))

# ---------------------------------------------------------------- replay certificates
certs, pending, unproven = [], [], set()
for r in L:
    for k in OURS:
        c = r[k]["certificate"]
        if not c["ok"]:  # rejected by the first checker version: re-run (deterministic) and re-checked
            rc = RC.get((r["name"], r["device"], k))
            c = rc["certificate"] if rc and rc["same_output"] else c
        (certs if c["ok"] else pending).append(c)
        if not c["ok"]:
            unproven.add((r["name"], r["device"]))
cmd("LargeOutputs", len(certs) + len(pending))
cmd("LargeCertOk", len(certs))
cmd("LargeCertPending", len(pending))
cmd("LargeCircuits", len({r["name"] for r in L}))
cmd("LargeCertRechecked", sum(1 for v in RC.values() if v["same_output"] and v["certificate"]["ok"]))
cmd("LargeCertFirstRejected", sum(not r[k]["certificate"]["ok"] for r in L for k in OURS))
cmd("LargeCertMaxSteps", f"{max(c['steps'] for c in certs):,}".replace(",", "{,}"))
cmd("LargeCertMaxSec", f"{max(c['seconds'] for c in certs):.1f}")
cmd("LargeCertMedSec", f"{st.median(c['seconds'] for c in certs):.2f}")
e = f"{max(c['err'] for c in certs):.0e}".split("e")
cmd("LargeCertMaxErr", f"10^{{{int(e[1]) + 1}}}")

# ---------------------------------------------------------------- comparison table (certified circuits only)
PIPES = [
    ("qiskit_L1", "Qiskit level 1"),
    ("tket", "TKET (FullPeepholeOptimise on L1)"),
    ("ours_exact_cal", "ours, exact, from L1"),
    ("ours_exact_cal_fromL3", "ours, exact, from L3"),
]
rows = {}
cmd("LargeEspMin", "10^{-3}")
for dev, D in DEV.items():
    A = [r for r in L if r["device"] == dev]
    R = [r for r in A if not pf(r) and (r["name"], dev) not in unproven]
    E = [r for r in R if r["qiskit_L3"]["success"] >= ESP_MIN]
    rows[dev] = {k: (sumratio(R, k, "twoq"), sumratio(R, k, "total"), gmratio(E, k, "success")) for k, _ in PIPES}
    cmd(f"Large{D}EspN", len(E))
    cmd(f"Large{D}N", len(R))
    cmd(f"Large{D}PfN", sum(map(pf, A)))
    cmd(f"Large{D}Pending", sum(not pf(r) and (r["name"], dev) in unproven for r in A))
    cmd(f"Large{D}MinQubits", min(r["n"] for r in R))
    cmd(f"Large{D}MaxQubits", max(r["n"] for r in R))
    cmd(f"Large{D}MaxInput", f"{max(r['input_size'] for r in R):,}".replace(",", "{,}"))
    f = [(r["qiskit_L3"]["twoq"], r["ours_exact_cal_fromL3"]["twoq"]) for r in R]
    cmd(f"Large{D}FromLThreeBetter", sum(b < a for a, b in f))
    cmd(f"Large{D}FromLThreeEqual", sum(b == a for a, b in f))
    cmd(f"Large{D}FromLThreeWorse", sum(b > a for a, b in f))
    cmd(f"Large{D}FromLThreeTotalUp", sum(r["ours_exact_cal_fromL3"]["total"] > r["qiskit_L3"]["total"] for r in R))
    cmd(
        f"Large{D}FromLThreeSuccDown",
        sum(r["ours_exact_cal_fromL3"]["success"] < r["qiskit_L3"]["success"] - 1e-12 for r in R),
    )
    cmd(
        f"Large{D}FromLThreeAllSaved",
        f"{100 * (1 - sumratio(R, 'ours_exact_cal_fromL3', 'total')) + 0:.0f}\\%".replace("-0\\%", "0\\%"),
    )
    cmd(f"Large{D}FromLThreeTwoqSaved", f"{100 * (1 - sumratio(R, 'ours_exact_cal_fromL3', 'twoq')):.1f}\\%")
    cmd(f"Large{D}FromLThreeEsp", f"{gmratio(E, 'ours_exact_cal_fromL3', 'success'):.2f}")
    cmd(
        f"Large{D}FromLThreeEspMedAll",
        f"{st.median(r['ours_exact_cal_fromL3']['success'] / r['qiskit_L3']['success'] for r in R):.2f}",
    )
    cmd(f"Large{D}LOneTwoqFewer", sum(r["ours_exact_cal"]["twoq"] < r["qiskit_L3"]["twoq"] for r in R))
    cmd(f"Large{D}LOneTwoqMore", sum(r["ours_exact_cal"]["twoq"] > r["qiskit_L3"]["twoq"] for r in R))
    cmd(f"Large{D}LOneAllFewer", sum(r["ours_exact_cal"]["total"] < r["qiskit_L3"]["total"] for r in R))
    cmd(f"Large{D}Legal", "all" if all(r[k]["legal"] for r in A for k in OURS) else "not all")

with open(EXT / "large_numbers.tex", "w") as fh:
    fh.write("% generated by experiments/make_large_assets.py -- do not edit\n")
    fh.write("\\newcommand{\\LargeRows}{%\n")
    for k, lab in PIPES:
        h, i = rows["heron"][k], rows["ion"][k]
        sep = "\\midrule\n" if k == "ours_exact_cal" else ""
        fh.write(f"{sep}{lab} & {x(h[0])} & {x(h[1])} & {x(h[2])} && {x(i[0])} & {x(i[1])} & {x(i[2], 3)}\\\\\n")
    fh.write("}\n")

# ---------------------------------------------------------------- run-time table (median seconds per circuit)
B = json.load(open(RESULTS / "general" / "bench_v2.json"))
BQ = jsonl("bqskit_v2.jsonl")
LB = jsonl("large_bqskit_v1.jsonl")  # BQSKit on the large suite (the smaller circuits only, the run was stopped)
TIMED = [
    ("qiskit_L1", "Qiskit level 1"),
    ("qiskit_L3", "Qiskit level 3"),
    ("tket", "TKET"),
    ("bqskit_L2", "BQSKit level 2 (8 workers)"),
    ("bqskit_L3", "BQSKit level 3 (8 workers)"),
    ("ours_exact_cal", "ours, exact, from L1"),
    ("ours_exact_cal_fromL3", "ours, exact, from L3"),
]


def med(R, k):
    S = [r[k]["seconds"] for r in R if k in r and "seconds" in r[k]]
    return sec(st.median(S)) if S else "--"


with open(EXT / "large_numbers.tex", "a") as fh:
    fh.write("\\newcommand{\\RuntimeRows}{%\n")
    for k, lab in TIMED:
        cells = []
        for dev in DEV:
            cells.append(med([r for r in (BQ if k.startswith("bqskit") else B) if r["device"] == dev], k))
        for dev in DEV:
            cells.append(med([r for r in (LB if k.startswith("bqskit") else L) if r["device"] == dev], k))
        sep = "\\midrule\n" if k == "ours_exact_cal" else ""
        fh.write(f"{sep}{lab} & {cells[0]} & {cells[1]} && {cells[2]} & {cells[3]}\\\\\n")
    fh.write("}\n")
    for dev, D in DEV.items():
        A = [r for r in L if r["device"] == dev]
        cmd(f"Large{D}MaxSec", f"{max(r[k]['seconds'] for r in A for k in OURS) / 3600:.1f}")
    for name, val in M.items():
        fh.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
print(f"wrote {EXT / 'large_numbers.tex'} ({len(M)} macros, {len(L)} rows)")
