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


"""Turn results/*.json into figures and LaTeX table snippets for the reports."""

import json
from collections import defaultdict

import matplotlib
from common import FIGS, RESULTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
TEX = RESULTS / "tex"
TEX.mkdir(exist_ok=True)


def load(name):
    p = RESULTS / name
    return json.load(open(p)) if p.exists() else None


def table2():
    rows = load("table2_stats.json")
    if rows is None:
        rows = [json.loads(l) for l in open(RESULTS / "table2_stats.log") if l.startswith("{")]
    d = defaultdict(list)
    for r in rows:
        d[r["variant"]].append(r)
    order = ["V1-RS", "V2-DR", "V3-RF", "V3-RF(t=0.2)"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    data = [[x["seconds"] for x in d[k]] for k in order]
    ax.boxplot(
        data,
        tick_labels=["V1-RS", "V2-DR", "V3-RF\n(t=0.5)", "V3-RF\n(t=0.2)"],
        flierprops=dict(marker="+", markeredgecolor="r"),
    )
    ax.set_ylabel("computation time (s)")
    ax.set_title("Fig. 7 reproduction: 100 random 5-qubit circuits, length 100 → 60", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "repro_boxplot.png", dpi=200)
    lines = []
    for k in order:
        s = np.array([x["seconds"] for x in d[k]])
        att = np.array([x["attempts"] for x in d[k]])
        lk = np.array([x["lookups"] for x in d[k]])
        lines.append(
            f"{k} & {len(s)} & {s.mean():.1f} & {s.std(ddof=1):.1f} & {np.median(s):.1f} & "
            f"{sum(x['reached'] for x in d[k])} & {np.mean(att):,.0f} & {np.mean(lk):,.0f}\\\\"
        )
    open(TEX / "table2.tex", "w").write("\n".join(lines))
    return d


def quality():
    rows = load("quality_all.json")
    if rows is None:
        return
    by = defaultdict(list)
    for r in rows:
        by[r["cfg"]].append(r)
    gates = {"ion": ["rx", "ry", "rz", "rxx"], "nisq": ["rx", "rz", "cz"]}
    out = {}
    for cfg, rs in by.items():
        rs.sort(key=lambda r: r["seed"])
        g = gates["ion" if "ion" in cfg else "nisq"]
        meth = ["original", "qiskit_L1", "qiskit_L2", "qiskit_L3", "ours"]
        lines = []
        for m in meth:
            first = [rs[0][m].get(x, 0) for x in g]
            arr = np.array([[r[m].get(x, 0) for x in g] for r in rs])
            tot = arr.sum(1)
            lines.append(
                m.replace("_", " ")
                + " & "
                + " & ".join(map(str, first))
                + f" & {sum(first)} & "
                + " & ".join(f"{a:.0f}$\\pm${b:.0f}" for a, b in zip(arr.mean(0), arr.std(0)))
                + f" & {tot.mean():.1f}$\\pm${tot.std():.1f}\\\\"
            )
        ok = all(r["ours_ok"] for r in rs) and all(r.get("qiskit_L3_ok", True) for r in rs)
        out[cfg] = dict(
            n=len(rs),
            ok=ok,
            ours_s=float(np.mean([r["ours_s"] for r in rs])),
            q3_s=float(np.mean([r["qiskit_L3_s"] for r in rs])),
            totals={m: float(np.mean([sum(r[m].values()) for r in rs])) for m in meth},
            twoq={m: float(np.mean([r[m].get(g[-1], 0) for r in rs])) for m in meth},
        )
        open(TEX / f"quality_{cfg}.tex", "w").write("\n".join(lines))
    json.dump(out, open(RESULTS / "quality_summary.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


def hardware():
    rows = load("hardware_sim.json")
    if rows is None:
        return
    lines = []
    summ = {}
    for name in ("Brisbane", "Kyiv"):
        for lvl in (0, 1, 3):
            rs = [r for r in rows if r["backend"] == name and r["level"] == lvl]
            f = lambda k: np.array([r[k] for r in rs], float)
            dl, ds = f("long_tvd"), f("short_tvd")
            diff = dl - ds
            lines.append(
                f"{name} & {lvl} & {f('len_long').mean():.0f} / {f('len_short').mean():.1f} & "
                f"{f('long_ops').mean():.0f} / {f('short_ops').mean():.0f} & "
                f"{f('long_2q').mean():.1f} / {f('short_2q').mean():.1f} & "
                f"{dl.mean():.3f}$\\pm${dl.std(ddof=1) / np.sqrt(len(dl)):.3f} & "
                f"{ds.mean():.3f}$\\pm${ds.std(ddof=1) / np.sqrt(len(ds)):.3f} & "
                f"{(diff > 0).sum()}/{len(diff)}\\\\"
            )
            summ[f"{name}_L{lvl}"] = dict(
                long=float(dl.mean()),
                short=float(ds.mean()),
                long_ops=float(f("long_ops").mean()),
                short_ops=float(f("short_ops").mean()),
                long_2q=float(f("long_2q").mean()),
                short_2q=float(f("short_2q").mean()),
                wins=int((diff > 0).sum()),
            )
    open(TEX / "hardware.tex", "w").write("\n".join(lines))
    json.dump(summ, open(RESULTS / "hardware_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


def report_numbers():
    """manuscripts/reproduction/numbers.tex"""
    from pathlib import Path

    rep = Path(__file__).resolve().parents[1] / "manuscripts" / "reproduction"
    M = {}
    t1 = load("table1.json") or []
    big = [r for r in t1 if r.get("qubits") == 3 and "build_seconds" in r]
    M["BuildSecondsBig"] = f"{big[0]['build_seconds']:.0f}" if big else "??"
    demo = load("demo_rerun.json")
    if demo:
        f = lambda v: f"{np.mean(v):.0f} (range {min(v)}--{max(v)})"
        M["DemoDfour"] = f(demo["ours_ion_d4"])
        M["DemoDfive"] = f(demo["ours_ion_pm_d5"])
    else:
        M["DemoDfour"] = M["DemoDfive"] = "??"
    rf = load("rf_eval.json")
    M["RFRecallHalf"] = f"{100 * rf['0.5']['recall']:.0f}\\%" if rf else "??"
    M["RFRecallTwo"] = f"{100 * rf['0.2']['recall']:.0f}\\%" if rf else "??"
    qs = load("quality_summary.json") or {}
    t3 = qs.get("T3_ion_3q")
    M["OursTthree"] = f"{t3['totals']['ours']:.1f}" if t3 else "??"
    M["NTthree"] = str(t3["n"]) if t3 else "??"
    for k in ("QualityFindings", "NJPFindings", "HardwareFindings"):
        p = rep / f"{k}.tex"
        M[k] = p.read_text() if p.exists() else ""
    with open(rep / "numbers.tex", "w") as fh:
        for k, v in M.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print({k: v for k, v in M.items() if "Findings" not in k})


if __name__ == "__main__":
    table2()
    quality()
    hardware()
    report_numbers()
