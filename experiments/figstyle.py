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


"""Shared look of every figure in the paper and the talks.

Figures are drawn at their final printed size (Springer text width W = 5.2 in), so font sizes are true point sizes.
Each method has one colour and one marker, used in every figure (the colour carries identity, the marker is the
secondary encoding required for the lighter colours). Legends sit outside the plotting area, above the axes, so they
never cover data. Circuit diagrams are drawn at a fixed scale with at most `CIRCUIT_FOLD` columns per row, so gate
labels print at 6.5 pt and angles at 5 pt in every panel.
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGS = ROOT / "manuscripts" / "paper" / "figs"
TALK_FIGS = ROOT / "manuscripts" / "talk" / "figs"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

W = 5.2  # text width in inches
INK, MUTED, FAINT, GRID, SHADE = "#1f1f1e", "#6b6a63", "#c9c8c1", "#e7e6e1", "#efeee9"

# entity -> (colour, marker, label). Colours: validated categorical slots 1-7, greys for the baselines.
METHOD = {
    "general": ("#2a78d6", "o", "ours (general optimiser)"),
    "original": ("#eb6834", "^", "original method"),
    "optimum": ("#1baf7a", "D", "proven optimum"),
    "cliff": ("#eda100", "s", "Clifford-specific (CW)"),
    "bqskit": ("#e87ba4", "P", "BQSKit L3"),
    "bq_ours": ("#008300", "X", "ours on BQSKit L3"),
    "tket": ("#6250d6", "v", "TKET"),
    "qiskit": ("#6b6a63", "s", "Qiskit L3"),
    "qiskit_L1": ("#a3a29b", "o", "Qiskit L1"),
    "input": ("#a3a29b", ".", "input"),
}


def col(k):
    return METHOD[k][0]


def mk(k):
    return METHOD[k][1]


def lab(k):
    return METHOD[k][2]


def apply():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.columnspacing": 1.2,
            "legend.borderaxespad": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.6,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )


apply()


def save(fig, name, talk=False):
    """Write <name>.pdf and <name>.png to the paper's figs/ (and copy both to the talk's figs/)."""
    for ext in ("pdf", "png"):
        fig.savefig(PAPER_FIGS / f"{name}.{ext}")
        if talk:
            TALK_FIGS.mkdir(parents=True, exist_ok=True)
            shutil.copy(PAPER_FIGS / f"{name}.{ext}", TALK_FIGS / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter, title=None):
    """Panel letter (bold) and an optional short title, left-aligned above the axes."""
    ax.set_title(f"$\\mathbf{{({letter})}}$ {title}" if title else f"$\\mathbf{{({letter})}}$", loc="left", pad=4)


def legend_above(ax, handles=None, ncol=3, pad=0.02, **kw):
    """Legend in a band directly above the axes (never over the data)."""
    return ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1 + pad),
        ncol=ncol,
        borderaxespad=0,
        handletextpad=0.4,
        **kw,
    )


def fig_legend(fig, handles, ncol, y=1.0):
    """One legend for the whole figure, centred above all panels."""
    return fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, borderaxespad=0, handletextpad=0.4
    )


def handle(k, line=True, **kw):
    c, m, label = METHOD[k]
    kw.setdefault("ls", "-" if line else "none")
    kw.setdefault("marker", m)
    return Line2D([], [], color=c, ms=4, label=kw.pop("label", label), **kw)


def light_grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ circuit diagrams
CIRCUIT_FOLD = 14  # columns per row: 14 columns at scale 0.5 fill the 5.2 in text width
_ONE = "#dbe8f8"  # pulsed single-qubit gates
_VIRT = "#eeeeea"  # virtual RZ (free on hardware)
_TWO = "#fbe1d6"  # two-qubit gates
CIRCUIT_STYLE = {
    "name": "bw",
    "fontsize": 13,
    "subfontsize": 10,
    "linecolor": INK,
    "textcolor": INK,
    "gatetextcolor": INK,
    "displaycolor": {
        "rx": [_ONE, INK],
        "ry": [_ONE, INK],
        "sx": [_ONE, INK],
        "x": [_ONE, INK],
        "h": [_ONE, INK],
        "rz": [_VIRT, INK],
        "rxx": [_TWO, INK],
        "cz": [INK, INK],
        "cx": [INK, INK],
    },
    "displaytext": {"rx": "$R_X$", "ry": "$R_Y$", "rz": "$R_Z$", "rxx": "$R_{XX}$", "sx": "$\\sqrt{X}$"},
}


def circuit(gates, n, name, fold=CIRCUIT_FOLD, talk=False):
    """Draw one circuit as its own figure at print scale (see the module docstring) and save it."""
    from qiskit.visualization import circuit_drawer

    from qcr.verify import to_qiskit

    fig = circuit_drawer(to_qiskit(gates, n), output="mpl", fold=fold, style=CIRCUIT_STYLE, scale=0.5)
    save(fig, name, talk=talk)
