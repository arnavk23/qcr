"""figures_tables.tex: every figure and table of paper.tex, grouped by section, one float per page.

Build after the make_*_assets.py scripts and make_paper_figs.py: pdflatex, bibtex, pdflatex x2.
"""

import re
from pathlib import Path

EXT = Path(__file__).resolve().parents[1] / "manuscripts" / "paper"
src = (EXT / "paper.tex").read_text()
pre = src[: src.index(r"\begin{document}")].replace(
    r"\raggedbottom", r"\usepackage{xr}\externaldocument{paper}"
)
body = src[src.index(r"\begin{document}") :]
appendix_at = body.find(r"\begin{appendices}")

out, sec = [], None
pat = re.compile(r"\\section\*?\{([^}]*)\}|\\begin\{(figure|table)\*?\}.*?\\end\{\2\*?\}", re.S)
for m in pat.finditer(body):
    if m.group(1) is not None:
        sec = ("Appendix: " if 0 <= appendix_at < m.start() else "") + m.group(1)
        continue
    env = re.sub(r"\\begin\{(figure|table)(\*?)\}\[[^\]]*\]", r"\\begin{\1\2}[p]", m.group(0))
    if sec:
        out.append(f"\\addcontentsline{{toc}}{{section}}{{{sec}}}\\markboth{{{sec}}}{{}}")
        out.append(f"\\begin{{center}}\\large\\bfseries {sec}\\end{{center}}")
        sec = None
    out.append(
        env.replace(r"\begin{figure}[p]", r"\begin{figure}[h!]").replace(r"\begin{table}[p]", r"\begin{table}[h!]")
    )
    out.append(r"\clearpage")

doc = (
    pre
    + r"""\begin{document}
\title{Figures and tables}
\author{\fnm{Arnav} \sur{Kapoor}}
\affil{\orgname{IISER Bhopal}}
\abstract{All figures and tables of ``Exact, hardware-aware local rewriting of quantum circuits, with certified ground
truth'', regenerated from the results in this repository. Values shown as ?? come from runs that have not finished.}
\maketitle
"""
    + "\n\n".join(out)
    + "\n\\bibliography{refs}\n\\end{document}\n"
)
(EXT / "figures_tables.tex").write_text(doc)
print("wrote figures_tables.tex,", sum(o.startswith(r"\begin{") for o in out), "floats")
