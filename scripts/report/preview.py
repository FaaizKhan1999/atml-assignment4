#!/usr/bin/env python3
"""Build report/figs/preview_all.pdf (every figure at its actual size, every table fragment at 8 pt in a NeurIPS-like
article: Times, 10 pt, 5.5 in text width; one item per page with its filename) and report/figs/MANIFEST.md from the
per-script manifests. Table widths are measured and written to report/figs/table_widths.txt."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "report/figs"


def main():
    items = sorted(p for p in FIGS.glob("task*/*") if p.suffix in (".pdf", ".tex"))
    body = []
    for p in items:
        rel = p.relative_to(FIGS).as_posix()
        name = rel.replace("_", "\\_")
        body.append(f"\\section*{{\\texttt{{{name}}}}}")
        if p.suffix == ".pdf":
            body.append(f"\\noindent\\includegraphics{{{rel}}}\\par")
        else:
            body.append(f"{{\\footnotesize\\setbox0=\\hbox{{\\input{{{rel}}}}}"
                        f"\\typeout{{TABWIDTH {rel} \\the\\wd0}}\\noindent\\box0}}\\par")
        body.append("\\clearpage")
    doc = "\n".join([
        "\\documentclass[10pt]{article}",
        "\\usepackage[paperwidth=8.5in,paperheight=11in,textwidth=5.5in,left=1.5in,top=1in,textheight=9in]{geometry}",
        "\\usepackage{times}\\usepackage{booktabs}\\usepackage{graphicx}\\usepackage{amssymb}\\usepackage{amsmath}",
        "\\pagestyle{plain}\\begin{document}", *body, "\\end{document}"])
    tex = FIGS / "preview_all.tex"
    tex.write_text(doc)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "preview_all.tex"], cwd=FIGS,
                       capture_output=True, text=True)
    log = (FIGS / "preview_all.log").read_text(errors="ignore")
    widths = re.findall(r"TABWIDTH (\S+) ([\d.]+)pt", log)
    lines = [f"{f}\t{float(w) / 72.27:.2f} in" + ("\tWIDER THAN 5.5 in" if float(w) / 72.27 > 5.5 else "")
             for f, w in widths]
    (FIGS / "table_widths.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if r.returncode != 0:
        print(log[-3000:])
        sys.exit("pdflatex failed")
    for ext in ("aux", "log"):
        (FIGS / f"preview_all.{ext}").unlink(missing_ok=True)
    man = []
    for m in sorted(FIGS.glob(".manifest_*.json")):
        man += json.load(open(m))
    md = ["# Report figure/table manifest", "", "| output | script | source files |", "|---|---|---|"]
    md += [f"| `{o}` | `{s}` | {', '.join(f'`{x}`' for x in src)} |" for o, s, src in sorted(man)]
    (FIGS / "MANIFEST.md").write_text("\n".join(md) + "\n")
    print("preview + manifest written:", len(items), "items")


if __name__ == "__main__":
    main()
