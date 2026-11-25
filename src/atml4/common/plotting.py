"""NeurIPS 2026 figure style: 10 pt Times body, 5.5 in text width.

Use apply_style() once, then savefig(fig, "name") to write a vector PDF into figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

FULL_W = 5.5   # inches, text width
HALF_W = 2.65  # inches, half-width panel
SERIF_CHAIN = ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "STIXGeneral"]
LABEL_PT = 8
TICK_PT = 7
LINE_W = 0.7
# Okabe-Ito colour-blind-safe palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def resolved_font() -> str:
    """Name of the first font in SERIF_CHAIN actually installed (what matplotlib will use)."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in SERIF_CHAIN:
        if n in names:
            return n
    return font_manager.FontProperties(family="serif").get_name()


def apply_style() -> str:
    from cycler import cycler

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF_CHAIN,
        "mathtext.fontset": "stix",
        "font.size": LABEL_PT,
        "axes.labelsize": LABEL_PT,
        "axes.titlesize": LABEL_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": TICK_PT,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": LINE_W + 0.1,
        "axes.linewidth": LINE_W,
        "xtick.major.width": LINE_W,
        "ytick.major.width": LINE_W,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "grid.linewidth": 0.6,
        "patch.linewidth": 0.6,
        "axes.prop_cycle": cycler(color=PALETTE),
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })
    return resolved_font()


def figure(width: float = FULL_W, height: float | None = None, **kw):
    return plt.subplots(figsize=(width, height or width * 0.62), **kw)


def savefig(fig, name: str, outdir: str | Path | None = None) -> Path:
    from .meta import repo_root

    outdir = Path(outdir) if outdir else repo_root() / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / (name if name.endswith(".pdf") else f"{name}.pdf")
    fig.savefig(path, format="pdf")
    return path
