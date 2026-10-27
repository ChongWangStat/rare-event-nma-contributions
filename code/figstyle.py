"""Shared plotting style for the manuscript figures (plotting only; no numbers are computed here).

Each figure is drawn at the width it is printed in the manuscript (fraction of a 6.5 in text width,
matching the \\includegraphics width in main_acs.tex), so the font sizes below are the printed sizes.
Colours are the Okabe-Ito blue / vermilion / bluish green with a Tol purple; every pair stays
distinguishable under protan, deutan and tritan simulation, and categories are also coded by marker shape.
Use the style inside `with plt.rc_context(RC):` so other figures made by the same script are unaffected.
"""
import matplotlib.pyplot as plt  # noqa: F401  (imported so callers can rely on pyplot being initialised)
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import offset_copy

TEXTWIDTH = 6.5  # inches: letter paper with 1 in margins


def width(frac):
    """Figure width in inches for a figure printed at `frac` x \\textwidth."""
    return frac * TEXTWIDTH


BLUE, VERMILION, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#AA3377"
INK, GREY, LIGHTGREY, GRID = "#222222", "#666666", "#BDBDBD", "#E6E6E6"

NET_COLOR = {"COPD": BLUE, "Cystic fibrosis": GREEN, "NMOSD": VERMILION, "SMA": PURPLE}
NET_MARKER = {"COPD": "o", "Cystic fibrosis": "s", "NMOSD": "^", "SMA": "D"}

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.sf": "sans",
    "mathtext.cal": "sans:italic",
    "mathtext.tt": "monospace",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GREY,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.2,
    "legend.columnspacing": 1.4,
    "legend.borderpad": 0.2,
    "legend.labelspacing": 0.35,
    "lines.linewidth": 1.0,
    "lines.markeredgewidth": 0.9,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}

# p-values and other log-scale quantities: plain decimals instead of powers of ten where they fit
plain_log = FuncFormatter(lambda v, pos: f"{v:g}")
thousands = FuncFormatter(lambda v, pos: f"{v:,.0f}")


def margins(fig, left, right, bottom, top, wspace=None):
    """Fix the axes area by margins given in inches (keeps the printed geometry exact)."""
    W, H = fig.get_size_inches()
    kw = {} if wspace is None else {"wspace": wspace}
    fig.subplots_adjust(left=left / W, right=1 - right / W, bottom=bottom / H, top=1 - top / H, **kw)


def legend_rows(ax, rows, x=0.0, gap=3.0, **kw):
    """Stack one single-row legend per entry of `rows` above `ax`, first row on top.

    rows: list of (handles, labels, row_height_pt).  Legends sit outside the plotting area, so they
    never cover data; leave a top margin of about sum(row heights) + gap points (see `margins`)."""
    y = gap
    for handles, labels, height in reversed(rows):
        tr = offset_copy(ax.transAxes, fig=ax.figure, x=0, y=y, units="points")
        leg = ax.legend(handles, labels, ncol=len(handles), loc="lower left", bbox_to_anchor=(x, 1.0),
                        bbox_transform=tr, borderaxespad=0, borderpad=0, **kw)
        ax.add_artist(leg)
        y += height


def mark(**kw):
    """Marker-only legend handle."""
    return Line2D([], [], ls="", **kw)
