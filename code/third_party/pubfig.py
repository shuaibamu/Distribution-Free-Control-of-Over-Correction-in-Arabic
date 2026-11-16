"""pubfig — reusable helpers implementing the figures4papers house style.

Style helpers following the figures4papers house style by Chen Liu
(https://github.com/ChenLiu-1996/figures4papers, CC BY-NC 4.0).

Usage:
    from pubfig import *
    apply_publication_style(FigureStyle(font_size=24, axes_linewidth=3))
    fig, axes = create_subplots(1, 4, figsize=(28, 6))
    ...
    finalize_figure(fig, "figures/result", formats=["png", "pdf"])
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

if matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps", "cairo") and not hasattr(
    __import__("sys"), "ps1"
):
    try:
        matplotlib.use("Agg")
    except Exception:  # pragma: no cover
        pass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

__all__ = [
    "PALETTE", "DEFAULT_COLORS", "BASELINE_SHADES", "FigureStyle",
    "apply_publication_style", "create_subplots", "finalize_figure",
    "make_grouped_bar", "make_method_bars", "annotate_bars", "make_trend",
    "make_heatmap", "make_scatter", "make_radar", "make_sphere_illustration",
    "legend_panel", "tighten_ylim", "is_dark", "outlined_text",
]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PALETTE = {
    "blue_main": "#0F4D92",       # proposed / key method
    "blue_secondary": "#3775BA",  # proposed variants, ablation (vary alpha)
    "blue_dark": "#0C2458",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "yellow_1": "#FFF4C4", "tan": "#D9A577", "lilac": "#D4B5D7",
    "sky": "#93D9F2", "magenta": "#EA84DD",
    "neutral": "#CFCECE", "gray_1": "#767676", "gray_2": "#4D4D4D", "gray_3": "#272727",
    "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
}

DEFAULT_COLORS = [PALETTE[k] for k in ("blue_main", "green_3", "red_strong", "teal", "violet", "neutral")]

# Light-to-strong red ramp used for baselines next to a blue "ours" bar
# (CellSpliceNet style: ours first in deep blue, baselines fade in red).
BASELINE_SHADES = ["#D86561", "#DD7C77", "#E2918C", "#E7A5A0", "#ECB8B4", "#F1CAC7", "#F6DCDA", "#FBEDEC"]

_SUPPORTED = {"pdf", "svg", "eps", "png", "jpg", "jpeg", "tif", "tiff"}


# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16                 # 24 for large bar panels, 15–16 compact
    axes_linewidth: float = 2.5         # 3 big bars, 2 compact
    use_tex: bool = False               # True only if LaTeX is installed
    font_family: tuple[str, ...] = ("Helvetica", "Arial", "Liberation Sans", "DejaVu Sans", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> FigureStyle:
    """Set rcParams for the house style. Call once before creating figures."""
    style = style or FigureStyle()
    use_tex = style.use_tex and shutil.which("latex") is not None
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(style.font_family),
        "font.size": style.font_size,
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": style.axes_linewidth * 0.8,
        "ytick.major.width": style.axes_linewidth * 0.8,
        "legend.frameon": False,
        "text.usetex": use_tex,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return style


def create_subplots(nrows: int = 1, ncols: int = 1, figsize=None, **kwargs):
    """Return (fig, axes) with axes flattened to a 1-D array."""
    if figsize is None:
        figsize = (7 * ncols, 6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, **kwargs)
    return fig, axes.ravel()


def finalize_figure(fig, out_path, formats: Sequence[str] | None = None, dpi: int = 300,
                    close: bool = True, pad: float = 2, tight: bool = True, **kwargs) -> list[Path]:
    """tight_layout(pad) then save to each format. Returns saved paths."""
    out = Path(out_path)
    if formats is None:
        formats = [out.suffix[1:]] if out.suffix[1:].lower() in _SUPPORTED else ["png", "pdf"]
    base = out.with_suffix("") if out.suffix[1:].lower() in _SUPPORTED else out
    base.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            fig.tight_layout(pad=pad)
        except Exception:
            pass
    saved = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        if fmt not in _SUPPORTED:
            raise ValueError(f"Unsupported format '{fmt}'. Use one of {sorted(_SUPPORTED)}")
        p = base.with_suffix("." + fmt)
        fig.savefig(p, dpi=dpi, **kwargs)
        saved.append(p)
    if close:
        plt.close(fig)
    return saved


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def is_dark(color, threshold: float = 0.5) -> bool:
    """True when a color is dark enough to need white text on top of it."""
    r, g, b = to_rgb(color)
    return (0.299 * r + 0.587 * g + 0.114 * b) < threshold


def outlined_text(ax, x, y, s, fontsize=20, color="#FFD700", stroke=4, **kw):
    """Text with a black stroke so it stays legible on any fill (Brainteaser style)."""
    t = ax.text(x, y, s, fontsize=fontsize, color=color, ha=kw.pop("ha", "center"),
                va=kw.pop("va", "center"), **kw)
    t.set_path_effects([patheffects.Stroke(linewidth=stroke, foreground="black"), patheffects.Normal()])
    return t


def tighten_ylim(ax, values, errors=None, margin: float = 0.5, floor=None, ceil=None):
    """Dynamic y-limits: [min - margin*std, max + margin*std] (house rule)."""
    v = np.asarray(values, float).ravel()
    e = np.zeros_like(v) if errors is None else np.asarray(errors, float).ravel()
    lo, hi = (v - e).min(), (v + e).max()
    spread = max(v.std(), (hi - lo) * 0.1, 1e-12)
    lo, hi = lo - margin * spread, hi + margin * spread
    if floor is not None:
        lo = max(lo, floor)
    if ceil is not None:
        hi = min(hi, ceil)
    ax.set_ylim(lo, hi)
    return lo, hi


def legend_panel(ax, handles=None, labels=None, source_ax=None, loc="center left", **kw):
    """Turn `ax` into a legend-only panel (dedicated legend subplot pattern)."""
    if handles is None:
        handles, labels = (source_ax or ax).get_legend_handles_labels()
    ax.set_axis_off()
    return ax.legend(handles, labels, loc=loc, frameon=False, **kw)


def _arr1d(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    return a


# --------------------------------------------------------------------------- #
# Bars
# --------------------------------------------------------------------------- #
def make_grouped_bar(ax, categories, series, labels, ylabel="Value", colors=None,
                     annotate=False, errors=None, hatches=None, edgecolor="black",
                     linewidth=1.5, width=0.8, fmt="{:.2f}", fontsize=None):
    """Grouped bars: one group per category, one bar per series."""
    cats = list(categories)
    S = np.asarray(series, dtype=float)
    if S.ndim == 1:
        S = S[None]
    if S.shape[1] != len(cats):
        raise ValueError(f"len(categories)={len(cats)} must equal series columns={S.shape[1]}")
    if len(labels) != S.shape[0]:
        raise ValueError("labels must have one entry per series")
    colors = colors or DEFAULT_COLORS
    n = S.shape[0]
    w = width / n
    x = np.arange(len(cats))
    last = None
    for i in range(n):
        last = ax.bar(x - width / 2 + w * (i + 0.5), S[i], w,
                      yerr=None if errors is None else np.asarray(errors)[i],
                      capsize=4 if errors is not None else 0,
                      color=colors[i % len(colors)], edgecolor=edgecolor, linewidth=linewidth,
                      hatch=None if hatches is None else hatches[i % len(hatches)], label=labels[i])
        if annotate:
            annotate_bars(ax, last, fmt=fmt, fontsize=fontsize or plt.rcParams["font.size"] * 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel(ylabel)
    return last


def make_method_bars(ax, methods, values, errors=None, colors=None, ylabel=None,
                     annotate=True, fmt="{:.2f}", hide_xticks=True, legend=True,
                     legend_ncol=2, alphas=None, hatches=None, edgecolor="none"):
    """One bar per method on a single metric (ImmunoStruct / CellSpliceNet style).

    Methods are named in the legend, x ticks hidden; put "ours" last (or first) in blue.
    """
    v = _arr1d(values, "values")
    if len(methods) != len(v):
        raise ValueError("methods and values length mismatch")
    if colors is None:
        colors = [PALETTE["blue_main"]] + BASELINE_SHADES[: len(v) - 1]
    x = np.arange(len(v))
    bars = []
    for i in range(len(v)):
        b = ax.bar(x[i], v[i], 0.8, yerr=None if errors is None else np.asarray(errors)[i],
                   capsize=8, error_kw=dict(elinewidth=2, capthick=2),
                   color=colors[i % len(colors)], alpha=1 if alphas is None else alphas[i],
                   edgecolor=edgecolor, hatch=None if hatches is None else hatches[i],
                   label=methods[i])
        bars.append(b)
    if annotate:
        top = v + (0 if errors is None else np.asarray(errors))
        for i in range(len(v)):
            ax.text(x[i], top[i], fmt.format(v[i]), ha="center", va="bottom",
                    fontsize=plt.rcParams["font.size"] * 0.75)
    if hide_xticks:
        ax.set_xticks([])
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc="upper right", ncol=legend_ncol, frameon=False,
                  fontsize=plt.rcParams["font.size"] * 0.8)
    return bars


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=10, padding=3, color="black"):
    """Print each bar's value just above it."""
    for rect in bars:
        h = rect.get_height()
        ax.annotate(fmt.format(h), (rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, padding if h >= 0 else -padding), textcoords="offset points",
                    ha="center", va="bottom" if h >= 0 else "top", fontsize=fontsize, color=color)


# --------------------------------------------------------------------------- #
# Lines / heatmap / scatter
# --------------------------------------------------------------------------- #
def make_trend(ax, x, y_series, labels, colors=None, ylabel=None, xlabel=None,
               show_shadow=True, y_err=None, markers=True, linewidth=3, baseline=None,
               baseline_label="Baseline"):
    """2–4 lines with optional uncertainty bands and a dashed gray reference."""
    x = _arr1d(x, "x")
    colors = colors or DEFAULT_COLORS
    for i, (y, lab) in enumerate(zip(y_series, labels)):
        y = _arr1d(y, f"y_series[{i}]")
        if len(y) != len(x):
            raise ValueError(f"y_series[{i}] length {len(y)} != len(x) {len(x)}")
        c = colors[i % len(colors)]
        ax.plot(x, y, color=c, lw=linewidth, label=lab,
                marker="o" if markers else None, markersize=8 if markers else 0)
        if show_shadow:
            band = (np.asarray(y_err[i]) if y_err is not None else
                    np.full_like(y, 0.03 * (np.ptp(y) or abs(y).mean() or 1)))
            ax.fill_between(x, y - band, y + band, color=c, alpha=0.18, lw=0)
    if baseline is not None:
        ax.axhline(baseline, color="#A0A0A0", ls="--", lw=2.5, label=baseline_label)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    return ax


def make_heatmap(ax, matrix, x_labels=None, y_labels=None, cmap="Reds", cbar_label=None,
                 annotate=False, fmt="{:.2f}", vmin=None, vmax=None, fontsize=None):
    """Heatmap with optional colorbar label and luminance-aware cell text."""
    M = np.asarray(matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("matrix must be 2-D")
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    if x_labels is not None:
        ax.set_xticks(range(M.shape[1]))
        ax.set_xticklabels(x_labels, rotation=30, ha="right")
    if y_labels is not None:
        ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels(y_labels)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    if cbar_label:
        cb.set_label(cbar_label)
    if annotate:
        norm = im.norm
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                rgba = im.cmap(norm(M[i, j]))
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center",
                        fontsize=fontsize or plt.rcParams["font.size"] * 0.6,
                        color="white" if is_dark(rgba[:3]) else "black")
    return im


def make_scatter(ax, x, y, label=None, color=None, size=50, alpha=0.7, edgecolor="white"):
    x, y = _arr1d(x, "x"), _arr1d(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must be same length")
    return ax.scatter(x, y, s=size, c=color or PALETTE["blue_main"], alpha=alpha,
                      edgecolors=edgecolor, linewidths=1, label=label)


# --------------------------------------------------------------------------- #
# Radar
# --------------------------------------------------------------------------- #
def make_radar(fig, axes_labels, series, labels, colors=None, rect=111, rmin=None, rmax=None,
               fill_alpha=0.08, linewidth=2.5):
    """Radar/polar comparison (VIGIL style). Returns the polar Axes."""
    S = np.asarray(series, dtype=float)
    k = len(axes_labels)
    if S.shape[1] != k:
        raise ValueError("each series needs one value per axis label")
    colors = colors or [PALETTE["red_2"], PALETTE["green_3"], PALETTE["blue_main"]]
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    ang_c = np.r_[ang, ang[:1]]
    ax = fig.add_subplot(rect, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for i, (row, lab) in enumerate(zip(S, labels)):
        c = colors[i % len(colors)]
        r = np.r_[row, row[:1]]
        ax.plot(ang_c, r, color=c, lw=linewidth, label=lab)
        ax.fill(ang_c, r, color=c, alpha=fill_alpha)
    ax.set_xticks(ang)
    ax.set_xticklabels(axes_labels, fontsize=plt.rcParams["font.size"] * 0.6)
    if rmin is not None or rmax is not None:
        ax.set_ylim(rmin, rmax)
    ax.spines["polar"].set_linewidth(1.5)
    ax.grid(color="#BBBBBB", lw=0.8)
    return ax


# --------------------------------------------------------------------------- #
# Conceptual sphere (Dispersion style)
# --------------------------------------------------------------------------- #
def make_sphere_illustration(ax, light_dir=(-0.5, 0.5, 0.8), resolution=256, alpha=0.6,
                             ambient=0.35, center=(0, 0), radius=1.0):
    """Lambert-shaded disk that reads as a 3-D sphere; axes turned off."""
    L = np.asarray(light_dir, float)
    L /= np.linalg.norm(L)
    t = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(t, t)
    R2 = X ** 2 + Y ** 2
    Z = np.sqrt(np.clip(1 - R2, 0, None))
    shade = np.clip(X * L[0] + Y * L[1] + Z * L[2], 0, 1)
    img = ambient + (1 - ambient) * shade
    img = np.where(R2 <= 1, img, np.nan)
    cx, cy = center
    ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1, alpha=alpha,
              extent=[cx - radius, cx + radius, cy - radius, cy + radius])
    ax.set_xlim(cx - 1.1 * radius, cx + 1.1 * radius)
    ax.set_ylim(cy - 1.1 * radius, cy + 1.1 * radius)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return ax


def proxy_handles(colors, labels, kind="patch", hatches=None):
    """Legend proxies: colored patches, hatch-only patches, or lines."""
    out = []
    for i, (c, l) in enumerate(zip(colors, labels)):
        if kind == "line":
            out.append(Line2D([], [], color=c, lw=3, marker="o", label=l))
        else:
            out.append(Patch(facecolor=c, edgecolor="black",
                             hatch=None if hatches is None else hatches[i], label=l))
    return out


__all__.append("proxy_handles")
