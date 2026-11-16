"""Architecture and methodology diagrams in the figures4papers house style
(PALETTE colours, sans-serif, frameless, tight bounding boxes)."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party"))
from pubfig import PALETTE, apply_publication_style, FigureStyle

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
apply_publication_style(FigureStyle(font_size=9, axes_linewidth=1))
plt.rcParams["pdf.fonttype"] = 42


def box(ax, x, y, w, h, text, fc, ec=None, fs=7.5, tc="black", bold=False, r=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}", fc=fc,
                       ec=ec or PALETTE["blue_main"], lw=1.0)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.15)
    return (x, y, w, h)


def arrow(ax, p, q, color="#333333", ls="-", lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


def architecture():
    fig = plt.figure(figsize=(7.1, 2.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    B, G, R, N = PALETTE["blue_main"], PALETTE["green_2"], PALETTE["red_1"], "#EEF3FA"
    # input
    box(ax, 0.005, 0.40, 0.095, 0.22, "Noisy\nsentence\n$x_1..x_n$", "white", fs=7.5)
    box(ax, 0.125, 0.36, 0.11, 0.30, "Word emb.\n+\nchar-CNN\n$\\mathbf{e}_i$", N, fs=7.5)
    arrow(ax, (0.10, 0.51), (0.125, 0.51))
    # branches
    box(ax, 0.27, 0.62, 0.15, 0.26, "Transformer\nencoder ($L$=4)\nMLM-pretrained\n$\\mathbf{h}^{T}_i$", N, fs=7.5)
    box(ax, 0.27, 0.12, 0.15, 0.26, "BiGRU\n(orthographic,\nlocal order)\n$\\mathbf{h}^{R}_i$", N, fs=7.5)
    arrow(ax, (0.235, 0.56), (0.27, 0.72)); arrow(ax, (0.235, 0.46), (0.27, 0.27))
    # gate
    box(ax, 0.455, 0.30, 0.13, 0.42, "Gated fusion\n\n$\\mathbf{g}_i=\\sigma(W_g[\\mathbf{h}^T_i;\\mathbf{h}^R_i])$\n\n$\\mathbf{h}_i=\\mathbf{g}_i\\odot\\mathbf{h}^T_i$\n\n$+\\,(\\mathbf{1}-\\mathbf{g}_i)\\odot\\mathbf{h}^R_i$",
        PALETTE["highlight"] + "55", fs=6.6)
    arrow(ax, (0.42, 0.75), (0.455, 0.64)); arrow(ax, (0.42, 0.25), (0.455, 0.38))
    # heads
    hx = 0.62
    box(ax, hx, 0.70, 0.135, 0.17, "Edit-tag head\n$p(t_i\\mid x)$, |T|=913", "white", fs=7)
    box(ax, hx, 0.43, 0.135, 0.17, "Detection head\n$p(e_i\\mid x)$", "white", fs=7)
    box(ax, hx, 0.16, 0.135, 0.17, "Rhetoric head\n$r_i=p(\\mathrm{rhet}_i\\mid x)$", R, ec=PALETTE["red_strong"], fs=7)
    for y in (0.785, 0.515, 0.245):
        arrow(ax, (0.585, 0.51), (hx, y))
    # decision
    box(ax, 0.79, 0.58, 0.2, 0.29, "Calibrated decision\n\n$\\hat p_i=\\mathrm{softmax}(z_i/T)_{\\hat t_i}$\n\nedit iff $\\hat p_i>\\hat\\tau+\\beta\\, r_i$", "#E8F1FB", fs=6.8)
    box(ax, 0.79, 0.16, 0.2, 0.31, "Conformal risk control\n\n$\\hat\\tau=\\inf\\{\\tau:\\ \\tilde R_n(\\tau)\\leq\\alpha\\}$\n\n$\\tilde R_n=(n\\hat R_n+1)/(n+1)$", G + "88", ec=PALETTE["green_3"], fs=6.8)
    arrow(ax, (hx + 0.135, 0.785), (0.79, 0.74))
    arrow(ax, (hx + 0.135, 0.245), (0.79, 0.62), color=PALETTE["red_strong"])
    arrow(ax, (0.89, 0.47), (0.89, 0.58), color=PALETTE["green_3"])
    ax.text(0.68, 0.37, "multi-task loss", fontsize=6, color="#555555", ha="center")
    ax.text(0.89, 0.06, "calibration split, exchangeable", fontsize=6, color="#555555", ha="center")
    ax.text(0.345, 0.93, "pretrained context", fontsize=6.3, color=B, ha="center")
    ax.text(0.345, 0.05, "recurrent local features", fontsize=6.3, color=B, ha="center")
    fig.savefig(os.path.join(OUT, "fig_arch.pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(os.path.join(OUT, "fig_arch.png"), bbox_inches="tight", pad_inches=0.01, dpi=300)


def methodology():
    fig = plt.figure(figsize=(7.1, 2.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    N = "#EEF3FA"
    # data lane
    ax.text(0.005, 0.95, "(a) Data and supervision", fontsize=7.5, fontweight="bold", va="top")
    box(ax, 0.005, 0.62, 0.13, 0.22, "OpenITI classical\nprose (4 works)", N, fs=7)
    box(ax, 0.165, 0.62, 0.14, 0.22, "Seeded error\ninjection (6 families)\n+ edit-tag derivation", N, fs=6.8)
    box(ax, 0.005, 0.30, 0.13, 0.22, "Tafsir: Kashshaf,\nTahrir wa-Tanwir", N, fs=7)
    box(ax, 0.165, 0.30, 0.14, 0.22, "Distant labels:\nverse tokens near\nisti'ara/majaz terms", PALETTE["red_1"], ec=PALETTE["red_strong"], fs=6.8)
    box(ax, 0.005, 0.02, 0.13, 0.20, "Tanzil Qur'an\n(error-free text)", N, fs=7)
    box(ax, 0.165, 0.02, 0.14, 0.20, "Faithfulness test:\nany edit = over-edit", "white", fs=6.8)
    for y in (0.73, 0.41, 0.12):
        arrow(ax, (0.135, y), (0.165, y))
    box(ax, 0.335, 0.30, 0.12, 0.54, "Train RACE\nand baselines\n(5 seeds)\n\nTemperature\nscaling (dev)\n\nCRC (cal)", "#E8F1FB", fs=6.8)
    arrow(ax, (0.305, 0.73), (0.335, 0.70)); arrow(ax, (0.305, 0.41), (0.335, 0.45))
    arrow(ax, (0.305, 0.12), (0.395, 0.30), ls="--")
    ax.text(0.37, 0.15, "evaluate", fontsize=6, color="#555555", rotation=38)
    ax.plot([0.475, 0.475], [0.02, 0.95], color="#BBBBBB", lw=0.8, ls="--")
    # serving lane
    ax.text(0.49, 0.95, "(b) Uncertainty-routed serving on a CPU cluster", fontsize=7.5, fontweight="bold", va="top")
    box(ax, 0.49, 0.40, 0.09, 0.24, "Requests\nPoisson $\\nu$", "white", fs=7)
    box(ax, 0.61, 0.40, 0.11, 0.24, "Light pool\n$c_L$ cores\n1$\\times$RACE", N, fs=7)
    box(ax, 0.75, 0.40, 0.1, 0.24, "Router\n$u=\\mathrm{max}_i\\,H(\\hat p_i)$\nroute if $u>\\theta$", PALETTE["highlight"] + "55", fs=6.8)
    box(ax, 0.88, 0.62, 0.115, 0.24, "Heavy pool\n$c_H$ cores\n4-model ensemble", PALETTE["green_2"] + "88", ec=PALETTE["green_3"], fs=6.8)
    box(ax, 0.88, 0.14, 0.115, 0.2, "Response\n(edits + conf.)", "white", fs=7)
    arrow(ax, (0.58, 0.52), (0.61, 0.52)); arrow(ax, (0.72, 0.52), (0.75, 0.52))
    arrow(ax, (0.85, 0.57), (0.88, 0.72)); ax.text(0.842, 0.70, "yes", fontsize=6)
    arrow(ax, (0.85, 0.47), (0.88, 0.25)); ax.text(0.842, 0.30, "no", fontsize=6)
    arrow(ax, (0.9375, 0.62), (0.9375, 0.34))
    ax.text(0.665, 0.13, "measured service times $\\rightarrow$ discrete-event cluster model\n(M/G/c per pool, 0.25 ms hop, p99 SLO)", fontsize=6.3, ha="center", color="#555555")
    fig.savefig(os.path.join(OUT, "fig_method.pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(os.path.join(OUT, "fig_method.png"), bbox_inches="tight", pad_inches=0.01, dpi=300)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    architecture(); methodology()
