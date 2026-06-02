"""
generate_hybrid_blackbox_figure.py
====================================

Regenerate the D-Wave Leap Hybrid Solver "black box" schematic
(Figure~\\ref{fig:hybrid_blackbox} in main.tex) with a clean and
consistent annotation of the three timing fields the SDK exposes.

The previous version of the figure left ``charge_time`` as an orphan
text box below the solver, with no visual referent.  This rewrite
makes ``charge_time`` a span arrow that mirrors ``run_time`` but sits
*below* the Hybrid Solver block, making the symmetry of the two
service-side wall-clock fields explicit.  ``qpu_access_time`` remains
a pointer to the QPU sub-solver block.

Run:
    python scripts/generate_hybrid_blackbox_figure.py

Outputs:
    manuscript/figures/hybrid_solver_black_box_paper.pdf
    manuscript/figures/hybrid_solver_black_box_paper.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "manuscript" / "figures"
OUT_PDF = OUT_DIR / "hybrid_solver_black_box_paper.pdf"
OUT_PNG = OUT_DIR / "hybrid_solver_black_box_paper.png"


def _round_box(ax, x, y, w, h, **kwargs):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.5",
        **kwargs,
    )
    ax.add_patch(box)


def _span_arrow(ax, x0, x1, y, color, label, label_offset=1.5,
                label_pos="above"):
    arrow = FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle="<->",
        mutation_scale=14,
        linewidth=1.6,
        color=color,
    )
    ax.add_patch(arrow)
    ya = y + label_offset if label_pos == "above" else y - label_offset
    va = "bottom" if label_pos == "above" else "top"
    ax.text(
        (x0 + x1) / 2, ya, label,
        ha="center", va=va,
        fontsize=11, color=color, family="monospace",
    )


def main() -> int:
    fig, ax = plt.subplots(figsize=(13.0, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # ----- Problem box (left) -----
    _round_box(ax, 2, 14, 18, 38,
               linewidth=2, facecolor="white", edgecolor="#64748b")
    ax.text(11, 49, "Problem", ha="center", fontsize=13, weight="bold")

    # BQM sub-box
    _round_box(ax, 4, 32, 14, 11,
               linewidth=2, facecolor="#fff5f5", edgecolor="#c0392b")
    ax.text(11, 39.5, "BQM:", ha="center", fontsize=11,
            weight="bold", color="#c0392b")
    ax.text(11, 35.0, "penalty-encoded $Q$ matrix", ha="center",
            fontsize=9.5, color="#c0392b")

    # CQM sub-box
    _round_box(ax, 4, 17, 14, 11,
               linewidth=2, facecolor="#f0fdf4", edgecolor="#2e8b57")
    ax.text(11, 24.5, "CQM:", ha="center", fontsize=11,
            weight="bold", color="#2e8b57")
    ax.text(11, 20.0, "objective + constraints", ha="center",
            fontsize=9.5, color="#2e8b57")

    # ----- D-Wave Leap Hybrid Solver block (center) -----
    _round_box(ax, 27, 11, 47, 38,
               linewidth=2, facecolor="#f5f5fa", edgecolor="#475569")
    ax.text(50.5, 46, "D-Wave Leap Hybrid Solver",
            ha="center", fontsize=13, weight="bold")

    # Classical decomposer
    _round_box(ax, 30, 28, 14, 10,
               linewidth=2, facecolor="#e2e8f0", edgecolor="#475569")
    ax.text(37, 34.5, "Classical", ha="center", fontsize=11, weight="bold")
    ax.text(37, 30.5, "decomposer", ha="center", fontsize=11)

    # QPU subproblem solver
    _round_box(ax, 48, 28, 14, 10,
               linewidth=2, facecolor="#dbeafe", edgecolor="#2563eb")
    ax.text(55, 34.5, "QPU subproblem", ha="center", fontsize=11,
            weight="bold", color="#2563eb")
    ax.text(55, 30.5, "solver", ha="center", fontsize=11, color="#2563eb")

    # Classical reassembler
    _round_box(ax, 39, 14, 14, 10,
               linewidth=2, facecolor="#e2e8f0", edgecolor="#475569")
    ax.text(46, 20.5, "Classical", ha="center", fontsize=11, weight="bold")
    ax.text(46, 16.5, "reassembler", ha="center", fontsize=11)

    # ----- Best feasible solution (right) -----
    _round_box(ax, 80, 24, 18, 14,
               linewidth=2, facecolor="white", edgecolor="#475569")
    ax.text(89, 33, "Best feasible", ha="center", fontsize=11, weight="bold")
    ax.text(89, 28.5, "solution", ha="center", fontsize=11, weight="bold")

    # ----- Inter-block arrows -----
    def _arrow(xy_to, xytext, color="#475569", lw=1.8):
        ax.annotate("", xy=xy_to, xytext=xytext,
                    arrowprops=dict(arrowstyle="->", lw=lw, color=color))

    _arrow((30, 33), (20, 32))             # Problem -> decomposer
    _arrow((48, 33), (44, 33))             # decomposer -> QPU
    _arrow((50, 24), (54, 28))             # QPU -> reassembler
    _arrow((38, 28), (40, 24))             # reassembler -> decomposer (loop)
    _arrow((80, 30), (53, 19))             # reassembler -> solution

    # ----- Timing-field annotations -----
    # run_time : span above the Hybrid Solver block
    _span_arrow(ax, x0=27, x1=74, y=52.0, color="#1f2937",
                label="run_time", label_offset=1.2, label_pos="above")

    # charge_time : span below the Hybrid Solver block (mirrors run_time
    # so the symmetry between the two service-side wall-clock fields is
    # visually explicit)
    _span_arrow(ax, x0=27, x1=74, y=8.0, color="#d97706",
                label="charge_time", label_offset=1.2, label_pos="below")

    # qpu_access_time : pointer to the QPU sub-solver
    ax.annotate(
        "", xy=(58, 38), xytext=(76, 50),
        arrowprops=dict(arrowstyle="->", lw=1.4, color="#2563eb"),
    )
    ax.text(77, 50.3, "qpu_access_time", ha="left", va="center",
            fontsize=11, color="#2563eb", family="monospace")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", dpi=300)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=300)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
