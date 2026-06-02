#!/usr/bin/env python3
"""Generate gap-to-Gurobi figure (Fig 4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "manuscript" / "figures"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def main():
    classical = load_jsonl(ROOT / "results" / "classical" / "d1d2_classical_baselines.jsonl")
    hybrid = load_jsonl(ROOT / "results" / "hybrid" / "a2a3_synthetic_hybrid.jsonl")

    # Build Gurobi optimal lookup (only proven optimal)
    gurobi_opt = {}
    for r in classical:
        if r["solver"] == "gurobi_miqp" and r.get("status") == "optimal":
            gurobi_opt[(r["family"], r["N"], r["seed"])] = r["objective_value"]

    a2 = [r for r in hybrid if r["experiment"] == "A2" and "objective_value" in r]
    a3 = [r for r in hybrid if r["experiment"] == "A3" and "objective_value" in r]
    sa = [r for r in classical if r["solver"] == "neal_sa" and "objective_value" in r]

    N_vals = [10, 20, 30, 50, 80, 120]  # where Gurobi has optimal

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, rows, color, marker in [
        ("Hybrid CQM", a3, "#4CAF50", "o"),
        ("Hybrid BQM", a2, "#F44336", "s"),
        ("Neal SA", sa, "#FF9800", "^"),
    ]:
        means, stds = [], []
        for N in N_vals:
            gaps = []
            for r in rows:
                if r["N"] != N:
                    continue
                key = (r["family"], r["N"], r["seed"])
                if key in gurobi_opt:
                    g = gurobi_opt[key]
                    if abs(g) > 1e-10:
                        gaps.append((r["objective_value"] - g) / abs(g))
            if gaps:
                means.append(np.mean(gaps))
                stds.append(np.std(gaps))
            else:
                means.append(float("nan"))
                stds.append(0)

        ax.errorbar(N_vals, means, yerr=stds, marker=marker, color=color,
                    label=label, capsize=3, linewidth=1.5)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Problem size $N$")
    ax.set_ylabel("Relative gap to Gurobi optimal")
    ax.set_title("Gap to proven optimum by solver")
    ax.legend()
    ax.set_ylim(-0.05, None)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_gap_to_gurobi.pdf")
    fig.savefig(FIG_DIR / "fig4_gap_to_gurobi.png")
    plt.close(fig)
    print("Fig 4 saved.")


if __name__ == "__main__":
    main()
