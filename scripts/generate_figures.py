#!/usr/bin/env python3
"""Generate the 3 core figures for Paper 2A.

Fig 1: Chain-break fraction + embedding overhead vs N (A1 data)
Fig 2: CQM vs BQM vs Gurobi vs SA objective gap vs N (A2/A3 + D1/D2 data)
Fig 3: Budget response curves — BQM vs CQM (B2 data)

Output: manuscript/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "paper1" / "src"))

FIG_DIR = ROOT / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def fig1_chain_break_embedding(a1_rows):
    """Chain-break fraction and embedding overhead vs N, split by topology."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    N_vals = sorted(set(r["N"] for r in a1_rows))

    for topo, marker, color in [("pegasus", "o", "#2196F3"), ("zephyr", "s", "#FF9800")]:
        cbfs_mean, cbfs_std = [], []
        overheads = []
        for N in N_vals:
            subset = [r for r in a1_rows if r["N"] == N and r["topology"] == topo
                       and r.get("chain_break_fraction") is not None]
            if subset:
                vals = [r["chain_break_fraction"] for r in subset]
                cbfs_mean.append(np.mean(vals))
                cbfs_std.append(np.std(vals))
                emb = [r["embedded_qubits"] for r in subset if r.get("embedded_qubits")]
                overheads.append(np.mean(emb) / N if emb else 0)

        ax1.errorbar(N_vals, cbfs_mean, yerr=cbfs_std, marker=marker, color=color,
                     label=topo.capitalize(), capsize=3, linewidth=1.5)
        ax2.plot(N_vals, overheads, marker=marker, color=color,
                 label=topo.capitalize(), linewidth=1.5)

    ax1.set_xlabel("Problem size $N$")
    ax1.set_ylabel("Mean chain-break fraction")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.legend()
    ax1.set_title("(a) Chain-break fraction")

    ax2.set_xlabel("Problem size $N$")
    ax2.set_ylabel("Embedding overhead (phys/logical)")
    ax2.legend()
    ax2.set_title("(b) Embedding overhead")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_chain_break_embedding.pdf")
    fig.savefig(FIG_DIR / "fig1_chain_break_embedding.png")
    plt.close(fig)
    print("Fig 1 saved.")


def fig2_objective_gap(a23_rows, classical_rows=None):
    """Per-solver mean objective vs N, averaged across families."""
    fig, ax = plt.subplots(figsize=(7, 5))

    solvers = {
        "hybrid_cqm": ("Hybrid CQM", "#4CAF50", "o"),
        "hybrid_bqm": ("Hybrid BQM", "#F44336", "s"),
    }

    a2 = [r for r in a23_rows if r["experiment"] == "A2" and "objective_value" in r]
    a3 = [r for r in a23_rows if r["experiment"] == "A3" and "objective_value" in r]
    N_hybrid = sorted(set(r["N"] for r in a2))

    # BQM
    bqm_means = [np.mean([r["objective_value"] for r in a2 if r["N"] == N]) for N in N_hybrid]
    ax.plot(N_hybrid, bqm_means, marker="s", color="#F44336", label="Hybrid BQM", linewidth=1.5)

    # CQM
    cqm_means = [np.mean([r["objective_value"] for r in a3 if r["N"] == N]) for N in N_hybrid]
    ax.plot(N_hybrid, cqm_means, marker="o", color="#4CAF50", label="Hybrid CQM", linewidth=1.5)

    # Classical baselines if available
    if classical_rows:
        gurobi = [r for r in classical_rows if r["solver"] == "gurobi_miqp" and "objective_value" in r]
        sa = [r for r in classical_rows if r["solver"] == "neal_sa" and "objective_value" in r]

        if gurobi:
            N_gurobi = sorted(set(r["N"] for r in gurobi))
            gurobi_means = [np.mean([r["objective_value"] for r in gurobi if r["N"] == N]) for N in N_gurobi]
            ax.plot(N_gurobi, gurobi_means, marker="D", color="#9C27B0", label="Gurobi MIQP",
                    linewidth=1.5, linestyle="--")

        if sa:
            N_sa = sorted(set(r["N"] for r in sa))
            sa_means = [np.mean([r["objective_value"] for r in sa if r["N"] == N]) for N in N_sa]
            ax.plot(N_sa, sa_means, marker="^", color="#FF9800", label="Neal SA",
                    linewidth=1.5, linestyle="--")

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Problem size $N$")
    ax.set_ylabel("Mean objective value (lower = better)")
    ax.set_title("Solver comparison across problem sizes")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_objective_vs_N.pdf")
    fig.savefig(FIG_DIR / "fig2_objective_vs_N.png")
    plt.close(fig)
    print("Fig 2 saved.")


def fig3_budget_curves(b2_rows):
    """Budget response curves for BQM and CQM, dense family."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax_idx, family in enumerate(["block", "dense"]):
        ax = axes[ax_idx]
        N_vals = sorted(set(r["N"] for r in b2_rows if r["family"] == family))

        for N in N_vals:
            for solver, style, color_base in [
                ("hybrid_bqm", "--", plt.cm.Reds),
                ("hybrid_cqm", "-", plt.cm.Greens),
            ]:
                subset = [r for r in b2_rows if r["N"] == N and r["family"] == family
                          and r["solver"] == solver and "objective_value" in r]
                if not subset:
                    continue
                subset.sort(key=lambda r: r["time_limit"])
                tls = [r["time_limit"] for r in subset]
                objs = [r["objective_value"] for r in subset]
                # Color by N
                norm_n = (N_vals.index(N) + 1) / len(N_vals)
                color = color_base(0.3 + 0.6 * norm_n)
                label_prefix = "BQM" if "bqm" in solver else "CQM"
                ax.plot(tls, objs, marker="o" if "cqm" in solver else "s",
                        linestyle=style, color=color, markersize=4,
                        label=f"{label_prefix} N={N}", linewidth=1.2)

        ax.set_xlabel("Time limit (s)")
        ax.set_ylabel("Objective value")
        ax.set_title(f"({chr(97+ax_idx)}) {family.capitalize()} family")
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_budget_curves.pdf")
    fig.savefig(FIG_DIR / "fig3_budget_curves.png")
    plt.close(fig)
    print("Fig 3 saved.")


def main():
    print("Loading results...")
    a1 = load_jsonl(ROOT / "results" / "qpu" / "a1_synthetic_qpu.jsonl")
    a23 = load_jsonl(ROOT / "results" / "hybrid" / "a2a3_synthetic_hybrid.jsonl")
    b2 = load_jsonl(ROOT / "results" / "hybrid" / "b2_budget_sweep.jsonl")

    classical_path = ROOT / "results" / "classical" / "d1d2_classical_baselines.jsonl"
    classical = load_jsonl(classical_path) if classical_path.exists() else None

    print(f"A1: {len(a1)} rows, A2/A3: {len(a23)} rows, B2: {len(b2)} rows")
    if classical:
        print(f"Classical: {len(classical)} rows")

    fig1_chain_break_embedding(a1)
    fig2_objective_gap(a23, classical)
    fig3_budget_curves(b2)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
