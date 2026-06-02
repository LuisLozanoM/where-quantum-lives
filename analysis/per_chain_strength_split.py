"""
per_chain_strength_split.py
============================

Phase 4 — re-tabulate Table 1 (chain-break fractions + embedding overhead)
split by chain strength {0.5, 1.0, 2.0} instead of averaged over all.

King-style reviewer point: chain strengths 0.5 and 1.0 sit at or below
|J_max| = A = 4.0, so they are known-broken regimes; only 2.0 is plausibly
in a working regime, and even 2.0 is below the rule-of-thumb 1.5*|J_max|.
Averaging over all three inflates the CBF means.  Per-strength reporting
shows the actual behavior.

Inputs: results/qpu/a1_synthetic_qpu.jsonl
Outputs:
    results/analysis/per_chain_strength_split.csv
    results/analysis/per_chain_strength_split_table.tex
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
JSONL = REPO_ROOT / "results" / "qpu" / "a1_synthetic_qpu.jsonl"
OUT_DIR = REPO_ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    rows: list[dict] = []
    with JSONL.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} rows from a1_synthetic_qpu.jsonl")

    # Group by (N, chain_strength) — average over family, seed, topology
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("embedding_success", True):
            continue
        cbf = r.get("chain_break_fraction")
        if cbf is None:
            continue
        key = (r["N"], r["chain_strength"])
        groups[key].append(r)

    out_rows: list[dict] = []
    for (n_val, cs_val), cell in sorted(groups.items()):
        cbf_arr = np.asarray([r["chain_break_fraction"] for r in cell], dtype=float)
        phys_arr = np.asarray([r.get("embedded_qubits", float("nan")) for r in cell],
                              dtype=float)
        out_rows.append({
            "N": n_val,
            "chain_strength": cs_val,
            "n_runs": len(cell),
            "mean_cbf": float(np.mean(cbf_arr)),
            "std_cbf": float(np.std(cbf_arr, ddof=1)),
            "mean_phys_qubits": float(np.nanmean(phys_arr)),
            "mean_overhead": float(np.nanmean(phys_arr) / n_val) if n_val > 0 else float("nan"),
        })

    csv_path = OUT_DIR / "per_chain_strength_split.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {csv_path} ({len(out_rows)} cells)")

    # LaTeX snippet
    tex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Direct-QPU chain-break fraction split by chain strength $J_c$",
        r"(mean $\pm$ sample std over density families, seeds, and topologies).",
        r"At $J_c \leq 1.0$ the chain strength sits at or below the penalty",
        r"magnitude $|J_{\max}| = A = 4.0$, so chain breaks are guaranteed by",
        r"construction; only $J_c = 2.0$ is plausibly in a working regime, and",
        r"even $J_c = 2.0$ is below the rule-of-thumb $1.5\,|J_{\max}| = 6.0$.",
        r"This per-strength split replaces the chain-strength-averaged",
        r"Table~\ref{tab:a1_cbf}.}",
        r"\label{tab:cbf_per_strength}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$N$ & $J_c$ & runs & CBF mean $\pm$ std & Phys.\ overhead \\",
        r"\midrule",
    ]
    last_n = None
    for r in out_rows:
        if last_n is not None and r["N"] != last_n:
            tex_lines.append(r"\addlinespace")
        last_n = r["N"]
        tex_lines.append(
            f"{r['N']} & ${r['chain_strength']:.1f}$ & {r['n_runs']} & "
            f"${r['mean_cbf']:.3f} \\pm {r['std_cbf']:.3f}$ & "
            f"${r['mean_overhead']:.1f}\\times$ \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = OUT_DIR / "per_chain_strength_split_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"Wrote {tex_path}")

    # Quick console summary
    print()
    print("PER-STRENGTH CBF (mean):")
    print(f"{'N':>4}  {'J_c=0.5':>10}  {'J_c=1.0':>10}  {'J_c=2.0':>10}")
    by_n: dict[int, dict[float, float]] = defaultdict(dict)
    for r in out_rows:
        by_n[r["N"]][r["chain_strength"]] = r["mean_cbf"]
    for n_val in sorted(by_n):
        row = by_n[n_val]
        print(f"{n_val:>4}  "
              f"{row.get(0.5, float('nan')):>10.3f}  "
              f"{row.get(1.0, float('nan')):>10.3f}  "
              f"{row.get(2.0, float('nan')):>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
