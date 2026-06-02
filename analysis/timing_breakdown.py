"""
timing_breakdown.py
===================

Phase 4 — full three-field timing breakdown per (solver, N) cell from
`results/hybrid/a2a3_synthetic_hybrid.jsonl`.

Produces the §5.3 "Full Timing Breakdown" table.  The paper recommends
three-field reporting (run_time, charge_time, qpu_access_time) as part of
the audit protocol; this script extracts all three fields from the saved
synthetic-hybrid runs.

a2a3 has all three fields per row (in microseconds).

Outputs:
    results/analysis/timing_breakdown.csv
    results/analysis/timing_breakdown_table.tex   (LaTeX snippet)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
JSONL = REPO_ROOT / "results" / "hybrid" / "a2a3_synthetic_hybrid.jsonl"
OUT_DIR = REPO_ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    rows: list[dict] = []
    with JSONL.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["solver"], r["N"])
        groups[key].append(r)

    out_rows: list[dict] = []
    for (solver, n_val), cell in sorted(groups.items()):
        run_us = np.array([r["hybrid_run_time"] for r in cell], dtype=float)
        chg_us = np.array([r["hybrid_charge_time"] for r in cell], dtype=float)
        qpu_us = np.array([r["hybrid_qpu_access_time"] for r in cell], dtype=float)
        wall_s = np.array([r["wall_clock_total"] for r in cell], dtype=float)
        out_rows.append({
            "solver": solver,
            "N": n_val,
            "n_runs": len(cell),
            "mean_run_time_s": float(np.mean(run_us) / 1e6),
            "mean_charge_time_s": float(np.mean(chg_us) / 1e6),
            "mean_qpu_access_s": float(np.mean(qpu_us) / 1e6),
            "median_qpu_access_s": float(np.median(qpu_us) / 1e6),
            "iqr_qpu_access_s": float(
                (np.percentile(qpu_us, 75) - np.percentile(qpu_us, 25)) / 1e6
            ),
            "mean_r_qpu_pct": float(100.0 * np.mean(qpu_us / run_us)),
            "mean_wall_clock_total_s": float(np.mean(wall_s)),
        })

    csv_path = OUT_DIR / "timing_breakdown.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {csv_path} ({len(out_rows)} cells)")

    # LaTeX snippet
    tex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Full timing breakdown per $(N,\,\text{solver})$ cell on the"
        r" synthetic-hybrid runs (mean over 9 instances per cell: 3 density"
        r" families $\times$ 3 seeds; 162 runs total).  All timing fields in"
        r" seconds.  $r_{\mathrm{QPU}}$ is the within-cell mean wall-clock"
        r" fraction $t_{\mathrm{QPU}}/t_{\mathrm{run}}$.}",
        r"\label{tab:timing_breakdown}",
        r"\begin{tabular}{rlrrrr}",
        r"\toprule",
        r"$N$ & Solver & $t_{\mathrm{run}}$ (s) & $t_{\mathrm{charge}}$ (s) & "
        r"$t_{\mathrm{QPU}}$ (s) & $r_{\mathrm{QPU}}$ (\%) \\",
        r"\midrule",
    ]
    for r in out_rows:
        tex_lines.append(
            f"{r['N']} & {r['solver'].replace('_', '-')} & "
            f"${r['mean_run_time_s']:.3f}$ & "
            f"${r['mean_charge_time_s']:.3f}$ & "
            f"${r['mean_qpu_access_s']:.4f}$ & "
            f"${r['mean_r_qpu_pct']:.2f}$ \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = OUT_DIR / "timing_breakdown_table.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"Wrote {tex_path}")

    # Quick summary
    print()
    print("CQM cells (sorted by N):")
    for r in [x for x in out_rows if x["solver"] == "hybrid_cqm"]:
        print(f"  N={r['N']:>3}  run={r['mean_run_time_s']:.3f}s  "
              f"charge={r['mean_charge_time_s']:.3f}s  "
              f"qpu={r['mean_qpu_access_s']*1000:.2f}ms  "
              f"r_QPU={r['mean_r_qpu_pct']:.2f}%")
    print()
    print("BQM cells (sorted by N):")
    for r in [x for x in out_rows if x["solver"] == "hybrid_bqm"]:
        print(f"  N={r['N']:>3}  run={r['mean_run_time_s']:.3f}s  "
              f"charge={r['mean_charge_time_s']:.3f}s  "
              f"qpu={r['mean_qpu_access_s']*1000:.2f}ms  "
              f"r_QPU={r['mean_r_qpu_pct']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
