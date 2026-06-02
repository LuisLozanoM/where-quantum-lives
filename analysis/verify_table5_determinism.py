"""
verify_table5_determinism.py
============================

Verify the manuscript claim (main.tex §5.7, lines 583-588):

    "CQM returns IDENTICAL solutions (zero variance) across all 10 repetitions
     at every tested (N, ρ, B) combination, at both 5 s and 300 s."

Grouping the data at the (solver, N, family, time_limit) cell level — finer
than my earlier (solver, N) aggregate which may have conflated variation
across budgets.

Inputs:
    results/hybrid/b2_budget_sweep.jsonl   (78 rows; budget sweep, single-seed)
    results/hybrid/repeated_hybrid.jsonl   (228 rows; 10-rep validation)

Outputs:
    results/analysis/table5_cell_verification.csv
        columns: source, solver, N, family, time_limit, n_runs,
                 objective_min, objective_max, objective_std,
                 n_unique_objectives, identical (bool)
    results/analysis/table5_cell_verification_summary.txt
        Human-readable summary of where the "identical" claim holds.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "analysis"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def verify_cells(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Group rows by (solver, N, family, time_limit); report objective stats per cell."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("solver"), r.get("N"), r.get("family"), r.get("time_limit"))
        groups[key].append(r)

    out = []
    for (solver, n_val, family, budget), group_rows in groups.items():
        objs = [r["objective_value"] for r in group_rows if r.get("objective_value") is not None]
        if not objs:
            continue
        objs_arr = np.asarray(objs, dtype=float)
        n_runs = len(objs_arr)
        std = float(np.std(objs_arr, ddof=0))
        obj_min = float(np.min(objs_arr))
        obj_max = float(np.max(objs_arr))
        n_unique = len(set(np.round(objs_arr, 10)))
        identical = bool(n_unique == 1)
        out.append({
            "source": source,
            "solver": solver,
            "N": n_val,
            "family": family,
            "time_limit_s": budget,
            "n_runs": n_runs,
            "objective_min": obj_min,
            "objective_max": obj_max,
            "objective_range": obj_max - obj_min,
            "objective_std": std,
            "n_unique_objectives": n_unique,
            "identical": identical,
        })
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = [
        ("b2_budget_sweep", REPO_ROOT / "results" / "hybrid" / "b2_budget_sweep.jsonl"),
        ("repeated_hybrid", REPO_ROOT / "results" / "hybrid" / "repeated_hybrid.jsonl"),
    ]

    all_rows = []
    for name, path in sources:
        if not path.exists():
            print(f"SKIP: {path} not found", file=sys.stderr)
            continue
        rows = load_jsonl(path)
        cells = verify_cells(rows, name)
        all_rows.extend(cells)
        print(f"OK: {name}: {len(rows)} rows → {len(cells)} cells")

    if not all_rows:
        return 1

    # Write CSV
    import csv
    csv_path = OUT_DIR / "table5_cell_verification.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {csv_path}")

    # Summary
    cqm_cells = [r for r in all_rows if r["solver"] == "hybrid_cqm"]
    bqm_cells = [r for r in all_rows if r["solver"] == "hybrid_bqm"]
    cqm_identical = [r for r in cqm_cells if r["identical"]]
    cqm_nonidentical = [r for r in cqm_cells if not r["identical"]]
    bqm_identical = [r for r in bqm_cells if r["identical"]]
    bqm_nonidentical = [r for r in bqm_cells if not r["identical"]]

    summary_lines = [
        "TABLE 5 DETERMINISM VERIFICATION",
        "=" * 60,
        f"Total cells analysed: {len(all_rows)}",
        f"  CQM cells:           {len(cqm_cells)}",
        f"    identical:         {len(cqm_identical)}",
        f"    non-identical:     {len(cqm_nonidentical)}",
        f"  BQM cells:           {len(bqm_cells)}",
        f"    identical:         {len(bqm_identical)}",
        f"    non-identical:     {len(bqm_nonidentical)}",
        "",
        "CQM CELLS WHERE THE MANUSCRIPT 'IDENTICAL' CLAIM FAILS:",
    ]
    if cqm_nonidentical:
        for r in sorted(cqm_nonidentical,
                        key=lambda x: (x["source"], x["N"] or 0,
                                       x["family"] or "", x["time_limit_s"] or 0)):
            summary_lines.append(
                f"  [{r['source']:<18}] N={r['N']:>3}  family={r['family']:<8}  "
                f"budget={r['time_limit_s']:>4}s  n={r['n_runs']:>2}  "
                f"unique_objs={r['n_unique_objectives']}  range={r['objective_range']:.4e}  "
                f"std={r['objective_std']:.4e}"
            )
    else:
        summary_lines.append("  (none — the 'identical' claim holds at every cell)")

    summary_lines += ["", "BQM CELLS — for comparison (manuscript says BQM has variance):"]
    for r in sorted(bqm_nonidentical,
                    key=lambda x: (x["source"], x["N"] or 0,
                                   x["family"] or "", x["time_limit_s"] or 0))[:20]:
        summary_lines.append(
            f"  [{r['source']:<18}] N={r['N']:>3}  family={r['family']:<8}  "
            f"budget={r['time_limit_s']:>4}s  n={r['n_runs']:>2}  "
            f"unique_objs={r['n_unique_objectives']}  range={r['objective_range']:.4e}  "
            f"std={r['objective_std']:.4e}"
        )

    summary_text = "\n".join(summary_lines)
    summary_path = OUT_DIR / "table5_cell_verification_summary.txt"
    summary_path.write_text(summary_text)
    print(summary_text)
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
