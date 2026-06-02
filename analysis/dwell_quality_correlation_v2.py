"""
dwell_quality_correlation_v2.py
================================

Phase 4 — refined dwell-vs-quality analysis at the proper grouping level.

Two analyses:

(A) Within-cell correlation: for each (solver, N, family, budget) cell with
    ≥5 runs and non-zero objective variance, compute Spearman ρ and Pearson r
    between hybrid_qpu_access_time and objective_value within that cell.
    Only BQM cells qualify; CQM cells are deterministic (verified by
    verify_table5_determinism.py — all 52 CQM cells are identical).

(B) Cross-cell correlation at fixed N: aggregate cell means of qpu_access and
    objective across (family, budget) cells, then compute Spearman across the
    cell means within each (solver, N) bucket.  This asks: "across the
    different (family, budget) cells at the same N, does the cell with more
    QPU dwell produce a better objective?"  Both CQM and BQM admit this.

Interpretation:
    * Within-cell positive correlation → service-internal randomness produces
      better outcomes when more QPU is used → QPU is doing useful work
      within a single (N, ρ, B) configuration.
    * Cross-cell negative Spearman (more QPU ↔ better obj, since obj
      minimized) → classical decomposer routes more QPU time to harder
      sub-problems → QPU usage is hardness-correlated, not random.

Inputs: same JSONLs as v1.
Outputs:
    results/analysis/dwell_within_cell.csv
    results/analysis/dwell_cross_cell.csv
    results/analysis/dwell_correlation_v2_summary.txt
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "analysis"


def _corr_statistic(result: Any) -> float:
    if hasattr(result, "statistic"):
        return float(result.statistic)
    return float(result[0])


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n: int, method: str,
                 seed: int = 20260531) -> tuple[float, float]:
    if len(x) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=float)
    corr_fn = spearmanr if method == "spearman" else pearsonr
    for k in range(n):
        idx = rng.integers(0, len(x), size=len(x))
        try:
            r = _corr_statistic(corr_fn(x[idx], y[idx]))
            samples[k] = r if np.isfinite(r) else 0.0
        except (ValueError, TypeError):
            samples[k] = 0.0
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def within_cell_analysis(rows: list[dict], source: str, n_boot: int = 2000) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("solver"), r.get("N"), r.get("family"), r.get("time_limit"))
        groups[key].append(r)
    out = []
    for (solver, n_val, family, budget), cell_rows in groups.items():
        valid = [r for r in cell_rows
                 if r.get("hybrid_qpu_access_time") is not None
                 and r.get("objective_value") is not None]
        if len(valid) < 5:
            continue
        x = np.asarray([r["hybrid_qpu_access_time"] for r in valid], dtype=float)
        y = np.asarray([r["objective_value"] for r in valid], dtype=float)
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            continue
        rho = _corr_statistic(spearmanr(x, y))
        pr = _corr_statistic(pearsonr(x, y))
        s_lo, s_hi = bootstrap_ci(x, y, n_boot, "spearman")
        out.append({
            "source": source,
            "solver": solver,
            "N": n_val,
            "family": family,
            "time_limit_s": budget,
            "n_runs": len(valid),
            "spearman_rho": rho,
            "spearman_ci_low": s_lo,
            "spearman_ci_high": s_hi,
            "pearson_r": pr,
            "mean_qpu_access_us": float(np.mean(x)),
            "mean_objective": float(np.mean(y)),
            "obj_std": float(np.std(y, ddof=0)),
        })
    return out


def cross_cell_analysis(rows: list[dict], source: str, n_boot: int = 2000) -> list[dict]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("solver"), r.get("N"), r.get("family"), r.get("time_limit"))
        if all(v is not None for v in key):
            cells[key].append(r)
    cell_means: list[dict] = []
    for (solver, n_val, family, budget), cell_rows in cells.items():
        qpu = np.mean([r["hybrid_qpu_access_time"] for r in cell_rows
                       if r.get("hybrid_qpu_access_time") is not None])
        obj = np.mean([r["objective_value"] for r in cell_rows
                       if r.get("objective_value") is not None])
        cell_means.append({
            "solver": solver, "N": n_val, "family": family, "budget": budget,
            "mean_qpu": float(qpu), "mean_obj": float(obj),
        })
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cell_means:
        groups[(c["solver"], c["N"])].append(c)
    out = []
    for (solver, n_val), bucket in groups.items():
        if len(bucket) < 5:
            continue
        x = np.asarray([c["mean_qpu"] for c in bucket], dtype=float)
        y = np.asarray([c["mean_obj"] for c in bucket], dtype=float)
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            continue
        rho = _corr_statistic(spearmanr(x, y))
        pr = _corr_statistic(pearsonr(x, y))
        s_lo, s_hi = bootstrap_ci(x, y, n_boot, "spearman")
        out.append({
            "source": source,
            "solver": solver,
            "N": n_val,
            "n_cells": len(bucket),
            "spearman_rho": rho,
            "spearman_ci_low": s_lo,
            "spearman_ci_high": s_hi,
            "pearson_r": pr,
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    import csv
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        ("a2a3_synthetic_hybrid", RESULTS_DIR / "hybrid" / "a2a3_synthetic_hybrid.jsonl"),
        ("repeated_hybrid", RESULTS_DIR / "hybrid" / "repeated_hybrid.jsonl"),
        ("b2_budget_sweep", RESULTS_DIR / "hybrid" / "b2_budget_sweep.jsonl"),
    ]

    within_rows: list[dict] = []
    cross_rows: list[dict] = []
    for name, path in sources:
        if not path.exists():
            continue
        rows = load_jsonl(path)
        within_rows.extend(within_cell_analysis(rows, name))
        cross_rows.extend(cross_cell_analysis(rows, name))
        print(f"OK: {name}: {len(rows)} rows analysed")

    write_csv(OUT_DIR / "dwell_within_cell.csv", within_rows)
    write_csv(OUT_DIR / "dwell_cross_cell.csv", cross_rows)

    # Summary
    lines = ["DWELL-VS-QUALITY CORRELATION (v2)", "=" * 60]
    lines += [
        "",
        "(A) WITHIN-CELL — service-internal randomness",
        f"    cells analysed: {len(within_rows)} (only BQM cells with variance qualify)",
    ]
    if within_rows:
        cqm_w = [c for c in within_rows if c["solver"] == "hybrid_cqm"]
        bqm_w = [c for c in within_rows if c["solver"] == "hybrid_bqm"]
        lines.append(f"      CQM: {len(cqm_w)}  BQM: {len(bqm_w)}")
        for sub in [bqm_w]:
            sub.sort(key=lambda c: (c["N"] or 0, c["family"] or "",
                                    c["time_limit_s"] or 0))
            for r in sub:
                lines.append(
                    f"      [{r['solver']}] N={r['N']:>3} fam={r['family']:<6} "
                    f"B={r['time_limit_s']:>4}s n={r['n_runs']:>2} "
                    f"ρ={r['spearman_rho']:>+.3f} CI=[{r['spearman_ci_low']:>+.2f},{r['spearman_ci_high']:>+.2f}]"
                )

    lines += ["", "(B) CROSS-CELL at fixed (solver, N) — hardness allocation",
              f"    buckets analysed: {len(cross_rows)}"]
    if cross_rows:
        for r in sorted(cross_rows, key=lambda c: (c["source"], c["solver"], c["N"] or 0)):
            lines.append(
                f"      [{r['source']:<20}] {r['solver']:<10} N={r['N']:>3} "
                f"cells={r['n_cells']:>2} ρ={r['spearman_rho']:>+.3f} "
                f"CI=[{r['spearman_ci_low']:>+.2f},{r['spearman_ci_high']:>+.2f}]"
            )

    # Aggregate verdict
    cross_neg = [c for c in cross_rows if c["spearman_ci_high"] < 0 and c["spearman_rho"] < 0]
    cross_pos = [c for c in cross_rows if c["spearman_ci_low"] > 0 and c["spearman_rho"] > 0]
    lines += [
        "",
        "VERDICT",
        f"  cross-cell significant NEGATIVE Spearman (more QPU ↔ better obj):  {len(cross_neg)} buckets",
        f"  cross-cell significant POSITIVE Spearman (more QPU ↔ worse obj):  {len(cross_pos)} buckets",
        f"  cross-cell consistent with zero:                                  {len(cross_rows) - len(cross_neg) - len(cross_pos)} buckets",
    ]
    within_bqm_neg = [c for c in within_rows if c["solver"] == "hybrid_bqm"
                      and c["spearman_ci_high"] < 0 and c["spearman_rho"] < 0]
    within_bqm_pos = [c for c in within_rows if c["solver"] == "hybrid_bqm"
                      and c["spearman_ci_low"] > 0 and c["spearman_rho"] > 0]
    lines += [
        f"  within-cell BQM significant NEGATIVE:  {len(within_bqm_neg)} cells",
        f"  within-cell BQM significant POSITIVE:  {len(within_bqm_pos)} cells",
    ]
    summary = "\n".join(lines)
    (OUT_DIR / "dwell_correlation_v2_summary.txt").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
