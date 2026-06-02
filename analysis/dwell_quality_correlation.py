"""
dwell_quality_correlation.py
============================

Phase 4 — within-N correlation between QPU dwell time and final solver
objective value, across all saved hybrid runs.

Purpose: pre-empt the strongest reviewer attack on paper2 ("qpu_access_time
is a billing field, not a contribution metric"). If, conditional on N,
runs with more QPU dwell do NOT produce better objective values, then
the 0.7% wall-clock fraction is genuinely incremental and not the
visible tip of a hidden QPU contribution.

Input files:
    results/hybrid/a2a3_synthetic_hybrid.jsonl   (162 rows; has full timing)
    results/hybrid/repeated_hybrid.jsonl         (228 rows; qpu_access + wall only)
    results/hybrid/b2_budget_sweep.jsonl         (78 rows;  qpu_access + wall only)
    results/hybrid/penalty_robustness.jsonl      (36 rows)

Output:
    results/analysis/dwell_quality_correlation.csv
        columns: source, solver, N, n_runs, spearman_rho, spearman_ci_low,
                 spearman_ci_high, pearson_r, pearson_ci_low, pearson_ci_high,
                 mean_qpu_access, mean_objective
    results/analysis/dwell_quality_correlation_summary.json
        bootstrap CI from 10_000 resamples per (source, solver, N) cell

Usage:
    python analysis/dwell_quality_correlation.py
        [--bootstrap-samples 10000]
        [--out results/analysis/]

Methodology:
    1. Load each JSONL into a DataFrame.
    2. Group by (source, solver, N).
    3. Within each group with >= 5 runs, compute:
        - Spearman rho between hybrid_qpu_access_time and objective_value
        - Pearson r (after log-transforming objective if negative, per
          Wunderlich-style normalization)
        - 95% bootstrap CI on both
    4. Aggregate.

Predicted finding (per NeurIPS-AC reviewer):
    Within-N Spearman correlations near zero or weakly negative.
    Interpretation: more QPU dwell does not buy quality. The 0.7%
    figure is incremental, not the lower bound of a hidden channel.

NO new QPU experiments. Pure re-analysis of saved data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _corr_statistic(result: Any) -> float:
    """Extract correlation coefficient from a scipy spearmanr/pearsonr result.

    New-API SignificanceResult has a `.statistic` attribute; older API returns
    a (correlation, pvalue) tuple. This helper handles both.
    """
    if hasattr(result, "statistic"):
        return float(result.statistic)
    return float(result[0])


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUT_DIR = RESULTS_DIR / "analysis"


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame, skipping malformed lines."""
    rows: list[dict] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"WARN: {path.name}:{line_no} JSON parse failed: {exc}",
                    file=sys.stderr,
                )
    return pd.DataFrame(rows)


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    method: str,
    seed: int = 20260531,
) -> tuple[float, float]:
    """Bootstrap a 95% CI for Spearman or Pearson correlation.

    Returns (ci_low, ci_high) at the 2.5/97.5 percentiles. The point
    estimate is reported separately by the caller; this returns CI only.
    """
    if len(x) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(x)
    samples = np.empty(n_samples, dtype=float)
    corr_fn = spearmanr if method == "spearman" else pearsonr
    for k in range(n_samples):
        idx = rng.integers(0, n, size=n)
        try:
            r = _corr_statistic(corr_fn(x[idx], y[idx]))
            samples[k] = r if np.isfinite(r) else 0.0
        except (ValueError, TypeError):
            samples[k] = 0.0
    return (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))


def analyze_source(
    df: pd.DataFrame,
    source_name: str,
    solver_col: str,
    n_col: str,
    qpu_col: str,
    obj_col: str,
    bootstrap_samples: int,
) -> pd.DataFrame:
    """Within-(source, solver, N) correlations between QPU dwell and objective."""
    rows: list[dict] = []
    if solver_col not in df.columns:
        print(f"WARN: {source_name}: missing solver column '{solver_col}'", file=sys.stderr)
        return pd.DataFrame(rows)
    for key, group_obj in df.groupby([solver_col, n_col], dropna=False):
        solver, n_val = cast(tuple[Any, Any], key)
        group_df = cast(pd.DataFrame, group_obj)
        valid = group_df[[qpu_col, obj_col]].dropna()
        if len(valid) < 5:
            continue
        x = np.asarray(valid[qpu_col], dtype=float)
        y = np.asarray(valid[obj_col], dtype=float)
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            continue
        spearman_rho = _corr_statistic(spearmanr(x, y))
        pearson_r = _corr_statistic(pearsonr(x, y))
        s_lo, s_hi = bootstrap_corr_ci(x, y, bootstrap_samples, "spearman")
        p_lo, p_hi = bootstrap_corr_ci(x, y, bootstrap_samples, "pearson")
        rows.append(
            {
                "source": source_name,
                "solver": str(solver),
                "N": int(n_val) if pd.notna(n_val) else None,
                "n_runs": int(len(valid)),
                "spearman_rho": spearman_rho,
                "spearman_ci_low": s_lo,
                "spearman_ci_high": s_hi,
                "pearson_r": pearson_r,
                "pearson_ci_low": p_lo,
                "pearson_ci_high": p_hi,
                "mean_qpu_access_s": float(np.mean(x)),
                "mean_objective": float(np.mean(y)),
            }
        )
    return pd.DataFrame(rows)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.out.mkdir(parents=True, exist_ok=True)

    sources = [
        (
            "a2a3_synthetic_hybrid",
            RESULTS_DIR / "hybrid" / "a2a3_synthetic_hybrid.jsonl",
            {"solver_col": "solver", "n_col": "N",
             "qpu_col": "hybrid_qpu_access_time", "obj_col": "objective_value"},
        ),
        (
            "repeated_hybrid",
            RESULTS_DIR / "hybrid" / "repeated_hybrid.jsonl",
            {"solver_col": "solver", "n_col": "N",
             "qpu_col": "hybrid_qpu_access_time", "obj_col": "objective_value"},
        ),
        (
            "b2_budget_sweep",
            RESULTS_DIR / "hybrid" / "b2_budget_sweep.jsonl",
            {"solver_col": "solver", "n_col": "N",
             "qpu_col": "hybrid_qpu_access_time", "obj_col": "objective_value"},
        ),
        (
            "penalty_robustness",
            RESULTS_DIR / "hybrid" / "penalty_robustness.jsonl",
            {"solver_col": "solver", "n_col": "N",
             "qpu_col": "hybrid_qpu_access_time", "obj_col": "objective_value"},
        ),
    ]

    all_rows: list[pd.DataFrame] = []
    for source_name, path, cols in sources:
        if not path.exists():
            print(f"SKIP: {path} not found", file=sys.stderr)
            continue
        df = load_jsonl(path)
        if df.empty:
            print(f"SKIP: {path} loaded empty", file=sys.stderr)
            continue
        result = analyze_source(
            df,
            source_name=source_name,
            bootstrap_samples=args.bootstrap_samples,
            **cols,
        )
        if not result.empty:
            all_rows.append(result)
            print(f"OK: {source_name}: {len(result)} (solver, N) cells analysed")
        else:
            print(f"EMPTY: {source_name}: no cells with >= 5 valid runs", file=sys.stderr)

    if not all_rows:
        print("ERROR: no data analysed; check column names in JSONL", file=sys.stderr)
        return 1

    combined = pd.concat(all_rows, ignore_index=True)
    csv_path = args.out / "dwell_quality_correlation.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(combined)} rows)")

    s_hi = combined["spearman_ci_high"].to_numpy(dtype=float)
    s_lo = combined["spearman_ci_low"].to_numpy(dtype=float)
    s_rho = combined["spearman_rho"].to_numpy(dtype=float)
    summary = {
        "n_cells_total": int(len(combined)),
        "n_cells_with_significant_negative_spearman": int(
            np.sum((s_hi < 0) & (s_rho < 0))
        ),
        "n_cells_with_significant_positive_spearman": int(
            np.sum((s_lo > 0) & (s_rho > 0))
        ),
        "n_cells_consistent_with_zero": int(
            np.sum((s_lo <= 0) & (s_hi >= 0))
        ),
        "median_spearman_rho": float(np.median(s_rho)),
        "median_abs_spearman_rho": float(np.median(np.abs(s_rho))),
    }
    summary_path = args.out / "dwell_quality_correlation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
