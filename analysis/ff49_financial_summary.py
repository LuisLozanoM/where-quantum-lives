"""
ff49_financial_summary.py
==========================

Phase 3 — financial overlay summary on the reconstructed FF-49 daily P&L
time series produced by `reconstruct_pnl.py`.

Computes:
    - Annualised Sharpe ratio
    - Max drawdown
    - Probabilistic Sharpe Ratio (PSR, Bailey & López de Prado 2012)
    - Deflated Sharpe Ratio (DSR, Bailey & López de Prado 2014)
        with n_trials = number of QPU configurations tried per (window, N)
    - 1/N equal-weight baseline (all 49 industries each window's eval month)

Outputs:
    results/analysis/ff49_financial_summary.csv
    results/analysis/ff49_financial_summary.txt
    results/analysis/ff49_financial_summary_table.tex
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from financial_metrics import (
    sharpe,
    max_drawdown,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
)
from reconstruct_pnl import load_ff49_daily, eval_window_returns

REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO_ROOT / "results" / "analysis" / "daily_pnl_index.csv"
PNL_DIR = REPO_ROOT / "results" / "analysis" / "daily_pnl"
OUT_DIR = REPO_ROOT / "results" / "analysis"
FF49_ZIP = REPO_ROOT / "data" / "raw" / "equities" / "49_Industry_Portfolios_Daily.zip"


def compute_metrics(returns: pd.Series, n_trials: int) -> dict[str, float]:
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 2:
        return {
            "n_days": 0,
            "sharpe_annualised": float("nan"),
            "max_drawdown": float("nan"),
            "psr": float("nan"),
            "dsr": float("nan"),
        }
    return {
        "n_days": int(len(r)),
        "sharpe_annualised": sharpe(r, periods=252),
        "max_drawdown": max_drawdown(r),
        "psr": probabilistic_sharpe_ratio(r, sr_benchmark=0.0, periods=252),
        "dsr": deflated_sharpe_ratio(r, n_trials=n_trials, sr_benchmark=0.0, periods=252),
    }


def main() -> int:
    idx = pd.read_csv(INDEX_PATH)
    print(f"Loaded {len(idx)} reconstructed portfolios from {INDEX_PATH.name}")

    # Compute equal-weight 1/N baseline (all 49 industries) per window
    ff49 = load_ff49_daily(FF49_ZIP)

    # Map window_idx → eval-window FF-49 mean (1/49) return series
    baseline_returns: dict[int, pd.Series] = {}
    for w in sorted(idx["window_idx"].unique()):
        rebalance_str = idx[idx["window_idx"] == w]["rebalance_date"].iloc[0]
        rebalance_date = pd.Timestamp(rebalance_str)
        eval_df = eval_window_returns(ff49, rebalance_date)
        baseline_returns[int(w)] = eval_df.mean(axis=1, skipna=True)
        print(f"  window {w}: rebalance={rebalance_date.date()}, "
              f"eval-days={len(eval_df)}, 1/49 mean={eval_df.mean(axis=1).mean():+.5f}")

    out_rows: list[dict[str, Any]] = []

    # Configuration-level metrics for each saved portfolio
    # n_trials per portfolio: number of competing configurations in the same
    # (window, N) cell that could have been "the best".  Configurations
    # vary in topology × chain_strength = 2 × 3 = 6.
    N_TRIALS_PER_CELL = 6

    for _, row in idx.iterrows():
        csv_path = REPO_ROOT / str(row["csv_path"])
        if not csv_path.exists():
            continue
        series = pd.read_csv(csv_path, index_col=0)["portfolio_return"]
        metrics = compute_metrics(series, n_trials=N_TRIALS_PER_CELL)
        out_rows.append({
            "instance_id": row["instance_id"],
            "window_idx": int(row["window_idx"]),
            "N": int(row["N"]),
            "K": int(row["K"]),
            "topology": row["topology"],
            "chain_strength": row["chain_strength"],
            **metrics,
        })

    # 1/N baselines (one per window)
    for w, series in baseline_returns.items():
        bm = compute_metrics(series, n_trials=1)  # n_trials=1 because the 1/N strategy is a single design choice
        out_rows.append({
            "instance_id": f"1over49_baseline_w{w}",
            "window_idx": w,
            "N": 49,
            "K": 49,
            "topology": "BASELINE_1OVERN",
            "chain_strength": float("nan"),
            **bm,
        })

    csv_path = OUT_DIR / "ff49_financial_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {csv_path} ({len(out_rows)} rows)")

    # Aggregate per (N, window) for QPU portfolios
    qpu_rows = [r for r in out_rows if r["topology"] != "BASELINE_1OVERN"]
    baseline_rows = [r for r in out_rows if r["topology"] == "BASELINE_1OVERN"]

    summary_lines = [
        "FF-49 FINANCIAL OVERLAY — PHASE 3",
        "=" * 70,
        f"QPU portfolios: {len(qpu_rows)} (5 windows × 3 N × 2 topologies × 3 chain strengths)",
        f"1/N baselines:  {len(baseline_rows)} (one per window)",
        f"DSR n_trials per configuration: {N_TRIALS_PER_CELL} (topology × chain strength)",
        "",
    ]

    # Per (window, N) cell summary
    cells = {}
    for r in qpu_rows:
        cells.setdefault((r["window_idx"], r["N"]), []).append(r)
    summary_lines.append("WINDOW × N CELL MEANS (over 6 QPU configurations):")
    summary_lines.append(
        f"  {'win':>3}  {'N':>3}  "
        f"{'Sharpe':>8}  {'MaxDD':>8}  {'PSR':>6}  {'DSR':>6}"
    )
    for key in sorted(cells):
        bucket = cells[key]
        s = np.nanmean([r["sharpe_annualised"] for r in bucket])
        mdd = np.nanmean([r["max_drawdown"] for r in bucket])
        psr = np.nanmean([r["psr"] for r in bucket])
        dsr = np.nanmean([r["dsr"] for r in bucket])
        summary_lines.append(
            f"  {key[0]:>3}  {key[1]:>3}  {s:>+8.3f}  {mdd:>+8.4f}  "
            f"{psr:>6.3f}  {dsr:>6.3f}"
        )
    summary_lines.append("")
    summary_lines.append("1/N BASELINE (49 equal-weight, same windows):")
    for r in baseline_rows:
        summary_lines.append(
            f"  win={r['window_idx']}  Sharpe={r['sharpe_annualised']:+.3f}  "
            f"MaxDD={r['max_drawdown']:+.4f}  PSR={r['psr']:.3f}"
        )
    summary_lines.append("")
    summary_lines.append("AVERAGES OVER ALL WINDOWS:")
    s_qpu = np.nanmean([r["sharpe_annualised"] for r in qpu_rows])
    s_bl = np.nanmean([r["sharpe_annualised"] for r in baseline_rows])
    summary_lines.append(f"  QPU-selected portfolios:  mean Sharpe = {s_qpu:+.3f}")
    summary_lines.append(f"  1/N baseline:             mean Sharpe = {s_bl:+.3f}")
    summary_lines.append(f"  QPU vs 1/N delta:                       {s_qpu - s_bl:+.3f}")

    summary_text = "\n".join(summary_lines)
    (OUT_DIR / "ff49_financial_summary.txt").write_text(summary_text)
    print()
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
