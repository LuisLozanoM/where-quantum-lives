"""
reconstruct_pnl.py
==================

Phase 3 — reconstruct daily out-of-sample P&L for every direct-QPU portfolio
saved in `results/qpu/a4_equity_qpu.jsonl`, using the public FF-49 daily
returns at `paper1/data/raw/equities/49_Industry_Portfolios_Daily.zip`.

Methodology (mirroring the paper1 FF-49 windowing code):
    1. Parse `instance_id` (e.g., "equities_19270531_n10") → rebalance date.
    2. Load FF-49 daily returns (percent → decimal).
    3. Evaluation window = the calendar month FOLLOWING the rebalance date
       (standard walk-forward — no look-ahead).
    4. For each saved portfolio: equal-weight the selected_industries over
       the evaluation window's daily returns.
    5. Save (date, portfolio_return) per saved portfolio + summary stats.

NOTE: This is the out-of-sample P&L, not the in-sample estimation-window P&L.
Industry selection is based on |μ̂| over the prior 252 days; the resulting
portfolio is then evaluated on the next month — a proper backtest split.

Outputs:
    results/analysis/daily_pnl/<instance_id>__<topology>__<chain_strength>.csv
    results/analysis/daily_pnl_index.csv           (one row per portfolio)
    results/analysis/daily_pnl_summary.txt
"""

from __future__ import annotations

import csv
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
A4_PATH = REPO_ROOT / "results" / "qpu" / "a4_equity_qpu.jsonl"
FF49_ZIP = REPO_ROOT / "data" / "raw" / "equities" / "49_Industry_Portfolios_Daily.zip"
OUT_DIR = REPO_ROOT / "results" / "analysis" / "daily_pnl"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NA_SENTINELS = (-99.99, -999.0)


def load_ff49_daily(zip_path: Path) -> pd.DataFrame:
    """Load the 49 Industry Portfolios daily return CSV (percent → decimal).

    Mirrors paper1.datasets.load_kenneth_french_49_daily.  Returns a DataFrame
    indexed by date with one column per industry (decimal returns; NaN where
    the source has -99.99 / -999.0 sentinels).
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as fh:
            raw = fh.read().decode("utf-8", errors="replace")

    # Find the daily-returns header.  The Kenneth French CSV begins with a
    # short metadata preamble; the daily portion's header row begins with a
    # comma followed by the industry-name columns.  Detect by scanning for the
    # first line that starts with "," and is followed by an 8-digit date line.
    lines = raw.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith(",") and i + 1 < len(lines):
            nxt = lines[i + 1].split(",")[0].strip()
            if nxt.isdigit() and len(nxt) == 8:
                header_idx = i
                break
    if header_idx is None:
        raise RuntimeError("Could not locate FF-49 daily header in the CSV")

    # Read from the header row, stopping at the first non-data line
    # (annual / equal-weight sections start after a blank line).
    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1:]:
        s = line.strip()
        if not s:
            break
        first = s.split(",")[0].strip()
        if not (first.isdigit() and len(first) == 8):
            break
        data_lines.append(line)

    import io
    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m%d")
    df = df.set_index("date").sort_index()

    # Sentinel → NaN, then percent → decimal
    df = df.replace(list(NA_SENTINELS), np.nan)
    df = df.astype(float) / 100.0
    return df


def parse_instance_id(instance_id: str) -> tuple[pd.Timestamp, int]:
    """'equities_19270531_n10' → (Timestamp('1927-05-31'), 10)."""
    parts = instance_id.split("_")
    if len(parts) < 3 or parts[0] != "equities" or not parts[-1].startswith("n"):
        raise ValueError(f"Unrecognized instance_id format: {instance_id}")
    date_str = parts[1]
    n_str = parts[-1].lstrip("n")
    return pd.Timestamp(date_str), int(n_str)


def eval_window_returns(
    ff49: pd.DataFrame, rebalance_date: pd.Timestamp
) -> pd.DataFrame:
    """Return the FF-49 daily returns for the calendar month following rebalance_date.

    The rebalance happens at the close of `rebalance_date`; we evaluate from
    the next trading day through the end of the following calendar month.
    """
    start = rebalance_date + pd.Timedelta(days=1)
    # End of the calendar month *after* the rebalance month
    if start.month == 12:
        end = pd.Timestamp(year=start.year + 1, month=1, day=31)
    else:
        # First day of two months after start, minus one day
        nm = start.month + 1
        ny = start.year
        if nm > 12:
            nm -= 12
            ny += 1
        end_first_of_next = pd.Timestamp(year=ny, month=nm, day=1)
        end = end_first_of_next - pd.Timedelta(days=1)
    mask = (ff49.index >= start) & (ff49.index <= end)
    return ff49.loc[mask]


def main() -> int:
    if not FF49_ZIP.exists():
        print(f"ERROR: FF-49 data file missing at {FF49_ZIP}", file=sys.stderr)
        print("(Symlink target: data/raw/equities/ → paper1/data/raw/equities/)",
              file=sys.stderr)
        return 1

    ff49 = load_ff49_daily(FF49_ZIP)
    print(f"Loaded FF-49 daily: {len(ff49)} dates, "
          f"{ff49.index.min().date()} to {ff49.index.max().date()}, "
          f"{len(ff49.columns)} industries")

    rows: list[dict] = []
    with A4_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Loaded {len(rows)} a4 rows")

    index_rows: list[dict] = []
    skipped = 0
    for r in rows:
        try:
            rebalance_date, n_val = parse_instance_id(r["instance_id"])
        except ValueError as e:
            skipped += 1
            continue
        eval_returns = eval_window_returns(ff49, rebalance_date)
        if eval_returns.empty:
            skipped += 1
            continue
        # Subset to selected_industries
        selected = r.get("selected_industries", [])
        cols_available = [c for c in selected if c in eval_returns.columns]
        if not cols_available:
            skipped += 1
            continue
        port_returns = eval_returns[cols_available].mean(axis=1, skipna=True)
        # Filename: instance_id + topology + chain_strength + objective for uniqueness
        topo = r.get("topology", "na")
        cs = r.get("chain_strength", float("nan"))
        out_name = f"{r['instance_id']}__{topo}__cs{cs}.csv"
        out_path = OUT_DIR / out_name
        port_returns.to_frame("portfolio_return").to_csv(out_path)
        index_rows.append({
            "instance_id": r["instance_id"],
            "window_idx": r.get("window_idx"),
            "N": n_val,
            "K": r.get("K"),
            "topology": topo,
            "chain_strength": cs,
            "rebalance_date": rebalance_date.date().isoformat(),
            "eval_start": eval_returns.index.min().date().isoformat(),
            "eval_end": eval_returns.index.max().date().isoformat(),
            "eval_n_days": len(eval_returns),
            "n_selected_industries": len(cols_available),
            "n_missing_from_ff49": len(selected) - len(cols_available),
            "mean_daily_return": float(port_returns.mean()),
            "std_daily_return": float(port_returns.std(ddof=1)),
            "objective_value": r.get("objective_value"),
            "csv_path": str(out_path.relative_to(REPO_ROOT)),
        })

    idx_path = REPO_ROOT / "results" / "analysis" / "daily_pnl_index.csv"
    with idx_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Wrote {idx_path} ({len(index_rows)} portfolios; skipped {skipped})")

    # Summary
    by_n = {}
    for r in index_rows:
        by_n.setdefault(r["N"], []).append(r)
    summary_lines = [
        "P&L RECONSTRUCTION — FF-49 OUT-OF-SAMPLE (Phase 3)",
        "=" * 60,
        f"Total portfolios reconstructed: {len(index_rows)}",
        f"Skipped (no eval window or unparseable id): {skipped}",
        f"Unique rebalance dates: {len({r['rebalance_date'] for r in index_rows})}",
        "",
    ]
    for n_val in sorted(by_n):
        bucket = by_n[n_val]
        means = [b["mean_daily_return"] for b in bucket]
        stds = [b["std_daily_return"] for b in bucket]
        summary_lines.append(
            f"  N={n_val:>3}  n_portfolios={len(bucket):>3}  "
            f"mean(daily_return) avg={np.mean(means):>+.5f}  "
            f"std(daily_return) avg={np.mean(stds):.5f}"
        )
    summary_path = REPO_ROOT / "results" / "analysis" / "daily_pnl_summary.txt"
    summary_path.write_text("\n".join(summary_lines))
    print()
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
