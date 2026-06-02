"""
wilcoxon_cqm_vs_bqm.py
======================

Phase 0 close — paired Wilcoxon signed-rank test of CQM vs BQM objective
values at each N, on the synthetic-hybrid runs in
`results/hybrid/a2a3_synthetic_hybrid.jsonl`.

Used to back the Table 2 caption claim that a Wilcoxon signed-rank test
rejects equality of paired CQM and BQM objectives at every N >= 20
(p < 0.05).

Pairing: by (family, seed) at each N.  There are 9 instances per N
(3 families × 3 seeds).

Output: results/analysis/wilcoxon_cqm_vs_bqm.csv
        results/analysis/wilcoxon_cqm_vs_bqm_summary.txt
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
A2A3_PATH = REPO_ROOT / "results" / "hybrid" / "a2a3_synthetic_hybrid.jsonl"
OUT_DIR = REPO_ROOT / "results" / "analysis"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(A2A3_PATH)

    # Build pairs indexed by (N, family, seed)
    by_key: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["N"], r["family"], r["seed"])
        by_key[key][r["solver"]] = r["objective_value"]

    # Group by N (we only need the first key element, but keep all for stability)
    pairs_by_n: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for key_tuple, pair in by_key.items():
        n_val = key_tuple[0]
        if "hybrid_cqm" in pair and "hybrid_bqm" in pair:
            pairs_by_n[n_val].append((pair["hybrid_cqm"], pair["hybrid_bqm"]))

    results: list[dict] = []
    for n_val in sorted(pairs_by_n):
        pairs = pairs_by_n[n_val]
        cqm = np.array([p[0] for p in pairs])
        bqm = np.array([p[1] for p in pairs])
        diff = cqm - bqm  # negative = CQM better
        n_strict_wins = int(np.sum(diff < -1e-6))
        n_ties = int(np.sum(np.abs(diff) <= 1e-6))
        n_bqm_wins = int(np.sum(diff > 1e-6))

        # Wilcoxon requires non-zero differences (or zero_method handling)
        nonzero = diff[np.abs(diff) > 1e-12]
        if len(nonzero) < 1:
            p_val = float("nan")
            stat = float("nan")
        else:
            res = cast(Any, wilcoxon(nonzero, alternative="less", zero_method="wilcox"))
            stat = float(getattr(res, "statistic", res[0]))
            p_val = float(getattr(res, "pvalue", res[1]))

        results.append({
            "N": n_val,
            "n_pairs": len(pairs),
            "cqm_strict_wins": n_strict_wins,
            "ties": n_ties,
            "bqm_strict_wins": n_bqm_wins,
            "mean_diff_cqm_minus_bqm": float(np.mean(diff)),
            "median_diff": float(np.median(diff)),
            "wilcoxon_statistic": stat,
            "wilcoxon_p_value": p_val,
            "rejects_at_p005": bool(p_val < 0.05),
        })

    # CSV
    import csv
    csv_path = OUT_DIR / "wilcoxon_cqm_vs_bqm.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # Summary
    lines = [
        "PAIRED WILCOXON SIGNED-RANK TEST — CQM vs BQM (one-sided: CQM < BQM)",
        "=" * 70,
    ]
    for r in results:
        lines.append(
            f"  N={r['N']:>3}  pairs={r['n_pairs']:>2}  "
            f"strict CQM={r['cqm_strict_wins']}/{r['n_pairs']}  "
            f"ties={r['ties']}  BQM={r['bqm_strict_wins']}  "
            f"median(CQM-BQM)={r['median_diff']:>+.4f}  "
            f"p={r['wilcoxon_p_value']:.4e}  "
            f"reject_p005={'YES' if r['rejects_at_p005'] else 'no'}"
        )
    rejected_at = [r['N'] for r in results if r['rejects_at_p005']]
    not_rejected = [r['N'] for r in results if not r['rejects_at_p005']]
    lines += [
        "",
        f"Rejects H_0 (CQM ≥ BQM) at p<0.05 at N values: {rejected_at}",
        f"Does NOT reject at:                            {not_rejected}",
    ]
    summary = "\n".join(lines)
    (OUT_DIR / "wilcoxon_cqm_vs_bqm_summary.txt").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
