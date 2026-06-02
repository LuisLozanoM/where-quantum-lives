#!/usr/bin/env python3
"""Appendix A: Penalty robustness check A={2,4,8}.

Runs hybrid BQM at 3 penalty values on N={20,50,80,120}, families={diagonal,block,dense}.
Output: results/hybrid/penalty_robustness.jsonl
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "paper1" / "src"))

from paper1.metrics import project_to_exact_k_feasible
from paper2.datasets import generate_instance_family
from paper2.formulations import build_mvt_bqm, evaluate_mvt_objective

N_VALUES = [20, 50, 80, 120]
FAMILIES = ["diagonal", "block", "dense"]
PENALTIES = [2.0, 4.0, 8.0]
SEED = 0
K_RATIO = 0.3
RISK_AVERSION = 0.5
CONFIG_FILE = str(ROOT / "dwave.conf")


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return obj


def main():
    from paper2.solvers.dwave_hybrid_bqm import solve_hybrid_bqm

    output = ROOT / "results" / "hybrid" / "penalty_robustness.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        for family in FAMILIES:
            for N in N_VALUES:
                K = max(2, round(N * K_RATIO))
                extra = {}
                if family in ("block", "block_cross"):
                    for nb in (4, 5, 2, 3):
                        if N % nb == 0:
                            extra["num_blocks"] = nb
                            break
                inst = generate_instance_family(family, N, seed=SEED, **extra)
                labels = inst.labels

                print(f"\n--- {family} N={N} K={K} ---")

                for A in PENALTIES:
                    try:
                        problem = build_mvt_bqm(
                            inst.mu, inst.sigma, RISK_AVERSION, K, A, labels=labels,
                        )
                        r = solve_hybrid_bqm(problem, time_limit=5, config_file=CONFIG_FILE)
                        proj = project_to_exact_k_feasible(problem, r.sample, K)
                        obj = evaluate_mvt_objective(inst.mu, inst.sigma, RISK_AVERSION, proj)
                        feas_raw = int(sum(r.sample)) == K

                        row = {
                            "family": family, "N": N, "K": K,
                            "penalty": A,
                            "objective_value": obj,
                            "raw_feasible": feas_raw,
                            "raw_sum": int(sum(r.sample)),
                            "projected_sum": int(sum(proj)),
                            "wall_clock": r.wall_clock_total,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        print(f"  A={A}: obj={obj:.4f} raw_feas={feas_raw} raw_sum={int(sum(r.sample))}")

                    except Exception as e:
                        print(f"  A={A}: FAILED ({e})")

    print(f"\nDone. Output: {output}")


if __name__ == "__main__":
    main()
