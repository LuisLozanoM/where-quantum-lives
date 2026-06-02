#!/usr/bin/env python3
"""Stochastic validation: repeated hybrid CQM/BQM runs for confidence intervals.

Runs 10 reps at N={50,100,200}, families={block,dense}, budgets={5,300}s.
Output: results/hybrid/repeated_hybrid.jsonl
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
from paper2.formulations import build_mvt_bqm, build_mvt_cqm, evaluate_mvt_objective
from paper2.solvers.dwave_hybrid_bqm import solve_hybrid_bqm
from paper2.solvers.dwave_hybrid_cqm import solve_hybrid_cqm

N_VALUES = [50, 100, 200]
FAMILIES = ["block", "dense"]
TIME_LIMITS = [5, 300]
REPS = 10
K_RATIO = 0.3
RISK_AVERSION = 0.5
PENALTY = 4.0
SEED = 0
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
    output = ROOT / "results" / "hybrid" / "repeated_hybrid.jsonl"
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

                for tl in TIME_LIMITS:
                    print(f"\n--- {family} N={N} tl={tl}s ---")

                    for rep in range(REPS):
                        # CQM
                        try:
                            cqm = build_mvt_cqm(inst.mu, inst.sigma, RISK_AVERSION, K, labels=labels)
                            r = solve_hybrid_cqm(cqm, time_limit=tl, config_file=CONFIG_FILE)
                            obj = evaluate_mvt_objective(inst.mu, inst.sigma, RISK_AVERSION, r.sample)
                            row = {
                                "solver": "hybrid_cqm", "family": family, "N": N, "K": K,
                                "time_limit": tl, "rep": rep, "objective_value": obj,
                                "feasible": r.feasible,
                                "hybrid_qpu_access_time": r.hybrid_qpu_access_time,
                                "wall_clock_total": r.wall_clock_total,
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                            f.write(json.dumps(_json_safe(row)) + "\n")
                            f.flush()
                            print(f"  CQM rep={rep}: obj={obj:.4f}")
                        except Exception as e:
                            print(f"  CQM rep={rep}: FAILED ({e})")

                        # BQM
                        try:
                            problem = build_mvt_bqm(inst.mu, inst.sigma, RISK_AVERSION, K, PENALTY, labels=labels)
                            r = solve_hybrid_bqm(problem, time_limit=tl, config_file=CONFIG_FILE)
                            proj = project_to_exact_k_feasible(problem, r.sample, K)
                            obj = evaluate_mvt_objective(inst.mu, inst.sigma, RISK_AVERSION, proj)
                            row = {
                                "solver": "hybrid_bqm", "family": family, "N": N, "K": K,
                                "time_limit": tl, "rep": rep, "objective_value": obj,
                                "hybrid_qpu_access_time": r.hybrid_qpu_access_time,
                                "wall_clock_total": r.wall_clock_total,
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                            f.write(json.dumps(_json_safe(row)) + "\n")
                            f.flush()
                            print(f"  BQM rep={rep}: obj={obj:.4f}")
                        except Exception as e:
                            print(f"  BQM rep={rep}: FAILED ({e})")

    print(f"\nDone. Output: {output}")


if __name__ == "__main__":
    main()
