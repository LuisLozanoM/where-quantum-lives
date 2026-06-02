#!/usr/bin/env python3
"""Experiment B2: Hybrid budget sweep.

For N={50,100,200,400}, runs hybrid BQM + CQM with time_limit={5,30,60,180,300}s.
Maps quality-vs-time frontier for hybrid solvers.

Output: results/hybrid/b2_budget_sweep.jsonl
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

# --- Configuration ---
N_VALUES = [50, 100, 200, 400]
TIME_LIMITS = [5, 30, 60, 180, 300]
FAMILIES = ["block", "dense"]  # most interesting for budget sweep
SEED = 0
K_RATIO = 0.3
RISK_AVERSION = 0.5
PENALTY = 4.0
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


def main() -> None:
    output_path = ROOT / "results" / "hybrid" / "b2_budget_sweep.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0

    with open(output_path, "a") as f:
        for family in FAMILIES:
            for N in N_VALUES:
                K = max(2, round(N * K_RATIO))
                extra_kwargs = {}
                if family in ("block", "block_cross"):
                    for nb in (4, 5, 2, 3):
                        if N % nb == 0:
                            extra_kwargs["num_blocks"] = nb
                            break
                instance = generate_instance_family(family, N, seed=SEED, **extra_kwargs)
                labels = instance.labels

                print(f"\n--- {family} N={N} K={K} density={instance.density:.3f} ---")

                for tl in TIME_LIMITS:
                    # Hybrid BQM
                    try:
                        problem = build_mvt_bqm(
                            instance.mu, instance.sigma,
                            RISK_AVERSION, K, PENALTY, labels=labels,
                        )
                        r = solve_hybrid_bqm(problem, time_limit=tl, config_file=CONFIG_FILE)
                        projected = project_to_exact_k_feasible(problem, r.sample, K)
                        obj_val = evaluate_mvt_objective(
                            instance.mu, instance.sigma, RISK_AVERSION, projected,
                        )

                        row = {
                            "experiment": "B2",
                            "solver": "hybrid_bqm",
                            "family": family,
                            "N": N, "K": K,
                            "time_limit": tl,
                            "density": instance.density,
                            "objective_value": obj_val,
                            "wall_clock_total": r.wall_clock_total,
                            "hybrid_run_time": r.hybrid_run_time,
                            "hybrid_qpu_access_time": r.hybrid_qpu_access_time,
                            "qpu_contributed": r.extras.get("qpu_contributed"),
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1
                        print(f"  BQM tl={tl}s: obj={obj_val:.4f} wall={r.wall_clock_total:.1f}s")

                    except Exception as e:
                        print(f"  BQM tl={tl}s: FAILED ({e})")

                    # Hybrid CQM
                    try:
                        cqm = build_mvt_cqm(
                            instance.mu, instance.sigma,
                            RISK_AVERSION, K, labels=labels,
                        )
                        r = solve_hybrid_cqm(cqm, time_limit=tl, config_file=CONFIG_FILE)
                        obj_val = evaluate_mvt_objective(
                            instance.mu, instance.sigma, RISK_AVERSION, r.sample,
                        )

                        row = {
                            "experiment": "B2",
                            "solver": "hybrid_cqm",
                            "family": family,
                            "N": N, "K": K,
                            "time_limit": tl,
                            "density": instance.density,
                            "objective_value": obj_val,
                            "feasible": r.feasible,
                            "wall_clock_total": r.wall_clock_total,
                            "hybrid_run_time": r.hybrid_run_time,
                            "hybrid_qpu_access_time": r.hybrid_qpu_access_time,
                            "qpu_contributed": r.extras.get("qpu_contributed"),
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1
                        print(f"  CQM tl={tl}s: obj={obj_val:.4f} feasible={r.feasible} wall={r.wall_clock_total:.1f}s")

                    except Exception as e:
                        print(f"  CQM tl={tl}s: FAILED ({e})")

    print(f"\nDone. Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
