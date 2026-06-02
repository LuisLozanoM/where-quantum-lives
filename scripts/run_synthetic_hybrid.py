#!/usr/bin/env python3
"""Experiments A2 + A3: Hybrid BQM and CQM on synthetic instances.

A2: LeapHybridBQMSampler on N={10,20,30,50,80,120,200,400,640}
A3: LeapHybridCQMSampler on same N range (constraint-native, no penalty)

Both run on 3 density families {diagonal, block, dense}, 3 seeds.

Output: results/hybrid/a2a3_synthetic_hybrid.jsonl
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

from paper2.datasets import generate_instance_family
from paper2.formulations import build_mvt_bqm, build_mvt_cqm, evaluate_mvt_objective
from paper2.solvers.dwave_hybrid_bqm import solve_hybrid_bqm
from paper2.solvers.dwave_hybrid_cqm import solve_hybrid_cqm

# --- Configuration ---
N_VALUES = [10, 20, 30, 50, 80, 120, 200, 400, 640]
FAMILIES = ["diagonal", "block", "dense"]
SEEDS = [0, 1, 2]
K_RATIO = 0.3
RISK_AVERSION = 0.5
PENALTY = 4.0
TIME_LIMIT = 5.0  # seconds (minimum for hybrid)
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
    output_path = ROOT / "results" / "hybrid" / "a2a3_synthetic_hybrid.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0

    with open(output_path, "a") as f:
        for family in FAMILIES:
            for N in N_VALUES:
                K = max(2, round(N * K_RATIO))
                for seed in SEEDS:
                    extra_kwargs = {}
                    if family in ("block", "block_cross"):
                        for nb in (4, 5, 2, 3):
                            if N % nb == 0:
                                extra_kwargs["num_blocks"] = nb
                                break
                    instance = generate_instance_family(family, N, seed=seed, **extra_kwargs)
                    labels = instance.labels

                    print(f"\n--- {family} N={N} K={K} seed={seed} density={instance.density:.3f} ---")

                    # --- A2: Hybrid BQM ---
                    try:
                        problem = build_mvt_bqm(
                            instance.mu, instance.sigma,
                            RISK_AVERSION, K, PENALTY, labels=labels,
                        )
                        result = solve_hybrid_bqm(
                            problem, time_limit=TIME_LIMIT, config_file=CONFIG_FILE,
                        )

                        # Post-process to exact-K
                        from paper1.metrics import project_to_exact_k_feasible
                        projected = project_to_exact_k_feasible(problem, result.sample, K)
                        obj_val = evaluate_mvt_objective(
                            instance.mu, instance.sigma, RISK_AVERSION, projected,
                        )

                        row = {
                            "experiment": "A2",
                            "solver": "hybrid_bqm",
                            "family": family,
                            "N": N,
                            "K": K,
                            "seed": seed,
                            "density": instance.density,
                            "raw_sample_sum": int(sum(result.sample)),
                            "projected_sample_sum": int(sum(projected)),
                            "raw_energy": result.energy,
                            "objective_value": obj_val,
                            "wall_clock_total": result.wall_clock_total,
                            "hybrid_run_time": result.hybrid_run_time,
                            "hybrid_charge_time": result.hybrid_charge_time,
                            "hybrid_qpu_access_time": result.hybrid_qpu_access_time,
                            "qpu_contributed": result.extras.get("qpu_contributed"),
                            "time_limit": TIME_LIMIT,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1
                        print(f"  A2 BQM: obj={obj_val:.4f} sum={int(sum(projected))} qpu={result.extras.get('qpu_contributed')}")

                    except Exception as e:
                        print(f"  A2 BQM: FAILED ({e})")
                        row = {
                            "experiment": "A2",
                            "solver": "hybrid_bqm",
                            "family": family,
                            "N": N,
                            "K": K,
                            "seed": seed,
                            "error": str(e),
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1

                    # --- A3: Hybrid CQM ---
                    try:
                        cqm = build_mvt_cqm(
                            instance.mu, instance.sigma,
                            RISK_AVERSION, K, labels=labels,
                        )
                        result = solve_hybrid_cqm(
                            cqm, time_limit=TIME_LIMIT, config_file=CONFIG_FILE,
                        )

                        obj_val = evaluate_mvt_objective(
                            instance.mu, instance.sigma, RISK_AVERSION, result.sample,
                        )

                        row = {
                            "experiment": "A3",
                            "solver": "hybrid_cqm",
                            "family": family,
                            "N": N,
                            "K": K,
                            "seed": seed,
                            "density": instance.density,
                            "sample_sum": int(sum(result.sample)),
                            "feasible": result.feasible,
                            "energy": result.energy,
                            "objective_value": obj_val,
                            "wall_clock_total": result.wall_clock_total,
                            "hybrid_run_time": result.hybrid_run_time,
                            "hybrid_charge_time": result.hybrid_charge_time,
                            "hybrid_qpu_access_time": result.hybrid_qpu_access_time,
                            "qpu_contributed": result.extras.get("qpu_contributed"),
                            "time_limit": TIME_LIMIT,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1
                        print(f"  A3 CQM: obj={obj_val:.4f} sum={int(sum(result.sample))} feasible={result.feasible} qpu={result.extras.get('qpu_contributed')}")

                    except Exception as e:
                        print(f"  A3 CQM: FAILED ({e})")
                        row = {
                            "experiment": "A3",
                            "solver": "hybrid_cqm",
                            "family": family,
                            "N": N,
                            "K": K,
                            "seed": seed,
                            "error": str(e),
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1

    print(f"\nDone. Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
