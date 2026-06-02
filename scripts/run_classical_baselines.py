#!/usr/bin/env python3
"""Classical baselines: Gurobi MIQP + neal SA on synthetic instances.

Same grid as A2/A3: N={10,20,30,50,80,120,200,400,640}, 3 families, 3 seeds.
Gurobi solves to optimality (gold standard). Neal SA is a heuristic baseline.

Output: results/classical/d1d2_classical_baselines.jsonl
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "paper1" / "src"))

from paper2.datasets import generate_instance_family
from paper2.formulations import build_mvt_bqm, build_mvt_objective, evaluate_mvt_objective

# --- Configuration ---
N_VALUES = [10, 20, 30, 50, 80, 120, 200, 400, 640]
FAMILIES = ["diagonal", "block", "dense"]
SEEDS = [0, 1, 2]
K_RATIO = 0.3
RISK_AVERSION = 0.5
PENALTY = 4.0
SA_READS = 1000
SA_SWEEPS = 1000


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


def solve_gurobi(mu, sigma, risk_aversion, K, labels):
    """Solve MVT as native MIQP with Gurobi. Returns (sample, objective, wall_clock, status)."""
    import gurobipy as gp

    N = len(mu)
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("mvt", env=env)
    m.setParam("TimeLimit", 300)

    z = m.addVars(N, vtype=gp.GRB.BINARY, name="z")

    # Objective: min -mu^T z + lambda z^T Sigma z
    obj = gp.QuadExpr()
    for i in range(N):
        obj += -mu[i] * z[i]
    for i in range(N):
        for j in range(N):
            obj += risk_aversion * sigma[i, j] * z[i] * z[j]
    m.setObjective(obj, gp.GRB.MINIMIZE)

    # Cardinality constraint: sum(z) == K
    m.addConstr(gp.quicksum(z[i] for i in range(N)) == K, "cardinality")

    start = perf_counter()
    m.optimize()
    wall = perf_counter() - start

    if m.status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL):
        sample = np.array([int(round(z[i].X)) for i in range(N)], dtype=int)
        obj_val = evaluate_mvt_objective(mu, sigma, risk_aversion, sample)
        gap = m.MIPGap
        status = "optimal" if m.status == gp.GRB.OPTIMAL else "suboptimal"
    else:
        sample = np.zeros(N, dtype=int)
        obj_val = float("inf")
        gap = None
        status = f"status_{m.status}"

    env.dispose()
    return sample, obj_val, wall, status, gap


def solve_neal_sa(problem, K, mu, sigma, risk_aversion):
    """Solve with D-Wave neal simulated annealing. Returns (sample, objective, wall_clock)."""
    from dwave.samplers import SimulatedAnnealingSampler
    from paper1.formulations import to_dimod_bqm
    from paper1.metrics import project_to_exact_k_feasible

    sampler = SimulatedAnnealingSampler()
    bqm = to_dimod_bqm(problem)

    start = perf_counter()
    sampleset = sampler.sample(bqm, num_reads=SA_READS, num_sweeps=SA_SWEEPS)
    wall = perf_counter() - start

    # Get best sample and project to exact-K
    best_idx = int(np.argmin(sampleset.record.energy))
    record = sampleset.record[best_idx]
    sample_dict = {var: int(record.sample[i]) for i, var in enumerate(sampleset.variables)}
    sample = np.array([sample_dict.get(i, 0) for i in range(problem.size)], dtype=int)

    projected = project_to_exact_k_feasible(problem, sample, K)
    obj_val = evaluate_mvt_objective(mu, sigma, risk_aversion, projected)

    return projected, obj_val, wall


def main() -> None:
    output_path = ROOT / "results" / "classical" / "d1d2_classical_baselines.jsonl"
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

                    print(f"\n--- {family} N={N} K={K} seed={seed} ---")

                    # --- D1: Gurobi MIQP ---
                    try:
                        sample, obj_val, wall, status, gap = solve_gurobi(
                            instance.mu, instance.sigma, RISK_AVERSION, K, instance.labels,
                        )
                        row = {
                            "experiment": "D1",
                            "solver": "gurobi_miqp",
                            "family": family,
                            "N": N, "K": K, "seed": seed,
                            "density": instance.density,
                            "objective_value": obj_val,
                            "sample_sum": int(sum(sample)),
                            "wall_clock": wall,
                            "status": status,
                            "mip_gap": gap,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1
                        print(f"  Gurobi: obj={obj_val:.4f} wall={wall:.3f}s status={status}")
                    except Exception as e:
                        print(f"  Gurobi: FAILED ({e})")
                        row = {
                            "experiment": "D1", "solver": "gurobi_miqp",
                            "family": family, "N": N, "K": K, "seed": seed,
                            "error": str(e),
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1

                    # --- D2: Neal SA ---
                    try:
                        problem = build_mvt_bqm(
                            instance.mu, instance.sigma,
                            RISK_AVERSION, K, PENALTY,
                        )
                        sample, obj_val, wall = solve_neal_sa(
                            problem, K, instance.mu, instance.sigma, RISK_AVERSION,
                        )
                        row = {
                            "experiment": "D2",
                            "solver": "neal_sa",
                            "family": family,
                            "N": N, "K": K, "seed": seed,
                            "density": instance.density,
                            "objective_value": obj_val,
                            "sample_sum": int(sum(sample)),
                            "wall_clock": wall,
                            "num_reads": SA_READS,
                            "num_sweeps": SA_SWEEPS,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1
                        print(f"  Neal SA: obj={obj_val:.4f} wall={wall:.3f}s")
                    except Exception as e:
                        print(f"  Neal SA: FAILED ({e})")
                        row = {
                            "experiment": "D2", "solver": "neal_sa",
                            "family": family, "N": N, "K": K, "seed": seed,
                            "error": str(e),
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1

    print(f"\nDone. Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
