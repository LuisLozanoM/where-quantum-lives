#!/usr/bin/env python3
"""Experiment A1: Direct QPU forward anneal on synthetic instances.

Runs across N={10,20,30,50,80}, 3 density families {diagonal, block, dense},
3 seeds, on both Pegasus and Zephyr. Records SolverResult + embedding stats.

Also runs T2 (penalty calibration) and T4 (embedding stress) implicitly:
embedding success/failure and chain stats are captured per run.

Output: results/qpu/a1_synthetic_qpu.jsonl
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "paper1" / "src"))

from paper1.embedding import find_embedding_for_graph, logical_graph_from_qubo
from paper1.metrics import project_to_exact_k_feasible
from paper1.qpu import get_live_sampler, sampler_working_graph

from paper2.datasets import generate_instance_family
from paper2.formulations import build_mvt_bqm, evaluate_mvt_objective
from paper2.solvers.dwave_qpu import solve_qpu_forward
from paper2.solvers.embedding_cache import EmbeddingCache

# --- Configuration ---
N_VALUES = [10, 20, 30, 50, 80]
FAMILIES = ["diagonal", "block", "dense"]
SEEDS = [0, 1, 2]
K_RATIO = 0.3  # K = round(N * K_RATIO)
RISK_AVERSION = 0.5
PENALTY = 4.0
READS = 1000
ANNEAL_TIME = 20.0
CHAIN_STRENGTHS = [0.5, 1.0, 2.0]
EMBEDDING_SEEDS = (0, 1, 2, 3, 4)
EMBEDDING_TRIES = 20

SOLVERS = {
    "pegasus": "Advantage_system4.1",
    "zephyr": "Advantage2_system1",  # was Advantage2_system1.13 at time of original runs (D-Wave rename 2026-04-10)
}
CONFIG_FILE = str(ROOT / "dwave.conf")


def _json_safe(obj):
    """Make an object JSON-serializable."""
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
    output_path = ROOT / "results" / "qpu" / "a1_synthetic_qpu.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Connecting to solvers...")
    samplers = {}
    for topo, solver_name in SOLVERS.items():
        try:
            s = get_live_sampler(solver=solver_name, config_file=CONFIG_FILE)
            samplers[topo] = s
            props = s.properties
            print(f"  {topo}: {props.get('chip_id')}, {props.get('num_qubits')} qubits")
        except Exception as e:
            print(f"  {topo}: FAILED ({e})")

    if not samplers:
        print("No solvers available. Exiting.")
        return

    cache = EmbeddingCache()
    rows_written = 0

    with open(output_path, "a") as f:
        for family in FAMILIES:
            for N in N_VALUES:
                K = max(2, round(N * K_RATIO))
                for seed in SEEDS:
                    # Auto-select num_blocks that divides N for block families
                    extra_kwargs = {}
                    if family in ("block", "block_cross"):
                        for nb in (4, 5, 2, 3):
                            if N % nb == 0:
                                extra_kwargs["num_blocks"] = nb
                                break
                    instance = generate_instance_family(family, N, seed=seed, **extra_kwargs)
                    problem = build_mvt_bqm(
                        instance.mu, instance.sigma,
                        RISK_AVERSION, K, PENALTY,
                        labels=instance.labels,
                    )
                    logical = logical_graph_from_qubo(problem)
                    n_edges = logical.number_of_edges()

                    print(f"\n--- {family} N={N} K={K} seed={seed} edges={n_edges} ---")

                    for topo, sampler in samplers.items():
                        # Find embedding
                        embedding = cache.get_or_find(
                            problem, sampler, topo,
                            seeds=EMBEDDING_SEEDS, tries=EMBEDDING_TRIES,
                        )
                        if embedding is None:
                            print(f"  {topo}: embedding FAILED")
                            row = {
                                "experiment": "A1",
                                "family": family,
                                "N": N,
                                "K": K,
                                "seed": seed,
                                "topology": topo,
                                "embedding_success": False,
                                "n_logical_edges": n_edges,
                                "density": instance.density,
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                            f.write(json.dumps(_json_safe(row)) + "\n")
                            f.flush()
                            rows_written += 1
                            continue

                        chain_lengths = [len(c) for c in embedding.values()]
                        print(f"  {topo}: embedded, mean_chain={np.mean(chain_lengths):.1f}, max_chain={max(chain_lengths)}")

                        # Run at each chain strength
                        for cs in CHAIN_STRENGTHS:
                            try:
                                result = solve_qpu_forward(
                                    problem, embedding,
                                    sampler=sampler,
                                    reads=READS,
                                    chain_strength=cs,
                                    anneal_time=ANNEAL_TIME,
                                )

                                # Post-process to exact-K
                                projected = project_to_exact_k_feasible(problem, result.sample, K)
                                proj_energy = float(problem.energy(projected))
                                obj_val = evaluate_mvt_objective(
                                    instance.mu, instance.sigma, RISK_AVERSION, projected,
                                )

                                row = {
                                    "experiment": "A1",
                                    "family": family,
                                    "N": N,
                                    "K": K,
                                    "seed": seed,
                                    "topology": topo,
                                    "chain_strength": cs,
                                    "embedding_success": True,
                                    "n_logical_edges": n_edges,
                                    "density": instance.density,
                                    "raw_sample_sum": int(sum(result.sample)),
                                    "raw_energy": result.energy,
                                    "projected_sample_sum": int(sum(projected)),
                                    "projected_energy": proj_energy,
                                    "objective_value": obj_val,
                                    "wall_clock_total": result.wall_clock_total,
                                    "qpu_access_time": result.qpu_access_time,
                                    "qpu_programming_time": result.qpu_programming_time,
                                    "qpu_sampling_time": result.qpu_sampling_time,
                                    "chain_break_fraction": result.chain_break_fraction,
                                    "embedded_qubits": result.embedded_qubits,
                                    "chain_length_mean": result.chain_length_mean,
                                    "chain_length_max": result.chain_length_max,
                                    "num_reads": READS,
                                    "anneal_time": ANNEAL_TIME,
                                    "timestamp": datetime.now(UTC).isoformat(),
                                }
                                f.write(json.dumps(_json_safe(row)) + "\n")
                                f.flush()
                                rows_written += 1

                                print(f"    cs={cs}: obj={obj_val:.4f} cbf={result.chain_break_fraction:.3f} sum={int(sum(projected))}")

                            except Exception as e:
                                print(f"    cs={cs}: FAILED ({e})")
                                row = {
                                    "experiment": "A1",
                                    "family": family,
                                    "N": N,
                                    "K": K,
                                    "seed": seed,
                                    "topology": topo,
                                    "chain_strength": cs,
                                    "embedding_success": True,
                                    "error": str(e),
                                    "timestamp": datetime.now(UTC).isoformat(),
                                }
                                f.write(json.dumps(_json_safe(row)) + "\n")
                                f.flush()
                                rows_written += 1

    # Close samplers
    for sampler in samplers.values():
        try:
            sampler.client.close()
        except Exception:
            pass

    print(f"\nDone. Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
