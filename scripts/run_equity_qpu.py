#!/usr/bin/env python3
"""Experiment A4: Direct QPU on FF49 equity subsets.

Runs on N={10,20,30} industry subsets from FF49 rolling windows,
K=round(N*0.3), on Pegasus + Zephyr with 3 chain strengths.

Output: results/qpu/a4_equity_qpu.jsonl
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

from paper1.datasets import (
    RAW_EQUITIES_DIR,
    build_equity_rolling_instances,
    load_kenneth_french_49_daily,
    subset_portfolio_instance,
)
from paper1.embedding import logical_graph_from_qubo
from paper1.metrics import project_to_exact_k_feasible
from paper1.qpu import get_live_sampler

from paper2.formulations import build_mvt_bqm, evaluate_mvt_objective
from paper2.solvers.dwave_qpu import solve_qpu_forward
from paper2.solvers.embedding_cache import EmbeddingCache

# --- Configuration ---
N_SUBSETS = [10, 20, 30]
NUM_WINDOWS = 5  # evenly spaced rolling windows
RISK_AVERSION = 0.5
PENALTY = 4.0
K_RATIO = 0.3
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
    output_path = ROOT / "results" / "qpu" / "a4_equity_qpu.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading FF49 data...")
    equity_returns = load_kenneth_french_49_daily(
        RAW_EQUITIES_DIR / "49_Industry_Portfolios_Daily.zip"
    )
    all_windows = build_equity_rolling_instances(
        equity_returns, estimation_days=252, rebalance_frequency="M"
    )
    # Select evenly spaced windows
    indices = np.linspace(0, len(all_windows) - 1, NUM_WINDOWS, dtype=int)
    windows = [all_windows[i] for i in indices]
    print(f"Selected {len(windows)} windows from {len(all_windows)} total")

    print("Connecting to solvers...")
    samplers = {}
    for topo, solver_name in SOLVERS.items():
        try:
            s = get_live_sampler(solver=solver_name, config_file=CONFIG_FILE)
            samplers[topo] = s
            print(f"  {topo}: {s.properties.get('chip_id')}")
        except Exception as e:
            print(f"  {topo}: FAILED ({e})")

    cache = EmbeddingCache()
    rows_written = 0

    with open(output_path, "a") as f:
        for w_idx, window in enumerate(windows):
            for N in N_SUBSETS:
                K = max(2, round(N * K_RATIO))
                subset = subset_portfolio_instance(window, N, method="top_abs_signal")
                labels = subset.labels
                problem = build_mvt_bqm(
                    subset.mu, subset.sigma,
                    RISK_AVERSION, K, PENALTY, labels=labels,
                )
                logical = logical_graph_from_qubo(problem)
                n_edges = logical.number_of_edges()

                print(f"\n--- window={w_idx} ({subset.instance_id}) N={N} K={K} edges={n_edges} ---")

                for topo, sampler in samplers.items():
                    embedding = cache.get_or_find(
                        problem, sampler, topo,
                        seeds=EMBEDDING_SEEDS, tries=EMBEDDING_TRIES,
                    )
                    if embedding is None:
                        print(f"  {topo}: embedding FAILED")
                        row = {
                            "experiment": "A4",
                            "instance_id": subset.instance_id,
                            "window_idx": w_idx,
                            "N": N, "K": K,
                            "topology": topo,
                            "embedding_success": False,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                        f.write(json.dumps(_json_safe(row)) + "\n")
                        f.flush()
                        rows_written += 1
                        continue

                    for cs in CHAIN_STRENGTHS:
                        try:
                            result = solve_qpu_forward(
                                problem, embedding, sampler=sampler,
                                reads=READS, chain_strength=cs, anneal_time=ANNEAL_TIME,
                            )
                            projected = project_to_exact_k_feasible(problem, result.sample, K)
                            obj_val = evaluate_mvt_objective(
                                subset.mu, subset.sigma, RISK_AVERSION, projected,
                            )

                            row = {
                                "experiment": "A4",
                                "instance_id": subset.instance_id,
                                "window_idx": w_idx,
                                "N": N, "K": K,
                                "topology": topo,
                                "chain_strength": cs,
                                "embedding_success": True,
                                "selected_industries": [labels[i] for i in np.where(projected == 1)[0]],
                                "objective_value": obj_val,
                                "chain_break_fraction": result.chain_break_fraction,
                                "embedded_qubits": result.embedded_qubits,
                                "chain_length_mean": result.chain_length_mean,
                                "qpu_access_time": result.qpu_access_time,
                                "wall_clock_total": result.wall_clock_total,
                                "timestamp": datetime.now(UTC).isoformat(),
                            }
                            f.write(json.dumps(_json_safe(row)) + "\n")
                            f.flush()
                            rows_written += 1
                            print(f"  {topo} cs={cs}: obj={obj_val:.4f} cbf={result.chain_break_fraction:.3f}")

                        except Exception as e:
                            print(f"  {topo} cs={cs}: FAILED ({e})")

    for sampler in samplers.values():
        try:
            sampler.client.close()
        except Exception:
            pass

    print(f"\nDone. Wrote {rows_written} rows to {output_path}")


if __name__ == "__main__":
    main()
