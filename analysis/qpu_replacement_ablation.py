"""
qpu_replacement_ablation.py
============================

Phase 4.5 — CPU-only counterfactual to bound the causal contribution of
the QPU to the LeapHybridCQM/BQM solver quality.

Motivation: a reviewer can reasonably demand an ablation showing the QPU's
causal contribution to solution quality.  Without live QPU access we cannot
run a true LeapHybridCQM-internal ablation, but a CPU-only counterfactual on
the same penalty-encoded BQM at matched wall-clock provides an upper bound
on the share of solution quality attributable to quantum sampling.

What we run:
    - A strong classical heuristic (TabuSampler) on the full penalty-encoded
      QUBO at matched wall-clock (5 s).
    - Compared to the actual hybrid CQM and hybrid BQM objectives at 5 s.

What we CANNOT run (proprietary):
    - The exact LeapHybridCQM decomposer with the QPU sub-solver swapped out.
      D-Wave's hybrid pipeline internals are not public.

Interpretation:
    - If TabuSampler at 5 s wall-clock matches or beats the hybrid path,
      the hybrid's quantum contribution to *solution quality* cannot exceed
      this gap.  Combined with the 0.7% QPU dwell fraction (§5.5), this
      brackets the causal-vs-time-share question.
    - If TabuSampler underperforms hybrid by a large margin, the hybrid
      pipeline (decomposer + projector + small QPU contribution) is doing
      non-trivial work that the classical solver alone cannot match.

Grid: N ∈ {50, 80, 120, 200}, family ∈ {block, dense}, seed ∈ {0, 1, 2}
      = 24 ablation instances (mirrors the a2a3 grid for direct
      head-to-head comparison).

Output:
    results/analysis/qpu_replacement_ablation.csv
    results/analysis/qpu_replacement_ablation_summary.txt
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT.parent / "paper1" / "src"))

import dimod
import paper2.datasets as d2
import paper2.formulations as fm
from dwave.samplers import TabuSampler

LAMBDA_RISK = 0.5
PENALTY = 4.0
TIMEOUT_MS = 5000  # 5 seconds wall-clock per instance


def make_instance(N: int, family: str, seed: int):
    # Match run_synthetic_hybrid.py's num_blocks choice for block families:
    # try (4, 5, 2, 3) and pick the first that divides N.
    extra_kwargs = {}
    if family in ("block", "block_cross"):
        for nb in (4, 5, 2, 3):
            if N % nb == 0:
                extra_kwargs["num_blocks"] = nb
                break
    inst = d2.generate_instance_family(family, N=N, seed=seed, **extra_kwargs)
    K = max(2, round(N * 0.3))
    return inst, K


def qubo_problem_to_bqm(qubo) -> dimod.BinaryQuadraticModel:
    """Convert paper2.formulations.QuboProblem (symmetric matrix + offset) to a
    dimod BinaryQuadraticModel.

    Under the symmetric convention H = z^T Q z = sum_i Q_ii z_i +
    sum_{i<j} (Q_ij + Q_ji) z_i z_j; since Q is symmetric, the off-diagonal
    contribution to the unordered z_i z_j coefficient is 2*Q[i,j].
    """
    Q = qubo.matrix
    n = Q.shape[0]
    linear = {i: float(Q[i, i]) for i in range(n)}
    quadratic: dict[tuple, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            v = float(Q[i, j] + Q[j, i])
            if v != 0.0:
                quadratic[(i, j)] = v
    return dimod.BinaryQuadraticModel(linear, quadratic,
                                       offset=float(qubo.offset),
                                       vartype="BINARY")


def project_to_exact_K(sample_dict: dict, K: int, mu: np.ndarray,
                       variable_order: list) -> np.ndarray:
    """Greedy projection: take the top-K variables by (sample_val + tiebreak score)."""
    n = len(variable_order)
    arr = np.zeros(n, dtype=int)
    for var, val in sample_dict.items():
        idx = variable_order.index(var) if not isinstance(var, int) else int(var)
        arr[idx] = int(val)
    selected_idx = np.where(arr == 1)[0]
    if len(selected_idx) == K:
        return arr
    if len(selected_idx) > K:
        # Keep top-K by mu (drop lowest-mu among selected)
        scores = mu[selected_idx]
        keep = selected_idx[np.argsort(-scores)[:K]]
        out = np.zeros(n, dtype=int)
        out[keep] = 1
        return out
    # Too few selected: add top-mu among unselected until we hit K
    unselected = np.where(arr == 0)[0]
    add_count = K - len(selected_idx)
    scores = mu[unselected]
    add = unselected[np.argsort(-scores)[:add_count]]
    out = arr.copy()
    out[add] = 1
    return out


def run_ablation_instance(N: int, family: str, seed: int):
    inst, K = make_instance(N, family, seed)
    mu = inst.mu.astype(float)
    sigma = inst.sigma.astype(float)

    qubo = fm.build_mvt_bqm(
        mu=mu,
        sigma=sigma,
        risk_aversion=LAMBDA_RISK,
        cardinality=K,
        penalty=PENALTY,
    )
    bqm = qubo_problem_to_bqm(qubo)
    variable_order = list(bqm.variables)

    t0 = time.perf_counter()
    sampler = TabuSampler()
    ss = sampler.sample(bqm, timeout=TIMEOUT_MS, num_reads=1)
    t_wall = time.perf_counter() - t0
    # SampleSet.first is a namedtuple-like (sample, energy, num_occurrences, ...)
    best_sample = ss.first.sample
    best_energy = float(ss.first.energy)
    z_raw = np.zeros(N, dtype=int)
    for var, val in best_sample.items():
        idx = variable_order.index(var) if not isinstance(var, int) else int(var)
        z_raw[idx] = int(val)
    raw_sum = int(z_raw.sum())
    raw_obj = fm.evaluate_mvt_objective(mu, sigma, LAMBDA_RISK, z_raw)
    z_proj = project_to_exact_K(best_sample, K, mu, variable_order)
    proj_sum = int(z_proj.sum())
    proj_obj = fm.evaluate_mvt_objective(mu, sigma, LAMBDA_RISK, z_proj)

    return {
        "N": N,
        "family": family,
        "seed": seed,
        "K": K,
        "raw_energy": best_energy,
        "raw_obj": float(raw_obj),
        "raw_sum": raw_sum,
        "raw_feasible": raw_sum == K,
        "projected_obj": float(proj_obj),
        "projected_sum": proj_sum,
        "wall_clock_s": float(t_wall),
        "timeout_ms": TIMEOUT_MS,
    }


def load_hybrid_baseline() -> dict[tuple, dict[str, float]]:
    """Map (N, family, seed) → {hybrid_bqm_obj, hybrid_cqm_obj}."""
    path = REPO_ROOT / "results" / "hybrid" / "a2a3_synthetic_hybrid.jsonl"
    out: dict[tuple, dict[str, float]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["N"], r["family"], r["seed"])
            out.setdefault(key, {})[r["solver"]] = float(r["objective_value"])
    return out


def load_gurobi_baseline() -> dict[tuple, float]:
    path = REPO_ROOT / "results" / "classical" / "d1d2_classical_baselines.jsonl"
    out: dict[tuple, float] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("solver") != "gurobi_miqp":
                continue
            if r.get("status") != "optimal":
                continue
            key = (r["N"], r["family"], r["seed"])
            out[key] = float(r["objective_value"])
    return out


def main() -> int:
    out_dir = REPO_ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    hybrid = load_hybrid_baseline()
    gurobi = load_gurobi_baseline()

    grid = [(N, fam, seed)
            for N in (50, 80, 120, 200)
            for fam in ("block", "dense")
            for seed in (0, 1, 2)]

    rows: list[dict] = []
    for (N, fam, seed) in grid:
        print(f"  running N={N:>3} family={fam:<6} seed={seed} ... ", end="", flush=True)
        try:
            r = run_ablation_instance(N, fam, seed)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        # Attach baselines
        key = (N, fam, seed)
        r["hybrid_bqm_obj"] = hybrid.get(key, {}).get("hybrid_bqm", float("nan"))
        r["hybrid_cqm_obj"] = hybrid.get(key, {}).get("hybrid_cqm", float("nan"))
        r["gurobi_obj"] = gurobi.get(key, float("nan"))
        r["tabu_minus_hybrid_bqm"] = r["projected_obj"] - r["hybrid_bqm_obj"]
        r["tabu_minus_hybrid_cqm"] = r["projected_obj"] - r["hybrid_cqm_obj"]
        r["tabu_minus_gurobi"] = r["projected_obj"] - r["gurobi_obj"]
        rows.append(r)
        print(f"obj={r['projected_obj']:+.4f}  vs CQM={r['hybrid_cqm_obj']:+.4f}  "
              f"vs BQM={r['hybrid_bqm_obj']:+.4f}  vs Gurobi={r['gurobi_obj']:+.4f}  "
              f"wall={r['wall_clock_s']:.2f}s")

    csv_path = out_dir / "qpu_replacement_ablation.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows)")

    # Summary
    summary_lines = [
        "QPU-REPLACEMENT ABLATION (Tabu at matched 5s) — PHASE 4.5",
        "=" * 70,
        f"Instances: {len(rows)} (N ∈ {{50,80,120,200}} × family ∈ {{block,dense}} × seed ∈ {{0,1,2}})",
        f"Classical solver: TabuSampler (dwave.samplers) with timeout={TIMEOUT_MS} ms",
        f"Comparison anchors: hybrid_cqm, hybrid_bqm (both at 5 s budget), Gurobi optimal",
        "",
        "PER-(N, family) MEANS:",
        f"  {'N':>3}  {'family':<6}  {'Tabu':>10}  {'CQM':>10}  {'BQM':>10}  {'Gurobi':>10}  {'Tabu-CQM':>10}",
    ]
    from collections import defaultdict
    cells = defaultdict(list)
    for r in rows:
        cells[(r["N"], r["family"])].append(r)
    for key in sorted(cells):
        bucket = cells[key]
        t = np.mean([r["projected_obj"] for r in bucket])
        c = np.nanmean([r["hybrid_cqm_obj"] for r in bucket])
        b = np.nanmean([r["hybrid_bqm_obj"] for r in bucket])
        g = np.nanmean([r["gurobi_obj"] for r in bucket])
        d = t - c
        summary_lines.append(
            f"  {key[0]:>3}  {key[1]:<6}  {t:>+10.4f}  {c:>+10.4f}  {b:>+10.4f}  {g:>+10.4f}  {d:>+10.4f}"
        )
    summary_lines.append("")
    summary_lines.append("HEADLINE:")
    deltas_cqm = [r["tabu_minus_hybrid_cqm"] for r in rows if not np.isnan(r["hybrid_cqm_obj"])]
    deltas_bqm = [r["tabu_minus_hybrid_bqm"] for r in rows if not np.isnan(r["hybrid_bqm_obj"])]
    deltas_gur = [r["tabu_minus_gurobi"] for r in rows if not np.isnan(r["gurobi_obj"])]
    if deltas_cqm:
        summary_lines.append(f"  Tabu − Hybrid_CQM: mean = {np.mean(deltas_cqm):+.4f}  "
                              f"(positive = Tabu worse, lower-better objective)")
    if deltas_bqm:
        summary_lines.append(f"  Tabu − Hybrid_BQM: mean = {np.mean(deltas_bqm):+.4f}")
    if deltas_gur:
        summary_lines.append(f"  Tabu − Gurobi:     mean = {np.mean(deltas_gur):+.4f}")

    summary = "\n".join(summary_lines)
    (out_dir / "qpu_replacement_ablation_summary.txt").write_text(summary)
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
