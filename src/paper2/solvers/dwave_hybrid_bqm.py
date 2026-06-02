"""D-Wave Leap hybrid BQM solver wrapper for Paper 2A."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

from paper1.formulations import QuboProblem, to_dimod_bqm
from paper2.solvers.base import SolverResult

_DEFAULT_CONFIG = "./dwave.conf"


def _resolve_config(config_file: str | None) -> dict[str, str]:
    """Build config kwargs for Leap hybrid samplers."""
    if config_file is not None:
        return {"config_file": config_file}
    if Path(_DEFAULT_CONFIG).exists():
        return {"config_file": _DEFAULT_CONFIG}
    return {}


def _resolve_sampleset(sampleset):
    if hasattr(sampleset, "resolve"):
        sampleset.resolve()
    return sampleset


def _extract_hybrid_timing(sampleset) -> dict[str, float]:
    """Extract hybrid timing metadata in microseconds."""
    info = getattr(sampleset, "info", {}) or {}
    nested_timing = info.get("timing", {}) if isinstance(info.get("timing"), dict) else {}

    timing: dict[str, float] = {}
    for key in ("run_time", "charge_time", "qpu_access_time"):
        value = info.get(key, nested_timing.get(key, 0.0))
        timing[key] = float(value) if isinstance(value, (int, float)) else 0.0
    return timing


def _best_sample_from_sampleset(sampleset, size: int) -> np.ndarray:
    """Extract the lowest-energy sample as a dense binary vector."""
    if len(sampleset) == 0:
        raise ValueError("Hybrid BQM solver returned no samples.")

    best_idx = int(np.argmin(sampleset.record.energy))
    record = sampleset.record[best_idx]
    sample_dict = {
        var: int(record.sample[idx])
        for idx, var in enumerate(sampleset.variables)
    }
    return np.array([sample_dict.get(i, 0) for i in range(size)], dtype=int)


def solve_hybrid_bqm(
    problem: QuboProblem,
    *,
    time_limit: float = 5.0,
    config_file: str | None = None,
) -> SolverResult:
    """Solve a QUBO with D-Wave's Leap hybrid BQM sampler."""
    from dwave.system import LeapHybridBQMSampler

    wall_start = perf_counter()
    sampler = LeapHybridBQMSampler(**_resolve_config(config_file))

    try:
        bqm = to_dimod_bqm(problem)
        sampleset = sampler.sample(bqm, time_limit=float(time_limit))
        sampleset = _resolve_sampleset(sampleset)

        wall_total = perf_counter() - wall_start
        timing = _extract_hybrid_timing(sampleset)
        sample = _best_sample_from_sampleset(sampleset, problem.size)
        qpu_access_time = timing["qpu_access_time"]
    except Exception:
        if hasattr(sampler, "client"):
            sampler.client.close()
        raise

    if hasattr(sampler, "client"):
        sampler.client.close()

    return SolverResult(
        sample=sample,
        energy=float(problem.energy(sample)),
        feasible=False,
        solver_family="hybrid",
        solver_name="hybrid_bqm",
        wall_clock_total=wall_total,
        wall_clock_solve=timing["run_time"] / 1e6,
        solver_timing=timing,
        hybrid_run_time=timing["run_time"],
        hybrid_charge_time=timing["charge_time"],
        hybrid_qpu_access_time=qpu_access_time,
        extras={"qpu_contributed": qpu_access_time > 0.0},
    )
