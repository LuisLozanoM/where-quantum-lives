"""D-Wave Leap hybrid CQM solver wrapper for Paper 2A."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np

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


def _choose_best_cqm_sample(sampleset, cqm) -> tuple[dict, bool]:
    """Return the best feasible sample, or the best infeasible sample if none are feasible."""
    if len(sampleset) == 0:
        raise ValueError("Hybrid CQM solver returned no samples.")

    best_sample: dict | None = None
    best_energy: float | None = None
    best_feasible_sample: dict | None = None
    best_feasible_energy: float | None = None

    for datum in sampleset.data():
        sample_dict = {var: int(value) for var, value in datum.sample.items()}
        energy = float(datum.energy)
        is_feasible = getattr(datum, "is_feasible", None)
        if is_feasible is None:
            is_feasible = cqm.check_feasible(sample_dict)
        feasible = bool(is_feasible)

        if best_energy is None or energy < best_energy:
            best_sample = sample_dict
            best_energy = energy

        if feasible and (best_feasible_energy is None or energy < best_feasible_energy):
            best_feasible_sample = sample_dict
            best_feasible_energy = energy

    if best_feasible_sample is not None:
        return best_feasible_sample, True
    if best_sample is None:
        raise ValueError("Hybrid CQM solver returned no usable samples.")
    return best_sample, False


def solve_hybrid_cqm(
    cqm,
    *,
    time_limit: float = 5.0,
    config_file: str | None = None,
) -> SolverResult:
    """Solve a constrained model with D-Wave's Leap hybrid CQM sampler."""
    from dwave.system import LeapHybridCQMSampler

    wall_start = perf_counter()
    sampler = LeapHybridCQMSampler(**_resolve_config(config_file))

    try:
        sampleset = sampler.sample_cqm(cqm, time_limit=float(time_limit))
        sampleset = _resolve_sampleset(sampleset)

        wall_total = perf_counter() - wall_start
        timing = _extract_hybrid_timing(sampleset)
        sample_dict, feasible = _choose_best_cqm_sample(sampleset, cqm)
        variables = list(cqm.variables)
        sample = np.array([sample_dict.get(var, 0) for var in variables], dtype=int)
        qpu_access_time = timing["qpu_access_time"]
    except Exception:
        if hasattr(sampler, "client"):
            sampler.client.close()
        raise

    if hasattr(sampler, "client"):
        sampler.client.close()

    return SolverResult(
        sample=sample,
        energy=float(cqm.objective.energy(sample_dict)),
        feasible=feasible,
        solver_family="hybrid",
        solver_name="hybrid_cqm",
        wall_clock_total=wall_total,
        wall_clock_solve=timing["run_time"] / 1e6,
        solver_timing=timing,
        hybrid_run_time=timing["run_time"],
        hybrid_charge_time=timing["charge_time"],
        hybrid_qpu_access_time=qpu_access_time,
        extras={"qpu_contributed": qpu_access_time > 0.0},
    )
