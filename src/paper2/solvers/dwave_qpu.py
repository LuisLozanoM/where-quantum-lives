"""Direct QPU solver wrappers: forward anneal (Q1), cached embedding (Q2), reverse anneal (Q3).

Extends paper1's QPU interface with Paper 2A's SolverResult schema,
full metadata logging, and reverse-anneal support.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from paper1.formulations import QuboProblem, to_qubo_dict
from paper2.solvers.base import SolverResult

_DEFAULT_CONFIG = "./dwave.conf"


def _resolve_config(config_file: str | None) -> dict:
    """Build config kwargs for DWaveSampler."""
    if config_file is not None:
        return {"config_file": config_file}
    if Path(_DEFAULT_CONFIG).exists():
        return {"config_file": _DEFAULT_CONFIG}
    return {}


def _resolve_sampleset(sampleset):
    if hasattr(sampleset, "resolve"):
        sampleset.resolve()
    return sampleset


def _extract_chain_stats(sampleset) -> dict:
    """Extract chain-break statistics from a sampleset."""
    stats: dict[str, Any] = {}
    if not hasattr(sampleset, "record"):
        return stats

    chain_breaks = []
    for record in sampleset.record:
        cbf = getattr(record, "chain_break_fraction", None)
        if cbf is not None:
            chain_breaks.append(float(cbf))

    if chain_breaks:
        stats["chain_break_fraction"] = float(np.mean(chain_breaks))
    return stats


def _extract_timing(sampleset) -> dict[str, float]:
    """Extract QPU timing from sampleset.info."""
    info = getattr(sampleset, "info", {}) or {}
    timing = info.get("timing", {}) or {}
    return {
        k: float(v)
        for k, v in timing.items()
        if isinstance(v, (int, float))
    }


def _labeled_qubo_dict(problem: QuboProblem) -> dict[tuple, float]:
    """Build QUBO dict using labels as variable names when present.

    logical_graph_from_qubo uses labels when available, so the embedding
    maps label -> [physical_qubits]. The QUBO dict must use the same
    namespace for FixedEmbeddingComposite to match them.
    """
    qubo, _ = to_qubo_dict(problem)
    if problem.labels is None:
        return qubo
    # Remap integer keys to label keys
    labels = problem.labels
    return {(labels[i], labels[j]): v for (i, j), v in qubo.items()}


def _best_sample_from_sampleset(sampleset, problem: QuboProblem) -> tuple[np.ndarray, float]:
    """Extract the lowest-energy sample as a numpy array."""
    best_idx = int(np.argmin(sampleset.record.energy))
    record = sampleset.record[best_idx]

    sample_dict = {}
    for idx, var in enumerate(sampleset.variables):
        sample_dict[var] = int(record.sample[idx])

    # Map back from labels to positional array
    if problem.labels is not None:
        sample = np.array(
            [sample_dict.get(lbl, 0) for lbl in problem.labels], dtype=int
        )
    else:
        sample = np.array(
            [sample_dict.get(i, 0) for i in range(problem.size)], dtype=int
        )
    return sample, float(record.energy)


def _sampler_metadata(sampler) -> dict[str, Any]:
    """Extract topology and hardware metadata from sampler properties."""
    props = getattr(sampler, "properties", {}) or {}
    topo = props.get("topology", {}) if isinstance(props.get("topology"), dict) else {}

    # Use num_active_qubits if available, fall back to nodelist length
    num_qubits = props.get("num_active_qubits")
    if num_qubits is None and hasattr(sampler, "nodelist"):
        num_qubits = len(list(sampler.nodelist))

    num_couplers = props.get("num_active_couplers")
    if num_couplers is None and hasattr(sampler, "edgelist"):
        num_couplers = len(list(sampler.edgelist))

    return {
        "topology_family": topo.get("type"),
        "graph_id": props.get("graph_id") or props.get("chip_id"),
        "num_active_qubits": num_qubits,
        "num_active_couplers": num_couplers,
    }


def solve_qpu_forward(
    problem: QuboProblem,
    embedding: Mapping[Any, Any],
    *,
    sampler=None,
    solver: str | None = None,
    reads: int = 1000,
    chain_strength: float | None = None,
    anneal_time: float | None = None,
    spin_reversal_transforms: int = 0,
    config_file: str | None = None,
) -> SolverResult:
    """Q1: Forward anneal with fixed embedding.

    Args:
        problem: QuboProblem to solve.
        embedding: Pre-computed minor embedding {logical_var: [physical_qubits]}.
        sampler: Optional pre-created DWaveSampler. Created if None.
        solver: Solver name (e.g. "Advantage_system4.1"). Ignored if sampler provided.
        reads: Number of annealing reads.
        chain_strength: Chain coupling strength. Auto-scaled if None.
        anneal_time: Annealing time in microseconds.
        spin_reversal_transforms: Number of spin-reversal transforms (gauge transforms).
        config_file: Path to dwave.conf.

    Returns:
        SolverResult with full QPU metadata.
    """
    from dwave.system import DWaveSampler, FixedEmbeddingComposite

    wall_start = perf_counter()
    owns_sampler = sampler is None

    try:
        if sampler is None:
            config = _resolve_config(config_file)
            if solver:
                config["solver"] = solver
            sampler = DWaveSampler(**config)

        composite = FixedEmbeddingComposite(sampler, embedding=embedding)

        # Apply spin-reversal if requested
        if spin_reversal_transforms > 0:
            from dwave.preprocessing.composites import SpinReversalTransformComposite
            composite = SpinReversalTransformComposite(composite)

        qubo = _labeled_qubo_dict(problem)

        # Build kwargs respecting composite's supported parameters
        supported = set(getattr(composite, "parameters", {}).keys())
        sample_kwargs: dict[str, Any] = {}
        if "num_reads" in supported:
            sample_kwargs["num_reads"] = reads
        if chain_strength is not None and "chain_strength" in supported:
            sample_kwargs["chain_strength"] = chain_strength
        if anneal_time is not None and "annealing_time" in supported:
            sample_kwargs["annealing_time"] = anneal_time
        if spin_reversal_transforms > 0 and "num_spin_reversal_transforms" in supported:
            sample_kwargs["num_spin_reversal_transforms"] = spin_reversal_transforms

        sampleset = composite.sample_qubo(qubo, **sample_kwargs)
        sampleset = _resolve_sampleset(sampleset)

        wall_total = perf_counter() - wall_start

        # Extract metadata
        timing = _extract_timing(sampleset)
        chain_stats = _extract_chain_stats(sampleset)
        sample, _ = _best_sample_from_sampleset(sampleset, problem)

        embedded_qubits = sum(len(chain) for chain in embedding.values())
        chain_lengths = [len(chain) for chain in embedding.values()]
        meta = _sampler_metadata(sampler)

        return SolverResult(
            sample=sample,
            energy=float(problem.energy(sample)),
            feasible=False,  # caller must check cardinality
            solver_family="direct_qpu",
            solver_name="dwave_qpu_forward",
            wall_clock_total=wall_total,
            wall_clock_solve=timing.get("qpu_access_time", 0.0) / 1e6,
            solver_timing=timing,
            topology_family=meta["topology_family"],
            graph_id=meta["graph_id"],
            num_active_qubits=meta["num_active_qubits"],
            num_active_couplers=meta["num_active_couplers"],
            embedded_qubits=embedded_qubits,
            chain_length_mean=float(np.mean(chain_lengths)) if chain_lengths else None,
            chain_length_max=max(chain_lengths) if chain_lengths else None,
            chain_break_fraction=chain_stats.get("chain_break_fraction"),
            chain_strength=chain_strength,
            anneal_time=anneal_time,
            num_reads=reads,
            qpu_programming_time=timing.get("qpu_programming_time"),
            qpu_sampling_time=timing.get("qpu_sampling_time"),
            qpu_access_time=timing.get("qpu_access_time"),
            total_service_time=timing.get("total_real_time"),
            extras={"spin_reversal_transforms": spin_reversal_transforms},
        )
    finally:
        if owns_sampler and sampler is not None:
            try:
                sampler.client.close()
            except Exception:
                pass


def solve_qpu_reverse(
    problem: QuboProblem,
    embedding: Mapping[Any, Any],
    initial_state: np.ndarray,
    *,
    sampler=None,
    solver: str | None = None,
    reads: int = 1000,
    chain_strength: float | None = None,
    reverse_schedule: list[list[float]] | None = None,
    config_file: str | None = None,
) -> SolverResult:
    """Q3: Reverse anneal from an initial state for local refinement.

    initial_state is in logical variable space; FixedEmbeddingComposite
    maps it to physical chains internally.

    Args:
        problem: QuboProblem to solve.
        embedding: Pre-computed minor embedding.
        initial_state: Binary vector z_{t-1} to start from, shape (N,).
        sampler: Optional pre-created DWaveSampler.
        solver: Solver name.
        reads: Number of reads.
        chain_strength: Chain coupling strength.
        reverse_schedule: Custom anneal schedule [[t, s], ...].
            Default: ramp down to s=0.4, hold, ramp back up.
        config_file: Path to dwave.conf.

    Returns:
        SolverResult with reverse-anneal metadata.
    """
    from dwave.system import DWaveSampler, FixedEmbeddingComposite

    wall_start = perf_counter()
    owns_sampler = sampler is None

    try:
        if sampler is None:
            config = _resolve_config(config_file)
            if solver:
                config["solver"] = solver
            sampler = DWaveSampler(**config)

        composite = FixedEmbeddingComposite(sampler, embedding=embedding)
        qubo = _labeled_qubo_dict(problem)

        # initial_state in logical variable space (matching QUBO variable keys)
        if problem.labels is not None:
            init_dict = {lbl: int(initial_state[i]) for i, lbl in enumerate(problem.labels)}
        else:
            init_dict = {i: int(initial_state[i]) for i in range(problem.size)}

        if reverse_schedule is None:
            reverse_schedule = [[0.0, 1.0], [5.0, 0.4], [15.0, 0.4], [20.0, 1.0]]

        sample_kwargs: dict[str, Any] = {
            "num_reads": reads,
            "initial_state": init_dict,
            "anneal_schedule": reverse_schedule,
            "reinitialize_state": True,
        }
        if chain_strength is not None:
            sample_kwargs["chain_strength"] = chain_strength

        sampleset = composite.sample_qubo(qubo, **sample_kwargs)
        sampleset = _resolve_sampleset(sampleset)

        wall_total = perf_counter() - wall_start

        timing = _extract_timing(sampleset)
        chain_stats = _extract_chain_stats(sampleset)
        sample, _ = _best_sample_from_sampleset(sampleset, problem)

        embedded_qubits = sum(len(chain) for chain in embedding.values())
        chain_lengths = [len(chain) for chain in embedding.values()]
        meta = _sampler_metadata(sampler)

        return SolverResult(
            sample=sample,
            energy=float(problem.energy(sample)),
            feasible=False,
            solver_family="direct_qpu",
            solver_name="dwave_qpu_reverse",
            wall_clock_total=wall_total,
            wall_clock_solve=timing.get("qpu_access_time", 0.0) / 1e6,
            solver_timing=timing,
            topology_family=meta["topology_family"],
            graph_id=meta["graph_id"],
            num_active_qubits=meta["num_active_qubits"],
            num_active_couplers=meta["num_active_couplers"],
            embedded_qubits=embedded_qubits,
            chain_length_mean=float(np.mean(chain_lengths)) if chain_lengths else None,
            chain_length_max=max(chain_lengths) if chain_lengths else None,
            chain_break_fraction=chain_stats.get("chain_break_fraction"),
            chain_strength=chain_strength,
            anneal_time=None,
            num_reads=reads,
            qpu_programming_time=timing.get("qpu_programming_time"),
            qpu_sampling_time=timing.get("qpu_sampling_time"),
            qpu_access_time=timing.get("qpu_access_time"),
            total_service_time=timing.get("total_real_time"),
            extras={
                "reverse_schedule": reverse_schedule,
                "initial_energy": float(problem.energy(initial_state)),
            },
        )
    finally:
        if owns_sampler and sampler is not None:
            try:
                sampler.client.close()
            except Exception:
                pass
