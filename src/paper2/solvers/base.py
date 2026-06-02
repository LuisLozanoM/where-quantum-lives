"""Unified solver interface and result schema for all solver families."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SolverResult:
    """Standardized result returned by every solver wrapper."""

    sample: np.ndarray
    energy: float
    feasible: bool
    solver_family: str  # "cpu_exact", "cpu_heuristic", "gpu", "direct_qpu", "hybrid"
    solver_name: str  # e.g. "gurobi_miqp", "neal_sa", "dwave_qpu_forward", "hybrid_cqm"
    timestamp: str = field(default_factory=_utc_timestamp)

    # Timing (seconds)
    wall_clock_total: float = 0.0  # end-to-end from client
    wall_clock_solve: float = 0.0  # solver-only portion

    # Solver-internal timing (populated where available)
    solver_timing: dict[str, float] = field(default_factory=dict)

    # --- QPU-specific (None for classical solvers) ---
    topology_family: str | None = None  # "pegasus", "zephyr"
    graph_id: str | None = None
    num_active_qubits: int | None = None
    num_active_couplers: int | None = None
    embedded_qubits: int | None = None
    chain_length_mean: float | None = None
    chain_length_max: int | None = None
    chain_break_fraction: float | None = None
    chain_strength: float | None = None
    anneal_time: float | None = None
    num_reads: int | None = None
    qpu_programming_time: float | None = None  # microseconds
    qpu_sampling_time: float | None = None  # microseconds
    qpu_access_time: float | None = None  # microseconds
    total_service_time: float | None = None  # microseconds

    # --- Hybrid-specific ---
    hybrid_run_time: float | None = None  # microseconds
    hybrid_charge_time: float | None = None  # microseconds
    hybrid_qpu_access_time: float | None = None  # microseconds; 0 = no QPU contribution

    # Extra metadata
    extras: dict[str, Any] = field(default_factory=dict)
