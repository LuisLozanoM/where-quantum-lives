"""Benchmark-level result row wrapping SolverResult with experiment metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from paper2.solvers.base import SolverResult


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class BenchmarkResultRow:
    """One row in the benchmark results table."""

    # Experiment identity
    run_id: str
    experiment_id: str  # e.g. "A1", "B2", "D1"
    instance_id: str
    instance_family: str  # "synthetic_diagonal", "ff49_rolling", "predmarket"

    # Problem parameters
    N: int  # number of variables
    K: int  # cardinality constraint
    density: float  # graph density rho
    constraint_level: str  # "L1", "L2", "L3", "L4"
    drift_level: str | None = None  # "low", "medium", "high", None
    wall_clock_budget: float | None = None  # seconds

    # Turnover context
    prev_portfolio: bool = False  # whether this is a re-optimization from prior solution
    turnover_cost: float = 0.0

    # Solver result
    solver_result: SolverResult | None = None

    # Aggregated metrics (populated during analysis)
    objective_gap: float | None = None  # gap vs best known / Gurobi optimal
    feasible_rate: float | None = None  # fraction of reads that were feasible

    timestamp: str = field(default_factory=_utc_timestamp)
    extras: dict[str, Any] = field(default_factory=dict)
