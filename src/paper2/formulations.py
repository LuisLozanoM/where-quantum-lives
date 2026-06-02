"""MVT QUBO/CQM/Kelly-Taylor formulation builders.

Extends paper1.formulations with mean-variance-turnover (MVT) builders
and CQM construction for hybrid solvers.

Binary turnover encoding:
    For binary z and fixed binary z_{t-1}, |z_i - z_{t-1,i}| is linear:
        z_{t-1,i} = 0  =>  |z_i - 0| = z_i
        z_{t-1,i} = 1  =>  |z_i - 1| = 1 - z_i
    This absorbs into the QUBO diagonal as:
        turnover_shift_i = tau_i * (1 - 2 * z_{t-1,i})
    with offset contribution: sum(tau * z_{t-1}).
    NO ancillas, NO pairwise terms, NO density increase.
"""

from __future__ import annotations

import numpy as np

from paper1.formulations import QuboProblem


def _validate_inputs(
    mu: np.ndarray, sigma: np.ndarray, labels: tuple[str, ...] | None = None,
) -> None:
    """Basic shape and sanity checks matching paper1's validation."""
    if mu.ndim != 1:
        raise ValueError(f"mu must be 1-d, got shape {mu.shape}")
    N = len(mu)
    if sigma.shape != (N, N):
        raise ValueError(f"sigma must be ({N}, {N}), got {sigma.shape}")
    if labels is not None and len(labels) != N:
        raise ValueError(f"labels length {len(labels)} != N={N}")


def _turnover_shift(
    N: int,
    prev_portfolio: np.ndarray | None,
    turnover_cost: float | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute diagonal shift and offset from binary turnover term.

    Returns (shift vector of length N, scalar offset).
    """
    if prev_portfolio is None:
        return np.zeros(N), 0.0
    prev = np.asarray(prev_portfolio, dtype=float)
    if prev.shape != (N,):
        raise ValueError(f"prev_portfolio must have shape ({N},), got {prev.shape}")
    if not np.all((prev == 0) | (prev == 1)):
        raise ValueError("prev_portfolio must be binary (0 or 1 only)")
    tau = np.broadcast_to(np.asarray(turnover_cost, dtype=float), N).copy()
    shift = tau * (1.0 - 2.0 * prev)
    offset = float(tau @ prev)
    return shift, offset


def build_mvt_bqm(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    cardinality: int,
    penalty: float,
    prev_portfolio: np.ndarray | None = None,
    turnover_cost: float | np.ndarray = 0.0,
    labels: tuple[str, ...] | None = None,
) -> QuboProblem:
    """Build binary mean-variance-turnover QUBO with exact-K penalty.

    Minimizes:
        -mu^T z + lambda z^T Sigma z + A(1^T z - K)^2 + tau^T |z - z_{t-1}|

    Args:
        mu: Expected returns, shape (N,).
        sigma: Covariance matrix, shape (N, N).
        risk_aversion: Lambda coefficient on the risk term.
        cardinality: K, the exact number of assets to select.
        penalty: A, penalty weight for the cardinality constraint.
        prev_portfolio: Previous binary portfolio z_{t-1}, shape (N,). None for first period.
        turnover_cost: Scalar or per-asset turnover cost tau.
        labels: Optional variable labels for stable identity across windows.

    Returns:
        QuboProblem with the full penalized MVT QUBO.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = 0.5 * (sigma + sigma.T)  # symmetrize
    N = len(mu)
    _validate_inputs(mu, sigma, labels)
    if cardinality < 1 or cardinality > N:
        raise ValueError(f"cardinality must be in [1, {N}], got {cardinality}")
    if penalty < 0:
        raise ValueError(f"penalty must be non-negative, got {penalty}")

    # Objective: -diag(mu) + lambda * Sigma
    Q = -np.diag(mu) + risk_aversion * sigma

    # Exact-K penalty: A * ones(N,N) - 2AK * I
    Q += penalty * np.ones((N, N)) - 2.0 * penalty * cardinality * np.eye(N)
    penalty_offset = penalty * cardinality ** 2

    # Turnover: diagonal shift only
    t_shift, t_offset = _turnover_shift(N, prev_portfolio, turnover_cost)
    Q += np.diag(t_shift)

    return QuboProblem(
        matrix=Q,
        offset=penalty_offset + t_offset,
        labels=labels,
    )


def build_mvt_objective(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    prev_portfolio: np.ndarray | None = None,
    turnover_cost: float | np.ndarray = 0.0,
    labels: tuple[str, ...] | None = None,
) -> QuboProblem:
    """Build the penalty-free MVT objective QUBO (no cardinality penalty).

    This is the objective used inside CQM (where cardinality is a native constraint)
    and also useful for evaluating solution quality at feasible points.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = 0.5 * (sigma + sigma.T)  # symmetrize
    N = len(mu)
    _validate_inputs(mu, sigma, labels)

    Q = -np.diag(mu) + risk_aversion * sigma

    t_shift, t_offset = _turnover_shift(N, prev_portfolio, turnover_cost)
    Q += np.diag(t_shift)

    return QuboProblem(matrix=Q, offset=t_offset, labels=labels)


def build_mvt_cqm(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    cardinality: int,
    prev_portfolio: np.ndarray | None = None,
    turnover_cost: float | np.ndarray = 0.0,
    labels: tuple[str, ...] | None = None,
):
    """Build CQM with native cardinality constraint for hybrid solvers.

    The objective is the same as build_mvt_objective (no penalty encoding).
    The cardinality constraint sum(z) == K is added natively.

    Returns:
        dimod.ConstrainedQuadraticModel
    """
    import dimod

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = 0.5 * (sigma + sigma.T)  # symmetrize
    N = len(mu)
    _validate_inputs(mu, sigma, labels)
    if cardinality < 1 or cardinality > N:
        raise ValueError(f"cardinality must be in [1, {N}], got {cardinality}")

    var_labels = list(labels) if labels is not None else list(range(N))

    t_shift, t_offset = _turnover_shift(N, prev_portfolio, turnover_cost)

    # Build objective as QuadraticModel
    obj = dimod.QuadraticModel()
    for i in range(N):
        obj.add_variable("BINARY", var_labels[i])

    # Linear terms: -mu_i + risk_aversion * sigma_ii + turnover_shift_i
    for i in range(N):
        obj.set_linear(
            var_labels[i],
            -mu[i] + risk_aversion * sigma[i, i] + t_shift[i],
        )

    # Quadratic terms (upper triangle, symmetric so double the coefficient)
    for i in range(N):
        for j in range(i + 1, N):
            coeff = 2.0 * risk_aversion * sigma[i, j]
            if abs(coeff) > 1e-15:
                obj.set_quadratic(var_labels[i], var_labels[j], coeff)

    obj.offset = t_offset

    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(obj)

    # Native cardinality constraint: sum(z) == K
    constraint_lhs = dimod.QuadraticModel()
    for i in range(N):
        constraint_lhs.add_variable("BINARY", var_labels[i])
        constraint_lhs.set_linear(var_labels[i], 1.0)
    cqm.add_constraint(constraint_lhs == cardinality, label="cardinality")

    return cqm


def evaluate_mvt_objective(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    z: np.ndarray,
    prev_portfolio: np.ndarray | None = None,
    turnover_cost: float | np.ndarray = 0.0,
) -> float:
    """Evaluate MVT objective value at a given binary solution.

    This is the penalty-free objective: -mu^T z + lambda z^T Sigma z + tau^T |z - z_{t-1}|.
    Useful for comparing solutions across formulations.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    z = np.asarray(z, dtype=float)
    N = len(mu)

    val = float(-mu @ z + risk_aversion * z @ sigma @ z)

    if prev_portfolio is not None:
        prev = np.asarray(prev_portfolio, dtype=float)
        tau = np.broadcast_to(np.asarray(turnover_cost, dtype=float), N)
        val += float(tau @ np.abs(z - prev))

    return val
