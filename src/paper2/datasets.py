"""Synthetic covariance generators and Paper 1 dataset re-exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from paper1.datasets import (
    PortfolioInstance,
    build_equity_rolling_instances,
    load_kenneth_french_49_daily,
    subset_portfolio_instance,
)


_RIDGE_EPS = 1e-6
_BANDED_BASE_VAR = 0.04


@dataclass(frozen=True)
class SyntheticInstance:
    mu: np.ndarray
    sigma: np.ndarray
    labels: tuple[str, ...]
    family: str
    N: int
    density: float
    metadata: dict[str, Any]


def covariance_density(sigma: np.ndarray, threshold: float = 1e-10) -> float:
    """Return the fraction of off-diagonal entries above a numerical threshold."""

    matrix = np.asarray(sigma, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("sigma must be a square matrix.")
    n_assets = matrix.shape[0]
    if n_assets < 2:
        return 0.0

    off_diagonal_mask = ~np.eye(n_assets, dtype=bool)
    numerator = np.count_nonzero(np.abs(matrix[off_diagonal_mask]) > threshold)
    denominator = n_assets * (n_assets - 1)
    return float(numerator / denominator)


def generate_diagonal_instance(N: int, *, epsilon: float = 0.02, seed: int = 0) -> SyntheticInstance:
    """Generate a diagonal or near-diagonal covariance matrix."""

    _validate_dimension(N)
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    rng = np.random.default_rng(seed)
    base_variances = _sample_variances(rng, N)
    sigma = np.diag(base_variances.copy())
    std = np.sqrt(base_variances)

    if epsilon > 0 and N > 1:
        local_noise = rng.uniform(0.0, epsilon, size=N - 1)
        for index, strength in enumerate(local_noise):
            sigma[index, index] += strength * base_variances[index]
            sigma[index + 1, index + 1] += strength * base_variances[index + 1]
            covariance = strength * std[index] * std[index + 1]
            sigma[index, index + 1] += covariance
            sigma[index + 1, index] += covariance

    mu = _sample_mu(rng, N)
    return _build_instance(
        family="diagonal",
        N=N,
        seed=seed,
        mu=mu,
        sigma=sigma,
        epsilon=epsilon,
    )


def generate_block_instance(
    N: int,
    *,
    num_blocks: int = 4,
    intra_block_rho: float = 0.35,
    seed: int = 0,
) -> SyntheticInstance:
    """Generate a block-diagonal covariance matrix with equicorrelated blocks."""

    _validate_block_arguments(N, num_blocks, intra_block_rho)

    rng = np.random.default_rng(seed)
    base_variances = _sample_variances(rng, N)
    sigma = _build_block_covariance(base_variances, num_blocks, intra_block_rho)
    mu = _sample_mu(rng, N)
    return _build_instance(
        family="block",
        N=N,
        seed=seed,
        mu=mu,
        sigma=sigma,
        num_blocks=num_blocks,
        intra_block_rho=intra_block_rho,
    )


def generate_block_cross_instance(
    N: int,
    *,
    num_blocks: int = 4,
    intra_block_rho: float = 0.35,
    cross_link_density: float = 0.1,
    seed: int = 0,
) -> SyntheticInstance:
    """Generate block structure with sparse cross-block links."""

    _validate_block_arguments(N, num_blocks, intra_block_rho)
    if not 0.0 <= cross_link_density <= 1.0:
        raise ValueError("cross_link_density must lie in [0, 1].")

    rng = np.random.default_rng(seed)
    base_variances = _sample_variances(rng, N)
    sigma = _build_block_covariance(base_variances, num_blocks, intra_block_rho)
    std = np.sqrt(base_variances)
    block_size = N // num_blocks
    block_ids = np.repeat(np.arange(num_blocks), block_size)

    upper_i, upper_j = np.triu_indices(N, k=1)
    cross_mask = block_ids[upper_i] != block_ids[upper_j]
    cross_i = upper_i[cross_mask]
    cross_j = upper_j[cross_mask]
    selected = rng.random(cross_i.size) < cross_link_density

    link_rho = min(max(0.05, 0.35 * intra_block_rho), 0.2)
    selected_links = 0
    for i, j in zip(cross_i[selected], cross_j[selected], strict=False):
        sigma[i, i] += link_rho * base_variances[i]
        sigma[j, j] += link_rho * base_variances[j]
        covariance = link_rho * std[i] * std[j]
        sigma[i, j] += covariance
        sigma[j, i] += covariance
        selected_links += 1

    mu = _sample_mu(rng, N)
    return _build_instance(
        family="block_cross",
        N=N,
        seed=seed,
        mu=mu,
        sigma=sigma,
        num_blocks=num_blocks,
        intra_block_rho=intra_block_rho,
        cross_link_density=cross_link_density,
        cross_link_rho=link_rho,
        selected_cross_links=selected_links,
    )


def generate_banded_instance(N: int, *, rho: float = 0.7, seed: int = 0) -> SyntheticInstance:
    """Generate an AR(1)-style banded covariance matrix."""

    _validate_dimension(N)
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1).")

    rng = np.random.default_rng(seed)
    distances = np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    sigma = _BANDED_BASE_VAR * np.power(rho, distances, dtype=float)
    mu = _sample_mu(rng, N)
    return _build_instance(
        family="banded",
        N=N,
        seed=seed,
        mu=mu,
        sigma=sigma,
        rho=rho,
        base_var=_BANDED_BASE_VAR,
    )


def generate_dense_instance(
    N: int,
    *,
    degrees_of_freedom: int | None = None,
    seed: int = 0,
) -> SyntheticInstance:
    """Generate a dense covariance matrix from a Wishart draw."""

    _validate_dimension(N)
    df = N + 5 if degrees_of_freedom is None else int(degrees_of_freedom)
    if df < N:
        raise ValueError("degrees_of_freedom must be at least N.")

    rng = np.random.default_rng(seed)
    draw = rng.normal(size=(df, N))
    wishart = (draw.T @ draw) / df
    std = np.sqrt(np.diag(wishart))
    corr = wishart / np.outer(std, std)

    target_variances = _sample_variances(rng, N)
    target_std = np.sqrt(target_variances)
    sigma = corr * np.outer(target_std, target_std)
    mu = _sample_mu(rng, N)
    return _build_instance(
        family="dense",
        N=N,
        seed=seed,
        mu=mu,
        sigma=sigma,
        degrees_of_freedom=df,
    )


def generate_instance_family(family: str, N: int, seed: int = 0, **kwargs: Any) -> SyntheticInstance:
    """Dispatch to a synthetic instance generator by family name."""

    normalized = family.strip().lower()
    generators: dict[str, Callable[..., SyntheticInstance]] = {
        "diagonal": generate_diagonal_instance,
        "block": generate_block_instance,
        "block_cross": generate_block_cross_instance,
        "banded": generate_banded_instance,
        "dense": generate_dense_instance,
    }
    if normalized not in generators:
        supported = ", ".join(sorted(generators))
        raise ValueError(f"Unknown family {family!r}. Expected one of: {supported}.")
    return generators[normalized](N=N, seed=seed, **kwargs)


def _build_block_covariance(
    base_variances: np.ndarray,
    num_blocks: int,
    intra_block_rho: float,
) -> np.ndarray:
    n_assets = base_variances.size
    block_size = n_assets // num_blocks
    sigma = np.zeros((n_assets, n_assets), dtype=float)
    std = np.sqrt(base_variances)

    for block_index in range(num_blocks):
        start = block_index * block_size
        stop = start + block_size
        block_std = std[start:stop]
        covariance_block = intra_block_rho * np.outer(block_std, block_std)
        covariance_block[np.diag_indices(block_size)] = base_variances[start:stop]
        sigma[start:stop, start:stop] = covariance_block

    return sigma


def _build_instance(
    *,
    family: str,
    N: int,
    seed: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    **params: Any,
) -> SyntheticInstance:
    sigma_spd, ridge = _ensure_spd(sigma)
    metadata = {"seed": seed, "ridge": ridge, **params}
    return SyntheticInstance(
        mu=np.asarray(mu, dtype=float),
        sigma=sigma_spd,
        labels=tuple(f"asset_{index}" for index in range(N)),
        family=family,
        N=N,
        density=covariance_density(sigma_spd),
        metadata=metadata,
    )


def _ensure_spd(sigma: np.ndarray, ridge: float = _RIDGE_EPS) -> tuple[np.ndarray, float]:
    matrix = np.asarray(sigma, dtype=float).copy()
    matrix = 0.5 * (matrix + matrix.T)
    min_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
    diagonal_shift = ridge if min_eigenvalue > ridge else (-min_eigenvalue + ridge)
    matrix += np.eye(matrix.shape[0], dtype=float) * diagonal_shift
    return matrix, diagonal_shift


def _sample_mu(rng: np.random.Generator, N: int) -> np.ndarray:
    return rng.normal(loc=0.05, scale=0.02, size=N)


def _sample_variances(rng: np.random.Generator, N: int) -> np.ndarray:
    return rng.uniform(0.02, 0.08, size=N)


def _validate_dimension(N: int) -> None:
    if int(N) != N or N <= 0:
        raise ValueError("N must be a positive integer.")


def _validate_block_arguments(N: int, num_blocks: int, intra_block_rho: float) -> None:
    _validate_dimension(N)
    if int(num_blocks) != num_blocks or num_blocks <= 0:
        raise ValueError("num_blocks must be a positive integer.")
    if N % num_blocks != 0:
        raise ValueError("N must be divisible by num_blocks for equal-sized blocks.")
    if not 0.0 <= intra_block_rho < 1.0:
        raise ValueError("intra_block_rho must lie in [0, 1).")


__all__ = [
    "PortfolioInstance",
    "SyntheticInstance",
    "build_equity_rolling_instances",
    "covariance_density",
    "generate_banded_instance",
    "generate_block_cross_instance",
    "generate_block_instance",
    "generate_dense_instance",
    "generate_diagonal_instance",
    "generate_instance_family",
    "load_kenneth_french_49_daily",
    "subset_portfolio_instance",
]
