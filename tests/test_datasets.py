"""Tests for synthetic covariance generators."""

from __future__ import annotations

import numpy as np
import pytest

from paper2.datasets import covariance_density, generate_instance_family


FAMILY_KWARGS = {
    "diagonal": {"epsilon": 0.02},
    "block": {"num_blocks": 3, "intra_block_rho": 0.35},
    "block_cross": {
        "num_blocks": 3,
        "intra_block_rho": 0.35,
        "cross_link_density": 0.25,
    },
    "banded": {"rho": 0.7},
    "dense": {"degrees_of_freedom": 20},
}


def _instance(family: str, *, N: int = 12, seed: int = 7):
    return generate_instance_family(family=family, N=N, seed=seed, **FAMILY_KWARGS[family])


@pytest.mark.parametrize("family", tuple(FAMILY_KWARGS))
def test_family_generators_produce_positive_definite_covariances(family: str) -> None:
    instance = _instance(family)
    eigenvalues = np.linalg.eigvalsh(instance.sigma)
    assert np.all(eigenvalues > 0.0)


@pytest.mark.parametrize("family", tuple(FAMILY_KWARGS))
def test_family_generators_return_symmetric_covariances(family: str) -> None:
    instance = _instance(family)
    np.testing.assert_allclose(instance.sigma, instance.sigma.T)


@pytest.mark.parametrize("family", tuple(FAMILY_KWARGS))
def test_family_generators_return_expected_mu_shape(family: str) -> None:
    instance = _instance(family)
    assert instance.mu.shape == (instance.N,)


def test_same_seed_produces_identical_output() -> None:
    first = _instance("block_cross", seed=19)
    second = _instance("block_cross", seed=19)

    np.testing.assert_allclose(first.mu, second.mu)
    np.testing.assert_allclose(first.sigma, second.sigma)
    assert first.labels == second.labels
    assert first.family == second.family
    assert first.N == second.N
    assert first.density == second.density
    assert first.metadata == second.metadata


def test_density_ordering_matches_expected_structure() -> None:
    diagonal = _instance("diagonal", N=12, seed=11)
    block = _instance("block", N=12, seed=11)
    block_cross = _instance("block_cross", N=12, seed=11)
    dense = _instance("dense", N=12, seed=11)

    assert diagonal.density == covariance_density(diagonal.sigma)
    assert diagonal.density < block.density < block_cross.density < dense.density


@pytest.mark.parametrize("family", tuple(FAMILY_KWARGS))
def test_dispatcher_preserves_requested_size_and_labels(family: str) -> None:
    instance = _instance(family, N=15 if family == "dense" else 12, seed=5)
    assert instance.N == (15 if family == "dense" else 12)
    assert instance.sigma.shape == (instance.N, instance.N)
    assert instance.labels == tuple(f"asset_{index}" for index in range(instance.N))
