"""T1 correctness tests: verify MVT BQM and CQM recover same optimum as brute-force.

Tests at N=8 (K=3) and N=10 (K=4), with and without turnover costs.
"""

from __future__ import annotations

import numpy as np
import pytest

from paper1.formulations import solve_qubo_bruteforce
from paper2.formulations import (
    build_mvt_bqm,
    build_mvt_cqm,
    build_mvt_objective,
    evaluate_mvt_objective,
)


def _random_psd(N: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random symmetric positive definite matrix."""
    A = rng.standard_normal((N, N))
    return (A @ A.T) / N + 0.01 * np.eye(N)


def _random_instance(N: int, seed: int = 42):
    """Generate a random MVT instance for testing."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.05, 0.02, size=N)
    sigma = _random_psd(N, rng)
    labels = tuple(f"asset_{i}" for i in range(N))
    return mu, sigma, labels


def _solve_cqm_bruteforce(cqm, N: int, cardinality: int, labels) -> tuple[dict, float]:
    """Brute-force solve a CQM by enumerating all K-subsets."""
    from itertools import combinations

    best_energy = float("inf")
    best_sample = None

    for combo in combinations(range(N), cardinality):
        sample = {labels[i]: 0 for i in range(N)}
        for i in combo:
            sample[labels[i]] = 1

        # Check all constraints satisfied
        feasible = True
        for label in cqm.constraint_labels:
            if cqm.check_feasible(sample) is False:
                feasible = False
                break

        if feasible:
            energy = cqm.objective.energy(sample)
            if energy < best_energy:
                best_energy = energy
                best_sample = sample

    return best_sample, best_energy


class TestBuildMvtBqm:
    """Tests for build_mvt_bqm."""

    @pytest.mark.parametrize("N,K", [(8, 3), (10, 4)])
    def test_returns_quboproblem(self, N, K):
        mu, sigma, labels = _random_instance(N)
        problem = build_mvt_bqm(mu, sigma, 0.5, K, 4.0, labels=labels)
        assert problem.matrix.shape == (N, N)
        assert problem.labels == labels

    @pytest.mark.parametrize("N,K", [(8, 3), (10, 4)])
    def test_symmetric_matrix(self, N, K):
        mu, sigma, labels = _random_instance(N)
        problem = build_mvt_bqm(mu, sigma, 0.5, K, 4.0, labels=labels)
        np.testing.assert_allclose(problem.matrix, problem.matrix.T, atol=1e-12)

    def test_no_turnover_matches_paper1(self):
        """Without turnover, build_mvt_bqm should produce the same QUBO as paper1."""
        from paper1.formulations import build_exact_k_portfolio_qubo

        N, K = 8, 3
        mu, sigma, labels = _random_instance(N)
        lam, penalty = 0.5, 4.0

        p1 = build_exact_k_portfolio_qubo(mu, sigma, lam, K, penalty, labels=labels)
        p2 = build_mvt_bqm(mu, sigma, lam, K, penalty, labels=labels)

        np.testing.assert_allclose(p2.matrix, p1.matrix, atol=1e-12)
        assert abs(p2.offset - p1.offset) < 1e-12

    @pytest.mark.parametrize("N,K", [(8, 3), (10, 4)])
    def test_turnover_only_modifies_diagonal(self, N, K):
        """Turnover should only change diagonal entries, not off-diagonal."""
        mu, sigma, labels = _random_instance(N)
        rng = np.random.default_rng(99)
        prev = (rng.random(N) > 0.5).astype(int)

        q_no_to = build_mvt_bqm(mu, sigma, 0.5, K, 4.0, labels=labels)
        q_with_to = build_mvt_bqm(
            mu, sigma, 0.5, K, 4.0,
            prev_portfolio=prev, turnover_cost=0.1, labels=labels,
        )

        # Off-diagonal should be identical
        mask = ~np.eye(N, dtype=bool)
        np.testing.assert_allclose(
            q_with_to.matrix[mask], q_no_to.matrix[mask], atol=1e-12,
        )

        # Diagonal should differ where prev has nonzero entries or turnover applies
        diag_diff = np.diag(q_with_to.matrix) - np.diag(q_no_to.matrix)
        expected_shift = 0.1 * (1.0 - 2.0 * prev)
        np.testing.assert_allclose(diag_diff, expected_shift, atol=1e-12)


class TestBqmCqmEquivalence:
    """T1 core: BQM and CQM find the same optimum at feasible solutions."""

    @pytest.mark.parametrize("N,K,seed", [
        (8, 3, 42),
        (8, 3, 123),
        (10, 4, 42),
        (10, 4, 99),
    ])
    def test_same_optimum_no_turnover(self, N, K, seed):
        mu, sigma, labels = _random_instance(N, seed=seed)
        lam, penalty = 0.5, 4.0

        # BQM path: brute-force solve the penalized QUBO, constrained to K
        bqm = build_mvt_bqm(mu, sigma, lam, K, penalty, labels=labels)
        bf_result = solve_qubo_bruteforce(bqm, cardinality=K)

        # CQM path: brute-force solve the CQM
        cqm = build_mvt_cqm(mu, sigma, lam, K, labels=labels)
        cqm_sample, cqm_energy = _solve_cqm_bruteforce(cqm, N, K, list(labels))

        # Both should select the same assets
        bqm_selection = set(np.where(bf_result.sample == 1)[0])
        cqm_selection = {i for i, lbl in enumerate(labels) if cqm_sample[lbl] == 1}
        assert bqm_selection == cqm_selection, (
            f"BQM selected {bqm_selection}, CQM selected {cqm_selection}"
        )

        # Evaluate penalty-free objective at both solutions — should match
        bqm_obj = evaluate_mvt_objective(mu, sigma, lam, bf_result.sample)
        cqm_z = np.array([cqm_sample[lbl] for lbl in labels])
        cqm_obj = evaluate_mvt_objective(mu, sigma, lam, cqm_z)
        assert abs(bqm_obj - cqm_obj) < 1e-10

    @pytest.mark.parametrize("N,K,seed", [
        (8, 3, 42),
        (10, 4, 42),
    ])
    def test_same_optimum_with_turnover(self, N, K, seed):
        mu, sigma, labels = _random_instance(N, seed=seed)
        lam, penalty = 0.5, 4.0
        rng = np.random.default_rng(seed + 1000)
        prev = (rng.random(N) > 0.5).astype(int)
        turnover_cost = 0.05

        # BQM path
        bqm = build_mvt_bqm(
            mu, sigma, lam, K, penalty,
            prev_portfolio=prev, turnover_cost=turnover_cost, labels=labels,
        )
        bf_result = solve_qubo_bruteforce(bqm, cardinality=K)

        # CQM path
        cqm = build_mvt_cqm(
            mu, sigma, lam, K,
            prev_portfolio=prev, turnover_cost=turnover_cost, labels=labels,
        )
        cqm_sample, cqm_energy = _solve_cqm_bruteforce(cqm, N, K, list(labels))

        # Same selection
        bqm_selection = set(np.where(bf_result.sample == 1)[0])
        cqm_selection = {i for i, lbl in enumerate(labels) if cqm_sample[lbl] == 1}
        assert bqm_selection == cqm_selection

        # Same objective value
        bqm_obj = evaluate_mvt_objective(
            mu, sigma, lam, bf_result.sample,
            prev_portfolio=prev, turnover_cost=turnover_cost,
        )
        cqm_z = np.array([cqm_sample[lbl] for lbl in labels])
        cqm_obj = evaluate_mvt_objective(
            mu, sigma, lam, cqm_z,
            prev_portfolio=prev, turnover_cost=turnover_cost,
        )
        assert abs(bqm_obj - cqm_obj) < 1e-10


class TestEvaluateMvtObjective:
    """Tests for evaluate_mvt_objective."""

    def test_matches_quboproblem_energy_at_feasible(self):
        """At feasible z, BQM energy (incl. offset) equals the penalty-free objective.

        bf.energy = offset + z^T Q z. At feasible sum(z)=K the penalty matrix
        contributes -AK^2 which cancels the AK^2 offset, leaving just the objective.
        """
        N, K = 8, 3
        mu, sigma, labels = _random_instance(N)
        lam, penalty = 0.5, 4.0

        bqm = build_mvt_bqm(mu, sigma, lam, K, penalty, labels=labels)
        bf = solve_qubo_bruteforce(bqm, cardinality=K)

        obj_val = evaluate_mvt_objective(mu, sigma, lam, bf.sample)
        assert abs(bf.energy - obj_val) < 1e-10, (
            f"BQM energy {bf.energy} != objective {obj_val}"
        )

    def test_turnover_cost_increases_for_changed_positions(self):
        N = 6
        mu = np.zeros(N)
        sigma = np.eye(N) * 0.01
        prev = np.array([1, 1, 1, 0, 0, 0])

        # Same portfolio: zero turnover
        val_same = evaluate_mvt_objective(mu, sigma, 0.5, prev, prev, turnover_cost=0.1)
        # Different portfolio: some turnover
        new = np.array([0, 0, 0, 1, 1, 1])
        val_diff = evaluate_mvt_objective(mu, sigma, 0.5, new, prev, turnover_cost=0.1)
        assert val_diff > val_same

    def test_per_asset_turnover_cost(self):
        N = 4
        mu = np.zeros(N)
        sigma = np.zeros((N, N))
        prev = np.array([1, 0, 0, 0])
        z = np.array([0, 1, 0, 0])  # sell asset 0, buy asset 1
        costs = np.array([0.1, 0.2, 0.0, 0.0])

        val = evaluate_mvt_objective(mu, sigma, 0.0, z, prev, turnover_cost=costs)
        # Turnover: |0-1|*0.1 + |1-0|*0.2 = 0.1 + 0.2 = 0.3
        assert abs(val - 0.3) < 1e-12
