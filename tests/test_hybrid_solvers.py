"""Tests for Paper 2A D-Wave hybrid solver wrappers."""

from __future__ import annotations

import sys
from types import ModuleType

import dimod
import numpy as np
import pytest

from paper1.formulations import QuboProblem, to_dimod_bqm
from paper2.formulations import build_mvt_cqm
from paper2.solvers.dwave_hybrid_bqm import solve_hybrid_bqm
from paper2.solvers.dwave_hybrid_cqm import solve_hybrid_cqm


def _install_fake_dwave_system(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bqm_sampleset=None,
    cqm_sampleset=None,
) -> dict:
    captured: dict[str, object] = {}

    class FakeLeapHybridBQMSampler:
        def __init__(self, **kwargs):
            captured["bqm_init_kwargs"] = kwargs

        def sample(self, bqm, *, time_limit):
            captured["bqm"] = bqm
            captured["bqm_time_limit"] = time_limit
            if bqm_sampleset is None:
                raise AssertionError("Unexpected BQM sampler call.")
            return bqm_sampleset

    class FakeLeapHybridCQMSampler:
        def __init__(self, **kwargs):
            captured["cqm_init_kwargs"] = kwargs

        def sample_cqm(self, cqm, *, time_limit):
            captured["cqm"] = cqm
            captured["cqm_time_limit"] = time_limit
            if cqm_sampleset is None:
                raise AssertionError("Unexpected CQM sampler call.")
            return cqm_sampleset

    dwave_module = ModuleType("dwave")
    system_module = ModuleType("dwave.system")
    system_module.LeapHybridBQMSampler = FakeLeapHybridBQMSampler
    system_module.LeapHybridCQMSampler = FakeLeapHybridCQMSampler
    dwave_module.system = system_module

    monkeypatch.setitem(sys.modules, "dwave", dwave_module)
    monkeypatch.setitem(sys.modules, "dwave.system", system_module)
    return captured


def test_solve_hybrid_bqm_converts_problem_and_extracts_timing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dwave.conf").write_text("[defaults]\nsolver = hybrid\n", encoding="utf-8")

    problem = QuboProblem(
        matrix=np.array([[-1.0, 0.5], [0.5, -0.25]], dtype=float),
        offset=1.25,
    )
    sampleset = dimod.SampleSet.from_samples(
        [{0: 0, 1: 1}, {0: 1, 1: 1}],
        vartype="BINARY",
        energy=[1.0, -2.0],
        info={
            "run_time": 120_000.0,
            "charge_time": 34_000.0,
            "qpu_access_time": 0.0,
        },
    )

    captured = _install_fake_dwave_system(monkeypatch, bqm_sampleset=sampleset)
    result = solve_hybrid_bqm(problem, time_limit=7.5)

    expected_bqm = to_dimod_bqm(problem)
    assert captured["bqm_init_kwargs"] == {"config_file": "./dwave.conf"}
    assert captured["bqm_time_limit"] == 7.5
    assert captured["bqm"].linear == expected_bqm.linear
    assert captured["bqm"].quadratic == expected_bqm.quadratic
    assert captured["bqm"].offset == pytest.approx(expected_bqm.offset)

    np.testing.assert_array_equal(result.sample, np.array([1, 1]))
    assert result.energy == pytest.approx(problem.energy(result.sample))
    assert result.feasible is False
    assert result.solver_family == "hybrid"
    assert result.solver_name == "hybrid_bqm"
    assert result.hybrid_run_time == pytest.approx(120_000.0)
    assert result.hybrid_charge_time == pytest.approx(34_000.0)
    assert result.hybrid_qpu_access_time == pytest.approx(0.0)
    assert result.wall_clock_solve == pytest.approx(0.12)
    assert result.wall_clock_total >= 0.0
    assert result.solver_timing == {
        "run_time": 120_000.0,
        "charge_time": 34_000.0,
        "qpu_access_time": 0.0,
    }
    assert result.extras == {"qpu_contributed": False}


def test_solve_hybrid_cqm_prefers_best_feasible_sample(monkeypatch):
    mu = np.array([1.0, 0.8, 0.3], dtype=float)
    sigma = np.zeros((3, 3), dtype=float)
    cqm = build_mvt_cqm(mu, sigma, risk_aversion=0.0, cardinality=1, labels=("a", "b", "c"))

    infeasible_sample = {"a": 1, "b": 1, "c": 1}
    best_feasible_sample = {"a": 1, "b": 0, "c": 0}
    worse_feasible_sample = {"a": 0, "b": 1, "c": 0}
    sampleset = dimod.SampleSet.from_samples(
        [infeasible_sample, best_feasible_sample, worse_feasible_sample],
        vartype="BINARY",
        energy=[
            cqm.objective.energy(infeasible_sample),
            cqm.objective.energy(best_feasible_sample),
            cqm.objective.energy(worse_feasible_sample),
        ],
        is_feasible=[False, True, True],
        info={"timing": {"run_time": 200_000.0, "charge_time": 40_000.0, "qpu_access_time": 5_000.0}},
    )

    captured = _install_fake_dwave_system(monkeypatch, cqm_sampleset=sampleset)
    result = solve_hybrid_cqm(cqm, time_limit=9.0, config_file="custom.conf")

    assert captured["cqm_init_kwargs"] == {"config_file": "custom.conf"}
    assert captured["cqm_time_limit"] == 9.0
    np.testing.assert_array_equal(result.sample, np.array([1, 0, 0]))
    assert result.energy == pytest.approx(cqm.objective.energy(best_feasible_sample))
    assert result.feasible is True
    assert result.solver_family == "hybrid"
    assert result.solver_name == "hybrid_cqm"
    assert result.hybrid_run_time == pytest.approx(200_000.0)
    assert result.hybrid_charge_time == pytest.approx(40_000.0)
    assert result.hybrid_qpu_access_time == pytest.approx(5_000.0)
    assert result.wall_clock_solve == pytest.approx(0.2)
    assert result.wall_clock_total >= 0.0
    assert result.extras == {"qpu_contributed": True}


def test_solve_hybrid_cqm_returns_best_infeasible_when_no_feasible(monkeypatch):
    mu = np.array([1.0, 0.8, 0.3], dtype=float)
    sigma = np.zeros((3, 3), dtype=float)
    cqm = build_mvt_cqm(mu, sigma, risk_aversion=0.0, cardinality=1, labels=("a", "b", "c"))

    best_infeasible_sample = {"a": 1, "b": 1, "c": 1}
    worse_infeasible_sample = {"a": 0, "b": 0, "c": 0}
    sampleset = dimod.SampleSet.from_samples(
        [best_infeasible_sample, worse_infeasible_sample],
        vartype="BINARY",
        energy=[
            cqm.objective.energy(best_infeasible_sample),
            cqm.objective.energy(worse_infeasible_sample),
        ],
        is_feasible=[False, False],
        info={
            "run_time": 80_000.0,
            "charge_time": 25_000.0,
            "qpu_access_time": 0.0,
        },
    )

    _install_fake_dwave_system(monkeypatch, cqm_sampleset=sampleset)
    result = solve_hybrid_cqm(cqm, time_limit=3.0)

    np.testing.assert_array_equal(result.sample, np.array([1, 1, 1]))
    assert result.energy == pytest.approx(cqm.objective.energy(best_infeasible_sample))
    assert result.feasible is False
    assert result.hybrid_run_time == pytest.approx(80_000.0)
    assert result.hybrid_charge_time == pytest.approx(25_000.0)
    assert result.hybrid_qpu_access_time == pytest.approx(0.0)
    assert result.wall_clock_solve == pytest.approx(0.08)
    assert result.extras == {"qpu_contributed": False}
