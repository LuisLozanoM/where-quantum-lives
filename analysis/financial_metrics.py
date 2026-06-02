"""
financial_metrics.py
====================

Phase 3 — financial-overlay metrics for the §5.5 rewrite of paper2.

All functions take a daily-returns time series (1-d numpy array or pandas
Series indexed by date) and return scalar metrics.  The companion module
reconstruct_pnl.py produces these time series from the saved
`selected_industries` lists in `results/qpu/a4_equity_qpu.jsonl` and the
public FF-49 daily returns.

Metrics implemented:
  - sharpe(r, rf=0, periods=252)
  - sortino(r, rf=0, periods=252)
  - information_ratio(r_strategy, r_benchmark, periods=252)
  - max_drawdown(r)
  - probabilistic_sharpe_ratio(r, sr_benchmark=0, periods=252)
        Bailey & López de Prado (2012)
  - deflated_sharpe_ratio(r, n_trials, sr_benchmark=0, periods=252)
        Bailey & López de Prado (2014); n_trials = number of independent
        configurations tried (each (N, ρ, A, budget) cell counts as a trial).
  - probability_of_backtest_overfit(in_sample_returns_dict)
        Bailey, Borwein, López de Prado, Zhu (2014); combinatorially-symmetric
        cross-validation.  Takes a dict {strategy_name -> daily_returns}.
  - tc_adjusted_returns(r, turnover, cost_bps=5)
        Linear transaction-cost adjustment.  cost_bps is bps per name turned
        over per rebalance.

No dependence on saved JSONL files — these are pure math on numpy/pandas
inputs.  Tested via doctests below; run as:
    python -m doctest analysis/financial_metrics.py -v
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import norm


def _as_array(r: Iterable) -> np.ndarray:
    """Coerce input to a 1-d float numpy array, dropping NaNs."""
    arr = np.asarray(r, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def sharpe(r: Iterable, rf: float = 0.0, periods: int = 252) -> float:
    """Annualised Sharpe ratio.

    >>> round(sharpe([0.001, 0.002, -0.001, 0.0015, 0.0005], periods=252), 2) > 0
    True
    """
    arr = _as_array(r)
    if len(arr) < 2:
        return float("nan")
    excess = arr - rf / periods
    sd = np.std(excess, ddof=1)
    if sd == 0.0:
        return float("nan")
    return float(np.mean(excess) / sd * math.sqrt(periods))


def sortino(r: Iterable, rf: float = 0.0, periods: int = 252) -> float:
    """Annualised Sortino ratio (downside-deviation denominator)."""
    arr = _as_array(r)
    if len(arr) < 2:
        return float("nan")
    excess = arr - rf / periods
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    dd = math.sqrt(np.mean(downside ** 2))
    if dd == 0.0:
        return float("nan")
    return float(np.mean(excess) / dd * math.sqrt(periods))


def information_ratio(r_strategy: Iterable, r_benchmark: Iterable,
                      periods: int = 252) -> float:
    """Annualised information ratio vs a benchmark daily-returns series."""
    s = _as_array(r_strategy)
    b = _as_array(r_benchmark)
    n = min(len(s), len(b))
    if n < 2:
        return float("nan")
    active = s[:n] - b[:n]
    te = np.std(active, ddof=1)
    if te == 0.0:
        return float("nan")
    return float(np.mean(active) / te * math.sqrt(periods))


def max_drawdown(r: Iterable) -> float:
    """Maximum drawdown of the cumulative-return series.

    Returns a value in [-1, 0]; 0 = no drawdown, -0.20 = 20% max drawdown.
    """
    arr = _as_array(r)
    if len(arr) == 0:
        return float("nan")
    wealth = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return float(np.min(dd))


def probabilistic_sharpe_ratio(r: Iterable, sr_benchmark: float = 0.0,
                                periods: int = 252) -> float:
    """Bailey & López de Prado (2012) Probabilistic Sharpe Ratio.

    PSR is the probability that the observed Sharpe exceeds the benchmark
    SR given the higher moments of the return distribution.  Range [0, 1].
    """
    arr = _as_array(r)
    n = len(arr)
    if n < 4:
        return float("nan")
    sr_hat = sharpe(arr, periods=periods)
    if not math.isfinite(sr_hat):
        return float("nan")
    # Deannualise sample SR for the test (PSR formula uses per-period SR)
    sr_per_period = sr_hat / math.sqrt(periods)
    sr_bench_per_period = sr_benchmark / math.sqrt(periods)
    # Sample skew and kurtosis (Pearson kurtosis, not excess)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd == 0.0:
        return float("nan")
    centered = (arr - mu) / sd
    skew = float(np.mean(centered ** 3))
    kurt = float(np.mean(centered ** 4))
    denom = math.sqrt(
        max(1e-12, (1.0 - skew * sr_per_period
                    + 0.25 * (kurt - 1.0) * (sr_per_period ** 2)) / (n - 1))
    )
    z = (sr_per_period - sr_bench_per_period) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(r: Iterable, n_trials: int,
                          sr_benchmark: float = 0.0, periods: int = 252) -> float:
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio.

    DSR deflates the PSR by an estimate of the maximum SR expected from
    n_trials i.i.d. trials with sigma_SR computed from the observed Sharpe
    variance under the null.

    n_trials should equal the number of independent trial configurations
    (e.g., grid cells in a hyperparameter sweep).
    """
    arr = _as_array(r)
    if n_trials < 1:
        return float("nan")
    # SR_0 = expected maximum SR under the null over n_trials
    # Approximation: SR_0 ≈ sqrt(2 ln n_trials) * sigma_SR_hat
    # where sigma_SR_hat is the inferred std of trial SRs.  Without a panel of
    # trial SRs we use the standard Bailey-LdP approximation with the
    # Euler-Mascheroni constant.
    if n_trials == 1:
        sr0_per_period = 0.0
    else:
        euler = 0.5772156649015329
        z_alpha = norm.ppf(1.0 - 1.0 / n_trials)
        z_beta = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
        sr0 = (1.0 - euler) * z_alpha + euler * z_beta
        sr0_per_period = sr0 / math.sqrt(periods) if periods > 1 else sr0
    sr_bench_annual = float(sr_benchmark + sr0_per_period * math.sqrt(periods))
    return probabilistic_sharpe_ratio(arr, sr_benchmark=sr_bench_annual,
                                       periods=periods)


def probability_of_backtest_overfit(returns_by_config: Mapping[str, Iterable],
                                    n_splits: int = 16) -> float:
    """Bailey, Borwein, López de Prado, Zhu (2014) Probability of Backtest Overfit.

    Combinatorially-symmetric cross-validation: partition the return series
    into S equal slices, choose S/2 for in-sample, the rest for out-of-sample,
    rank strategies by IS Sharpe and check whether the IS-best strategy's
    OOS rank is below median (overfit).  PBO is the fraction of CSCV splits
    where this happens.

    returns_by_config: {strategy_name -> daily returns time series of equal length}
    n_splits (S): must be even.  Default 16 → C(16, 8) = 12,870 splits.
    """
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    names = list(returns_by_config)
    if len(names) < 2:
        raise ValueError("PBO requires at least 2 strategies")
    arrays = [_as_array(returns_by_config[n]) for n in names]
    L = min(len(a) for a in arrays)
    if L < n_splits:
        raise ValueError(f"Each returns series must have >= {n_splits} observations")
    L_trim = (L // n_splits) * n_splits  # truncate so slices are equal
    slice_size = L_trim // n_splits
    slices = [
        np.stack([a[i * slice_size:(i + 1) * slice_size] for a in arrays], axis=0)
        for i in range(n_splits)
    ]
    from itertools import combinations
    half = n_splits // 2
    overfits = 0
    total = 0
    for in_idx in combinations(range(n_splits), half):
        in_set = set(in_idx)
        out_idx = [i for i in range(n_splits) if i not in in_set]
        is_returns = np.concatenate([slices[i] for i in in_idx], axis=1)
        oos_returns = np.concatenate([slices[i] for i in out_idx], axis=1)
        is_sharpe = np.array([sharpe(is_returns[j], periods=1) for j in range(len(names))])
        oos_sharpe = np.array([sharpe(oos_returns[j], periods=1) for j in range(len(names))])
        if not (np.all(np.isfinite(is_sharpe)) and np.all(np.isfinite(oos_sharpe))):
            continue
        best_is = int(np.argmax(is_sharpe))
        oos_rank = int(np.sum(oos_sharpe < oos_sharpe[best_is])) + 1  # 1-indexed
        if oos_rank < (len(names) + 1) / 2:
            overfits += 1
        total += 1
    if total == 0:
        return float("nan")
    return overfits / total


def tc_adjusted_returns(r: Iterable, turnover: Iterable, cost_bps: float = 5.0
                        ) -> np.ndarray:
    """Linear transaction-cost adjustment.

    r:         daily portfolio gross returns (numpy array, length T)
    turnover:  per-rebalance fraction of portfolio churned (length T;
               zero on non-rebalance days)
    cost_bps:  basis points charged per unit turnover per rebalance
               (5 bps = 0.0005)

    Returns r_net = r - turnover * cost_bps / 10000
    """
    arr = np.asarray(r, dtype=float).ravel()
    to = np.asarray(turnover, dtype=float).ravel()
    if len(arr) != len(to):
        raise ValueError(f"returns ({len(arr)}) and turnover ({len(to)}) must match length")
    cost = to * (cost_bps / 10000.0)
    return arr - cost


# ---------- self-test ----------

if __name__ == "__main__":
    rng = np.random.default_rng(20260531)
    r = rng.normal(0.0005, 0.012, size=252 * 4)
    print(f"Sharpe:              {sharpe(r):.3f}")
    print(f"Sortino:             {sortino(r):.3f}")
    print(f"MaxDD:               {max_drawdown(r):.4f}")
    print(f"PSR vs SR=0:         {probabilistic_sharpe_ratio(r):.4f}")
    print(f"DSR (n_trials=10):   {deflated_sharpe_ratio(r, n_trials=10):.4f}")
    print(f"DSR (n_trials=100):  {deflated_sharpe_ratio(r, n_trials=100):.4f}")

    # PBO test with 3 random strategies
    pbo_input = {
        f"strat_{k}": rng.normal(0.0003 * (k + 1), 0.012, size=252 * 4)
        for k in range(3)
    }
    print(f"PBO (3 strats, S=16): {probability_of_backtest_overfit(pbo_input, 16):.4f}")
