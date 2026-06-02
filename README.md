# Where the Quantum Lives in D-Wave Hybrid Portfolio Optimization

Reproducibility repository for the paper

> **Where the Quantum Lives in D-Wave Hybrid Portfolio Optimization: An Operational Decomposition Audit**
> Luis Lozano. Preprint: [arXiv:2605.17623](https://arxiv.org/abs/2605.17623). Under peer review.

## What this repository contains

```
src/paper2/      Python package — QUBO/CQM construction, D-Wave hybrid
                 + direct-QPU + classical solver wrappers, MVT objective
                 evaluators, synthetic-instance generators
scripts/         Driver scripts for every experiment in the paper
                 (see "Reproducing results" below)
analysis/        Phase 3 / Phase 4 / Phase 4.5 post-hoc analysis scripts:
                 timing breakdowns, dwell-vs-quality correlation,
                 Wilcoxon tests, Tabu ablation, FF-49 P&L reconstruction
                 + financial-metrics summary
configs/         Solver configuration files
tests/           Pytest suite (offline + live-QPU markers; live tests
                 gated by marker, skipped without D-Wave credentials)
data/raw/        Public datasets used in the paper
                   - equities/ → symlink to paper1's Fama-French 49
                     industry daily portfolios
pyproject.toml   Package metadata and dependencies
LICENSE          MIT (covers the code in this repository only)
```

The compiled manuscript, supplement, and working draft notes
(`manuscript/`, `manuscript/drafts/`) are intentionally not tracked here;
this repository is the code and result tables needed to reproduce the
experimental results.

## Datasets

The Fama-French 49 industry daily portfolio data is publicly available
from the [Kenneth R. French Data
Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html).
The copy used by this paper is referenced via symlink to the companion
penalty-encoding paper (paper 1) repository.

## Quick start

This package depends on a small set of utilities from the companion paper's
repository (paper 1, *A Penalty-Free Pipeline for Direct Quantum-Annealer
Portfolio Optimization*, [arXiv:2605.17628](https://arxiv.org/abs/2605.17628)).
Clone both repositories side by side:

```bash
# Companion package (paper 1) — provides QUBO/embedding utilities reused here
git clone https://github.com/LuisLozanoM/penalty-free-portfolio.git

# This repository (paper 2)
git clone https://github.com/LuisLozanoM/where-quantum-lives.git
cd where-quantum-lives

# Install in editable mode (with dev dependencies for pytest)
python -m venv .venv && source .venv/bin/activate
pip install -e ../penalty-free-portfolio    # install paper 1 first
pip install -e ".[dev]"

# Run the offline test suite
pytest -m "not live_qpu and not live_hybrid"

# Run the offline analyses on already-saved JSONLs (no QPU access required)
python analysis/timing_breakdown.py
python analysis/per_chain_strength_split.py
python analysis/verify_table5_determinism.py
python analysis/wilcoxon_cqm_vs_bqm.py
python analysis/dwell_quality_correlation_v2.py
python analysis/qpu_replacement_ablation.py       # ~2 minutes wall-clock
python analysis/reconstruct_pnl.py
python analysis/ff49_financial_summary.py
```

Live-QPU experiments require D-Wave Leap credentials in a `dwave.conf`
at the project root (this file is gitignored). To install the optional
D-Wave SDK extras:

```bash
pip install -e ".[qpu]"      # direct QPU + minor-embedding stack
pip install -e ".[hybrid]"   # hybrid CQM / BQM samplers only
pip install -e ".[classical]"  # Gurobi + OR-Tools
```

## Reproducing results in the paper

| Paper result | Script(s) |
|---|---|
| §5.1 Direct-QPU chain-break tables (Table 1 + per-strength split) | `scripts/run_synthetic_qpu.py` → `analysis/per_chain_strength_split.py` |
| §5.2 Hybrid CQM vs BQM (Table 2 + Wilcoxon p-values) | `scripts/run_synthetic_hybrid.py` → `analysis/wilcoxon_cqm_vs_bqm.py` |
| §5.3 Full timing breakdown (Table 3) | `analysis/timing_breakdown.py` |
| §5.4 Gap to Gurobi (Table 4) | `scripts/run_classical_baselines.py` |
| §5.5 Budget sweep + 10-rep determinism (Table 5 + cell verification) | `scripts/run_hybrid_budget_sweep.py` + `scripts/run_repeated_hybrid.py` → `analysis/verify_table5_determinism.py` |
| §5.6 QPU-dwell vs solution-quality correlation | `analysis/dwell_quality_correlation_v2.py` |
| §5.7 QPU-replacement ablation (TabuSampler @ 5 s) | `analysis/qpu_replacement_ablation.py` |
| §5.8 FF-49 out-of-sample financial overlay | `analysis/reconstruct_pnl.py` → `analysis/ff49_financial_summary.py` |
| §A Penalty robustness | `scripts/run_penalty_robustness.py` |
| Figure regeneration | `scripts/generate_figures.py`, `scripts/generate_gap_figure.py` |

All scripts accept standard `argparse` options; pass `--help` for the
flag surface. The Methods section of the paper documents every solver
setting that is not a default of the underlying SDK (D-Wave Ocean,
Gurobi 13.0, `dwave.samplers`).

## Licensing

The **code** in this repository is released under the
[MIT License](LICENSE), Copyright © 2026 Luis Lozano.

The **Fama--French 49 industry daily portfolios** in `data/raw/equities/`
are third-party data provided by the
[Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/)
and remain subject to that library's usage terms.  Re-users should cite
the upstream data provider in addition to the present repository.

## Citation

If you use this code, please cite the paper (preprint form until the
peer-reviewed version is published):

```bibtex
@misc{lozano2026wherequantum,
  author       = {Lozano, Luis},
  title        = {Where the Quantum Lives in {D-Wave} Hybrid Portfolio Optimization: An Operational Decomposition Audit},
  year         = {2026},
  eprint       = {2605.17623},
  archivePrefix = {arXiv},
  primaryClass = {quant-ph},
  note         = {Preprint, under peer review}
}
```

## Contact

Luis Lozano — `lalozanom@tec.mx` — EGADE Business School, Tecnológico de
Monterrey, Campus Santa Fe, Mexico City, Mexico.
ORCID: [0000-0001-7202-3437](https://orcid.org/0000-0001-7202-3437).
