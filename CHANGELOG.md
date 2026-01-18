# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-18

### Added

#### Core Estimation
- **SMM Estimation** (`smm_estimate`)
  - User-defined simulation and moment functions
  - Common Random Numbers (CRN) for smooth objectives
  - Identity and optimal (two-step) weighting
  - Automatic standard error computation
  - Support for pre-drawn shocks
- **GMM Estimation** (`gmm_estimate`)
  - Per-observation moment conditions
  - Support for array and dictionary data formats
  - Identity and optimal (two-step) weighting
  - Asymptotic standard errors
- **Low-level API**
  - `SMMEngine`: Full control over SMM computation
  - `GMMEngine`: Full control over GMM computation
  - `EstimationSetup`: Configuration dataclass
  - `estimate`: Core estimation function
  - `global_search`: Sobol/LHS sampling for global optimization
  - `local_optimize`: scipy.optimize wrapper with bounds

#### Inference
- **Bootstrap inference** (`bootstrap`)
  - Parallel bootstrap replications
  - Standard errors as std of bootstrap estimates
  - Percentile confidence intervals
  - Support for both SMM and GMM
- **Statistical tests**
  - `j_test`: Hansen's J-test for overidentification
  - `wald_test`: Wald test for parameter restrictions
  - `confidence_interval`: Asymptotic confidence intervals
  - `sandwich_covariance`: Robust covariance estimation
  - `asymptotic_se`: Standard error computation

#### Output Tools
- **Tables**
  - `table_estimates`: Parameter estimates with SEs and CIs
  - `table_moments`: Data vs model moment comparison
  - `table_bootstrap`: Bootstrap results summary
  - `summary`: Comprehensive estimation summary
- **Diagnostic Plots**
  - `plot_objective_landscape`: 3D surface/contour of objective
  - `plot_moment_contributions`: Per-moment contribution to objective
  - `plot_identification`: Moment-parameter relationships
  - `plot_convergence`: Optimization path visualization
  - `plot_bootstrap_distribution`: Bootstrap histograms with CIs
  - `plot_moment_comparison`: Data vs model bar charts
  - `plot_marginal_objective`: 1D slices around optimum
  - `plot_sanity`: Multi-start convergence diagnostics

#### Educational Resources
- **Built-in Datasets**
  - `load_econ381`: Test scores (truncated normal, n=161)
  - `load_consumption`: FRED consumption data (Hall 1978)
  - `load_labor_supply`: PSID labor supply (MaCurdy 1981)
  - `load_asset_pricing`: Ken French + consumption (CCAPM)
  - `DatasetBundle`: Container with data, documentation, exercises
- **Data Generating Processes (DGPs)**
  - `linear_iv`: Linear model with endogeneity and instruments
  - `consumption_savings`: Two-period consumption-savings model
  - `dynamic_discrete_choice`: Simplified Rust (1987) DDC model
  - `DGPResult`: Container with data, true parameters, moment function
- **Tutorials** (Jupyter notebooks)
  - 01_gmm_basics: GMM theory and linear IV
  - 02_smm_basics: SMM theory and consumption model
  - 03_optimal_weighting: Two-step efficient estimation
  - 04_bootstrap_inference: Bootstrap SE and CIs
  - 05_diagnostics: Visualization and model checking
  - 06_advanced_models: Dynamic models and DDC
  - 07_real_data_applications: Real-world examples

#### Testing
- Property-based tests using Hypothesis
- Econometric validation tests (DGP parameter recovery)
- Placebo tests (flat objectives, noise-only moments)
- Monte Carlo tests (bias, coverage, efficiency)
- Comprehensive unit tests for all modules

#### Documentation
- Detailed README with API reference
- GMM methodology guide (`docs/gmm.md`)
- SMM methodology guide (`docs/smm.md`)
- Working examples with truncated normal distribution
- Inline docstrings with examples

### Technical Details
- Pure Python implementation (no C++ dependencies)
- NumPy/SciPy for numerical computation
- joblib for parallel bootstrap
- matplotlib for plotting (optional)
- Type hints throughout
- Python 3.9+ support

[Unreleased]: https://github.com/haytug/momentest/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/haytug/momentest/releases/tag/v0.1.0
