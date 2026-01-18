# momentest

**Simulated Method of Moments (SMM) and Generalized Method of Moments (GMM) Estimation in Python**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`momentest` provides a clean, high-level API for moment-based estimation methods commonly used in econometrics and structural estimation. It handles the boilerplate of criterion functions, weighting matrices, two-step procedures, and inference so you can focus on specifying your model.

### Key Features

- **Simple API**: Just pass your moment function and bounds - get estimates in 5 lines
- **Two-step optimal weighting**: Automatic efficient GMM/SMM estimation
- **Common Random Numbers (CRN)**: Smooth SMM objectives for reliable optimization
- **Bootstrap inference**: Standard errors and confidence intervals
- **Diagnostic visualizations**: Objective landscapes, moment contributions, convergence plots
- **Educational resources**: Built-in datasets, DGPs, and interactive tutorials
- **Publication-ready output**: Tables and plots for papers

## Installation

```bash
pip install momentest
```

For development (includes testing and plotting dependencies):
```bash
git clone https://github.com/haytug/momentest.git
cd momentest
pip install -e ".[dev]"
```

## Quick Start

### GMM Example

Estimate parameters of a truncated normal distribution using GMM:

```python
import numpy as np
from momentest import gmm_estimate, load_econ381

# Load example data (test scores bounded 0-450)
dataset = load_econ381()
data = dataset['data']

def moment_func(data, theta):
    """Per-observation moment conditions for mean and variance."""
    mu, sigma = theta
    n = len(data)
    moments = np.zeros((n, 2))
    moments[:, 0] = data - mu                    # E[X - μ] = 0
    moments[:, 1] = (data - mu)**2 - sigma**2    # E[(X-μ)² - σ²] = 0
    return moments

result = gmm_estimate(
    data=data,
    moment_func=moment_func,
    bounds=[(0, 1000), (1, 500)],
    k=2,
    weighting="optimal",
)

print(f"μ = {result.theta[0]:.2f} (SE: {result.se[0]:.2f})")
print(f"σ = {result.theta[1]:.2f} (SE: {result.se[1]:.2f})")
```

### SMM Example

Estimate parameters when moments must be simulated:

```python
import numpy as np
from momentest import smm_estimate

def sim_func(theta, shocks):
    """Simulate data: X = μ + σ * ε where ε ~ N(0,1)."""
    mu, sigma = theta
    return mu + sigma * shocks

def moment_func(sim_data):
    """Compute mean and variance from simulated data."""
    return np.column_stack([sim_data, sim_data**2])

# Target moments from data
data_moments = np.array([5.0, 26.0])  # E[X]=5, E[X²]=26 → μ=5, σ=1

result = smm_estimate(
    sim_func=sim_func,
    moment_func=moment_func,
    data_moments=data_moments,
    bounds=[(0, 10), (0.1, 5)],
    n_sim=1000,
    shock_dim=1,
    weighting="optimal",
)

print(f"μ = {result.theta[0]:.4f}, σ = {result.theta[1]:.4f}")
```

## Methodology

### GMM vs SMM

| | GMM | SMM |
|---|---|---|
| **Moments** | Analytical (computed from data) | Simulated (from model) |
| **Use when** | Closed-form moment conditions exist | Model requires simulation |
| **Randomness** | Deterministic objective | Uses CRN for smooth objectives |
| **Variance** | $(D'WD)^{-1}$ | $(1+1/S)(D'WD)^{-1}$ |

### The Estimators

**GMM** minimizes the quadratic form:

$$\hat{\theta}_{GMM} = \arg\min_\theta \bar{g}(\theta)' W \bar{g}(\theta)$$

where $\bar{g}(\theta) = \frac{1}{n}\sum_{i=1}^n g(y_i, \theta)$ are sample moment conditions.

**SMM** minimizes:

$$\hat{\theta}_{SMM} = \arg\min_\theta [m^{data} - \bar{m}(\theta)]' W [m^{data} - \bar{m}(\theta)]$$

where $\bar{m}(\theta) = \frac{1}{S}\sum_{s=1}^S m(\tilde{y}_s(\theta))$ are simulated moments.

### Weighting Matrices

- `"identity"`: W = I. Simple, consistent estimates.
- `"optimal"`: W = S⁻¹. Two-step efficient estimation using inverse of moment covariance.
- `"user"`: Custom weighting matrix.

### Identification

- **k** = number of moments, **p** = number of parameters
- Need k ≥ p (order condition for identification)
- When k > p (overidentified): can test model specification with J-test

## API Reference

### High-Level Functions (Recommended)

#### `gmm_estimate`

```python
from momentest import gmm_estimate

result = gmm_estimate(
    data,                    # Data array or dict of arrays
    moment_func,             # g(data, theta) -> (n, k) array
    bounds,                  # [(lb, ub), ...] for each parameter
    k,                       # Number of moment conditions
    weighting="optimal",     # "identity", "optimal", or "user"
    n_global=50,             # Global search candidates
    seed=42,                 # Random seed
)
```

#### `smm_estimate`

```python
from momentest import smm_estimate

result = smm_estimate(
    sim_func,                # sim_func(theta, shocks) -> simulated data
    moment_func,             # moment_func(sim_data) -> (n_sim, k) moments
    data_moments,            # Target moments from data
    bounds,                  # [(lb, ub), ...] for each parameter
    n_sim=1000,              # Number of simulations
    shock_dim=1,             # Dimension of shocks per simulation
    seed=42,                 # Random seed (for CRN)
    weighting="optimal",     # "identity", "optimal", or "user"
    n_global=50,             # Global search candidates
)
```

**Returns**: Result object with:
- `theta`: Parameter estimates (numpy array)
- `se`: Standard errors
- `objective`: Minimized criterion value
- `converged`: Whether optimization converged

### Bootstrap Inference

```python
from momentest import bootstrap, EstimationSetup, SMMEngine

# Create engine and setup
engine = SMMEngine(k=2, p=2, n_sim=1000, shock_dim=1,
                   sim_func=sim_func, moment_func=moment_func, seed=42)

setup = EstimationSetup(
    mode="SMM", model_name="my_model", moment_type="custom",
    k=2, p=2, n_sim=1000, shock_dim=1, seed=42, weighting="optimal"
)

# Run bootstrap
boot_result = bootstrap(
    setup=setup,
    data_moments=data_moments,
    bounds=bounds,
    n_boot=200,              # Number of bootstrap replications
    alpha=0.05,              # For 95% confidence intervals
    n_jobs=-1,               # Parallel (-1 = all cores)
    engine=engine,
)

print(f"Bootstrap SE: {boot_result.se}")
print(f"95% CI: [{boot_result.ci_lower}, {boot_result.ci_upper}]")
```

### Inference Tools

```python
from momentest import j_test, confidence_interval, wald_test, asymptotic_se

# J-test for overidentification (k > p)
j_result = j_test(objective, n, k, p)
print(f"J-stat: {j_result.statistic:.2f}, p-value: {j_result.pvalue:.4f}")

# Confidence intervals
ci_lower, ci_upper = confidence_interval(theta, se, alpha=0.05)

# Wald test for H0: theta = null_value
wald_result = wald_test(theta, se, null_value=0)
```

### Output Tools

#### Tables

```python
from momentest import table_estimates, table_moments, table_bootstrap, summary

# Parameter estimates table
print(table_estimates(theta, se, param_names=["μ", "σ"], 
                      ci_lower=ci_lower, ci_upper=ci_upper))

# Moment comparison (data vs model)
print(table_moments(data_moments, model_moments, 
                    moment_names=["Mean", "Variance"]))

# Bootstrap summary
print(table_bootstrap(boot_estimates, theta_hat, param_names=["μ", "σ"]))

# Full estimation summary
print(summary(theta, se, objective, data_moments, model_moments,
              k=2, p=2, n=161, converged=True, method="GMM"))
```

#### Diagnostic Plots

```python
from momentest import (
    plot_moment_comparison,
    plot_bootstrap_distribution,
    plot_objective_landscape,
    plot_moment_contributions,
    plot_convergence,
    plot_sanity,
)

# Data vs model moments
plot_moment_comparison(data_moments, model_moments, 
                       moment_names=["Mean", "Var"])

# Bootstrap distribution
plot_bootstrap_distribution(boot_estimates, theta_hat,
                            param_names=["μ", "σ"])

# 3D objective landscape
plot_objective_landscape(engine, data_moments, W, bounds,
                         param_names=["μ", "σ"])

# Moment contributions to objective
plot_moment_contributions(engine, theta_hat, data_moments, W,
                          moment_names=["Mean", "Var"])

# Optimization convergence
plot_convergence(history, param_names=["μ", "σ"])

# Multi-start sanity check
plot_sanity(trial_results, param_names=["μ", "σ"])
```

### Built-in Datasets

```python
from momentest import list_datasets, load_dataset, load_econ381

# List available datasets
print(list_datasets())
# ['econ381', 'consumption', 'labor_supply', 'asset_pricing']

# Load as DatasetBundle (includes exercises, documentation)
dataset = load_dataset('consumption')
print(dataset.description)
print(dataset.exercises)

# Quick load for econ381
data = load_econ381()['data']
```

| Dataset | Description | k | p | Difficulty |
|---------|-------------|---|---|------------|
| `econ381` | Test scores (truncated normal) | 2 | 2 | Beginner |
| `consumption` | FRED consumption data (Hall 1978) | 2 | 2 | Beginner |
| `labor_supply` | PSID labor supply (MaCurdy 1981) | 4 | 3 | Intermediate |
| `asset_pricing` | Ken French + consumption (CCAPM) | 3 | 2 | Intermediate |

### Data Generating Processes (DGPs)

For learning and validation - generate synthetic data with known true parameters:

```python
from momentest import list_dgps, linear_iv, consumption_savings, load_dgp

# List available DGPs
print(list_dgps())
# ['linear_iv', 'consumption_savings', 'dynamic_discrete_choice']

# Generate data from linear IV model
dgp = linear_iv(n=1000, seed=42, beta0=1.0, beta1=2.0, rho=0.5)
print(f"True parameters: {dgp.true_theta}")
print(f"Data keys: {dgp.data.keys()}")

# Use the moment function for estimation
result = gmm_estimate(
    data=dgp.data,
    moment_func=dgp.moment_function,
    bounds=[(-10, 10), (-10, 10)],
    k=dgp.k,
)
print(f"Estimated: {result.theta}, True: {dgp.true_theta}")
```

| DGP | Description | Difficulty |
|-----|-------------|------------|
| `linear_iv` | Linear model with endogeneity and instruments | Beginner |
| `consumption_savings` | Two-period consumption-savings model | Intermediate |
| `dynamic_discrete_choice` | Simplified Rust (1987) bus engine model | Advanced |

## Tutorials

Interactive Jupyter notebooks in [`tutorials/`](tutorials/):

| Tutorial | Description |
|----------|-------------|
| [01_gmm_basics.ipynb](tutorials/01_gmm_basics.ipynb) | GMM theory and linear IV example |
| [02_smm_basics.ipynb](tutorials/02_smm_basics.ipynb) | SMM theory and consumption model |
| [03_optimal_weighting.ipynb](tutorials/03_optimal_weighting.ipynb) | Two-step efficient estimation |
| [04_bootstrap_inference.ipynb](tutorials/04_bootstrap_inference.ipynb) | Bootstrap SE and confidence intervals |
| [05_diagnostics.ipynb](tutorials/05_diagnostics.ipynb) | Visualization and model checking |
| [06_advanced_models.ipynb](tutorials/06_advanced_models.ipynb) | Dynamic models and DDC |
| [07_real_data_applications.ipynb](tutorials/07_real_data_applications.ipynb) | Real-world applications |

## Examples

Working examples in [`examples/`](examples/):

| Example | Description |
|---------|-------------|
| [`truncated_normal_gmm.py`](examples/truncated_normal_gmm.py) | GMM with 2 and 4 moments |
| [`truncated_normal_smm.py`](examples/truncated_normal_smm.py) | SMM with CRN |

Run an example:
```bash
python examples/truncated_normal_gmm.py
```

## Advanced Usage

### Custom Engine

For full control over the estimation process:

```python
from momentest import SMMEngine, EstimationSetup, estimate

# Create engine with custom settings
engine = SMMEngine(
    k=2, p=2, n_sim=2000, shock_dim=1,
    sim_func=sim_func, moment_func=moment_func, seed=42
)

# Create setup
setup = EstimationSetup(
    mode="SMM",
    model_name="my_model",
    moment_type="mean_variance",
    k=2, p=2, n_sim=2000, shock_dim=1,
    seed=42,
    weighting="optimal",
)

# Run estimation with custom options
result = estimate(
    setup=setup,
    data_moments=data_moments,
    bounds=bounds,
    n_global=100,
    local_method="L-BFGS-B",
    engine=engine,
)
```

### Using Pre-drawn Shocks

For reproducibility or comparing methods with identical randomness:

```python
import numpy as np
from momentest import SMMEngine, smm_estimate

# Pre-draw shocks
rng = np.random.default_rng(42)
shocks = rng.standard_normal((1000, 1))

# Use same shocks for different estimations
result = smm_estimate(
    sim_func=sim_func,
    moment_func=moment_func,
    data_moments=data_moments,
    bounds=bounds,
    n_sim=1000,
    shock_dim=1,
    shocks=shocks,  # Pass pre-drawn shocks
)
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{momentest,
  title={momentest: SMM and GMM Estimation in Python},
  author={Aytug, Harry},
  year={2025},
  url={https://github.com/haytug/momentest}
}
```

## References

- **Hansen, L.P. (1982).** Large Sample Properties of Generalized Method of Moments Estimators. *Econometrica*, 50(4), 1029-1054.
- **McFadden, D. (1989).** A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration. *Econometrica*, 57(5), 995-1026.
- **Pakes, A. & Pollard, D. (1989).** Simulation and the Asymptotics of Optimization Estimators. *Econometrica*, 57(5), 1027-1057.
- **Newey, W.K. & McFadden, D. (1994).** Large Sample Estimation and Hypothesis Testing. *Handbook of Econometrics*, Vol. 4.
- **Hall, P. & Horowitz, J.L. (1996).** Bootstrap Critical Values for Tests Based on Generalized-Method-of-Moments Estimators. *Econometrica*, 64(4), 891-916.

## License

MIT License - see [LICENSE](LICENSE) for details.
