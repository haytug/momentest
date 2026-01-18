"""
momentest - SMM and GMM estimation for econometrics.

A Python package for Simulated Method of Moments (SMM) and Generalized Method
of Moments (GMM) estimation. Provides a clean API for structural estimation
with Common Random Numbers (CRN) support for smooth objective functions.

Simple usage (recommended):

    from momentest import smm_estimate
    
    result = smm_estimate(
        sim_func=my_simulation,
        moment_func=my_moments,
        data_moments=[5.0, 26.0],
        bounds=[(0, 10), (0.1, 5)]
    )
    print(result.theta)

Advanced usage (full control):

    from momentest import SMMEngine, EstimationSetup, estimate
    
    engine = SMMEngine(k=2, p=2, n_sim=1000, shock_dim=100,
                       sim_func=my_simulation, moment_func=my_moments)
    result = estimate(setup, data_moments, bounds, engine=engine)
"""

__version__ = "0.1.0"
__author__ = "Harry Aytug"

# High-level API (recommended for most users)
from .smm import smm_estimate, SMMResult
from .gmm import gmm_estimate, GMMEstimateResult

# Engine classes (for advanced usage)
from .smm import SMMEngine
from .gmm import GMMEngine, GMMResult

# Low-level estimation module
from .estimation import (
    EstimationSetup,
    EstimationResult,
    BootstrapResult,
    objective,
    global_search,
    local_optimize,
    estimate,
    compute_optimal_weighting,
    bootstrap,
)

# Inference module
from .inference import (
    JTestResult,
    j_test,
    sandwich_covariance,
    asymptotic_se,
    wald_test,
    confidence_interval,
)

# Output module (tables and plots)
from .output import (
    table_estimates,
    table_moments,
    table_bootstrap,
    plot_objective_landscape,
    plot_moment_contributions,
    plot_identification,
    plot_marginal_objective,
    plot_moment_comparison,
    plot_bootstrap_distribution,
    plot_convergence,
    plot_optimization_history,
    plot_sanity,
    summary,
)

# Datasets module
from .datasets import (
    DatasetBundle,
    load_econ381,
    load_econ381_bundle,
    load_consumption,
    load_labor_supply,
    load_asset_pricing,
    load_dataset,
    list_datasets,
)

# DGP module (data generating processes for learning/validation)
from .dgp import (
    DGPResult,
    list_dgps,
    linear_iv,
    consumption_savings,
    dynamic_discrete_choice,
    load_dgp,
)

# Public API exports
__all__ = [
    # Version
    "__version__",
    # High-level API (recommended)
    "smm_estimate",
    "gmm_estimate",
    "SMMResult",
    "GMMEstimateResult",
    # Low-level engines
    "SMMEngine",
    "GMMEngine",
    "GMMResult",
    # Low-level estimation
    "EstimationSetup",
    "EstimationResult",
    "BootstrapResult",
    "objective",
    "global_search",
    "local_optimize",
    "estimate",
    "compute_optimal_weighting",
    "bootstrap",
    # Inference
    "JTestResult",
    "j_test",
    "sandwich_covariance",
    "asymptotic_se",
    "wald_test",
    "confidence_interval",
    # Output
    "table_estimates",
    "table_moments",
    "table_bootstrap",
    "plot_objective_landscape",
    "plot_moment_contributions",
    "plot_identification",
    "plot_marginal_objective",
    "plot_moment_comparison",
    "plot_bootstrap_distribution",
    "plot_convergence",
    "plot_optimization_history",
    "plot_sanity",
    "summary",
    # Datasets
    "DatasetBundle",
    "load_econ381",
    "load_econ381_bundle",
    "load_consumption",
    "load_labor_supply",
    "load_asset_pricing",
    "load_dataset",
    "list_datasets",
    # DGPs
    "DGPResult",
    "list_dgps",
    "linear_iv",
    "consumption_savings",
    "dynamic_discrete_choice",
    "load_dgp",
]
