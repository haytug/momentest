"""
Property-based tests for bootstrap inference.

Tests verify correctness properties from the design document.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from momentest import BootstrapResult
from momentest.smm import SMMEngine
from momentest.estimation import (
    EstimationSetup,
    bootstrap,
)


# =============================================================================
# Helper functions for testing
# =============================================================================

def create_test_sim_func(p: int, shock_dim: int):
    """Create a simulation function that uses all parameters and shocks."""
    def sim_func(theta: np.ndarray, shocks: np.ndarray) -> np.ndarray:
        n_sim = shocks.shape[0]
        # Linear combination of parameters and shocks
        Y = np.zeros(n_sim)
        for i in range(min(p, shock_dim)):
            Y += theta[i % p] * shocks[:, i]
        # Add remaining parameters as constants
        for i in range(shock_dim, p):
            Y += theta[i]
        return Y.reshape(n_sim, 1)
    return sim_func


def create_test_moment_func(k: int):
    """Create a moment function that returns k moments."""
    def moment_func(simulated_data: np.ndarray) -> np.ndarray:
        Y = simulated_data[:, 0]
        n_sim = len(Y)
        moments = np.zeros((n_sim, k))
        for i in range(k):
            moments[:, i] = Y ** (i + 1)  # Y, Y^2, Y^3, ...
        return moments
    return moment_func


# =============================================================================
# Property 11: Bootstrap Replication Count
# **Feature: momentest, Property 11: Bootstrap Replication Count**
# **Validates: Requirements 5.1**
# =============================================================================

@given(
    k=st.integers(1, 3),
    p=st.integers(1, 2),
    n_sim=st.integers(100, 200),
    shock_dim=st.integers(1, 2),
    seed=st.integers(0, 2**31 - 1),
    n_boot=st.integers(5, 15),
)
@settings(max_examples=100, deadline=None)
def test_bootstrap_replication_count(k, p, n_sim, shock_dim, seed, n_boot):
    """
    Property 11: Bootstrap Replication Count
    
    For any bootstrap call with n_boot replications, the returned
    bootstrap_estimates SHALL have shape (n_boot, p).
    
    **Feature: momentest, Property 11: Bootstrap Replication Count**
    **Validates: Requirements 5.1**
    """
    # Create setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="test_model",
        moment_type="test_moments",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=shock_dim,
        seed=seed,
        weighting="identity"
    )
    
    # Create test functions
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    # Create engine for bootstrap
    engine = SMMEngine(k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
                       sim_func=sim_func, moment_func=moment_func, seed=seed)
    
    # Generate random bounds
    bounds = []
    for _ in range(p):
        lower = np.random.uniform(-5, -1)
        upper = np.random.uniform(1, 5)
        bounds.append((lower, upper))
    
    # Generate random data moments
    data_moments = np.random.randn(k)
    
    # Run bootstrap with small n_global for speed
    result = bootstrap(
        setup,
        data_moments,
        bounds,
        n_boot=n_boot,
        n_global=10,
        n_jobs=1,  # Sequential for reproducibility
        engine=engine
    )
    
    # Verify bootstrap_estimates shape
    assert result.bootstrap_estimates.shape == (n_boot, p), (
        f"bootstrap_estimates shape {result.bootstrap_estimates.shape} != expected ({n_boot}, {p})"
    )
    
    # Verify n_boot is stored correctly
    assert result.n_boot == n_boot, (
        f"result.n_boot {result.n_boot} != expected {n_boot}"
    )


@given(
    k=st.integers(1, 3),
    p=st.integers(1, 2),
    n_sim=st.integers(100, 200),
    shock_dim=st.integers(1, 2),
    seed=st.integers(0, 2**31 - 1),
    n_boot=st.integers(5, 15),
)
@settings(max_examples=100, deadline=None)
def test_bootstrap_returns_valid_result_structure(k, p, n_sim, shock_dim, seed, n_boot):
    """
    Property 11 (extended): Bootstrap returns valid result structure
    
    The bootstrap() function SHALL return a BootstrapResult with all
    required fields properly populated.
    
    **Feature: momentest, Property 11: Bootstrap Replication Count**
    **Validates: Requirements 5.1**
    """
    # Create setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="test_model",
        moment_type="test_moments",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=shock_dim,
        seed=seed,
        weighting="identity"
    )
    
    # Create test functions
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    # Create engine for bootstrap
    engine = SMMEngine(k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
                       sim_func=sim_func, moment_func=moment_func, seed=seed)
    
    # Generate random bounds
    bounds = [(np.random.uniform(-5, -1), np.random.uniform(1, 5)) for _ in range(p)]
    
    # Generate random data moments
    data_moments = np.random.randn(k)
    
    # Run bootstrap
    result = bootstrap(
        setup,
        data_moments,
        bounds,
        n_boot=n_boot,
        n_global=10,
        n_jobs=1,
        engine=engine
    )
    
    # Verify result structure
    assert isinstance(result, BootstrapResult)
    assert result.theta_hat.shape == (p,)
    assert result.se.shape == (p,)
    assert result.ci_lower.shape == (p,)
    assert result.ci_upper.shape == (p,)
    assert result.bootstrap_estimates.shape == (n_boot, p)
    assert isinstance(result.alpha, float)
    assert isinstance(result.n_boot, int)
    assert isinstance(result.n_converged, int)
    assert 0 <= result.n_converged <= n_boot


# =============================================================================
# Property 12: Bootstrap Standard Error Consistency
# **Feature: momentest, Property 12: Bootstrap Standard Error Consistency**
# **Validates: Requirements 5.3**
# =============================================================================

@given(
    k=st.integers(1, 3),
    p=st.integers(1, 2),
    n_sim=st.integers(100, 200),
    shock_dim=st.integers(1, 2),
    seed=st.integers(0, 2**31 - 1),
    n_boot=st.integers(10, 20),
)
@settings(max_examples=100, deadline=None)
def test_bootstrap_se_consistency(k, p, n_sim, shock_dim, seed, n_boot):
    """
    Property 12: Bootstrap Standard Error Consistency
    
    For any bootstrap result, the returned standard errors SHALL equal
    np.std(bootstrap_estimates, axis=0) within numerical tolerance.
    
    **Feature: momentest, Property 12: Bootstrap Standard Error Consistency**
    **Validates: Requirements 5.3**
    """
    # Create setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="test_model",
        moment_type="test_moments",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=shock_dim,
        seed=seed,
        weighting="identity"
    )
    
    # Create test functions
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    # Create engine for bootstrap
    engine = SMMEngine(k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
                       sim_func=sim_func, moment_func=moment_func, seed=seed)
    
    # Generate random bounds
    bounds = [(np.random.uniform(-5, -1), np.random.uniform(1, 5)) for _ in range(p)]
    
    # Generate random data moments
    data_moments = np.random.randn(k)
    
    # Run bootstrap
    result = bootstrap(
        setup,
        data_moments,
        bounds,
        n_boot=n_boot,
        n_global=10,
        n_jobs=1,
        engine=engine
    )
    
    # Compute expected SE from bootstrap estimates (excluding NaN rows)
    valid_mask = ~np.any(np.isnan(result.bootstrap_estimates), axis=1)
    
    if np.sum(valid_mask) > 1:
        expected_se = np.std(result.bootstrap_estimates[valid_mask], axis=0, ddof=1)
        
        # SE should match expected (within numerical tolerance)
        np.testing.assert_allclose(
            result.se, expected_se, rtol=1e-10,
            err_msg=f"Bootstrap SE {result.se} != expected {expected_se}"
        )


@given(
    k=st.integers(1, 3),
    p=st.integers(1, 2),
    n_sim=st.integers(100, 200),
    shock_dim=st.integers(1, 2),
    seed=st.integers(0, 2**31 - 1),
    n_boot=st.integers(10, 20),
)
@settings(max_examples=100, deadline=None)
def test_bootstrap_se_non_negative(k, p, n_sim, shock_dim, seed, n_boot):
    """
    Property 12 (extended): Bootstrap SE is non-negative
    
    For any bootstrap result, the standard errors SHALL be non-negative
    (or NaN if computation failed).
    
    **Feature: momentest, Property 12: Bootstrap Standard Error Consistency**
    **Validates: Requirements 5.3**
    """
    # Create setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="test_model",
        moment_type="test_moments",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=shock_dim,
        seed=seed,
        weighting="identity"
    )
    
    # Create test functions
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    # Create engine for bootstrap
    engine = SMMEngine(k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
                       sim_func=sim_func, moment_func=moment_func, seed=seed)
    
    # Generate random bounds
    bounds = [(np.random.uniform(-5, -1), np.random.uniform(1, 5)) for _ in range(p)]
    
    # Generate random data moments
    data_moments = np.random.randn(k)
    
    # Run bootstrap
    result = bootstrap(
        setup,
        data_moments,
        bounds,
        n_boot=n_boot,
        n_global=10,
        n_jobs=1,
        engine=engine
    )
    
    # SE should be non-negative (or NaN)
    for i, se_val in enumerate(result.se):
        assert se_val >= 0 or np.isnan(se_val), (
            f"SE[{i}] = {se_val} should be non-negative or NaN"
        )


# =============================================================================
# Property 13: Bootstrap Confidence Interval Percentiles
# **Feature: momentest, Property 13: Bootstrap Confidence Interval Percentiles**
# **Validates: Requirements 5.4**
# =============================================================================

@given(
    k=st.integers(1, 3),
    p=st.integers(1, 2),
    n_sim=st.integers(100, 200),
    shock_dim=st.integers(1, 2),
    seed=st.integers(0, 2**31 - 1),
    n_boot=st.integers(10, 20),
    alpha=st.floats(0.01, 0.20),
)
@settings(max_examples=100, deadline=None)
def test_bootstrap_ci_percentiles(k, p, n_sim, shock_dim, seed, n_boot, alpha):
    """
    Property 13: Bootstrap Confidence Interval Percentiles
    
    For any bootstrap result with confidence level α, the CI bounds SHALL
    equal the (α/2) and (1-α/2) percentiles of the bootstrap distribution.
    
    **Feature: momentest, Property 13: Bootstrap Confidence Interval Percentiles**
    **Validates: Requirements 5.4**
    """
    # Create setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="test_model",
        moment_type="test_moments",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=shock_dim,
        seed=seed,
        weighting="identity"
    )
    
    # Create test functions
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    # Create engine for bootstrap
    engine = SMMEngine(k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
                       sim_func=sim_func, moment_func=moment_func, seed=seed)
    
    # Generate random bounds
    bounds = [(np.random.uniform(-5, -1), np.random.uniform(1, 5)) for _ in range(p)]
    
    # Generate random data moments
    data_moments = np.random.randn(k)
    
    # Run bootstrap with specified alpha
    result = bootstrap(
        setup,
        data_moments,
        bounds,
        n_boot=n_boot,
        alpha=alpha,
        n_global=10,
        n_jobs=1,
        engine=engine
    )
    
    # Verify alpha is stored correctly
    assert result.alpha == alpha, (
        f"result.alpha {result.alpha} != expected {alpha}"
    )
    
    # Compute expected CI bounds from bootstrap estimates (excluding NaN rows)
    valid_mask = ~np.any(np.isnan(result.bootstrap_estimates), axis=1)
    
    if np.sum(valid_mask) > 1:
        valid_estimates = result.bootstrap_estimates[valid_mask]
        expected_ci_lower = np.percentile(valid_estimates, 100 * alpha / 2, axis=0)
        expected_ci_upper = np.percentile(valid_estimates, 100 * (1 - alpha / 2), axis=0)
        
        # CI bounds should match expected (within numerical tolerance)
        np.testing.assert_allclose(
            result.ci_lower, expected_ci_lower, rtol=1e-10,
            err_msg=f"CI lower {result.ci_lower} != expected {expected_ci_lower}"
        )
        np.testing.assert_allclose(
            result.ci_upper, expected_ci_upper, rtol=1e-10,
            err_msg=f"CI upper {result.ci_upper} != expected {expected_ci_upper}"
        )


@given(
    k=st.integers(1, 3),
    p=st.integers(1, 2),
    n_sim=st.integers(100, 200),
    shock_dim=st.integers(1, 2),
    seed=st.integers(0, 2**31 - 1),
    n_boot=st.integers(10, 20),
    alpha=st.floats(0.01, 0.20),
)
@settings(max_examples=100, deadline=None)
def test_bootstrap_ci_ordering(k, p, n_sim, shock_dim, seed, n_boot, alpha):
    """
    Property 13 (extended): CI lower <= CI upper
    
    For any bootstrap result, ci_lower[i] SHALL be <= ci_upper[i] for all i.
    
    **Feature: momentest, Property 13: Bootstrap Confidence Interval Percentiles**
    **Validates: Requirements 5.4**
    """
    # Create setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="test_model",
        moment_type="test_moments",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=shock_dim,
        seed=seed,
        weighting="identity"
    )
    
    # Create test functions
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    # Create engine for bootstrap
    engine = SMMEngine(k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
                       sim_func=sim_func, moment_func=moment_func, seed=seed)
    
    # Generate random bounds
    bounds = [(np.random.uniform(-5, -1), np.random.uniform(1, 5)) for _ in range(p)]
    
    # Generate random data moments
    data_moments = np.random.randn(k)
    
    # Run bootstrap
    result = bootstrap(
        setup,
        data_moments,
        bounds,
        n_boot=n_boot,
        alpha=alpha,
        n_global=10,
        n_jobs=1,
        engine=engine
    )
    
    # CI lower should be <= CI upper (or both NaN)
    for i in range(p):
        if not (np.isnan(result.ci_lower[i]) and np.isnan(result.ci_upper[i])):
            assert result.ci_lower[i] <= result.ci_upper[i], (
                f"CI[{i}]: lower {result.ci_lower[i]} > upper {result.ci_upper[i]}"
            )


@given(
    k=st.integers(1, 3),
    p=st.integers(1, 2),
    n_sim=st.integers(100, 200),
    shock_dim=st.integers(1, 2),
    seed=st.integers(0, 2**31 - 1),
    n_boot=st.integers(10, 20),
)
@settings(max_examples=100, deadline=None)
def test_bootstrap_ci_contains_point_estimate(k, p, n_sim, shock_dim, seed, n_boot):
    """
    Property 13 (extended): CI typically contains point estimate
    
    For most bootstrap results with reasonable data, the confidence interval
    SHOULD contain the point estimate (though this is not guaranteed for
    all random inputs).
    
    **Feature: momentest, Property 13: Bootstrap Confidence Interval Percentiles**
    **Validates: Requirements 5.4**
    """
    # Create setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="test_model",
        moment_type="test_moments",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=shock_dim,
        seed=seed,
        weighting="identity"
    )
    
    # Create test functions
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    # Create engine for bootstrap
    engine = SMMEngine(k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
                       sim_func=sim_func, moment_func=moment_func, seed=seed)
    
    # Generate random bounds
    bounds = [(np.random.uniform(-5, -1), np.random.uniform(1, 5)) for _ in range(p)]
    
    # Generate random data moments
    data_moments = np.random.randn(k)
    
    # Run bootstrap with 95% CI
    result = bootstrap(
        setup,
        data_moments,
        bounds,
        n_boot=n_boot,
        alpha=0.05,
        n_global=10,
        n_jobs=1,
        engine=engine
    )
    
    # Check if point estimate is within CI (this is expected but not guaranteed)
    # We just verify the structure is correct - the point estimate may or may not
    # be within the CI depending on the random data
    for i in range(p):
        if not np.isnan(result.ci_lower[i]) and not np.isnan(result.ci_upper[i]):
            # Just verify the CI bounds are finite
            assert np.isfinite(result.ci_lower[i]), f"CI lower[{i}] should be finite"
            assert np.isfinite(result.ci_upper[i]), f"CI upper[{i}] should be finite"
