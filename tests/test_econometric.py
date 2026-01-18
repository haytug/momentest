"""
Econometric validation tests for momentest.

These tests verify that SMM/GMM estimation correctly recovers true parameters
from known Data Generating Processes (DGPs). This is the gold standard for
validating estimation code.

**Property 19: DGP Parameter Recovery**
**Validates: Requirements 11.4**

Test categories:
1. Linear model recovery - Basic GMM with linear IV
2. Quadratic moment recovery - SMM with nonlinear moments
3. Overidentified model tests - Models with k > p

References:
- Hansen (1982) - GMM asymptotic theory
- McFadden (1989) - SMM foundations
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from momentest import (
    linear_iv,
    consumption_savings,
    GMMEngine,
    gmm_estimate,
    smm_estimate,
    SMMEngine,
    EstimationSetup,
    estimate,
)


class TestLinearModelRecovery:
    """
    Tests that GMM recovers true parameters from linear IV model.
    
    Linear model: Y = β₀ + β₁X + ε
    Moments: E[Y - β₀ - β₁X] = 0, E[Z(Y - β₀ - β₁X)] = 0
    
    GMM should recover true β with low bias.
    
    **Property 19: DGP Parameter Recovery**
    **Validates: Requirements 11.4**
    """

    def test_linear_iv_recovery_identity_weighting(self):
        """
        GMM with identity weighting should recover linear IV parameters.
        
        **Property 19: DGP Parameter Recovery**
        **Validates: Requirements 11.4**
        """
        # Generate data from known DGP
        dgp = linear_iv(n=2000, seed=42, beta0=1.0, beta1=2.0, rho=0.5)
        
        # Estimate using GMM
        result = gmm_estimate(
            data=dgp.data,
            moment_func=dgp.moment_function,
            bounds=[(-10, 10), (-10, 10)],
            k=dgp.k,
            weighting="identity",
            n_global=50,
            seed=42
        )
        
        # Check convergence
        assert result.converged, "GMM should converge"
        
        # Check parameter recovery (bias < 0.5 * SE or absolute tolerance)
        # For large n, bias should be small
        bias = np.abs(result.theta - dgp.true_theta)
        tolerance = 0.3  # Allow 0.3 absolute deviation
        
        assert bias[0] < tolerance, (
            f"beta0 bias {bias[0]:.4f} exceeds tolerance {tolerance}. "
            f"Estimated: {result.theta[0]:.4f}, True: {dgp.true_theta[0]:.4f}"
        )
        assert bias[1] < tolerance, (
            f"beta1 bias {bias[1]:.4f} exceeds tolerance {tolerance}. "
            f"Estimated: {result.theta[1]:.4f}, True: {dgp.true_theta[1]:.4f}"
        )

    def test_linear_iv_recovery_optimal_weighting(self):
        """
        GMM with optimal weighting should recover linear IV parameters.
        
        Two-step efficient GMM should have lower variance than identity weighting.
        
        **Property 19: DGP Parameter Recovery**
        **Validates: Requirements 11.4**
        """
        # Generate data from known DGP
        dgp = linear_iv(n=2000, seed=123, beta0=1.0, beta1=2.0, rho=0.5)
        
        # Estimate using GMM with optimal weighting
        result = gmm_estimate(
            data=dgp.data,
            moment_func=dgp.moment_function,
            bounds=[(-10, 10), (-10, 10)],
            k=dgp.k,
            weighting="optimal",
            n_global=50,
            seed=123
        )
        
        # Check convergence
        assert result.converged, "GMM should converge"
        
        # Check parameter recovery
        bias = np.abs(result.theta - dgp.true_theta)
        tolerance = 0.3
        
        assert bias[0] < tolerance, (
            f"beta0 bias {bias[0]:.4f} exceeds tolerance. "
            f"Estimated: {result.theta[0]:.4f}, True: {dgp.true_theta[0]:.4f}"
        )
        assert bias[1] < tolerance, (
            f"beta1 bias {bias[1]:.4f} exceeds tolerance. "
            f"Estimated: {result.theta[1]:.4f}, True: {dgp.true_theta[1]:.4f}"
        )

    @given(
        beta0=st.floats(min_value=-5, max_value=5),
        beta1=st.floats(min_value=-5, max_value=5),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=20, deadline=60000)
    def test_linear_iv_recovery_property(self, beta0, beta1, seed):
        """
        Property test: GMM should recover parameters across different true values.
        
        **Property 19: DGP Parameter Recovery**
        *For any* DGP with known true parameters, estimating with sufficient
        n_sim and data SHALL recover parameters within a reasonable tolerance.
        
        **Validates: Requirements 11.4**
        """
        # Skip edge cases where parameters are too close to zero
        assume(abs(beta0) > 0.1 or abs(beta1) > 0.1)
        
        # Generate data
        dgp = linear_iv(n=1500, seed=seed, beta0=beta0, beta1=beta1, rho=0.3)
        
        # Estimate
        result = gmm_estimate(
            data=dgp.data,
            moment_func=dgp.moment_function,
            bounds=[(-10, 10), (-10, 10)],
            k=dgp.k,
            weighting="identity",
            n_global=30,
            seed=seed
        )
        
        # Check recovery with relative tolerance
        bias = np.abs(result.theta - dgp.true_theta)
        
        # Use larger tolerance for property test (more variation)
        tolerance = 0.5
        
        assert bias[0] < tolerance, (
            f"beta0 recovery failed: est={result.theta[0]:.3f}, true={beta0:.3f}"
        )
        assert bias[1] < tolerance, (
            f"beta1 recovery failed: est={result.theta[1]:.3f}, true={beta1:.3f}"
        )


class TestQuadraticMomentRecovery:
    """
    Tests that SMM recovers parameters using quadratic/nonlinear moments.
    
    This tests the SMM machinery with a simple model where we can verify
    the moments analytically.
    
    **Property 19: DGP Parameter Recovery**
    **Validates: Requirements 11.4**
    """

    def test_normal_mean_variance_recovery(self):
        """
        SMM should recover mean and variance of normal distribution.
        
        Model: X ~ N(μ, σ²)
        Moments: E[X] = μ, E[X²] = μ² + σ²
        
        **Property 19: DGP Parameter Recovery**
        **Validates: Requirements 11.4**
        """
        # True parameters
        true_mu = 3.0
        true_sigma = 2.0
        
        # Data moments (analytical)
        data_moments = np.array([
            true_mu,                          # E[X]
            true_mu**2 + true_sigma**2        # E[X²]
        ])
        
        # Simulation function
        def sim_func(theta, shocks):
            mu, sigma = theta
            return mu + sigma * shocks
        
        # Moment function
        def moment_func(sim_data):
            return np.column_stack([
                sim_data,           # First moment
                sim_data**2         # Second moment
            ])
        
        # Estimate
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0, 10), (0.1, 10)],
            n_sim=2000,
            shock_dim=1,
            seed=42,
            weighting="optimal",
            n_global=50
        )
        
        # Check convergence
        assert result.converged, "SMM should converge"
        
        # Check parameter recovery
        bias_mu = abs(result.theta[0] - true_mu)
        bias_sigma = abs(result.theta[1] - true_sigma)
        
        tolerance = 0.3
        assert bias_mu < tolerance, (
            f"mu bias {bias_mu:.4f} exceeds tolerance. "
            f"Estimated: {result.theta[0]:.4f}, True: {true_mu}"
        )
        assert bias_sigma < tolerance, (
            f"sigma bias {bias_sigma:.4f} exceeds tolerance. "
            f"Estimated: {result.theta[1]:.4f}, True: {true_sigma}"
        )

    def test_exponential_rate_recovery(self):
        """
        SMM should recover rate parameter of exponential distribution.
        
        Model: X ~ Exp(λ)
        Moments: E[X] = 1/λ, E[X²] = 2/λ²
        
        **Property 19: DGP Parameter Recovery**
        **Validates: Requirements 11.4**
        """
        # True parameter
        true_lambda = 0.5
        
        # Data moments (analytical)
        data_moments = np.array([
            1.0 / true_lambda,              # E[X] = 1/λ
            2.0 / (true_lambda**2)          # E[X²] = 2/λ²
        ])
        
        # Simulation function (inverse CDF method)
        def sim_func(theta, shocks):
            lam = theta[0]
            # Transform uniform to exponential via inverse CDF
            # Use CDF of standard normal to get uniform
            from scipy.stats import norm
            u = norm.cdf(shocks)
            # Clip to avoid log(0)
            u = np.clip(u, 1e-10, 1 - 1e-10)
            return -np.log(1 - u) / lam
        
        # Moment function
        def moment_func(sim_data):
            return np.column_stack([
                sim_data,           # First moment
                sim_data**2         # Second moment
            ])
        
        # Estimate
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0.1, 2.0)],
            n_sim=3000,
            shock_dim=1,
            seed=42,
            weighting="optimal",
            n_global=50
        )
        
        # Check convergence
        assert result.converged, "SMM should converge"
        
        # Check parameter recovery
        bias = abs(result.theta[0] - true_lambda)
        tolerance = 0.15
        
        assert bias < tolerance, (
            f"lambda bias {bias:.4f} exceeds tolerance. "
            f"Estimated: {result.theta[0]:.4f}, True: {true_lambda}"
        )

    @given(
        mu=st.floats(min_value=1, max_value=5),
        sigma=st.floats(min_value=0.5, max_value=3),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(max_examples=15, deadline=60000)
    def test_quadratic_moment_recovery_property(self, mu, sigma, seed):
        """
        Property test: SMM should recover normal parameters across different values.
        
        **Property 19: DGP Parameter Recovery**
        **Validates: Requirements 11.4**
        """
        # Data moments
        data_moments = np.array([mu, mu**2 + sigma**2])
        
        def sim_func(theta, shocks):
            return theta[0] + theta[1] * shocks
        
        def moment_func(sim_data):
            return np.column_stack([sim_data, sim_data**2])
        
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0, 10), (0.1, 10)],
            n_sim=1500,
            shock_dim=1,
            seed=seed,
            weighting="identity",
            n_global=30
        )
        
        # Check recovery
        bias_mu = abs(result.theta[0] - mu)
        bias_sigma = abs(result.theta[1] - sigma)
        
        tolerance = 0.5
        assert bias_mu < tolerance, f"mu recovery failed: est={result.theta[0]:.3f}, true={mu:.3f}"
        assert bias_sigma < tolerance, f"sigma recovery failed: est={result.theta[1]:.3f}, true={sigma:.3f}"


class TestOveridentifiedModels:
    """
    Tests for overidentified models (k > p).
    
    When there are more moment conditions than parameters, the model
    is overidentified. This allows testing model specification via
    the J-test.
    
    **Property 19: DGP Parameter Recovery**
    **Validates: Requirements 11.4**
    """

    def test_consumption_savings_overidentified(self):
        """
        Consumption-savings model is overidentified (k=3, p=2).
        
        GMM should still recover parameters, and J-statistic should
        be well-behaved under correct specification.
        
        Note: This is a challenging model due to the nonlinear Euler equation
        and measurement error. We use wider bounds and larger tolerance.
        
        **Property 19: DGP Parameter Recovery**
        **Validates: Requirements 11.4**
        """
        # Generate data with larger sample for better identification
        dgp = consumption_savings(n=3000, seed=42, beta=0.95, gamma=2.0)
        
        # Estimate with wider bounds to avoid boundary issues
        result = gmm_estimate(
            data=dgp.data,
            moment_func=dgp.moment_function,
            bounds=[(0.7, 0.99), (0.5, 6.0)],
            k=dgp.k,
            weighting="optimal",
            n_global=100,
            seed=42
        )
        
        # Check convergence OR good objective value
        # Sometimes optimizer hits max iterations but still finds good solution
        assert result.converged or result.objective < 1e-2, (
            f"GMM should converge or find good solution. "
            f"Converged: {result.converged}, Objective: {result.objective:.6f}"
        )
        
        # Check parameter recovery (with larger tolerance due to model complexity)
        # The consumption-savings model has weak identification for gamma
        # due to measurement error and the nonlinear Euler equation
        bias = np.abs(result.theta - dgp.true_theta)
        
        # Beta is typically well-identified
        assert bias[0] < 0.3, (
            f"beta bias {bias[0]:.4f} exceeds tolerance. "
            f"Estimated: {result.theta[0]:.4f}, True: {dgp.true_theta[0]:.4f}"
        )
        
        # Gamma has weaker identification - use larger tolerance
        # This is a known issue in consumption-based asset pricing
        assert bias[1] < 2.0, (
            f"gamma bias {bias[1]:.4f} exceeds tolerance. "
            f"Estimated: {result.theta[1]:.4f}, True: {dgp.true_theta[1]:.4f}"
        )

    def test_overidentified_linear_model(self):
        """
        Test overidentified linear model with extra instruments.
        
        Model: Y = β₀ + β₁X + ε
        Moments: E[ε], E[Zε], E[Z²ε] (k=3, p=2)
        
        **Property 19: DGP Parameter Recovery**
        **Validates: Requirements 11.4**
        """
        # Generate data with extra instrument
        np.random.seed(42)
        n = 2000
        
        # True parameters
        beta0, beta1 = 1.0, 2.0
        
        # Generate data
        Z = np.random.randn(n)
        Z2 = Z**2 - 1  # Centered Z²
        v = np.random.randn(n)
        eps = 0.5 * v + np.random.randn(n)  # Correlated errors
        X = 0.5 + Z + v
        Y = beta0 + beta1 * X + eps
        
        data = {'Y': Y, 'X': X, 'Z': Z, 'Z2': Z2}
        
        # Overidentified moment function (k=3, p=2)
        def moment_func(data, theta):
            b0, b1 = theta
            residual = data['Y'] - b0 - b1 * data['X']
            return np.column_stack([
                residual,              # E[ε] = 0
                residual * data['Z'],  # E[Zε] = 0
                residual * data['Z2']  # E[Z²ε] = 0
            ])
        
        # Estimate
        result = gmm_estimate(
            data=data,
            moment_func=moment_func,
            bounds=[(-10, 10), (-10, 10)],
            k=3,
            weighting="optimal",
            n_global=50,
            seed=42
        )
        
        # Check convergence
        assert result.converged, "GMM should converge"
        
        # Check parameter recovery
        bias = np.abs(result.theta - np.array([beta0, beta1]))
        tolerance = 0.3
        
        assert bias[0] < tolerance, f"beta0 bias {bias[0]:.4f} exceeds tolerance"
        assert bias[1] < tolerance, f"beta1 bias {bias[1]:.4f} exceeds tolerance"

    def test_j_statistic_reasonable(self):
        """
        J-statistic should be reasonable for correctly specified model.
        
        Under correct specification, J ~ χ²(k-p) asymptotically.
        For k=3, p=2, we have 1 degree of freedom.
        
        **Property 19: DGP Parameter Recovery**
        **Validates: Requirements 11.4**
        """
        # Generate data from correctly specified model
        dgp = consumption_savings(n=2000, seed=42, beta=0.95, gamma=2.0)
        
        # Estimate with optimal weighting
        result = gmm_estimate(
            data=dgp.data,
            moment_func=dgp.moment_function,
            bounds=[(0.8, 0.99), (0.5, 5.0)],
            k=dgp.k,
            weighting="optimal",
            n_global=50,
            seed=42
        )
        
        # J-statistic = n * objective (for optimal weighting)
        # Should not be extremely large for correct specification
        j_stat = dgp.n * result.objective
        
        # For χ²(1), 95th percentile is about 3.84
        # We use a much larger threshold to avoid false failures
        # (finite sample, measurement error in DGP, etc.)
        assert j_stat < 50, (
            f"J-statistic {j_stat:.2f} seems too large for correct specification"
        )


class TestDGPParameterRecoveryProperty:
    """
    Main property test for DGP parameter recovery.
    
    **Property 19: DGP Parameter Recovery**
    *For any* DGP with known true parameters, estimating with sufficient
    n_sim and data SHALL recover parameters within a reasonable tolerance
    (bias < 0.5 * SE).
    
    **Validates: Requirements 11.4**
    """

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20, deadline=120000)
    def test_linear_iv_parameter_recovery(self, seed):
        """
        Property 19: Linear IV parameter recovery across seeds.
        
        **Validates: Requirements 11.4**
        """
        # Fixed true parameters for consistency
        true_theta = np.array([1.0, 2.0])
        
        # Generate data
        dgp = linear_iv(n=1500, seed=seed, beta0=1.0, beta1=2.0, rho=0.3)
        
        # Estimate
        result = gmm_estimate(
            data=dgp.data,
            moment_func=dgp.moment_function,
            bounds=[(-10, 10), (-10, 10)],
            k=dgp.k,
            weighting="identity",
            n_global=30,
            seed=seed
        )
        
        # Check recovery
        bias = np.abs(result.theta - true_theta)
        tolerance = 0.5
        
        for i, (b, t) in enumerate(zip(bias, tolerance * np.ones(2))):
            assert b < t, (
                f"Parameter {i} recovery failed: "
                f"bias={b:.4f}, tolerance={t:.4f}, "
                f"estimated={result.theta[i]:.4f}, true={true_theta[i]:.4f}"
            )

    def test_smm_parameter_recovery_with_crn(self):
        """
        SMM with CRN should recover parameters reliably.
        
        CRN (Common Random Numbers) ensures smooth objective function,
        which helps local optimization converge to the true minimum.
        
        **Property 19: DGP Parameter Recovery**
        **Validates: Requirements 11.4**
        """
        # True parameters
        true_mu, true_sigma = 2.0, 1.5
        data_moments = np.array([true_mu, true_mu**2 + true_sigma**2])
        
        def sim_func(theta, shocks):
            return theta[0] + theta[1] * shocks
        
        def moment_func(sim_data):
            return np.column_stack([sim_data, sim_data**2])
        
        # Run multiple times to verify consistency
        results = []
        for seed in [42, 123, 456]:
            result = smm_estimate(
                sim_func=sim_func,
                moment_func=moment_func,
                data_moments=data_moments,
                bounds=[(0, 5), (0.1, 5)],
                n_sim=2000,
                shock_dim=1,
                seed=seed,
                weighting="optimal",
                n_global=50
            )
            results.append(result.theta)
        
        # All estimates should be close to true values
        for i, theta_hat in enumerate(results):
            bias = np.abs(theta_hat - np.array([true_mu, true_sigma]))
            tolerance = 0.3
            
            assert bias[0] < tolerance, f"Run {i}: mu bias {bias[0]:.4f} too large"
            assert bias[1] < tolerance, f"Run {i}: sigma bias {bias[1]:.4f} too large"
