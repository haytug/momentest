"""
Property-based tests for the GMM engine module.

Tests verify correctness properties from the design document.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from momentest.gmm import GMMEngine, GMMResult


# =============================================================================
# Property 14: GMM Sample Moments
# **Feature: momentest, Property 14: GMM Sample Moments**
# **Validates: Requirements 6.1, 6.3**
# =============================================================================

@given(
    n=st.integers(10, 200),
    k=st.integers(1, 10),
    p=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_gmm_sample_moments_shape(n, k, p, seed):
    """
    Property 14: GMM Sample Moments - Output Shape
    
    For any GMM engine with data of n observations, gbar(theta) SHALL return
    the mean of moment conditions across observations with shape (k,).
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    np.random.seed(seed)
    
    # Generate random data
    data = np.random.randn(n, 3)  # n observations, 3 features
    
    # Define a simple moment function: g(data, theta) = data[:, :k] - theta[:k]
    # This creates k moment conditions
    def moment_func(data, theta):
        # Each observation contributes k moment conditions
        # g_i = data_i[:k] - theta[:k] (broadcast theta across observations)
        result = np.zeros((data.shape[0], k))
        for i in range(k):
            if i < data.shape[1]:
                result[:, i] = data[:, i] - theta[i % p]
            else:
                result[:, i] = -theta[i % p]
        return result
    
    # Create engine
    engine = GMMEngine(data=data, k=k, p=p, moment_func=moment_func)
    
    # Generate random theta
    theta = np.random.randn(p)
    
    # Compute moments
    g_bar, S = engine.gbar(theta)
    
    # Verify shapes
    assert g_bar.shape == (k,), f"g_bar should have shape ({k},), got {g_bar.shape}"
    assert S.shape == (k, k), f"S should have shape ({k}, {k}), got {S.shape}"


@given(
    n=st.integers(10, 200),
    k=st.integers(1, 10),
    p=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_gmm_sample_moments_is_mean(n, k, p, seed):
    """
    Property 14: GMM Sample Moments - Is Sample Mean
    
    For any GMM engine, gbar(theta) SHALL return the sample mean of moment
    conditions: ḡ(θ) = (1/n) Σᵢ g(zᵢ, θ).
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    np.random.seed(seed)
    
    # Generate random data
    data = np.random.randn(n, 3)
    
    # Define a simple moment function
    def moment_func(data, theta):
        result = np.zeros((data.shape[0], k))
        for i in range(k):
            if i < data.shape[1]:
                result[:, i] = data[:, i] * theta[i % p]
            else:
                result[:, i] = theta[i % p]
        return result
    
    # Create engine
    engine = GMMEngine(data=data, k=k, p=p, moment_func=moment_func)
    
    # Generate random theta
    theta = np.random.randn(p)
    
    # Compute moments using engine
    g_bar, _ = engine.gbar(theta)
    
    # Compute expected mean manually
    g_i = moment_func(data, theta)
    expected_g_bar = np.mean(g_i, axis=0)
    
    # Should match
    np.testing.assert_allclose(
        g_bar, expected_g_bar, rtol=1e-10,
        err_msg=f"g_bar should equal sample mean of moment conditions"
    )


@given(
    n=st.integers(20, 200),
    k=st.integers(1, 8),
    p=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_gmm_covariance_is_sample_covariance(n, k, p, seed):
    """
    Property 14: GMM Sample Moments - Covariance Computation
    
    For any GMM engine, the returned S SHALL be the sample covariance of
    moment conditions: S = (1/n) Σᵢ (gᵢ - ḡ)(gᵢ - ḡ)'.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    np.random.seed(seed)
    
    # Generate random data
    data = np.random.randn(n, 3)
    
    # Define a moment function with some variation
    def moment_func(data, theta):
        result = np.zeros((data.shape[0], k))
        for i in range(k):
            if i < data.shape[1]:
                result[:, i] = data[:, i] ** 2 - theta[i % p]
            else:
                result[:, i] = theta[i % p] ** 2
        return result
    
    # Create engine
    engine = GMMEngine(data=data, k=k, p=p, moment_func=moment_func)
    
    # Generate random theta
    theta = np.random.randn(p)
    
    # Compute moments using engine
    g_bar, S = engine.gbar(theta)
    
    # Compute expected covariance manually
    g_i = moment_func(data, theta)
    g_centered = g_i - np.mean(g_i, axis=0)
    expected_S = (g_centered.T @ g_centered) / n
    
    # Should match
    np.testing.assert_allclose(
        S, expected_S, rtol=1e-10,
        err_msg=f"S should equal sample covariance of moment conditions"
    )


@given(
    n=st.integers(10, 200),
    k=st.integers(1, 8),
    p=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_gmm_covariance_is_symmetric(n, k, p, seed):
    """
    Property 14 (extended): GMM Covariance is Symmetric
    
    For any GMM engine, the returned covariance matrix S SHALL be symmetric.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    np.random.seed(seed)
    
    # Generate random data
    data = np.random.randn(n, 3)
    
    # Define a moment function
    def moment_func(data, theta):
        result = np.zeros((data.shape[0], k))
        for i in range(k):
            if i < data.shape[1]:
                result[:, i] = data[:, i] - theta[i % p]
            else:
                result[:, i] = -theta[i % p]
        return result
    
    # Create engine
    engine = GMMEngine(data=data, k=k, p=p, moment_func=moment_func)
    
    # Generate random theta
    theta = np.random.randn(p)
    
    # Compute moments
    _, S = engine.gbar(theta)
    
    # S should be symmetric
    np.testing.assert_allclose(
        S, S.T, rtol=1e-10,
        err_msg="Covariance matrix S should be symmetric"
    )


@given(
    n=st.integers(10, 200),
    k=st.integers(1, 8),
    p=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_gmm_covariance_is_positive_semidefinite(n, k, p, seed):
    """
    Property 14 (extended): GMM Covariance is Positive Semi-Definite
    
    For any GMM engine, the returned covariance matrix S SHALL be positive
    semi-definite (all eigenvalues >= 0).
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    np.random.seed(seed)
    
    # Generate random data
    data = np.random.randn(n, 3)
    
    # Define a moment function
    def moment_func(data, theta):
        result = np.zeros((data.shape[0], k))
        for i in range(k):
            if i < data.shape[1]:
                result[:, i] = data[:, i] - theta[i % p]
            else:
                result[:, i] = -theta[i % p]
        return result
    
    # Create engine
    engine = GMMEngine(data=data, k=k, p=p, moment_func=moment_func)
    
    # Generate random theta
    theta = np.random.randn(p)
    
    # Compute moments
    _, S = engine.gbar(theta)
    
    # Check eigenvalues are non-negative (within numerical tolerance)
    eigenvalues = np.linalg.eigvalsh(S)
    min_eigenvalue = np.min(eigenvalues)
    
    # Allow small negative eigenvalues due to numerical precision
    assert min_eigenvalue >= -1e-10, (
        f"S should be positive semi-definite, got min eigenvalue {min_eigenvalue}"
    )


@given(
    n=st.integers(10, 200),
    k=st.integers(1, 8),
    p=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_gmm_moments_alias(n, k, p, seed):
    """
    Property 14 (extended): moments() is alias for gbar()
    
    The moments() method SHALL return the same result as gbar() to provide
    a consistent interface with MomentEngine.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    np.random.seed(seed)
    
    # Generate random data
    data = np.random.randn(n, 3)
    
    # Define a moment function
    def moment_func(data, theta):
        result = np.zeros((data.shape[0], k))
        for i in range(k):
            if i < data.shape[1]:
                result[:, i] = data[:, i] - theta[i % p]
            else:
                result[:, i] = -theta[i % p]
        return result
    
    # Create engine
    engine = GMMEngine(data=data, k=k, p=p, moment_func=moment_func)
    
    # Generate random theta
    theta = np.random.randn(p)
    
    # Compute using both methods
    g_bar1, S1 = engine.gbar(theta)
    g_bar2, S2 = engine.moments(theta)
    
    # Should be identical
    np.testing.assert_array_equal(g_bar1, g_bar2)
    np.testing.assert_array_equal(S1, S2)


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_gmm_engine_validates_theta_dimension():
    """
    Test that GMMEngine raises ValueError for wrong theta dimension.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    data = np.random.randn(100, 3)
    k, p = 2, 3
    
    def moment_func(data, theta):
        return np.zeros((data.shape[0], k))
    
    engine = GMMEngine(data=data, k=k, p=p, moment_func=moment_func)
    
    # Wrong dimension should raise ValueError
    with pytest.raises(ValueError, match="theta must have shape"):
        engine.gbar(np.array([1.0, 2.0]))  # p=3, but only 2 elements


def test_gmm_engine_validates_empty_data():
    """
    Test that GMMEngine raises ValueError for empty data.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    def moment_func(data, theta):
        return np.zeros((data.shape[0], 2))
    
    with pytest.raises(ValueError, match="data cannot be empty"):
        GMMEngine(data=np.array([]), k=2, p=2, moment_func=moment_func)


def test_gmm_engine_validates_k_positive():
    """
    Test that GMMEngine raises ValueError for non-positive k.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    data = np.random.randn(100, 3)
    
    def moment_func(data, theta):
        return np.zeros((data.shape[0], 2))
    
    with pytest.raises(ValueError, match="k must be positive"):
        GMMEngine(data=data, k=0, p=2, moment_func=moment_func)


def test_gmm_engine_validates_p_positive():
    """
    Test that GMMEngine raises ValueError for non-positive p.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    data = np.random.randn(100, 3)
    
    def moment_func(data, theta):
        return np.zeros((data.shape[0], 2))
    
    with pytest.raises(ValueError, match="p must be positive"):
        GMMEngine(data=data, k=2, p=0, moment_func=moment_func)


def test_gmm_engine_validates_moment_func_output_shape():
    """
    Test that GMMEngine raises ValueError when moment_func returns wrong shape.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    data = np.random.randn(100, 3)
    k, p = 2, 3
    
    # Moment function returns wrong shape
    def bad_moment_func(data, theta):
        return np.zeros((data.shape[0], k + 1))  # Wrong k
    
    engine = GMMEngine(data=data, k=k, p=p, moment_func=bad_moment_func)
    
    with pytest.raises(ValueError, match="moment_func must return array of shape"):
        engine.gbar(np.random.randn(p))


# =============================================================================
# Jacobian Tests
# =============================================================================

@given(
    n=st.integers(50, 200),
    k=st.integers(1, 5),
    p=st.integers(1, 4),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_gmm_moments_jac_shape(n, k, p, seed):
    """
    Test that moments_jac returns correct shapes.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    np.random.seed(seed)
    
    # Generate random data
    data = np.random.randn(n, 3)
    
    # Define a moment function
    def moment_func(data, theta):
        result = np.zeros((data.shape[0], k))
        for i in range(k):
            if i < data.shape[1]:
                result[:, i] = data[:, i] - theta[i % p]
            else:
                result[:, i] = -theta[i % p]
        return result
    
    # Create engine
    engine = GMMEngine(data=data, k=k, p=p, moment_func=moment_func)
    
    # Generate random theta
    theta = np.random.randn(p)
    
    # Compute moments with Jacobian
    g_bar, S, D = engine.moments_jac(theta)
    
    # Verify shapes
    assert g_bar.shape == (k,), f"g_bar should have shape ({k},), got {g_bar.shape}"
    assert S.shape == (k, k), f"S should have shape ({k}, {k}), got {S.shape}"
    assert D.shape == (k, p), f"D should have shape ({k}, {p}), got {D.shape}"


@given(
    n=st.integers(50, 200),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_gmm_moments_jac_numerical_accuracy(n, seed):
    """
    Test that moments_jac computes accurate numerical Jacobian.
    
    For a linear moment function g(data, theta) = data - theta,
    the Jacobian should be -I (negative identity).
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1, 6.3**
    """
    np.random.seed(seed)
    
    k = p = 3  # Use same k and p for simplicity
    
    # Generate random data
    data = np.random.randn(n, k)
    
    # Linear moment function: g_i = data_i - theta
    # Jacobian: ∂g/∂θ = -I
    def moment_func(data, theta):
        return data - theta  # Broadcasting theta across rows
    
    # Create engine
    engine = GMMEngine(data=data, k=k, p=p, moment_func=moment_func)
    
    # Generate random theta
    theta = np.random.randn(p)
    
    # Compute Jacobian
    _, _, D = engine.moments_jac(theta)
    
    # Expected Jacobian is -I
    expected_D = -np.eye(k)
    
    # Should match (within numerical tolerance for finite differences)
    np.testing.assert_allclose(
        D, expected_D, rtol=1e-5, atol=1e-5,
        err_msg="Jacobian should be -I for linear moment function"
    )


# =============================================================================
# GMM Integration Tests with estimate()
# **Feature: momentest, Property 14: GMM Sample Moments**
# **Validates: Requirements 6.1**
# =============================================================================

from momentest.estimation import EstimationSetup, estimate


@given(
    n=st.integers(100, 500),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=50, deadline=None)
def test_gmm_estimate_integration(n, seed):
    """
    Test that GMM mode integrates correctly with estimate() function.
    
    Uses a simple linear model: Y = theta[0] + theta[1] * X + epsilon
    Moment conditions: E[Y - theta[0] - theta[1] * X] = 0
                       E[X * (Y - theta[0] - theta[1] * X)] = 0
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1**
    """
    np.random.seed(seed)
    
    # True parameters
    true_theta = np.array([1.0, 2.0])
    
    # Generate data
    X = np.random.randn(n)
    epsilon = np.random.randn(n) * 0.5
    Y = true_theta[0] + true_theta[1] * X + epsilon
    
    # Stack data
    data = np.column_stack([Y, X])
    
    # Define moment function
    def moment_func(data, theta):
        Y = data[:, 0]
        X = data[:, 1]
        residual = Y - theta[0] - theta[1] * X
        # Two moment conditions
        g = np.column_stack([residual, X * residual])
        return g
    
    # Setup
    k, p = 2, 2
    setup = EstimationSetup(
        mode="GMM",
        model_name="linear_model",
        moment_type="OLS moments",
        k=k,
        p=p,
        weighting="identity"
    )
    
    # Data moments for GMM are zeros (moment conditions should equal zero)
    data_moments = np.zeros(k)
    
    # Bounds
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    
    # Estimate
    result = estimate(
        setup,
        data_moments,
        bounds,
        n_global=20,
        data=data,
        moment_func=moment_func
    )
    
    # Verify result structure
    assert result.theta_hat.shape == (p,)
    assert result.m_bar.shape == (k,)
    assert result.S.shape == (k, k)
    
    # Verify theta_hat is within bounds
    for i, (lower, upper) in enumerate(bounds):
        assert lower <= result.theta_hat[i] <= upper


def test_gmm_estimate_requires_data():
    """
    Test that GMM mode raises error when data is not provided.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1**
    """
    setup = EstimationSetup(
        mode="GMM",
        model_name="test",
        moment_type="test",
        k=2,
        p=2
    )
    
    with pytest.raises(ValueError, match="data is required for GMM mode"):
        estimate(
            setup,
            data_moments=np.zeros(2),
            bounds=[(-1, 1), (-1, 1)],
            data=None,
            moment_func=lambda d, t: np.zeros((d.shape[0], 2))
        )


def test_gmm_estimate_requires_moment_func():
    """
    Test that GMM mode raises error when moment_func is not provided.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1**
    """
    setup = EstimationSetup(
        mode="GMM",
        model_name="test",
        moment_type="test",
        k=2,
        p=2
    )
    
    with pytest.raises(ValueError, match="moment_func is required for GMM mode"):
        estimate(
            setup,
            data_moments=np.zeros(2),
            bounds=[(-1, 1), (-1, 1)],
            data=np.random.randn(100, 2),
            moment_func=None
        )


@given(
    n=st.integers(200, 500),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_gmm_recovers_true_parameters(n, seed):
    """
    Test that GMM estimation recovers true parameters for a simple model.
    
    For a well-identified model with sufficient data, GMM should recover
    parameters close to the true values.
    
    **Feature: momentest, Property 14: GMM Sample Moments**
    **Validates: Requirements 6.1**
    """
    np.random.seed(seed)
    
    # True parameters
    true_theta = np.array([1.5, -0.5])
    
    # Generate data with low noise
    X = np.random.randn(n)
    epsilon = np.random.randn(n) * 0.1  # Low noise
    Y = true_theta[0] + true_theta[1] * X + epsilon
    
    data = np.column_stack([Y, X])
    
    def moment_func(data, theta):
        Y = data[:, 0]
        X = data[:, 1]
        residual = Y - theta[0] - theta[1] * X
        return np.column_stack([residual, X * residual])
    
    setup = EstimationSetup(
        mode="GMM",
        model_name="linear",
        moment_type="OLS",
        k=2,
        p=2,
        weighting="identity"
    )
    
    result = estimate(
        setup,
        data_moments=np.zeros(2),
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        n_global=30,
        data=data,
        moment_func=moment_func
    )
    
    # With low noise and sufficient data, estimates should be close to true values
    # Allow tolerance of 0.5 for robustness
    np.testing.assert_allclose(
        result.theta_hat, true_theta, atol=0.5,
        err_msg=f"GMM should recover true parameters. Got {result.theta_hat}, expected {true_theta}"
    )
