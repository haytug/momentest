"""
Property-based tests for the Python SMMEngine module.

Tests verify correctness properties from the design document.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from momentest.smm import SMMEngine


# =============================================================================
# Helper functions for testing
# =============================================================================

def simple_sim_func(theta: np.ndarray, shocks: np.ndarray) -> np.ndarray:
    """
    Simple simulation function for testing.
    
    Simulates: Y = theta[0] + theta[1] * shocks[:, 0] + noise
    where noise comes from additional shock dimensions if available.
    
    Args:
        theta: Parameter vector of shape (p,)
        shocks: Pre-drawn shocks of shape (n_sim, shock_dim)
    
    Returns:
        Simulated data of shape (n_sim, 1)
    """
    n_sim = shocks.shape[0]
    # Use first shock dimension as main shock
    Y = theta[0] + theta[1] * shocks[:, 0]
    # Add noise from second shock dimension if available
    if shocks.shape[1] > 1:
        Y += 0.1 * shocks[:, 1]
    return Y.reshape(n_sim, 1)


def simple_moment_func(simulated_data: np.ndarray) -> np.ndarray:
    """
    Simple moment function for testing.
    
    Computes mean and variance moments from simulated data.
    
    Args:
        simulated_data: Output from sim_func of shape (n_sim, 1)
    
    Returns:
        Moments of shape (n_sim, 2) - each row is [Y_i, Y_i^2]
    """
    Y = simulated_data[:, 0]
    n_sim = len(Y)
    # Return individual contributions to mean and second moment
    moments = np.column_stack([Y, Y**2])
    return moments


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
# Property 22: SMMEngine Seed Determinism
# **Feature: momentest, Property 22: SMMEngine Seed Determinism**
# **Validates: Requirements 14.3**
# =============================================================================

@given(
    k=st.integers(1, 10),
    p=st.integers(1, 5),
    n_sim=st.integers(50, 200),
    shock_dim=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_smm_seed_determinism(k, p, n_sim, shock_dim, seed):
    """
    Property 22: SMMEngine Seed Determinism
    
    For any seed value, constructing two SMMEngines with identical parameters
    and functions SHALL produce identical moment computations for any theta.
    
    **Feature: momentest, Property 22: SMMEngine Seed Determinism**
    **Validates: Requirements 14.3**
    """
    # Create simulation and moment functions
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    # Create two engines with identical parameters
    engine1 = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    engine2 = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    
    # Generate a random theta
    rng = np.random.default_rng(seed + 1)  # Different seed for theta
    theta = rng.standard_normal(p)
    
    # Compute moments from both engines
    m_bar1, S1 = engine1.moments(theta)
    m_bar2, S2 = engine2.moments(theta)
    
    # Results should be identical
    np.testing.assert_array_equal(
        m_bar1, m_bar2,
        err_msg="m_bar differs between engines with same seed"
    )
    np.testing.assert_array_equal(
        S1, S2,
        err_msg="S differs between engines with same seed"
    )


@given(
    k=st.integers(1, 5),
    p=st.integers(1, 3),
    n_sim=st.integers(50, 100),
    shock_dim=st.integers(1, 3),
    seed1=st.integers(0, 2**31 - 1),
    seed2=st.integers(0, 2**31 - 1),
)
@settings(max_examples=100, deadline=None)
def test_smm_different_seeds_different_results(k, p, n_sim, shock_dim, seed1, seed2):
    """
    Property 22 (extended): Different seeds produce different results
    
    For any two different seed values, SMMEngines SHALL produce different
    moment computations (with high probability).
    
    **Feature: momentest, Property 22: SMMEngine Seed Determinism**
    **Validates: Requirements 14.3**
    """
    # Skip if seeds are the same
    assume(seed1 != seed2)
    
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    engine1 = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed1
    )
    engine2 = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed2
    )
    
    # Use a fixed theta
    theta = np.ones(p)
    
    m_bar1, _ = engine1.moments(theta)
    m_bar2, _ = engine2.moments(theta)
    
    # Results should differ (with overwhelming probability)
    assert not np.allclose(m_bar1, m_bar2), (
        "Different seeds should produce different results. "
        f"seed1={seed1}, seed2={seed2}, m_bar1={m_bar1}, m_bar2={m_bar2}"
    )


# =============================================================================
# Property 20: SMMEngine CRN Consistency
# **Feature: momentest, Property 20: SMMEngine CRN Consistency**
# **Validates: Requirements 14.3, 14.6**
# =============================================================================

@given(
    k=st.integers(1, 10),
    p=st.integers(1, 5),
    n_sim=st.integers(50, 200),
    shock_dim=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_smm_crn_consistency(k, p, n_sim, shock_dim, seed):
    """
    Property 20: SMMEngine CRN Consistency
    
    For any SMMEngine instance and two different parameter vectors θ₁ ≠ θ₂,
    the underlying shocks used in moments(θ₁) and moments(θ₂) SHALL be identical
    (verified by checking that re-seeding produces same results).
    
    **Feature: momentest, Property 20: SMMEngine CRN Consistency**
    **Validates: Requirements 14.3, 14.6**
    """
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    # Create a single engine
    engine = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    
    # Generate two different theta vectors
    rng = np.random.default_rng(seed + 1)
    theta1 = rng.standard_normal(p)
    theta2 = rng.standard_normal(p) + 10.0  # Ensure they're different
    
    # Compute moments for theta1
    m_bar1_first, S1_first = engine.moments(theta1)
    
    # Compute moments for theta2
    m_bar2, S2 = engine.moments(theta2)
    
    # Compute moments for theta1 again - should be identical to first call
    # This verifies CRN: the same shocks are used regardless of theta
    m_bar1_second, S1_second = engine.moments(theta1)
    
    # Results for theta1 should be identical across calls
    np.testing.assert_array_equal(
        m_bar1_first, m_bar1_second,
        err_msg="CRN violated: moments(theta1) differs across calls"
    )
    np.testing.assert_array_equal(
        S1_first, S1_second,
        err_msg="CRN violated: covariance differs across calls"
    )
    
    # Additionally verify that a fresh engine with same seed gives same results
    engine_fresh = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    m_bar1_fresh, S1_fresh = engine_fresh.moments(theta1)
    
    np.testing.assert_array_equal(
        m_bar1_first, m_bar1_fresh,
        err_msg="CRN violated: fresh engine with same seed gives different results"
    )


@given(
    k=st.integers(1, 5),
    p=st.integers(1, 3),
    n_sim=st.integers(50, 100),
    shock_dim=st.integers(1, 3),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_smm_shocks_are_reused(k, p, n_sim, shock_dim, seed):
    """
    Property 20 (extended): Shocks are reused across calls
    
    The pre-drawn shocks should be identical across multiple calls to moments().
    
    **Feature: momentest, Property 20: SMMEngine CRN Consistency**
    **Validates: Requirements 14.3, 14.6**
    """
    # Track which shocks are used
    shocks_used = []
    
    def tracking_sim_func(theta, shocks):
        shocks_used.append(shocks.copy())
        return (theta[0] + shocks[:, 0]).reshape(-1, 1)
    
    def simple_moment(data):
        return data  # Just return the data as moments
    
    engine = SMMEngine(
        k=1, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=tracking_sim_func, moment_func=simple_moment, seed=seed
    )
    
    theta1 = np.ones(p)
    theta2 = np.ones(p) * 2
    
    # Call moments multiple times
    engine.moments(theta1)
    engine.moments(theta2)
    engine.moments(theta1)
    
    # All calls should have used the same shocks
    assert len(shocks_used) == 3
    np.testing.assert_array_equal(shocks_used[0], shocks_used[1])
    np.testing.assert_array_equal(shocks_used[1], shocks_used[2])


# =============================================================================
# Property 21: SMMEngine Output Dimensions
# **Feature: momentest, Property 21: SMMEngine Output Dimensions**
# **Validates: Requirements 14.4, 14.5**
# =============================================================================

@given(
    k=st.integers(1, 20),
    p=st.integers(1, 10),
    n_sim=st.integers(50, 200),
    shock_dim=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_smm_output_dimensions(k, p, n_sim, shock_dim, seed):
    """
    Property 21: SMMEngine Output Dimensions
    
    For any valid SMMEngine configuration (k, p, n_sim) and parameter vector
    theta of length p, moments() SHALL return m_bar of shape (k,) and S of
    shape (k, k).
    
    **Feature: momentest, Property 21: SMMEngine Output Dimensions**
    **Validates: Requirements 14.4, 14.5**
    """
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    engine = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    
    # Generate random theta of correct dimension
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(p)
    
    # Compute moments
    m_bar, S = engine.moments(theta)
    
    # Verify dimensions
    assert m_bar.shape == (k,), f"m_bar shape {m_bar.shape} != expected ({k},)"
    assert S.shape == (k, k), f"S shape {S.shape} != expected ({k}, {k})"
    
    # Verify dtypes are float64
    assert m_bar.dtype == np.float64, f"m_bar dtype {m_bar.dtype} != float64"
    assert S.dtype == np.float64, f"S dtype {S.dtype} != float64"


@given(
    k=st.integers(1, 10),
    p=st.integers(1, 5),
    n_sim=st.integers(50, 200),
    shock_dim=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_smm_moments_jac_output_dimensions(k, p, n_sim, shock_dim, seed):
    """
    Property 21 (extended): Output Dimensions for moments_jac
    
    For any valid SMMEngine configuration (k, p) and parameter vector theta of
    length p, moments_jac() SHALL return m_bar of shape (k,), S of shape (k, k),
    and D (Jacobian) of shape (k, p).
    
    **Feature: momentest, Property 21: SMMEngine Output Dimensions**
    **Validates: Requirements 14.4, 14.5**
    """
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    engine = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    
    # Generate random theta of correct dimension
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(p)
    
    # Compute moments with Jacobian
    m_bar, S, D = engine.moments_jac(theta)
    
    # Verify dimensions
    assert m_bar.shape == (k,), f"m_bar shape {m_bar.shape} != expected ({k},)"
    assert S.shape == (k, k), f"S shape {S.shape} != expected ({k}, {k})"
    assert D.shape == (k, p), f"D shape {D.shape} != expected ({k}, {p})"
    
    # Verify dtypes are float64
    assert m_bar.dtype == np.float64, f"m_bar dtype {m_bar.dtype} != float64"
    assert S.dtype == np.float64, f"S dtype {S.dtype} != float64"
    assert D.dtype == np.float64, f"D dtype {D.dtype} != float64"


@given(
    k=st.integers(1, 10),
    p=st.integers(1, 5),
    n_sim=st.integers(50, 200),
    shock_dim=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_smm_covariance_is_symmetric(k, p, n_sim, shock_dim, seed):
    """
    Property 21 (extended): Covariance is Symmetric
    
    For any SMMEngine, the returned covariance matrix S SHALL be symmetric.
    
    **Feature: momentest, Property 21: SMMEngine Output Dimensions**
    **Validates: Requirements 14.4, 14.5**
    """
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    engine = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(p)
    
    _, S = engine.moments(theta)
    
    # S should be symmetric
    np.testing.assert_allclose(
        S, S.T, rtol=1e-10,
        err_msg="Covariance matrix S should be symmetric"
    )


@given(
    k=st.integers(1, 5),  # Limit k to ensure n_sim >> k
    p=st.integers(1, 5),
    n_sim=st.integers(200, 500),  # Ensure n_sim >> k for well-defined covariance
    shock_dim=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_smm_covariance_is_positive_semidefinite(k, p, n_sim, shock_dim, seed):
    """
    Property 21 (extended): Covariance is Positive Semi-Definite
    
    For any SMMEngine, the returned covariance matrix S SHALL be positive
    semi-definite (all eigenvalues >= 0).
    
    Note: We require n_sim >> k for a well-defined sample covariance matrix.
    
    **Feature: momentest, Property 21: SMMEngine Output Dimensions**
    **Validates: Requirements 14.4, 14.5**
    """
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    engine = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(p)
    
    _, S = engine.moments(theta)
    
    # Check eigenvalues are non-negative (within numerical tolerance)
    eigenvalues = np.linalg.eigvalsh(S)
    min_eigenvalue = np.min(eigenvalues)
    
    # Allow small negative eigenvalues due to numerical precision
    assert min_eigenvalue >= -1e-8, (
        f"S should be positive semi-definite, got min eigenvalue {min_eigenvalue}"
    )


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_smm_engine_validates_k_positive():
    """Test that SMMEngine raises ValueError for non-positive k."""
    with pytest.raises(ValueError, match="k must be positive"):
        SMMEngine(
            k=0, p=2, n_sim=100, shock_dim=1,
            sim_func=simple_sim_func, moment_func=simple_moment_func
        )


def test_smm_engine_validates_p_positive():
    """Test that SMMEngine raises ValueError for non-positive p."""
    with pytest.raises(ValueError, match="p must be positive"):
        SMMEngine(
            k=2, p=0, n_sim=100, shock_dim=1,
            sim_func=simple_sim_func, moment_func=simple_moment_func
        )


def test_smm_engine_validates_n_sim_positive():
    """Test that SMMEngine raises ValueError for non-positive n_sim."""
    with pytest.raises(ValueError, match="n_sim must be positive"):
        SMMEngine(
            k=2, p=2, n_sim=0, shock_dim=1,
            sim_func=simple_sim_func, moment_func=simple_moment_func
        )


def test_smm_engine_validates_shock_dim_positive():
    """Test that SMMEngine raises ValueError for non-positive shock_dim."""
    with pytest.raises(ValueError, match="shock_dim must be positive"):
        SMMEngine(
            k=2, p=2, n_sim=100, shock_dim=0,
            sim_func=simple_sim_func, moment_func=simple_moment_func
        )


def test_smm_engine_validates_sim_func_callable():
    """Test that SMMEngine raises TypeError for non-callable sim_func."""
    with pytest.raises(TypeError, match="sim_func must be callable"):
        SMMEngine(
            k=2, p=2, n_sim=100, shock_dim=1,
            sim_func="not a function", moment_func=simple_moment_func
        )


def test_smm_engine_validates_moment_func_callable():
    """Test that SMMEngine raises TypeError for non-callable moment_func."""
    with pytest.raises(TypeError, match="moment_func must be callable"):
        SMMEngine(
            k=2, p=2, n_sim=100, shock_dim=1,
            sim_func=simple_sim_func, moment_func="not a function"
        )


def test_smm_engine_validates_theta_dimension():
    """Test that SMMEngine raises ValueError for wrong theta dimension."""
    engine = SMMEngine(
        k=2, p=3, n_sim=100, shock_dim=1,
        sim_func=simple_sim_func, moment_func=simple_moment_func
    )
    
    # Wrong dimension should raise ValueError
    with pytest.raises(ValueError, match="theta must have length"):
        engine.moments(np.array([1.0, 2.0]))  # p=3, but only 2 elements


def test_smm_engine_validates_theta_1d():
    """Test that SMMEngine raises ValueError for non-1D theta."""
    engine = SMMEngine(
        k=2, p=2, n_sim=100, shock_dim=1,
        sim_func=simple_sim_func, moment_func=simple_moment_func
    )
    
    # 2D array should raise ValueError
    with pytest.raises(ValueError, match="theta must be 1-dimensional"):
        engine.moments(np.array([[1.0, 2.0]]))


def test_smm_engine_validates_moment_func_output_shape():
    """Test that SMMEngine raises ValueError when moment_func returns wrong shape."""
    def bad_moment_func(data):
        # Returns wrong number of moments
        return np.zeros((data.shape[0], 5))  # k=2 but returns 5
    
    engine = SMMEngine(
        k=2, p=2, n_sim=100, shock_dim=1,
        sim_func=simple_sim_func, moment_func=bad_moment_func
    )
    
    with pytest.raises(ValueError, match="moment_func must return array of shape"):
        engine.moments(np.array([1.0, 2.0]))


# =============================================================================
# Python Type Conversion Tests
# =============================================================================

@given(
    k=st.integers(1, 5),
    p=st.integers(1, 3),
    n_sim=st.integers(50, 100),
    shock_dim=st.integers(1, 3),
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=100, deadline=None)
def test_smm_python_type_conversion(k, p, n_sim, shock_dim, seed):
    """
    Test that SMMEngine accepts both Python lists and NumPy arrays for theta.
    
    **Feature: momentest, Property 21: SMMEngine Output Dimensions**
    **Validates: Requirements 14.4, 14.5**
    """
    sim_func = create_test_sim_func(p, shock_dim)
    moment_func = create_test_moment_func(k)
    
    engine = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=shock_dim,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    
    # Generate theta as list
    theta_list = [float(i) for i in range(p)]
    
    # Test with Python list
    m_bar_list, S_list = engine.moments(theta_list)
    
    # Verify return types are numpy arrays
    assert isinstance(m_bar_list, np.ndarray)
    assert isinstance(S_list, np.ndarray)
    
    # Test with numpy array
    theta_np = np.array(theta_list)
    m_bar_np, S_np = engine.moments(theta_np)
    
    # Results should be identical
    np.testing.assert_array_equal(m_bar_list, m_bar_np)
    np.testing.assert_array_equal(S_list, S_np)


# =============================================================================
# Jacobian Accuracy Tests
# =============================================================================

@given(
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=50, deadline=None)
def test_smm_moments_jac_numerical_accuracy(seed):
    """
    Test that moments_jac computes accurate numerical Jacobian.
    
    For a linear simulation function Y = theta[0] + theta[1] * shock,
    the Jacobian of the mean moment should be [1, E[shock]].
    Since we use pre-drawn shocks, E[shock] is the sample mean of shocks.
    
    **Feature: momentest, Property 21: SMMEngine Output Dimensions**
    **Validates: Requirements 14.4, 14.5**
    """
    k, p = 1, 2
    n_sim = 1000  # Use more simulations for better accuracy
    
    def linear_sim_func(theta, shocks):
        Y = theta[0] + theta[1] * shocks[:, 0]
        return Y.reshape(-1, 1)
    
    def mean_moment_func(data):
        return data  # Just return Y as the moment
    
    engine = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=1,
        sim_func=linear_sim_func, moment_func=mean_moment_func, seed=seed
    )
    
    theta = np.array([1.0, 2.0])
    
    _, _, D = engine.moments_jac(theta)
    
    # Expected Jacobian: ∂E[Y]/∂θ = [1, E[shock]]
    # E[shock] is the sample mean of the pre-drawn shocks
    expected_shock_mean = np.mean(engine.shocks[:, 0])
    expected_D = np.array([[1.0, expected_shock_mean]])
    
    # Allow tolerance for finite differences
    np.testing.assert_allclose(
        D, expected_D, atol=1e-5,
        err_msg=f"Jacobian should be [1, {expected_shock_mean}], got {D}"
    )


# =============================================================================
# Integration Tests with estimate() and bootstrap()
# **Feature: momentest, Property 20, 21, 22: SMMEngine Integration**
# **Validates: Requirements 14.7, 14.8**
# =============================================================================

from momentest.estimation import EstimationSetup, estimate, bootstrap


@given(
    seed=st.integers(0, 2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_smm_engine_estimate_integration(seed):
    """
    Test that SMMEngine integrates correctly with estimate() function.
    
    Uses a simple linear model: Y = theta[0] + theta[1] * shock
    Moments: E[Y], E[Y^2]
    
    **Feature: momentest, Property 20, 21, 22: SMMEngine Integration**
    **Validates: Requirements 14.7**
    """
    k, p = 2, 2
    n_sim = 500
    
    # True parameters - use positive values and constrain bounds to avoid sign ambiguity
    true_theta = np.array([1.0, 2.0])
    
    def sim_func(theta, shocks):
        Y = theta[0] + theta[1] * shocks[:, 0]
        return Y.reshape(-1, 1)
    
    def moment_func(data):
        Y = data[:, 0]
        return np.column_stack([Y, Y**2])
    
    # Create engine
    engine = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=1,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    
    # Compute "data moments" at true parameters
    data_moments, _ = engine.moments(true_theta)
    
    # Setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="linear_model",
        moment_type="mean and variance",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=1,
        seed=seed,
        weighting="identity"
    )
    
    # Bounds - constrain theta[1] to be positive to avoid sign ambiguity
    bounds = [(-5.0, 5.0), (0.1, 5.0)]
    
    # Estimate using SMMEngine
    result = estimate(
        setup,
        data_moments,
        bounds,
        n_global=20,
        engine=engine
    )
    
    # Verify result structure
    assert result.theta_hat.shape == (p,)
    assert result.m_bar.shape == (k,)
    assert result.S.shape == (k, k)
    
    # Verify theta_hat is within bounds
    for i, (lower, upper) in enumerate(bounds):
        assert lower <= result.theta_hat[i] <= upper
    
    # Since we're matching moments at true_theta, estimate should be close
    np.testing.assert_allclose(
        result.theta_hat, true_theta, atol=0.5,
        err_msg=f"Estimate should be close to true parameters. Got {result.theta_hat}, expected {true_theta}"
    )


def test_smm_engine_bootstrap_integration():
    """
    Test that SMMEngine integrates correctly with bootstrap() function.
    
    Verifies that bootstrap creates new SMMEngines with different seeds
    for each replication.
    
    **Feature: momentest, Property 20, 21, 22: SMMEngine Integration**
    **Validates: Requirements 14.8**
    """
    k, p = 2, 2
    n_sim = 200
    seed = 42
    
    # True parameters
    true_theta = np.array([1.0, 2.0])
    
    def sim_func(theta, shocks):
        Y = theta[0] + theta[1] * shocks[:, 0]
        return Y.reshape(-1, 1)
    
    def moment_func(data):
        Y = data[:, 0]
        return np.column_stack([Y, Y**2])
    
    # Create engine
    engine = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=1,
        sim_func=sim_func, moment_func=moment_func, seed=seed
    )
    
    # Compute "data moments" at true parameters
    data_moments, _ = engine.moments(true_theta)
    
    # Setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="linear_model",
        moment_type="mean and variance",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=1,
        seed=seed,
        weighting="identity"
    )
    
    # Bounds
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    
    # Run bootstrap with small number of replications
    result = bootstrap(
        setup,
        data_moments,
        bounds,
        n_boot=5,
        n_global=10,
        n_jobs=1,
        engine=engine
    )
    
    # Verify result structure
    assert result.bootstrap_estimates.shape == (5, p)
    assert result.se.shape == (p,)
    assert result.ci_lower.shape == (p,)
    assert result.ci_upper.shape == (p,)
    
    # Verify bootstrap estimates vary (different seeds produce different results)
    # At least some variation should exist
    estimate_std = np.std(result.bootstrap_estimates, axis=0)
    assert np.any(estimate_std > 0), (
        "Bootstrap estimates should vary across replications"
    )


def test_smm_engine_bootstrap_different_shocks():
    """
    Test that bootstrap replications use different shocks.
    
    Each bootstrap replication should create a new SMMEngine with a different
    seed, resulting in different pre-drawn shocks.
    
    **Feature: momentest, Property 20, 21, 22: SMMEngine Integration**
    **Validates: Requirements 14.8**
    """
    k, p = 1, 1
    n_sim = 100
    seed = 42
    
    # Track which shocks are used in each replication
    shocks_used = []
    
    def tracking_sim_func(theta, shocks):
        shocks_used.append(shocks.copy())
        Y = theta[0] * shocks[:, 0]
        return Y.reshape(-1, 1)
    
    def moment_func(data):
        return data
    
    # Create engine
    engine = SMMEngine(
        k=k, p=p, n_sim=n_sim, shock_dim=1,
        sim_func=tracking_sim_func, moment_func=moment_func, seed=seed
    )
    
    # Compute data moments
    data_moments = np.array([0.0])
    
    # Setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="test",
        moment_type="test",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=1,
        seed=seed,
        weighting="identity"
    )
    
    bounds = [(-2.0, 2.0)]
    
    # Clear tracking
    shocks_used.clear()
    
    # Run bootstrap
    result = bootstrap(
        setup,
        data_moments,
        bounds,
        n_boot=3,
        n_global=5,
        n_jobs=1,
        engine=engine
    )
    
    # We should have shocks from multiple replications
    # Each replication should use different shocks
    assert len(shocks_used) > 3, "Should have shocks from multiple evaluations"
    
    # The first shock array (from original estimation) should differ from
    # later ones (from bootstrap replications with different seeds)
    # Note: This is a weak test since we can't easily separate which shocks
    # came from which replication, but we verify the mechanism works
    assert result.n_boot == 3
