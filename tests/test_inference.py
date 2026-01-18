"""
Tests for inference module.

Tests J-test, sandwich covariance, asymptotic SE, Wald test, and confidence intervals.
"""

import numpy as np
import pytest
from scipy import stats

from momentest.inference import (
    JTestResult,
    j_test,
    sandwich_covariance,
    asymptotic_se,
    wald_test,
    confidence_interval,
)


class TestJTest:
    """Tests for J-test for overidentifying restrictions."""
    
    def test_j_test_basic(self):
        """Test basic J-test computation."""
        result = j_test(objective=0.05, n=1000, k=5, p=2)
        
        assert isinstance(result, JTestResult)
        assert result.df == 3  # k - p = 5 - 2
        assert result.J_statistic == 50.0  # n * objective = 1000 * 0.05
        assert 0 <= result.p_value <= 1
        assert result.alpha == 0.05
    
    def test_j_test_exactly_identified(self):
        """Test J-test raises error for exactly identified model."""
        with pytest.raises(ValueError, match="overidentification"):
            j_test(objective=0.01, n=100, k=2, p=2)
    
    def test_j_test_underidentified(self):
        """Test J-test raises error for underidentified model."""
        with pytest.raises(ValueError, match="overidentification"):
            j_test(objective=0.01, n=100, k=2, p=3)
    
    def test_j_test_reject_null(self):
        """Test J-test correctly rejects when J is large."""
        # Large objective -> large J -> small p-value -> reject
        result = j_test(objective=0.5, n=1000, k=5, p=2, alpha=0.05)
        assert result.reject == True
        assert result.p_value < 0.05
    
    def test_j_test_fail_to_reject(self):
        """Test J-test correctly fails to reject when J is small."""
        # Small objective -> small J -> large p-value -> fail to reject
        result = j_test(objective=0.001, n=100, k=5, p=2, alpha=0.05)
        assert result.reject == False
        assert result.p_value > 0.05
    
    def test_j_test_repr(self):
        """Test J-test result string representation."""
        result = j_test(objective=0.05, n=1000, k=5, p=2)
        repr_str = repr(result)
        
        assert "J-Test" in repr_str
        assert "J-statistic" in repr_str
        assert "Degrees of freedom" in repr_str
        assert "P-value" in repr_str
    
    def test_j_test_custom_alpha(self):
        """Test J-test with custom significance level."""
        result = j_test(objective=0.05, n=1000, k=5, p=2, alpha=0.10)
        assert result.alpha == 0.10


class TestSandwichCovariance:
    """Tests for sandwich covariance matrix computation."""
    
    def test_sandwich_basic(self):
        """Test basic sandwich covariance computation."""
        k, p, n = 3, 2, 100
        D = np.random.randn(k, p)
        W = np.eye(k)
        S = np.eye(k) * 0.1
        
        cov = sandwich_covariance(D, W, S, n)
        
        assert cov.shape == (p, p)
        # Covariance should be symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)
        # Diagonal should be positive
        assert np.all(np.diag(cov) >= 0)
    
    def test_sandwich_shape_mismatch_W(self):
        """Test sandwich raises error for W shape mismatch."""
        D = np.random.randn(3, 2)
        W = np.eye(4)  # Wrong shape
        S = np.eye(3)
        
        with pytest.raises(ValueError, match="W must have shape"):
            sandwich_covariance(D, W, S, n=100)
    
    def test_sandwich_shape_mismatch_S(self):
        """Test sandwich raises error for S shape mismatch."""
        D = np.random.randn(3, 2)
        W = np.eye(3)
        S = np.eye(4)  # Wrong shape
        
        with pytest.raises(ValueError, match="S must have shape"):
            sandwich_covariance(D, W, S, n=100)
    
    def test_sandwich_optimal_weighting(self):
        """Test sandwich with optimal weighting W = S^{-1}."""
        k, p, n = 3, 2, 100
        D = np.array([[1.0, 0.5], [0.3, 1.2], [0.8, 0.4]])
        S = np.eye(k) * 0.5
        W = np.linalg.inv(S)  # Optimal weighting
        
        cov = sandwich_covariance(D, W, S, n)
        
        # With optimal weighting, sandwich simplifies to (D'WD)^{-1}/n
        DtWD = D.T @ W @ D
        expected = np.linalg.inv(DtWD) / n
        np.testing.assert_array_almost_equal(cov, expected)


class TestAsymptoticSE:
    """Tests for asymptotic standard error computation."""
    
    def test_asymptotic_se_basic(self):
        """Test basic asymptotic SE computation."""
        k, p, n = 3, 2, 100
        D = np.random.randn(k, p)
        W = np.eye(k)
        S = np.eye(k) * 0.1
        
        se = asymptotic_se(D, W, S, n)
        
        assert se.shape == (p,)
        assert np.all(se >= 0)  # SE should be non-negative
    
    def test_asymptotic_se_matches_cov_diagonal(self):
        """Test SE equals sqrt of covariance diagonal."""
        k, p, n = 3, 2, 100
        D = np.random.randn(k, p)
        W = np.eye(k)
        S = np.eye(k) * 0.1
        
        cov = sandwich_covariance(D, W, S, n)
        se = asymptotic_se(D, W, S, n)
        
        np.testing.assert_array_almost_equal(se, np.sqrt(np.diag(cov)))


class TestWaldTest:
    """Tests for Wald test."""
    
    def test_wald_test_basic(self):
        """Test basic Wald test computation."""
        theta = np.array([1.0, 2.0])
        se = np.array([0.1, 0.2])
        
        t_stats, p_values, reject = wald_test(theta, se)
        
        assert t_stats.shape == (2,)
        assert p_values.shape == (2,)
        assert reject.shape == (2,)
        
        # t-stat = theta / se when theta_0 = 0
        np.testing.assert_array_almost_equal(t_stats, theta / se)
    
    def test_wald_test_custom_null(self):
        """Test Wald test with custom null hypothesis."""
        theta = np.array([1.0, 2.0])
        se = np.array([0.1, 0.2])
        theta_0 = np.array([1.0, 2.0])  # Null = estimated values
        
        t_stats, p_values, reject = wald_test(theta, se, theta_0)
        
        # t-stats should be zero when theta = theta_0
        np.testing.assert_array_almost_equal(t_stats, np.zeros(2))
        # p-values should be 1 when t-stats are zero
        np.testing.assert_array_almost_equal(p_values, np.ones(2))
        # Should not reject
        assert not np.any(reject)
    
    def test_wald_test_reject(self):
        """Test Wald test correctly rejects."""
        theta = np.array([5.0])  # Far from zero
        se = np.array([0.1])  # Small SE
        
        t_stats, p_values, reject = wald_test(theta, se, alpha=0.05)
        
        assert reject[0] == True
        assert p_values[0] < 0.05


class TestConfidenceInterval:
    """Tests for confidence interval computation."""
    
    def test_ci_basic(self):
        """Test basic confidence interval computation."""
        theta = np.array([1.0, 2.0])
        se = np.array([0.1, 0.2])
        
        ci_lower, ci_upper = confidence_interval(theta, se)
        
        assert ci_lower.shape == (2,)
        assert ci_upper.shape == (2,)
        
        # CI should contain the point estimate
        assert np.all(ci_lower < theta)
        assert np.all(ci_upper > theta)
    
    def test_ci_95_percent(self):
        """Test 95% CI uses correct z-value."""
        theta = np.array([0.0])
        se = np.array([1.0])
        
        ci_lower, ci_upper = confidence_interval(theta, se, alpha=0.05)
        
        # 95% CI: theta +/- 1.96 * se
        z = stats.norm.ppf(0.975)  # ~1.96
        np.testing.assert_almost_equal(ci_lower[0], -z, decimal=5)
        np.testing.assert_almost_equal(ci_upper[0], z, decimal=5)
    
    def test_ci_90_percent(self):
        """Test 90% CI uses correct z-value."""
        theta = np.array([0.0])
        se = np.array([1.0])
        
        ci_lower, ci_upper = confidence_interval(theta, se, alpha=0.10)
        
        # 90% CI: theta +/- 1.645 * se
        z = stats.norm.ppf(0.95)  # ~1.645
        np.testing.assert_almost_equal(ci_lower[0], -z, decimal=5)
        np.testing.assert_almost_equal(ci_upper[0], z, decimal=5)
    
    def test_ci_width_increases_with_se(self):
        """Test CI width increases with standard error."""
        theta = np.array([1.0])
        
        ci_lower1, ci_upper1 = confidence_interval(theta, np.array([0.1]))
        ci_lower2, ci_upper2 = confidence_interval(theta, np.array([0.5]))
        
        width1 = ci_upper1[0] - ci_lower1[0]
        width2 = ci_upper2[0] - ci_lower2[0]
        
        assert width2 > width1
