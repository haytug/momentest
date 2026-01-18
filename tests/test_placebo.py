"""
Placebo tests for momentest.

These tests verify that the estimation code behaves sensibly when
moments are uninformative or the objective is flat. This helps catch
bugs where the optimizer might return garbage or crash.

Test categories:
1. Flat objective handling - When moments don't identify parameters
2. Noise-only moments - When simulated moments are pure noise

These are "sanity checks" that ensure the code doesn't produce
spurious results when the model is misspecified or uninformative.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from momentest import (
    smm_estimate,
    gmm_estimate,
    SMMEngine,
    GMMEngine,
    EstimationSetup,
    estimate,
)


class TestFlatObjectiveHandling:
    """
    Tests for handling flat or nearly-flat objective functions.
    
    When moments don't depend on parameters (or depend very weakly),
    the objective function is flat. The optimizer should:
    1. Not crash
    2. Return some estimate within bounds
    3. Indicate potential issues (e.g., large SE, non-convergence)
    """

    def test_constant_moments_returns_within_bounds(self):
        """
        When moments are constant (don't depend on theta), estimation
        should still return a result within bounds without crashing.
        """
        # Simulation function that ignores theta
        def sim_func(theta, shocks):
            # Returns constant regardless of theta
            return np.ones_like(shocks) * 5.0
        
        # Moment function
        def moment_func(sim_data):
            return np.column_stack([sim_data, sim_data**2])
        
        # Target moments (won't match, but that's okay)
        data_moments = np.array([3.0, 10.0])
        
        # This should not crash
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0, 10), (0.1, 5)],
            n_sim=500,
            shock_dim=1,
            seed=42,
            weighting="identity",
            n_global=20
        )
        
        # Result should be within bounds
        assert 0 <= result.theta[0] <= 10, "theta[0] should be within bounds"
        assert 0.1 <= result.theta[1] <= 5, "theta[1] should be within bounds"
        
        # Objective should be positive (can't match constant moments)
        assert result.objective > 0, "Objective should be positive for mismatched moments"

    def test_weakly_identified_model_no_crash(self):
        """
        When parameters are weakly identified, estimation should
        complete without crashing, even if results are imprecise.
        """
        # Simulation function where theta[1] has very weak effect
        def sim_func(theta, shocks):
            mu = theta[0]
            sigma = theta[1]
            # sigma has very weak effect (multiplied by small constant)
            return mu + 0.001 * sigma * shocks
        
        def moment_func(sim_data):
            return np.column_stack([sim_data, sim_data**2])
        
        # Target moments
        data_moments = np.array([2.0, 4.1])
        
        # Should not crash
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0, 5), (0.1, 5)],
            n_sim=500,
            shock_dim=1,
            seed=42,
            weighting="identity",
            n_global=20
        )
        
        # Result should be within bounds
        assert 0 <= result.theta[0] <= 5
        assert 0.1 <= result.theta[1] <= 5
        
        # mu should be close to target (it's well-identified)
        assert abs(result.theta[0] - 2.0) < 0.5, "mu should be close to target"

    def test_gmm_with_uninformative_moments(self):
        """
        GMM with moments that don't identify parameters should
        return results within bounds without crashing.
        """
        # Generate data
        np.random.seed(42)
        n = 500
        Y = np.random.randn(n)
        X = np.random.randn(n)  # X is independent of Y
        
        data = {'Y': Y, 'X': X}
        
        # Moment function that doesn't identify beta
        # (X is independent of Y, so E[X*Y] ≈ 0 regardless of beta)
        def moment_func(data, theta):
            beta = theta[0]
            # This moment is uninformative because X ⊥ Y
            residual = data['Y'] - beta * data['X']
            return residual.reshape(-1, 1)
        
        # Should not crash
        result = gmm_estimate(
            data=data,
            moment_func=moment_func,
            bounds=[(-10, 10)],
            k=1,
            weighting="identity",
            n_global=20,
            seed=42
        )
        
        # Result should be within bounds
        assert -10 <= result.theta[0] <= 10


class TestNoiseOnlyMoments:
    """
    Tests for when simulated moments are pure noise.
    
    When moments don't depend on theta at all (pure noise),
    estimates should be diffuse across the parameter space.
    """

    def test_pure_noise_moments_diffuse_estimates(self):
        """
        When moments are pure noise, estimates from different seeds
        should be spread across the parameter space.
        """
        # Simulation function that returns pure noise
        def sim_func(theta, shocks):
            # Completely ignores theta
            return shocks
        
        def moment_func(sim_data):
            return np.column_stack([sim_data, sim_data**2])
        
        # Target moments
        data_moments = np.array([0.0, 1.0])
        
        # Run estimation with different seeds
        estimates = []
        for seed in [42, 123, 456, 789, 1011]:
            result = smm_estimate(
                sim_func=sim_func,
                moment_func=moment_func,
                data_moments=data_moments,
                bounds=[(0, 10), (0.1, 5)],
                n_sim=500,
                shock_dim=1,
                seed=seed,
                weighting="identity",
                n_global=20
            )
            estimates.append(result.theta)
        
        estimates = np.array(estimates)
        
        # All estimates should be within bounds
        assert np.all(estimates[:, 0] >= 0) and np.all(estimates[:, 0] <= 10)
        assert np.all(estimates[:, 1] >= 0.1) and np.all(estimates[:, 1] <= 5)
        
        # Estimates should show some variation (not all identical)
        # This is a weak test - just checking the code doesn't always
        # return the same value
        std_theta0 = np.std(estimates[:, 0])
        std_theta1 = np.std(estimates[:, 1])
        
        # At least one parameter should show variation
        # (with pure noise, optimizer will find different local minima)
        assert std_theta0 > 0.01 or std_theta1 > 0.01, (
            "Estimates should show some variation with pure noise moments"
        )

    def test_noise_moments_objective_not_zero(self):
        """
        With noise-only moments, objective should generally not be zero
        (can't perfectly match random moments).
        """
        def sim_func(theta, shocks):
            return shocks  # Pure noise
        
        def moment_func(sim_data):
            return sim_data.reshape(-1, 1)
        
        # Target moment that won't be matched by noise
        data_moments = np.array([5.0])
        
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0, 10)],
            n_sim=500,
            shock_dim=1,
            seed=42,
            weighting="identity",
            n_global=20
        )
        
        # Objective should be positive (can't match target with noise)
        assert result.objective > 0, "Objective should be positive"

    def test_gmm_noise_data_no_crash(self):
        """
        GMM with pure noise data should not crash.
        """
        np.random.seed(42)
        n = 500
        
        # Pure noise data
        data = {
            'Y': np.random.randn(n),
            'X': np.random.randn(n),
            'Z': np.random.randn(n)
        }
        
        # Standard IV moment function
        def moment_func(data, theta):
            b0, b1 = theta
            residual = data['Y'] - b0 - b1 * data['X']
            return np.column_stack([residual, residual * data['Z']])
        
        # Should not crash
        result = gmm_estimate(
            data=data,
            moment_func=moment_func,
            bounds=[(-10, 10), (-10, 10)],
            k=2,
            weighting="identity",
            n_global=20,
            seed=42
        )
        
        # Results should be within bounds
        assert -10 <= result.theta[0] <= 10
        assert -10 <= result.theta[1] <= 10


class TestEdgeCases:
    """
    Tests for edge cases and boundary conditions.
    """

    def test_single_parameter_estimation(self):
        """
        Estimation with a single parameter should work.
        """
        def sim_func(theta, shocks):
            return theta[0] * shocks
        
        def moment_func(sim_data):
            return (sim_data**2).reshape(-1, 1)
        
        # E[X²] = theta² when X = theta * N(0,1)
        true_theta = 2.0
        data_moments = np.array([true_theta**2])
        
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0.1, 5)],
            n_sim=1000,
            shock_dim=1,
            seed=42,
            weighting="identity",
            n_global=30
        )
        
        # Should recover theta (up to sign, but we bounded to positive)
        assert abs(result.theta[0] - true_theta) < 0.5

    def test_single_moment_estimation(self):
        """
        Estimation with a single moment should work.
        """
        def sim_func(theta, shocks):
            mu, sigma = theta
            return mu + sigma * shocks
        
        def moment_func(sim_data):
            # Only use first moment
            return sim_data.reshape(-1, 1)
        
        # E[X] = mu
        true_mu = 3.0
        data_moments = np.array([true_mu])
        
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0, 10), (0.1, 5)],
            n_sim=1000,
            shock_dim=1,
            seed=42,
            weighting="identity",
            n_global=30
        )
        
        # mu should be recovered (sigma is not identified by first moment alone)
        assert abs(result.theta[0] - true_mu) < 0.5

    def test_tight_bounds_estimation(self):
        """
        Estimation with tight bounds should work and respect bounds.
        """
        def sim_func(theta, shocks):
            return theta[0] + theta[1] * shocks
        
        def moment_func(sim_data):
            return np.column_stack([sim_data, sim_data**2])
        
        data_moments = np.array([2.0, 5.0])
        
        # Very tight bounds
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(1.9, 2.1), (0.9, 1.1)],
            n_sim=500,
            shock_dim=1,
            seed=42,
            weighting="identity",
            n_global=20
        )
        
        # Results must be within tight bounds
        assert 1.9 <= result.theta[0] <= 2.1
        assert 0.9 <= result.theta[1] <= 1.1

    def test_many_parameters_estimation(self):
        """
        Estimation with many parameters should work.
        """
        p = 5  # 5 parameters
        
        def sim_func(theta, shocks):
            # Linear combination of parameters
            return np.sum(theta) + shocks[:, 0]
        
        def moment_func(sim_data):
            return np.column_stack([
                sim_data,
                sim_data**2,
                sim_data**3,
                sim_data**4,
                sim_data**5
            ])
        
        # Target moments
        true_sum = 5.0
        data_moments = np.array([
            true_sum,
            true_sum**2 + 1,  # E[X²] = μ² + σ²
            true_sum**3 + 3*true_sum,  # E[X³]
            true_sum**4 + 6*true_sum**2 + 3,  # E[X⁴]
            true_sum**5 + 10*true_sum**3 + 15*true_sum  # E[X⁵]
        ])
        
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0, 3)] * p,
            n_sim=1000,
            shock_dim=1,
            seed=42,
            weighting="identity",
            n_global=50
        )
        
        # Sum of parameters should be close to true_sum
        estimated_sum = np.sum(result.theta)
        assert abs(estimated_sum - true_sum) < 1.0, (
            f"Sum of parameters {estimated_sum:.2f} should be close to {true_sum}"
        )


class TestRobustness:
    """
    Tests for robustness to various conditions.
    """

    def test_large_n_sim_no_memory_issues(self):
        """
        Large n_sim should work without memory issues.
        """
        def sim_func(theta, shocks):
            return theta[0] + theta[1] * shocks
        
        def moment_func(sim_data):
            return np.column_stack([sim_data, sim_data**2])
        
        data_moments = np.array([2.0, 5.0])
        
        # Large n_sim
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0, 5), (0.1, 3)],
            n_sim=10000,
            shock_dim=1,
            seed=42,
            weighting="identity",
            n_global=20
        )
        
        # Should complete and give reasonable results
        assert result.converged or result.objective < 1.0

    def test_small_n_sim_still_works(self):
        """
        Small n_sim should still work (though with more noise).
        """
        def sim_func(theta, shocks):
            return theta[0] + theta[1] * shocks
        
        def moment_func(sim_data):
            return np.column_stack([sim_data, sim_data**2])
        
        data_moments = np.array([2.0, 5.0])
        
        # Small n_sim
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0, 5), (0.1, 3)],
            n_sim=50,
            shock_dim=1,
            seed=42,
            weighting="identity",
            n_global=20
        )
        
        # Should complete without crashing
        assert 0 <= result.theta[0] <= 5
        assert 0.1 <= result.theta[1] <= 3

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=10, deadline=30000)
    def test_different_seeds_no_crash(self, seed):
        """
        Estimation should not crash for any seed.
        """
        def sim_func(theta, shocks):
            return theta[0] + theta[1] * shocks
        
        def moment_func(sim_data):
            return np.column_stack([sim_data, sim_data**2])
        
        data_moments = np.array([2.0, 5.0])
        
        result = smm_estimate(
            sim_func=sim_func,
            moment_func=moment_func,
            data_moments=data_moments,
            bounds=[(0, 5), (0.1, 3)],
            n_sim=200,
            shock_dim=1,
            seed=seed,
            weighting="identity",
            n_global=10
        )
        
        # Should complete without crashing
        assert 0 <= result.theta[0] <= 5
        assert 0.1 <= result.theta[1] <= 3
