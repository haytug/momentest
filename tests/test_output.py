"""
Tests for output module.

Tests table generation and plotting functions.
"""

import numpy as np
import pytest

from momentest.output import (
    table_estimates,
    table_moments,
    table_bootstrap,
    summary,
)


class TestTableEstimates:
    """Tests for parameter estimates table."""
    
    def test_table_estimates_basic(self):
        """Test basic estimates table generation."""
        theta = np.array([1.0, 2.0])
        se = np.array([0.1, 0.2])
        
        table = table_estimates(theta, se)
        
        assert isinstance(table, str)
        assert "Parameter Estimates" in table
        assert "1.0000" in table
        assert "2.0000" in table
        assert "0.1000" in table
        assert "0.2000" in table
    
    def test_table_estimates_with_names(self):
        """Test estimates table with custom parameter names."""
        theta = np.array([1.0, 2.0])
        se = np.array([0.1, 0.2])
        param_names = ["mu", "sigma"]
        
        table = table_estimates(theta, se, param_names=param_names)
        
        assert "mu" in table
        assert "sigma" in table
    
    def test_table_estimates_with_ci(self):
        """Test estimates table with confidence intervals."""
        theta = np.array([1.0, 2.0])
        se = np.array([0.1, 0.2])
        ci_lower = np.array([0.8, 1.6])
        ci_upper = np.array([1.2, 2.4])
        
        table = table_estimates(theta, se, ci_lower=ci_lower, ci_upper=ci_upper)
        
        assert "CI" in table
        assert "0.8000" in table
        assert "1.2000" in table
    
    def test_table_estimates_nan_se(self):
        """Test estimates table handles NaN standard errors."""
        theta = np.array([1.0, 2.0])
        se = np.array([np.nan, 0.2])
        
        table = table_estimates(theta, se)
        
        assert "N/A" in table


class TestTableMoments:
    """Tests for moment comparison table."""
    
    def test_table_moments_basic(self):
        """Test basic moments table generation."""
        data_moments = np.array([1.0, 2.0, 3.0])
        model_moments = np.array([1.1, 1.9, 3.2])
        
        table = table_moments(data_moments, model_moments)
        
        assert isinstance(table, str)
        assert "Moment Comparison" in table
        assert "Data" in table
        assert "Model" in table
        assert "Diff" in table
    
    def test_table_moments_with_names(self):
        """Test moments table with custom moment names."""
        data_moments = np.array([1.0, 2.0])
        model_moments = np.array([1.1, 1.9])
        moment_names = ["mean", "variance"]
        
        table = table_moments(data_moments, model_moments, moment_names=moment_names)
        
        assert "mean" in table
        assert "variance" in table
    
    def test_table_moments_normalized(self):
        """Test moments table with normalized differences."""
        data_moments = np.array([10.0, 20.0])
        model_moments = np.array([11.0, 19.0])
        
        table = table_moments(data_moments, model_moments, normalize=True)
        
        assert "%Diff" in table
        assert "10.00%" in table  # (11-10)/10 * 100 = 10%


class TestTableBootstrap:
    """Tests for bootstrap results table."""
    
    def test_table_bootstrap_basic(self):
        """Test basic bootstrap table generation."""
        np.random.seed(42)
        bootstrap_estimates = np.random.randn(100, 2) + np.array([1.0, 2.0])
        theta_hat = np.array([1.0, 2.0])
        
        table = table_bootstrap(bootstrap_estimates, theta_hat)
        
        assert isinstance(table, str)
        assert "Bootstrap Results" in table
        assert "Boot.SE" in table
        assert "Bias" in table
    
    def test_table_bootstrap_with_names(self):
        """Test bootstrap table with custom parameter names."""
        np.random.seed(42)
        bootstrap_estimates = np.random.randn(100, 2) + np.array([1.0, 2.0])
        theta_hat = np.array([1.0, 2.0])
        param_names = ["mu", "sigma"]
        
        table = table_bootstrap(bootstrap_estimates, theta_hat, param_names=param_names)
        
        assert "mu" in table
        assert "sigma" in table
    
    def test_table_bootstrap_handles_nan(self):
        """Test bootstrap table handles NaN values."""
        bootstrap_estimates = np.array([
            [1.0, 2.0],
            [np.nan, np.nan],  # Invalid row
            [1.1, 2.1],
            [0.9, 1.9],
        ])
        theta_hat = np.array([1.0, 2.0])
        
        table = table_bootstrap(bootstrap_estimates, theta_hat)
        
        assert "n_valid=3" in table  # Should report 3 valid


class TestSummary:
    """Tests for comprehensive summary output."""
    
    def test_summary_basic(self):
        """Test basic summary generation."""
        theta = np.array([1.0, 2.0])
        se = np.array([0.1, 0.2])
        objective = 0.001
        data_moments = np.array([1.0, 2.0, 3.0])
        model_moments = np.array([1.0, 2.0, 3.0])
        
        output = summary(
            theta=theta,
            se=se,
            objective=objective,
            data_moments=data_moments,
            model_moments=model_moments,
            k=3,
            p=2,
            n=1000,
            converged=True
        )
        
        assert isinstance(output, str)
        assert "SMM Estimation Results" in output
        assert "Converged: True" in output
        assert "Parameter Estimates" in output
        assert "Moment Comparison" in output
    
    def test_summary_gmm(self):
        """Test summary for GMM method."""
        theta = np.array([1.0])
        se = np.array([0.1])
        
        output = summary(
            theta=theta,
            se=se,
            objective=0.001,
            data_moments=np.array([0.0]),
            model_moments=np.array([0.0]),
            k=1,
            p=1,
            n=100,
            converged=True,
            method="GMM"
        )
        
        assert "GMM Estimation Results" in output
    
    def test_summary_overidentified(self):
        """Test summary includes J-test for overidentified model."""
        theta = np.array([1.0])
        se = np.array([0.1])
        
        output = summary(
            theta=theta,
            se=se,
            objective=0.001,
            data_moments=np.array([0.0, 0.0, 0.0]),
            model_moments=np.array([0.0, 0.0, 0.0]),
            k=3,  # 3 moments
            p=1,  # 1 parameter -> overidentified
            n=100,
            converged=True
        )
        
        assert "J-Test" in output
    
    def test_summary_exactly_identified(self):
        """Test summary notes J-test not applicable for exactly identified."""
        theta = np.array([1.0, 2.0])
        se = np.array([0.1, 0.2])
        
        output = summary(
            theta=theta,
            se=se,
            objective=0.001,
            data_moments=np.array([0.0, 0.0]),
            model_moments=np.array([0.0, 0.0]),
            k=2,  # 2 moments
            p=2,  # 2 parameters -> exactly identified
            n=100,
            converged=True
        )
        
        assert "Not applicable" in output or "exactly identified" in output


class TestPlotFunctions:
    """Tests for plotting functions (basic smoke tests)."""
    
    @pytest.fixture
    def skip_if_no_matplotlib(self):
        """Skip test if matplotlib is not available."""
        pytest.importorskip("matplotlib")
    
    def test_plot_moment_comparison(self, skip_if_no_matplotlib):
        """Test moment comparison plot generation."""
        from momentest.output import plot_moment_comparison
        import matplotlib.pyplot as plt
        
        data_moments = np.array([1.0, 2.0, 3.0])
        model_moments = np.array([1.1, 1.9, 3.2])
        
        fig = plot_moment_comparison(data_moments, model_moments)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_bootstrap_distribution(self, skip_if_no_matplotlib):
        """Test bootstrap distribution plot generation."""
        from momentest.output import plot_bootstrap_distribution
        import matplotlib.pyplot as plt
        
        np.random.seed(42)
        bootstrap_estimates = np.random.randn(100, 2) + np.array([1.0, 2.0])
        theta_hat = np.array([1.0, 2.0])
        
        fig = plot_bootstrap_distribution(bootstrap_estimates, theta_hat)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_optimization_history(self, skip_if_no_matplotlib):
        """Test optimization history plot generation."""
        from momentest.output import plot_optimization_history
        import matplotlib.pyplot as plt
        
        history = [
            (np.array([0.5, 0.5]), 1.0),
            (np.array([0.8, 0.8]), 0.5),
            (np.array([1.0, 1.0]), 0.1),
        ]
        
        fig = plot_optimization_history(history)
        
        assert fig is not None
        plt.close(fig)
    
    def test_plot_sanity(self, skip_if_no_matplotlib):
        """Test sanity plot generation."""
        from momentest.output import plot_sanity
        import matplotlib.pyplot as plt
        
        # Simulate multiple optimization trials
        trial_results = [
            (np.array([1.0, 2.0]), 0.01),
            (np.array([1.1, 1.9]), 0.02),
            (np.array([0.9, 2.1]), 0.015),
            (np.array([1.05, 2.05]), 0.012),
        ]
        
        fig = plot_sanity(trial_results, param_names=["mu", "sigma"])
        
        assert fig is not None
        plt.close(fig)
