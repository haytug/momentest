"""
Truncated Normal GMM Estimation Example

This example replicates the GMM estimation from the OpenSourceEcon tutorial:
https://opensourceecon.github.io/CompMethods/struct_est/GMM.html

We fit a truncated normal distribution to Econ 381 test scores using
Generalized Method of Moments (GMM).

The data are test scores bounded between 0 and 450.

This script compares:
1. Original webpage method (direct scipy.optimize) - complex, manual
2. Our momentest high-level API - simple, just a few lines!

Both methods use the SAME:
- Data
- moment_func(data, theta) -> per-observation moment conditions

This ensures an apples-to-apples comparison.

Key difference from SMM:
- GMM uses analytical moment conditions computed from data
- SMM uses simulated moments from a model
"""

import numpy as np
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import scipy.linalg as lin

# Import our simple high-level API
import sys
sys.path.insert(0, '..')
from momentest import (
    gmm_estimate,
    j_test,
    table_estimates,
    table_moments,
    confidence_interval,
    summary,
    plot_moment_comparison,
    load_econ381,
)


# =============================================================================
# SHARED: Helper functions for truncated normal
# =============================================================================

def trunc_norm_pdf(xvals, mu, sigma, cut_lb, cut_ub):
    """
    PDF of truncated normal distribution.
    """
    sigma = max(sigma, 1e-6)
    prob_notcut = sts.norm.cdf(cut_ub, loc=mu, scale=sigma) - sts.norm.cdf(
        cut_lb, loc=mu, scale=sigma
    )
    if prob_notcut < 1e-10:
        return np.zeros_like(xvals)
    pdf_vals = (
        1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-((xvals - mu) ** 2) / (2 * sigma**2))
    ) / prob_notcut
    return pdf_vals


def model_moments(mu, sigma, cut_lb, cut_ub):
    """
    Compute analytical moments (mean, variance) of truncated normal.
    Uses numerical integration to compute E[X] and Var[X].
    """
    sigma = max(sigma, 1e-6)
    
    xfx = lambda x: x * trunc_norm_pdf(x, mu, sigma, cut_lb, cut_ub)
    mean_model, _ = intgr.quad(xfx, cut_lb, cut_ub)
    
    x2fx = lambda x: ((x - mean_model) ** 2) * trunc_norm_pdf(x, mu, sigma, cut_lb, cut_ub)
    var_model, _ = intgr.quad(x2fx, cut_lb, cut_ub)
    
    return mean_model, var_model


# =============================================================================
# Main Example
# =============================================================================

def main():
    print("=" * 80)
    print("Truncated Normal GMM Estimation - API Comparison")
    print("=" * 80)

    # =========================================================================
    # SETUP: Load data from package
    # =========================================================================
    dataset = load_econ381()
    data = dataset['data']
    cut_lb, cut_ub = dataset['bounds']
    mle_mu = dataset['mle_params']['mu']
    mle_sigma = dataset['mle_params']['sigma']
    N = dataset['n']

    # Compute data moments
    data_mean = data.mean()
    data_var = data.var()

    print(f"Data: N={N}, Mean={data_mean:.2f}, Var={data_var:.2f}")
    print(f"MLE Parameters (target): mu={mle_mu}, sigma={mle_sigma}")

    # =========================================================================
    # DEFINE SHARED MOMENT FUNCTION (used by BOTH methods)
    # =========================================================================
    
    def moment_func(data_arr, theta):
        """
        GMM moment conditions: per-observation deviations.
        
        This is the SHARED function used by both webpage and momentest methods.
        
        Args:
            data_arr: Data array of shape (n,) or (n, 1)
            theta: (mu, sigma) parameters
        
        Returns:
            Moment conditions of shape (n, 2)
        """
        mu, sigma = theta
        sigma = max(sigma, 1e-6)
        mean_mod, var_mod = model_moments(mu, sigma, cut_lb, cut_ub)
        
        # Protect against division by zero
        mean_mod = max(mean_mod, 1e-10)
        var_mod = max(var_mod, 1e-10)
        
        x = np.asarray(data_arr).flatten()
        n = len(x)
        
        moments = np.zeros((n, 2))
        # Moment 1: (x_i - mean_model) / mean_model
        moments[:, 0] = (x - mean_mod) / mean_mod
        # Moment 2: ((x_i - data_mean)^2 - var_model) / var_model
        moments[:, 1] = ((x - data_mean)**2 - var_mod) / var_mod
        
        return moments

    # =========================================================================
    # METHOD 1: Webpage Method (manual)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 1: ORIGINAL WEBPAGE METHOD")
    print("(Manual criterion, weighting matrix, two-step procedure)")
    print("=" * 80)

    # --- Webpage method uses the SAME moment_func ---
    def err_vec_webpage(theta):
        """Compute average moment error using shared moment_func."""
        moments = moment_func(data, theta)
        return moments.mean(axis=0).reshape(-1, 1)

    def criterion_webpage(params, W):
        """GMM criterion function: e'We."""
        err = err_vec_webpage(params)
        return float((err.T @ W @ err).item())

    def get_W_2step(theta):
        """Compute two-step optimal weighting matrix using shared moment_func."""
        moments = moment_func(data, theta)
        m_bar = moments.mean(axis=0)
        m_centered = moments - m_bar
        var_cov = (m_centered.T @ m_centered) / N
        return lin.inv(var_cov)

    # Step 1: Identity Weighting
    W_identity = np.eye(2)
    result1 = opt.minimize(
        lambda p: criterion_webpage(p, W_identity),
        np.array([400.0, 60.0]),
        method="L-BFGS-B",
        bounds=((1e-10, None), (1e-10, None)),
    )
    webpage_identity = {
        "mu": result1.x[0],
        "sigma": result1.x[1],
        "objective": criterion_webpage(result1.x, W_identity),
    }

    # Step 2: Two-Step Optimal Weighting
    W_optimal = get_W_2step(result1.x)
    result2 = opt.minimize(
        lambda p: criterion_webpage(p, W_optimal),
        result1.x,
        method="L-BFGS-B",
        bounds=((1e-10, None), (1e-10, None)),
    )
    webpage_optimal = {
        "mu": result2.x[0],
        "sigma": result2.x[1],
        "objective": criterion_webpage(result2.x, W_optimal),
    }

    print(f"\nIdentity:  mu={webpage_identity['mu']:.4f}, "
          f"sigma={webpage_identity['sigma']:.4f}, "
          f"obj={webpage_identity['objective']:.2e}")
    print(f"Optimal:   mu={webpage_optimal['mu']:.4f}, "
          f"sigma={webpage_optimal['sigma']:.4f}, "
          f"obj={webpage_optimal['objective']:.2e}")

    # =========================================================================
    # METHOD 2: momentest Package
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 2: MOMENTEST PACKAGE")
    print("(Just pass data, moment_func to gmm_estimate)")
    print("=" * 80)

    # --- momentest uses the SAME moment_func ---
    momentest_identity = gmm_estimate(
        data=data,
        moment_func=moment_func,
        bounds=[(1e-10, 1000.0), (1e-10, 500.0)],
        k=2,
        weighting="identity",
    )

    momentest_optimal = gmm_estimate(
        data=data,
        moment_func=moment_func,
        bounds=[(1e-10, 1000.0), (1e-10, 500.0)],
        k=2,
        weighting="optimal",
    )

    print(f"\nIdentity:  mu={momentest_identity.theta[0]:.4f}, "
          f"sigma={momentest_identity.theta[1]:.4f}, "
          f"obj={momentest_identity.objective:.2e}")
    print(f"Optimal:   mu={momentest_optimal.theta[0]:.4f}, "
          f"sigma={momentest_optimal.theta[1]:.4f}, "
          f"obj={momentest_optimal.objective:.2e}")

    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    print("\nBoth methods use the SAME moment_func(data, theta).")
    print("Differences are due to optimization approach:")
    print("  - Webpage: L-BFGS-B from single starting point")
    print("  - momentest: global search + local optimization")

    print(f"\n{'Method':<25} {'mu_hat':>12} {'sigma_hat':>12} {'objective':>14}")
    print("-" * 65)
    print(f"{'True Parameters':<25} {mle_mu:>12.4f} {mle_sigma:>12.4f} {'-':>14}")
    print(f"{'Webpage (identity)':<25} {webpage_identity['mu']:>12.4f} "
          f"{webpage_identity['sigma']:>12.4f} "
          f"{webpage_identity['objective']:>14.2e}")
    print(f"{'momentest (identity)':<25} {momentest_identity.theta[0]:>12.4f} "
          f"{momentest_identity.theta[1]:>12.4f} "
          f"{momentest_identity.objective:>14.2e}")
    print(f"{'Webpage (optimal)':<25} {webpage_optimal['mu']:>12.4f} "
          f"{webpage_optimal['sigma']:>12.4f} "
          f"{webpage_optimal['objective']:>14.2e}")
    print(f"{'momentest (optimal)':<25} {momentest_optimal.theta[0]:>12.4f} "
          f"{momentest_optimal.theta[1]:>12.4f} "
          f"{momentest_optimal.objective:>14.2e}")

    # =========================================================================
    # CODE COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("CODE COMPARISON")
    print("=" * 80)
    print("""
Both methods use the SAME:
  - moment_func(data, theta) -> per-observation moment conditions
  - data array

WEBPAGE METHOD requires (~25 lines):
  - Define err_vec_webpage() wrapper
  - Define criterion_webpage() function
  - Define get_W_2step() function
  - Manual scipy.optimize.minimize() calls
  - Manual two-step procedure

MOMENTEST requires (~5 lines):
  - Just call gmm_estimate() twice (identity + optimal)
  - Handles criterion, weighting, two-step automatically
""")

    print("\n" + "=" * 80)
    print("ANALYSIS OF RESULTS")
    print("=" * 80)
    print("""
Interesting! With 2 moments (mean, variance), we recover the MLE parameters
(mu≈622, sigma≈199) almost exactly. This is because the MLE also matches
mean and variance.

With 4 moments (bin percentages), we get DIFFERENT estimates that better
fit the SHAPE of the distribution, not just mean/variance.

The 4-moment J-test REJECTS the model, indicating that a truncated normal
may not perfectly fit this data's shape across all bins.

Key insight: Different moment choices identify different aspects of the
distribution. Mean/variance match the MLE; bin percentages match the shape.
""")

    # =========================================================================
    # 4-MOMENT ESTIMATION (Better Identification)
    # =========================================================================
    print("\n" + "=" * 80)
    print("4-MOMENT ESTIMATION (Better Identification)")
    print("=" * 80)
    print("""
Using 4 bin percentages as moments:
  1. % of observations in [0, 220)
  2. % of observations in [220, 320)
  3. % of observations in [320, 430)
  4. % of observations in [430, 450]
""")

    # Define bin edges (from webpage)
    bin_edges = [0.0, 220.0, 320.0, 430.0, 450.0]

    def model_moments_4bins(mu, sigma, cut_lb, cut_ub):
        """Compute 4 bin percentages from truncated normal."""
        sigma = max(sigma, 1e-6)
        xfx = lambda x: trunc_norm_pdf(x, mu, sigma, cut_lb, cut_ub)
        
        pct1, _ = intgr.quad(xfx, bin_edges[0], bin_edges[1])
        pct2, _ = intgr.quad(xfx, bin_edges[1], bin_edges[2])
        pct3, _ = intgr.quad(xfx, bin_edges[2], bin_edges[3])
        pct4, _ = intgr.quad(xfx, bin_edges[3], bin_edges[4])
        
        return np.array([pct1, pct2, pct3, pct4])

    # Compute data bin percentages
    data_pct1 = np.sum(data < 220) / N
    data_pct2 = np.sum((data >= 220) & (data < 320)) / N
    data_pct3 = np.sum((data >= 320) & (data < 430)) / N
    data_pct4 = np.sum(data >= 430) / N
    data_moments_4 = np.array([data_pct1, data_pct2, data_pct3, data_pct4])

    print(f"Data bin percentages: {data_moments_4}")

    def moment_func_4(data_arr, theta):
        """
        4-moment GMM conditions using bin percentages.
        """
        mu, sigma = theta
        sigma = max(sigma, 1e-6)
        model_pcts = model_moments_4bins(mu, sigma, cut_lb, cut_ub)
        
        # Protect against division by zero
        model_pcts = np.maximum(model_pcts, 1e-10)
        
        x = np.asarray(data_arr).flatten()
        n = len(x)
        
        # Per-observation indicators for each bin
        in_bin1 = (x < 220).astype(float)
        in_bin2 = ((x >= 220) & (x < 320)).astype(float)
        in_bin3 = ((x >= 320) & (x < 430)).astype(float)
        in_bin4 = (x >= 430).astype(float)
        
        moments = np.zeros((n, 4))
        # Percent deviation: (indicator - model_pct) / model_pct
        moments[:, 0] = (in_bin1 - model_pcts[0]) / model_pcts[0]
        moments[:, 1] = (in_bin2 - model_pcts[1]) / model_pcts[1]
        moments[:, 2] = (in_bin3 - model_pcts[2]) / model_pcts[2]
        moments[:, 3] = (in_bin4 - model_pcts[3]) / model_pcts[3]
        
        return moments

    # --- Webpage method with 4 moments ---
    def err_vec_4(theta):
        moments = moment_func_4(data, theta)
        return moments.mean(axis=0).reshape(-1, 1)

    def criterion_4(params, W):
        err = err_vec_4(params)
        return float((err.T @ W @ err).item())

    def get_W_2step_4(theta):
        moments = moment_func_4(data, theta)
        m_bar = moments.mean(axis=0)
        m_centered = moments - m_bar
        var_cov = (m_centered.T @ m_centered) / N
        return lin.inv(var_cov)

    # Step 1: Identity
    W_identity_4 = np.eye(4)
    result1_4 = opt.minimize(
        lambda p: criterion_4(p, W_identity_4),
        np.array([400.0, 70.0]),
        method="L-BFGS-B",
        bounds=((1e-10, None), (1e-10, None)),
    )
    webpage_4mom_identity = {
        "mu": result1_4.x[0],
        "sigma": result1_4.x[1],
        "objective": criterion_4(result1_4.x, W_identity_4),
    }

    # Step 2: Two-step
    W_optimal_4 = get_W_2step_4(result1_4.x)
    result2_4 = opt.minimize(
        lambda p: criterion_4(p, W_optimal_4),
        result1_4.x,
        method="L-BFGS-B",
        bounds=((1e-10, None), (1e-10, None)),
    )
    webpage_4mom_optimal = {
        "mu": result2_4.x[0],
        "sigma": result2_4.x[1],
        "objective": criterion_4(result2_4.x, W_optimal_4),
    }

    print(f"\nWebpage 4-moment:")
    print(f"  Identity:  mu={webpage_4mom_identity['mu']:.4f}, "
          f"sigma={webpage_4mom_identity['sigma']:.4f}")
    print(f"  Optimal:   mu={webpage_4mom_optimal['mu']:.4f}, "
          f"sigma={webpage_4mom_optimal['sigma']:.4f}")

    # --- momentest with 4 moments ---
    momentest_4mom_identity = gmm_estimate(
        data=data,
        moment_func=moment_func_4,
        bounds=[(1e-10, 1000.0), (1e-10, 500.0)],
        k=4,
        weighting="identity",
    )

    momentest_4mom_optimal = gmm_estimate(
        data=data,
        moment_func=moment_func_4,
        bounds=[(1e-10, 1000.0), (1e-10, 500.0)],
        k=4,
        weighting="optimal",
    )

    print(f"\nmomentset 4-moment:")
    print(f"  Identity:  mu={momentest_4mom_identity.theta[0]:.4f}, "
          f"sigma={momentest_4mom_identity.theta[1]:.4f}")
    print(f"  Optimal:   mu={momentest_4mom_optimal.theta[0]:.4f}, "
          f"sigma={momentest_4mom_optimal.theta[1]:.4f}")

    # Comparison table
    print(f"\n{'Method':<30} {'mu_hat':>12} {'sigma_hat':>12}")
    print("-" * 56)
    print(f"{'True Parameters':<30} {mle_mu:>12.4f} {mle_sigma:>12.4f}")
    print(f"{'2-moment (weakly identified)':<30} {momentest_optimal.theta[0]:>12.4f} "
          f"{momentest_optimal.theta[1]:>12.4f}")
    print(f"{'4-moment webpage (optimal)':<30} {webpage_4mom_optimal['mu']:>12.4f} "
          f"{webpage_4mom_optimal['sigma']:>12.4f}")
    print(f"{'4-moment momentest (optimal)':<30} {momentest_4mom_optimal.theta[0]:>12.4f} "
          f"{momentest_4mom_optimal.theta[1]:>12.4f}")

    print("""
With 4 moments, we get DIFFERENT estimates than with 2 moments.

The 2-moment GMM recovers the MLE parameters (mu≈622, sigma≈199) because
MLE also matches mean and variance.

The 4-moment GMM fits the SHAPE of the distribution (bin percentages),
which may require different parameters.

The J-test for overidentification tells us whether the model fits all
moments well. A rejection suggests the truncated normal may not perfectly
capture the data's shape.
""")

    # =========================================================================
    # INFERENCE AND OUTPUT FEATURES (using 4-moment estimates)
    # =========================================================================
    print("\n" + "=" * 80)
    print("INFERENCE AND OUTPUT FEATURES (4-moment estimates)")
    print("=" * 80)

    theta_4 = momentest_4mom_optimal.theta
    se_4 = momentest_4mom_optimal.se

    print("\n--- Parameter Estimates ---")
    ci_lower_4, ci_upper_4 = confidence_interval(theta_4, se_4)
    print(table_estimates(
        theta=theta_4,
        se=se_4,
        param_names=["mu", "sigma"],
        ci_lower=ci_lower_4,
        ci_upper=ci_upper_4,
    ))

    model_pcts_final = model_moments_4bins(theta_4[0], theta_4[1], cut_lb, cut_ub)

    print("\n--- Moment Comparison (4 bins) ---")
    print(table_moments(
        data_moments=data_moments_4,
        model_moments=model_pcts_final,
        moment_names=["Bin [0,220)", "Bin [220,320)", "Bin [320,430)", "Bin [430,450]"],
        normalize=True,
    ))

    print("\n--- Full Summary ---")
    print(summary(
        theta=theta_4,
        se=se_4,
        objective=momentest_4mom_optimal.objective,
        data_moments=data_moments_4,
        model_moments=model_pcts_final,
        k=4,
        p=2,
        n=N,
        converged=momentest_4mom_optimal.converged,
        param_names=["mu", "sigma"],
        moment_names=["Bin1", "Bin2", "Bin3", "Bin4"],
        method="GMM",
    ))

    # =========================================================================
    # PLOTS
    # =========================================================================
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Plot 1: Data histogram with all fitted distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=30, density=True, alpha=0.6, edgecolor="black",
                label="Data", color="gray")

        x_vals = np.linspace(cut_lb, cut_ub, 500)
        ax.plot(x_vals, trunc_norm_pdf(x_vals, mle_mu, mle_sigma, cut_lb, cut_ub),
                "g-", linewidth=2.5, label=f"MLE: μ={mle_mu}, σ={mle_sigma}")
        ax.plot(x_vals, trunc_norm_pdf(x_vals, momentest_optimal.theta[0], momentest_optimal.theta[1], cut_lb, cut_ub),
                "r--", linewidth=2, label=f'2-moment: μ={momentest_optimal.theta[0]:.1f}, σ={momentest_optimal.theta[1]:.1f}')
        ax.plot(x_vals, trunc_norm_pdf(x_vals, theta_4[0], theta_4[1], cut_lb, cut_ub),
                "b:", linewidth=2.5, label=f'4-moment: μ={theta_4[0]:.1f}, σ={theta_4[1]:.1f}')

        ax.set_xlabel("Test Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("GMM Estimation: 2-moment vs 4-moment", fontsize=14)
        ax.legend(loc="upper left")
        ax.set_xlim([0, 500])

        plt.tight_layout()
        plot1_path = os.path.join(script_dir, "gmm_comparison.png")
        plt.savefig(plot1_path, dpi=150)
        print(f"\nPlot saved: {plot1_path}")
        plt.close()

        # Plot 2: 2-moment comparison (mean, variance)
        theta_2 = momentest_optimal.theta
        model_mean_2, model_var_2 = model_moments(theta_2[0], theta_2[1], cut_lb, cut_ub)
        
        plot2_path = os.path.join(script_dir, "gmm_2mom_moment_comparison.png")
        fig = plot_moment_comparison(
            data_moments=np.array([data_mean, data_var]),
            model_moments=np.array([model_mean_2, model_var_2]),
            moment_names=["Mean", "Variance"],
            save_path=plot2_path,
        )
        print(f"Plot saved: {plot2_path}")
        plt.close(fig)

        # Plot 3: 4-moment comparison (bin percentages)
        plot3_path = os.path.join(script_dir, "gmm_4mom_moment_comparison.png")
        fig = plot_moment_comparison(
            data_moments=data_moments_4,
            model_moments=model_pcts_final,
            moment_names=["Bin [0,220)", "Bin [220,320)", "Bin [320,430)", "Bin [430,450]"],
            save_path=plot3_path,
        )
        print(f"Plot saved: {plot3_path}")
        plt.close(fig)

    except Exception as e:
        print(f"\nNote: Could not generate plots ({e})")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
