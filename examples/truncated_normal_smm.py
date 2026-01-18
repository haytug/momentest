"""
Truncated Normal SMM Estimation Example

This example replicates the SMM estimation from the OpenSourceEcon tutorial:
https://opensourceecon.github.io/CompMethods/struct_est/SMM.html

We fit a truncated normal distribution to Econ 381 test scores using
Simulated Method of Moments (SMM).

The data are test scores bounded between 0 and 450.

This script compares:
1. Original webpage method (direct scipy.optimize) - complex, manual
2. Our momentest high-level API - simple, just a few lines!

Both methods use the SAME:
- Data
- Random draws (shocks)
- Simulation function (sim_func)
- Moment function (moment_func)

This ensures an apples-to-apples comparison.
"""

import numpy as np
import scipy.stats as sts
import scipy.optimize as opt
import scipy.linalg as lin

# Import our simple high-level API
import sys
sys.path.insert(0, '..')
from momentest import (
    smm_estimate,
    j_test,
    table_estimates,
    table_moments,
    confidence_interval,
    summary,
    plot_moment_comparison,
    load_econ381,
)


# =============================================================================
# SHARED: Helper function for truncated normal
# =============================================================================

def trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub):
    """
    Draw from truncated normal using inverse CDF method.
    
    Args:
        unif_vals: Uniform(0,1) random values (any shape)
        mu: Mean of underlying normal
        sigma: Std dev of underlying normal
        cut_lb: Lower truncation bound
        cut_ub: Upper truncation bound
    
    Returns:
        Truncated normal draws (same shape as unif_vals)
    """
    sigma = max(sigma, 1e-6)
    cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
    cdf_range = cut_ub_cdf - cut_lb_cdf

    if cdf_range < 1e-10:
        return np.full_like(unif_vals, (cut_lb + cut_ub) / 2)

    unif_scaled = unif_vals * cdf_range + cut_lb_cdf
    unif_scaled = np.clip(unif_scaled, cut_lb_cdf + 1e-10, cut_ub_cdf - 1e-10)
    tnorm_draws = sts.norm.ppf(unif_scaled, loc=mu, scale=sigma)
    return np.clip(tnorm_draws, cut_lb, cut_ub)


def trunc_norm_pdf(xvals, mu, sigma, cut_lb, cut_ub):
    """PDF of truncated normal distribution."""
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


# =============================================================================
# Main Example
# =============================================================================

def main():
    print("=" * 80)
    print("Truncated Normal SMM Estimation - API Comparison")
    print("=" * 80)

    # =========================================================================
    # SETUP: Load data from package and random draws
    # =========================================================================
    S = 300  # Number of simulations
    seed = 25

    dataset = load_econ381()
    data = dataset['data']
    cut_lb, cut_ub = dataset['bounds']
    mle_mu = dataset['mle_params']['mu']
    mle_sigma = dataset['mle_params']['sigma']
    N = dataset['n']

    # Compute data moments (target)
    data_mean = np.mean(data)
    data_var = np.var(data)

    # Pre-draw shocks: (n_sim, n_obs) = (S, N)
    np.random.seed(seed)
    shocks = sts.uniform.rvs(0, 1, size=(S, N))

    print(f"\nData: N={N}, Mean={data_mean:.2f}, Var={data_var:.2f}")
    print(f"MLE Parameters (target): mu={mle_mu}, sigma={mle_sigma}")
    print(f"Simulations: S={S}")

    # =========================================================================
    # DEFINE SHARED FUNCTIONS (used by BOTH methods)
    # =========================================================================
    
    def sim_func(theta, shocks):
        """
        Simulate truncated normal data.
        
        Args:
            theta: (mu, sigma) parameters
            shocks: Uniform(0,1) values of shape (n_sim, n_obs)
        
        Returns:
            Simulated data of shape (n_sim, n_obs)
        """
        mu, sigma = theta
        sigma = max(sigma, 1.0)
        return trunc_norm_draws(shocks, mu, sigma, cut_lb, cut_ub)

    def moment_func(sim_data):
        """
        Compute moments (mean and variance) for each simulation.
        
        Args:
            sim_data: Simulated data of shape (n_sim, n_obs)
        
        Returns:
            Moments of shape (n_sim, 2) - [mean, variance] for each simulation
        """
        sim_means = np.mean(sim_data, axis=1)
        sim_vars = np.var(sim_data, axis=1)
        return np.column_stack([sim_means, sim_vars])

    # =========================================================================
    # METHOD 1: Webpage Method (manual)
    # =========================================================================
    print("\n" + "=" * 80)
    print("METHOD 1: ORIGINAL WEBPAGE METHOD")
    print("(Manual criterion, weighting matrix, two-step procedure)")
    print("=" * 80)

    # --- Webpage method inline (shows the boilerplate) ---
    # Target moments from data
    target_moments = np.array([data_mean, data_var])
    
    def criterion(params, W):
        """SMM criterion function: (m_bar - m_data)'W(m_bar - m_data)."""
        sim_data = sim_func(params, shocks)
        moments = moment_func(sim_data)
        m_bar = moments.mean(axis=0)
        g = (m_bar - target_moments).reshape(-1, 1)
        return float((g.T @ W @ g).item())

    def get_moment_cov(theta):
        """Compute covariance matrix of moments for optimal weighting."""
        sim_data = sim_func(theta, shocks)
        moments = moment_func(sim_data)
        m_bar = moments.mean(axis=0)
        m_centered = moments - m_bar
        return (m_centered.T @ m_centered) / moments.shape[0]

    # Step 1: Identity Weighting
    W_identity = np.eye(2)
    result1 = opt.minimize(
        lambda p: criterion(p, W_identity),
        np.array([300.0, 30.0]),
        method="L-BFGS-B",
        bounds=((1e-10, None), (1e-10, None)),
    )
    webpage_identity = {
        "mu": result1.x[0],
        "sigma": result1.x[1],
        "objective": criterion(result1.x, W_identity),
    }

    # Step 2: Two-Step Optimal Weighting
    S_cov = get_moment_cov(result1.x)
    W_optimal = lin.inv(S_cov)
    result2 = opt.minimize(
        lambda p: criterion(p, W_optimal),
        result1.x,
        method="L-BFGS-B",
        bounds=((1e-10, None), (1e-10, None)),
    )
    webpage_optimal = {
        "mu": result2.x[0],
        "sigma": result2.x[1],
        "objective": criterion(result2.x, W_optimal),
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
    print("(Just pass sim_func, moment_func, shocks to smm_estimate)")
    print("=" * 80)

    # --- momentest method (so simple!) ---
    momentest_identity = smm_estimate(
        sim_func=sim_func,
        moment_func=moment_func,
        data_moments=[data_mean, data_var],  # Actual data moments
        bounds=[(0.0, 1000.0), (1.0, 500.0)],
        n_sim=S,
        shock_dim=N,
        shocks=shocks,
        weighting="identity",
    )

    momentest_optimal = smm_estimate(
        sim_func=sim_func,
        moment_func=moment_func,
        data_moments=[data_mean, data_var],  # Actual data moments
        bounds=[(0.0, 1000.0), (1.0, 500.0)],
        n_sim=S,
        shock_dim=N,
        shocks=shocks,
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

    print("\nBoth methods use the SAME sim_func, moment_func, and shocks.")
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
  - sim_func(theta, shocks) -> simulated data
  - moment_func(sim_data) -> moments
  - shocks array

WEBPAGE METHOD requires (~30 lines):
  - Define criterion() function inside estimate function
  - Define get_moment_cov() function inside estimate function
  - Manual scipy.optimize.minimize() calls
  - Manual two-step procedure

MOMENTEST requires (~15 lines):
  - Just call smm_estimate() twice (identity + optimal)
  - Handles criterion, weighting, two-step automatically
""")

    # =========================================================================
    # INFERENCE AND OUTPUT FEATURES
    # =========================================================================
    print("\n" + "=" * 80)
    print("INFERENCE AND OUTPUT FEATURES")
    print("=" * 80)

    theta = momentest_optimal.theta
    se = momentest_optimal.se

    # Parameter estimates table
    print("\n--- Parameter Estimates ---")
    ci_lower, ci_upper = confidence_interval(theta, se)
    print(table_estimates(
        theta=theta,
        se=se,
        param_names=["mu", "sigma"],
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    ))

    # Compute model moments at estimated parameters
    sim_data_final = sim_func(theta, shocks)
    model_mean = np.mean(sim_data_final)
    model_var = np.var(sim_data_final)

    # Moment comparison table
    print("\n--- Moment Comparison ---")
    print(table_moments(
        data_moments=np.array([data_mean, data_var]),
        model_moments=np.array([model_mean, model_var]),
        moment_names=["Mean", "Variance"],
        normalize=True,
    ))

    # Full summary
    print("\n--- Full Summary ---")
    print(summary(
        theta=theta,
        se=se,
        objective=momentest_optimal.objective,
        data_moments=np.array([data_mean, data_var]),
        model_moments=np.array([model_mean, model_var]),
        k=2,
        p=2,
        n=S,
        converged=momentest_optimal.converged,
        param_names=["mu", "sigma"],
        moment_names=["Mean", "Variance"],
        method="SMM",
    ))

    # J-test example (hypothetical overidentified case)
    print("\n--- J-Test Example ---")
    print("(Hypothetical: if we had k=4 moments and p=2 parameters)")
    j_result = j_test(objective=0.05, n=100, k=4, p=2, alpha=0.05)
    print(j_result)

    # =========================================================================
    # PLOTS
    # =========================================================================
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Plot 1: Data histogram with fitted distributions
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(data, bins=30, density=True, alpha=0.6, edgecolor="black",
                label="Data", color="gray")

        x_vals = np.linspace(cut_lb, cut_ub, 500)

        ax.plot(x_vals, trunc_norm_pdf(x_vals, mle_mu, mle_sigma, cut_lb, cut_ub),
                "g-", linewidth=2.5, label=f"True: μ={mle_mu}, σ={mle_sigma}")

        ax.plot(x_vals, trunc_norm_pdf(x_vals, webpage_optimal["mu"], webpage_optimal["sigma"], cut_lb, cut_ub),
                "r--", linewidth=2, label=f'Webpage: μ={webpage_optimal["mu"]:.1f}, σ={webpage_optimal["sigma"]:.1f}')

        ax.plot(x_vals, trunc_norm_pdf(x_vals, theta[0], theta[1], cut_lb, cut_ub),
                "b:", linewidth=2.5, label=f'momentest: μ={theta[0]:.1f}, σ={theta[1]:.1f}')

        ax.set_xlabel("Test Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title("SMM Estimation: Webpage vs momentest", fontsize=14)
        ax.legend(loc="upper left")
        ax.set_xlim([0, 500])

        plt.tight_layout()
        plot1_path = os.path.join(script_dir, "smm_comparison.png")
        plt.savefig(plot1_path, dpi=150)
        print(f"\nPlot saved: {plot1_path}")
        plt.close()

        # Plot 2: Moment comparison
        plot2_path = os.path.join(script_dir, "smm_moment_comparison.png")
        fig = plot_moment_comparison(
            data_moments=np.array([data_mean, data_var]),
            model_moments=np.array([model_mean, model_var]),
            moment_names=["Mean", "Variance"],
            save_path=plot2_path,
        )
        print(f"Plot saved: {plot2_path}")
        plt.close(fig)

    except Exception as e:
        print(f"\nNote: Could not generate plots ({e})")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
