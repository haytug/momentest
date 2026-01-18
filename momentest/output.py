"""
Output generation for GMM/SMM estimation results.

Provides functions to create tables and figures displaying estimation results,
diagnostics, and inference statistics.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# Tables
# =============================================================================

def table_estimates(
    theta: np.ndarray,
    se: np.ndarray,
    param_names: Optional[List[str]] = None,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> str:
    """
    Create a formatted table of parameter estimates.
    
    Args:
        theta: Estimated parameters
        se: Standard errors
        param_names: Parameter names (default: θ[0], θ[1], ...)
        ci_lower: Lower CI bounds (optional)
        ci_upper: Upper CI bounds (optional)
        alpha: Significance level for CI
    
    Returns:
        Formatted string table
    """
    p = len(theta)
    
    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(p)]
    
    # Header
    lines = []
    lines.append("=" * 70)
    lines.append("Parameter Estimates")
    lines.append("=" * 70)
    
    if ci_lower is not None and ci_upper is not None:
        lines.append(f"{'Parameter':<15} {'Estimate':>12} {'Std.Err':>12} "
                     f"{'CI {:.0%}'.format(1-alpha):>24}")
        lines.append("-" * 70)
        for i in range(p):
            se_str = f"{se[i]:.4f}" if np.isfinite(se[i]) else "N/A"
            ci_str = f"[{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]" if np.isfinite(ci_lower[i]) else "N/A"
            lines.append(f"{param_names[i]:<15} {theta[i]:>12.4f} {se_str:>12} {ci_str:>24}")
    else:
        lines.append(f"{'Parameter':<15} {'Estimate':>12} {'Std.Err':>12}")
        lines.append("-" * 70)
        for i in range(p):
            se_str = f"{se[i]:.4f}" if np.isfinite(se[i]) else "N/A"
            lines.append(f"{param_names[i]:<15} {theta[i]:>12.4f} {se_str:>12}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def table_moments(
    data_moments: np.ndarray,
    model_moments: np.ndarray,
    moment_names: Optional[List[str]] = None,
    normalize: bool = False
) -> str:
    """
    Create a formatted table comparing data and model moments.
    
    Args:
        data_moments: Target moments from data
        model_moments: Fitted moments from model
        moment_names: Moment names (default: m[0], m[1], ...)
        normalize: Whether to show normalized differences
    
    Returns:
        Formatted string table
    """
    k = len(data_moments)
    
    if moment_names is None:
        moment_names = [f"m[{i}]" for i in range(k)]
    
    # Compute differences
    diff = model_moments - data_moments
    if normalize:
        pct_diff = 100 * diff / np.where(np.abs(data_moments) > 1e-10, data_moments, 1)
    
    lines = []
    lines.append("=" * 70)
    lines.append("Moment Comparison")
    lines.append("=" * 70)
    
    if normalize:
        lines.append(f"{'Moment':<15} {'Data':>12} {'Model':>12} {'Diff':>12} {'%Diff':>12}")
    else:
        lines.append(f"{'Moment':<15} {'Data':>12} {'Model':>12} {'Diff':>12}")
    lines.append("-" * 70)
    
    for i in range(k):
        if normalize:
            lines.append(f"{moment_names[i]:<15} {data_moments[i]:>12.4f} "
                        f"{model_moments[i]:>12.4f} {diff[i]:>12.4f} {pct_diff[i]:>11.2f}%")
        else:
            lines.append(f"{moment_names[i]:<15} {data_moments[i]:>12.4f} "
                        f"{model_moments[i]:>12.4f} {diff[i]:>12.4f}")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def table_bootstrap(
    bootstrap_estimates: np.ndarray,
    theta_hat: np.ndarray,
    param_names: Optional[List[str]] = None,
    alpha: float = 0.05
) -> str:
    """
    Create a formatted table of bootstrap results.
    
    Args:
        bootstrap_estimates: Bootstrap estimates of shape (n_boot, p)
        theta_hat: Point estimate
        param_names: Parameter names
        alpha: Significance level for CI
    
    Returns:
        Formatted string table
    """
    n_boot, p = bootstrap_estimates.shape
    
    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(p)]
    
    # Compute statistics
    valid_mask = ~np.any(np.isnan(bootstrap_estimates), axis=1)
    valid_estimates = bootstrap_estimates[valid_mask]
    n_valid = np.sum(valid_mask)
    
    se = np.std(valid_estimates, axis=0, ddof=1)
    ci_lower = np.percentile(valid_estimates, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(valid_estimates, 100 * (1 - alpha / 2), axis=0)
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"Bootstrap Results (n_boot={n_boot}, n_valid={n_valid})")
    lines.append("=" * 80)
    lines.append(f"{'Parameter':<12} {'Estimate':>10} {'Boot.SE':>10} "
                 f"{'CI {:.0%}'.format(1-alpha):>28} {'Bias':>10}")
    lines.append("-" * 80)
    
    boot_mean = np.mean(valid_estimates, axis=0)
    bias = boot_mean - theta_hat
    
    for i in range(p):
        lines.append(f"{param_names[i]:<12} {theta_hat[i]:>10.4f} {se[i]:>10.4f} "
                    f"[{ci_lower[i]:>10.4f}, {ci_upper[i]:>10.4f}] {bias[i]:>10.4f}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


# =============================================================================
# Figures
# =============================================================================

def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def plot_objective_landscape(
    engine,
    theta_hat: np.ndarray,
    data_moments: np.ndarray,
    W: np.ndarray,
    param_indices: Tuple[int, int] = (0, 1),
    param_names: Optional[List[str]] = None,
    n_points: int = 50,
    scale: float = 0.3,
    plot_type: str = "both",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
    cmap: str = "viridis"
):
    """
    Plot the objective function landscape as 3D surface and/or contour plot.
    
    Creates a visualization of the objective function over a 2D slice of the
    parameter space, holding other parameters fixed at their estimated values.
    This helps diagnose optimization issues and understand the objective surface.
    
    Args:
        engine: SMMEngine or GMMEngine instance
        theta_hat: Estimated parameters (used as center and for fixed params)
        data_moments: Target moments from data
        W: Weighting matrix
        param_indices: Tuple of two parameter indices to vary (default: (0, 1))
        param_names: Parameter names (default: θ[0], θ[1], ...)
        n_points: Number of grid points per dimension (default: 50)
        scale: Fraction of parameter value to vary around estimate (default: 0.3)
        plot_type: "surface", "contour", or "both" (default: "both")
        figsize: Figure size (default: (14, 5))
        save_path: Path to save figure (optional)
        cmap: Colormap for the plot (default: "viridis")
    
    Returns:
        matplotlib Figure object
    
    Requirements: 13.1
    
    Example:
        >>> fig = plot_objective_landscape(
        ...     engine, theta_hat, data_moments, W,
        ...     param_indices=(0, 1),
        ...     plot_type="both"
        ... )
    """
    _check_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D
    
    p = len(theta_hat)
    i, j = param_indices
    
    if i >= p or j >= p:
        raise ValueError(f"param_indices ({i}, {j}) out of range for p={p}")
    if i == j:
        raise ValueError("param_indices must be different")
    
    if param_names is None:
        param_names = [f"θ[{k}]" for k in range(p)]
    
    # Create grid around theta_hat
    center_i = theta_hat[i]
    center_j = theta_hat[j]
    width_i = max(abs(center_i) * scale, 0.1)
    width_j = max(abs(center_j) * scale, 0.1)
    
    grid_i = np.linspace(center_i - width_i, center_i + width_i, n_points)
    grid_j = np.linspace(center_j - width_j, center_j + width_j, n_points)
    
    # Compute objective on grid
    Z = np.zeros((n_points, n_points))
    for ii, val_i in enumerate(grid_i):
        for jj, val_j in enumerate(grid_j):
            theta_test = theta_hat.copy()
            theta_test[i] = val_i
            theta_test[j] = val_j
            m_bar, _ = engine.moments(theta_test)
            g = m_bar - data_moments
            Z[jj, ii] = float(g @ W @ g)
    
    X, Y = np.meshgrid(grid_i, grid_j)
    
    # Create figure based on plot_type
    if plot_type == "both":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                        subplot_kw={'projection': None})
        # Replace first subplot with 3D
        ax1.remove()
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        # 3D surface plot
        surf = ax1.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8,
                                 linewidth=0, antialiased=True)
        ax1.scatter([theta_hat[i]], [theta_hat[j]], 
                    [Z[n_points//2, n_points//2]], 
                    color='red', s=100, marker='*', label='Estimate')
        ax1.set_xlabel(param_names[i])
        ax1.set_ylabel(param_names[j])
        ax1.set_zlabel('Objective')
        ax1.set_title('Objective Surface')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
        
        # Contour plot
        contour = ax2.contourf(X, Y, Z, levels=30, cmap=cmap)
        ax2.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)
        ax2.scatter([theta_hat[i]], [theta_hat[j]], color='red', s=100, 
                    marker='*', label='Estimate', zorder=5)
        ax2.set_xlabel(param_names[i])
        ax2.set_ylabel(param_names[j])
        ax2.set_title('Objective Contours')
        ax2.legend()
        fig.colorbar(contour, ax=ax2)
        
    elif plot_type == "surface":
        fig = plt.figure(figsize=(figsize[0]//2, figsize[1]))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8,
                                linewidth=0, antialiased=True)
        ax.scatter([theta_hat[i]], [theta_hat[j]], 
                   [Z[n_points//2, n_points//2]], 
                   color='red', s=100, marker='*', label='Estimate')
        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        ax.set_zlabel('Objective')
        ax.set_title('Objective Surface')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        
    else:  # contour
        fig, ax = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
        contour = ax.contourf(X, Y, Z, levels=30, cmap=cmap)
        ax.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)
        ax.scatter([theta_hat[i]], [theta_hat[j]], color='red', s=100, 
                   marker='*', label='Estimate', zorder=5)
        ax.set_xlabel(param_names[i])
        ax.set_ylabel(param_names[j])
        ax.set_title('Objective Contours')
        ax.legend()
        fig.colorbar(contour, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_moment_contributions(
    engine,
    theta_hat: np.ndarray,
    data_moments: np.ndarray,
    W: np.ndarray,
    param_index: int = 0,
    param_names: Optional[List[str]] = None,
    moment_names: Optional[List[str]] = None,
    n_points: int = 50,
    scale: float = 0.3,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot each moment's contribution to the objective function.
    
    Shows how each individual moment condition contributes to the total
    objective as a parameter varies. This helps identify which moments
    are most informative for identifying each parameter.
    
    The contribution of moment m_i is: W[i,i] * (m_bar[i] - data_moments[i])^2
    plus cross-terms from off-diagonal W elements.
    
    Args:
        engine: SMMEngine or GMMEngine instance
        theta_hat: Estimated parameters
        data_moments: Target moments from data
        W: Weighting matrix
        param_index: Index of parameter to vary (default: 0)
        param_names: Parameter names (default: θ[0], θ[1], ...)
        moment_names: Moment names (default: m[0], m[1], ...)
        n_points: Number of grid points (default: 50)
        scale: Fraction of parameter value to vary (default: 0.3)
        figsize: Figure size (default: (12, 8))
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure object
    
    Requirements: 13.2
    
    Example:
        >>> fig = plot_moment_contributions(
        ...     engine, theta_hat, data_moments, W,
        ...     param_index=0
        ... )
    """
    _check_matplotlib()
    
    p = len(theta_hat)
    k = len(data_moments)
    
    if param_index >= p:
        raise ValueError(f"param_index {param_index} out of range for p={p}")
    
    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(p)]
    if moment_names is None:
        moment_names = [f"m[{i}]" for i in range(k)]
    
    # Create grid around theta_hat[param_index]
    center = theta_hat[param_index]
    width = max(abs(center) * scale, 0.1)
    grid = np.linspace(center - width, center + width, n_points)
    
    # Compute moment contributions at each point
    contributions = np.zeros((n_points, k))
    total_obj = np.zeros(n_points)
    
    for idx, val in enumerate(grid):
        theta_test = theta_hat.copy()
        theta_test[param_index] = val
        m_bar, _ = engine.moments(theta_test)
        g = m_bar - data_moments
        
        # Total objective
        total_obj[idx] = float(g @ W @ g)
        
        # Individual moment contributions (diagonal terms)
        # For a more accurate decomposition, we compute g_i * (W @ g)_i
        Wg = W @ g
        for m in range(k):
            contributions[idx, m] = g[m] * Wg[m]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    
    # Stacked area plot of contributions
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    
    # Plot individual contributions
    for m in range(k):
        ax1.plot(grid, contributions[:, m], label=moment_names[m], 
                 color=colors[m], linewidth=1.5)
    
    ax1.axvline(theta_hat[param_index], color='red', linestyle='--', 
                linewidth=1.5, label='Estimate')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.set_xlabel(param_names[param_index])
    ax1.set_ylabel('Moment Contribution')
    ax1.set_title(f'Individual Moment Contributions vs {param_names[param_index]}')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Bar chart of contributions at estimate
    m_bar_hat, _ = engine.moments(theta_hat)
    g_hat = m_bar_hat - data_moments
    Wg_hat = W @ g_hat
    contrib_at_estimate = g_hat * Wg_hat
    
    bars = ax2.bar(moment_names, contrib_at_estimate, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Moment')
    ax2.set_ylabel('Contribution at Estimate')
    ax2.set_title('Moment Contributions at θ̂')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-labels if many moments
    if k > 5:
        ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_identification(
    engine,
    theta_hat: np.ndarray,
    param_names: Optional[List[str]] = None,
    moment_names: Optional[List[str]] = None,
    eps: float = 1e-6,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    cmap: str = "RdBu_r"
):
    """
    Visualize moment-parameter relationships through the Jacobian matrix.
    
    Shows which moments are most sensitive to which parameters, helping
    diagnose identification issues. A moment that doesn't respond to any
    parameter is uninformative; a parameter that doesn't affect any moment
    is unidentified.
    
    The visualization shows:
    1. Heatmap of the Jacobian matrix D = ∂m/∂θ
    2. Normalized sensitivity (each column scaled to unit norm)
    
    Args:
        engine: SMMEngine or GMMEngine instance with moments_jac method
        theta_hat: Estimated parameters
        param_names: Parameter names (default: θ[0], θ[1], ...)
        moment_names: Moment names (default: m[0], m[1], ...)
        eps: Step size for numerical differentiation (default: 1e-6)
        figsize: Figure size (default: (10, 8))
        save_path: Path to save figure (optional)
        cmap: Colormap for heatmap (default: "RdBu_r")
    
    Returns:
        matplotlib Figure object
    
    Requirements: 13.3
    
    Example:
        >>> fig = plot_identification(engine, theta_hat)
    """
    _check_matplotlib()
    
    # Get Jacobian matrix
    _, _, D = engine.moments_jac(theta_hat, eps=eps)
    k, p = D.shape
    
    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(p)]
    if moment_names is None:
        moment_names = [f"m[{i}]" for i in range(k)]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Raw Jacobian heatmap
    vmax = np.max(np.abs(D))
    im1 = ax1.imshow(D, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
    ax1.set_xticks(range(p))
    ax1.set_xticklabels(param_names, rotation=45, ha='right')
    ax1.set_yticks(range(k))
    ax1.set_yticklabels(moment_names)
    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Moments')
    ax1.set_title('Jacobian Matrix D = ∂m/∂θ')
    fig.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Add text annotations for values
    for i in range(k):
        for j in range(p):
            text_color = 'white' if abs(D[i, j]) > vmax * 0.5 else 'black'
            ax1.text(j, i, f'{D[i, j]:.2f}', ha='center', va='center',
                     color=text_color, fontsize=8)
    
    # Normalized sensitivity (column-wise normalization)
    col_norms = np.linalg.norm(D, axis=0, keepdims=True)
    col_norms = np.where(col_norms > 1e-10, col_norms, 1)  # Avoid division by zero
    D_normalized = D / col_norms
    
    im2 = ax2.imshow(np.abs(D_normalized), cmap='YlOrRd', aspect='auto', 
                      vmin=0, vmax=1)
    ax2.set_xticks(range(p))
    ax2.set_xticklabels(param_names, rotation=45, ha='right')
    ax2.set_yticks(range(k))
    ax2.set_yticklabels(moment_names)
    ax2.set_xlabel('Parameters')
    ax2.set_ylabel('Moments')
    ax2.set_title('Normalized Sensitivity |D|/||D||')
    fig.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Add text annotations
    for i in range(k):
        for j in range(p):
            val = abs(D_normalized[i, j])
            text_color = 'white' if val > 0.5 else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                     color=text_color, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_marginal_objective(
    engine,
    theta_hat: np.ndarray,
    data_moments: np.ndarray,
    W: np.ndarray,
    param_names: Optional[List[str]] = None,
    n_points: int = 50,
    scale: float = 0.2,
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
):
    """
    Plot marginal objective function around the minimizer.
    
    For each parameter, varies that parameter while holding others fixed
    at their estimated values, showing how the objective changes.
    
    Args:
        engine: SMMEngine or GMMEngine instance
        theta_hat: Estimated parameters
        data_moments: Target moments
        W: Weighting matrix
        param_names: Parameter names
        n_points: Number of points per parameter
        scale: Fraction of parameter value to vary (default: 20%)
        figsize: Figure size (default: auto)
        save_path: Path to save figure (optional)
    
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    p = len(theta_hat)
    
    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(p)]
    
    if figsize is None:
        figsize = (4 * min(p, 3), 3 * ((p + 2) // 3))
    
    fig, axes = plt.subplots(
        nrows=(p + 2) // 3, 
        ncols=min(p, 3), 
        figsize=figsize,
        squeeze=False
    )
    axes = axes.flatten()
    
    for i in range(p):
        ax = axes[i]
        
        # Create grid around theta_hat[i]
        center = theta_hat[i]
        width = max(abs(center) * scale, 0.1)
        grid = np.linspace(center - width, center + width, n_points)
        
        # Compute objective at each point
        obj_vals = []
        for val in grid:
            theta_test = theta_hat.copy()
            theta_test[i] = val
            m_bar, _ = engine.moments(theta_test)
            g = m_bar - data_moments
            obj = float(g @ W @ g)
            obj_vals.append(obj)
        
        # Plot
        ax.plot(grid, obj_vals, 'b-', linewidth=1.5)
        ax.axvline(theta_hat[i], color='r', linestyle='--', linewidth=1, label='Estimate')
        ax.scatter([theta_hat[i]], [min(obj_vals)], color='r', s=50, zorder=5)
        ax.set_xlabel(param_names[i])
        ax.set_ylabel('Objective')
        ax.set_title(f'Marginal: {param_names[i]}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(p, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_moment_comparison(
    data_moments: np.ndarray,
    model_moments: np.ndarray,
    moment_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot comparison of data vs model moments.
    
    Args:
        data_moments: Target moments from data
        model_moments: Fitted moments from model
        moment_names: Moment names
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    k = len(data_moments)
    
    if moment_names is None:
        moment_names = [f"m[{i}]" for i in range(k)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart comparison
    x = np.arange(k)
    width = 0.35
    
    ax1.bar(x - width/2, data_moments, width, label='Data', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, model_moments, width, label='Model', color='coral', alpha=0.8)
    ax1.set_xlabel('Moment')
    ax1.set_ylabel('Value')
    ax1.set_title('Data vs Model Moments')
    ax1.set_xticks(x)
    ax1.set_xticklabels(moment_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Scatter plot (45-degree line)
    all_vals = np.concatenate([data_moments, model_moments])
    min_val, max_val = np.min(all_vals), np.max(all_vals)
    margin = (max_val - min_val) * 0.1
    
    ax2.scatter(data_moments, model_moments, s=80, c='steelblue', alpha=0.7, edgecolors='black')
    ax2.plot([min_val - margin, max_val + margin], 
             [min_val - margin, max_val + margin], 
             'r--', linewidth=1, label='45° line')
    ax2.set_xlabel('Data Moments')
    ax2.set_ylabel('Model Moments')
    ax2.set_title('Moment Fit')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_bootstrap_distribution(
    bootstrap_estimates: np.ndarray,
    theta_hat: np.ndarray,
    param_names: Optional[List[str]] = None,
    ci_alpha: float = 0.05,
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None,
    show_stats: bool = True
):
    """
    Plot bootstrap distribution of parameter estimates with CI markers.
    
    Creates histograms showing the distribution of bootstrap estimates
    for each parameter, with vertical lines marking the point estimate
    and confidence interval bounds.
    
    Args:
        bootstrap_estimates: Bootstrap estimates of shape (n_boot, p)
        theta_hat: Point estimate
        param_names: Parameter names (default: θ[0], θ[1], ...)
        ci_alpha: Significance level for CI bands (default: 0.05 for 95% CI)
        figsize: Figure size (default: auto)
        save_path: Path to save figure (optional)
        show_stats: Whether to show SE and bias statistics (default: True)
    
    Returns:
        matplotlib Figure object
    
    Requirements: 13.5
    
    Example:
        >>> result = bootstrap(setup, data_moments, bounds, n_boot=200)
        >>> fig = plot_bootstrap_distribution(
        ...     result.bootstrap_estimates, result.theta_hat
        ... )
    """
    _check_matplotlib()
    
    n_boot, p = bootstrap_estimates.shape
    
    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(p)]
    
    if figsize is None:
        figsize = (4 * min(p, 3), 4 * ((p + 2) // 3))
    
    fig, axes = plt.subplots(
        nrows=(p + 2) // 3,
        ncols=min(p, 3),
        figsize=figsize,
        squeeze=False
    )
    axes = axes.flatten()
    
    # Filter valid estimates
    valid_mask = ~np.any(np.isnan(bootstrap_estimates), axis=1)
    valid_estimates = bootstrap_estimates[valid_mask]
    n_valid = np.sum(valid_mask)
    
    for i in range(p):
        ax = axes[i]
        
        estimates_i = valid_estimates[:, i]
        
        # Histogram
        ax.hist(estimates_i, bins=30, density=True, alpha=0.7, 
                color='steelblue', edgecolor='black', linewidth=0.5)
        
        # Point estimate
        ax.axvline(theta_hat[i], color='red', linestyle='-', linewidth=2, 
                   label=f'Estimate: {theta_hat[i]:.3f}')
        
        # CI bounds
        ci_lower = np.percentile(estimates_i, 100 * ci_alpha / 2)
        ci_upper = np.percentile(estimates_i, 100 * (1 - ci_alpha / 2))
        ax.axvline(ci_lower, color='green', linestyle='--', linewidth=1.5)
        ax.axvline(ci_upper, color='green', linestyle='--', linewidth=1.5,
                   label=f'{100*(1-ci_alpha):.0f}% CI')
        
        # Shade CI region
        ylim = ax.get_ylim()
        ax.fill_betweenx(ylim, ci_lower, ci_upper, alpha=0.1, color='green')
        ax.set_ylim(ylim)
        
        ax.set_xlabel(param_names[i])
        ax.set_ylabel('Density')
        ax.set_title(f'Bootstrap: {param_names[i]}')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add statistics annotation
        if show_stats:
            se = np.std(estimates_i, ddof=1)
            bias = np.mean(estimates_i) - theta_hat[i]
            stats_text = f'SE: {se:.4f}\nBias: {bias:.4f}'
            ax.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                        ha='left', va='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Hide unused subplots
    for i in range(p, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title with sample info
    fig.suptitle(f'Bootstrap Distribution (n_boot={n_boot}, n_valid={n_valid})', 
                 fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_optimization_history(
    history: List[Tuple[np.ndarray, float]],
    param_names: Optional[List[str]] = None,
    n_global: int = 0,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Plot optimization history (sanity check).
    
    Shows how objective and parameters evolved during optimization.
    
    Args:
        history: List of (theta, objective) tuples
        param_names: Parameter names
        n_global: Number of global search points (to mark transition)
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if len(history) == 0:
        raise ValueError("History is empty")
    
    objectives = [obj for _, obj in history]
    thetas = np.array([theta for theta, _ in history])
    p = thetas.shape[1]
    
    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(p)]
    
    fig, axes = plt.subplots(p + 1, 1, figsize=figsize, sharex=True)
    
    x = np.arange(len(history))
    
    # Objective plot
    axes[0].semilogy(x, objectives, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Objective (log)')
    axes[0].set_title('Optimization History')
    axes[0].grid(True, alpha=0.3)
    
    if n_global > 0 and n_global < len(history):
        axes[0].axvline(n_global, color='red', linestyle='--', 
                        label='Global→Local', alpha=0.7)
        axes[0].legend()
    
    # Parameter plots
    for i in range(p):
        axes[i + 1].plot(x, thetas[:, i], 'b-', linewidth=0.5, alpha=0.7)
        axes[i + 1].axhline(thetas[-1, i], color='red', linestyle='--', 
                            linewidth=1, alpha=0.7)
        axes[i + 1].set_ylabel(param_names[i])
        axes[i + 1].grid(True, alpha=0.3)
        
        if n_global > 0 and n_global < len(history):
            axes[i + 1].axvline(n_global, color='red', linestyle='--', alpha=0.7)
    
    axes[-1].set_xlabel('Iteration')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_convergence(
    engine,
    history: List[Tuple[np.ndarray, float]],
    data_moments: np.ndarray,
    W: np.ndarray,
    param_indices: Tuple[int, int] = (0, 1),
    param_names: Optional[List[str]] = None,
    n_points: int = 50,
    scale: float = 0.3,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    cmap: str = "viridis"
):
    """
    Plot optimization path overlaid on the objective surface.
    
    Shows the trajectory of the optimizer through parameter space,
    overlaid on a contour plot of the objective function. This helps
    visualize how the optimization progressed and whether it found
    the global minimum.
    
    Args:
        engine: SMMEngine or GMMEngine instance
        history: List of (theta, objective) tuples from optimization
        data_moments: Target moments from data
        W: Weighting matrix
        param_indices: Tuple of two parameter indices to plot (default: (0, 1))
        param_names: Parameter names (default: θ[0], θ[1], ...)
        n_points: Number of grid points per dimension (default: 50)
        scale: Fraction of parameter range to show (default: 0.3)
        figsize: Figure size (default: (12, 5))
        save_path: Path to save figure (optional)
        cmap: Colormap for contours (default: "viridis")
    
    Returns:
        matplotlib Figure object
    
    Requirements: 13.4
    
    Example:
        >>> result = estimate(setup, data_moments, bounds)
        >>> fig = plot_convergence(engine, result.history, data_moments, W)
    """
    _check_matplotlib()
    
    if len(history) == 0:
        raise ValueError("History is empty")
    
    thetas = np.array([theta for theta, _ in history])
    objectives = np.array([obj for _, obj in history])
    p = thetas.shape[1]
    i, j = param_indices
    
    if i >= p or j >= p:
        raise ValueError(f"param_indices ({i}, {j}) out of range for p={p}")
    if i == j:
        raise ValueError("param_indices must be different")
    
    if param_names is None:
        param_names = [f"θ[{k}]" for k in range(p)]
    
    # Get final estimate
    theta_hat = thetas[-1]
    
    # Determine grid bounds from history
    theta_i_vals = thetas[:, i]
    theta_j_vals = thetas[:, j]
    
    center_i = theta_hat[i]
    center_j = theta_hat[j]
    
    # Use history range or scale around estimate, whichever is larger
    range_i = max(theta_i_vals.max() - theta_i_vals.min(), 
                  abs(center_i) * scale * 2, 0.2)
    range_j = max(theta_j_vals.max() - theta_j_vals.min(), 
                  abs(center_j) * scale * 2, 0.2)
    
    grid_i = np.linspace(center_i - range_i/2, center_i + range_i/2, n_points)
    grid_j = np.linspace(center_j - range_j/2, center_j + range_j/2, n_points)
    
    # Compute objective on grid
    Z = np.zeros((n_points, n_points))
    for ii, val_i in enumerate(grid_i):
        for jj, val_j in enumerate(grid_j):
            theta_test = theta_hat.copy()
            theta_test[i] = val_i
            theta_test[j] = val_j
            m_bar, _ = engine.moments(theta_test)
            g = m_bar - data_moments
            Z[jj, ii] = float(g @ W @ g)
    
    X, Y = np.meshgrid(grid_i, grid_j)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Contour plot with optimization path
    contour = ax1.contourf(X, Y, Z, levels=30, cmap=cmap, alpha=0.8)
    ax1.contour(X, Y, Z, levels=15, colors='white', alpha=0.3, linewidths=0.5)
    
    # Plot optimization path
    ax1.plot(theta_i_vals, theta_j_vals, 'w-', linewidth=1, alpha=0.7)
    ax1.scatter(theta_i_vals[:-1], theta_j_vals[:-1], c=np.arange(len(theta_i_vals)-1),
                cmap='cool', s=20, alpha=0.7, zorder=4)
    
    # Mark start and end
    ax1.scatter([theta_i_vals[0]], [theta_j_vals[0]], color='lime', s=100, 
                marker='o', label='Start', zorder=5, edgecolors='black')
    ax1.scatter([theta_hat[i]], [theta_hat[j]], color='red', s=150, 
                marker='*', label='Final', zorder=5, edgecolors='black')
    
    ax1.set_xlabel(param_names[i])
    ax1.set_ylabel(param_names[j])
    ax1.set_title('Optimization Path on Objective Surface')
    ax1.legend(loc='upper right')
    fig.colorbar(contour, ax=ax1)
    
    # Objective value over iterations
    ax2.semilogy(objectives, 'b-', linewidth=1.5, alpha=0.8)
    ax2.scatter(range(len(objectives)), objectives, c=np.arange(len(objectives)),
                cmap='cool', s=20, alpha=0.7)
    ax2.scatter([0], [objectives[0]], color='lime', s=100, marker='o', 
                label='Start', zorder=5, edgecolors='black')
    ax2.scatter([len(objectives)-1], [objectives[-1]], color='red', s=100, 
                marker='*', label='Final', zorder=5, edgecolors='black')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Objective (log scale)')
    ax2.set_title('Convergence Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_sanity(
    trial_results: List[Tuple[np.ndarray, float]],
    param_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = None,
    save_path: Optional[str] = None
):
    """
    Plot sanity check: parameter estimates across multiple optimization trials.
    
    This diagnostic plot shows whether different starting points converge to
    the same solution. If estimates cluster tightly, the optimization is robust.
    If they're scattered, there may be multiple local minima or identification issues.
    
    Args:
        trial_results: List of (theta, objective) tuples from multiple trials
        param_names: Parameter names
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()
    
    if len(trial_results) == 0:
        raise ValueError("trial_results is empty")
    
    thetas = np.array([theta for theta, _ in trial_results])
    objectives = np.array([obj for _, obj in trial_results])
    n_trials, p = thetas.shape
    
    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(p)]
    
    if figsize is None:
        figsize = (12, 3 * ((p + 1 + 2) // 3))
    
    n_plots = p + 1  # parameters + objective
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    trial_idx = np.arange(n_trials)
    
    # Plot each parameter
    for i in range(p):
        ax = axes[i]
        ax.scatter(trial_idx, thetas[:, i], c='steelblue', alpha=0.7, s=50, edgecolors='black')
        ax.axhline(np.median(thetas[:, i]), color='red', linestyle='--', 
                   linewidth=1.5, label=f'Median: {np.median(thetas[:, i]):.3f}')
        ax.set_xlabel('Trial')
        ax.set_ylabel(param_names[i])
        ax.set_title(f'{param_names[i]} across trials')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add std annotation
        std = np.std(thetas[:, i])
        ax.annotate(f'Std: {std:.4f}', xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot objective values
    ax = axes[p]
    ax.scatter(trial_idx, objectives, c='coral', alpha=0.7, s=50, edgecolors='black')
    ax.axhline(np.min(objectives), color='green', linestyle='--', 
               linewidth=1.5, label=f'Min: {np.min(objectives):.2e}')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Objective')
    ax.set_title('Objective across trials')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# Summary output
# =============================================================================

def summary(
    theta: np.ndarray,
    se: np.ndarray,
    objective: float,
    data_moments: np.ndarray,
    model_moments: np.ndarray,
    k: int,
    p: int,
    n: int,
    converged: bool,
    param_names: Optional[List[str]] = None,
    moment_names: Optional[List[str]] = None,
    method: str = "SMM"
) -> str:
    """
    Generate comprehensive summary of estimation results.
    
    Args:
        theta: Estimated parameters
        se: Standard errors
        objective: Final objective value
        data_moments: Target moments
        model_moments: Fitted moments
        k: Number of moments
        p: Number of parameters
        n: Sample size
        converged: Whether optimization converged
        param_names: Parameter names
        moment_names: Moment names
        method: "SMM" or "GMM"
    
    Returns:
        Formatted summary string
    """
    from .inference import j_test, confidence_interval
    
    if param_names is None:
        param_names = [f"θ[{i}]" for i in range(p)]
    
    ci_lower, ci_upper = confidence_interval(theta, se)
    
    lines = []
    lines.append("=" * 70)
    lines.append(f"{method} Estimation Results")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Converged: {converged}")
    lines.append(f"Objective: {objective:.6e}")
    lines.append(f"Sample size: {n}")
    lines.append(f"Moments (k): {k}")
    lines.append(f"Parameters (p): {p}")
    lines.append("")
    
    # Parameter estimates
    lines.append(table_estimates(theta, se, param_names, ci_lower, ci_upper))
    lines.append("")
    
    # Moment comparison
    lines.append(table_moments(data_moments, model_moments, moment_names))
    lines.append("")
    
    # J-test (if overidentified)
    if k > p:
        try:
            j_result = j_test(objective, n, k, p)
            lines.append(str(j_result))
        except Exception as e:
            lines.append(f"J-test: Could not compute ({e})")
    else:
        lines.append("J-test: Not applicable (exactly identified)")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)
