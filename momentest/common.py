"""
Common utilities shared between SMM and GMM modules.
"""

from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np


def numerical_jacobian(
    moments_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    theta: np.ndarray,
    k: int,
    p: int,
    eps: float = 1e-6
) -> np.ndarray:
    """
    Compute numerical Jacobian of moment function via central differences.
    
    D[i,j] = (m_bar(θ + εeⱼ) - m_bar(θ - εeⱼ)) / (2ε)
    
    Args:
        moments_func: Function that takes theta and returns (m_bar, S)
        theta: Parameter vector of length p
        k: Number of moment conditions
        p: Number of parameters
        eps: Step size for numerical differentiation (default: 1e-6)
    
    Returns:
        Jacobian matrix D of shape (k, p)
    """
    D = np.zeros((k, p))
    
    for j in range(p):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[j] += eps
        theta_minus[j] -= eps
        
        m_plus, _ = moments_func(theta_plus)
        m_minus, _ = moments_func(theta_minus)
        
        D[:, j] = (m_plus - m_minus) / (2 * eps)
    
    return D


def format_estimate_result(
    name: str,
    theta: np.ndarray,
    se: np.ndarray,
    objective: float,
    converged: bool,
    objective_name: str = "Objective"
) -> str:
    """
    Format estimation result for display.
    
    Args:
        name: Result class name (e.g., "SMMResult")
        theta: Estimated parameters
        se: Standard errors
        objective: Objective function value
        converged: Whether optimization converged
        objective_name: Label for objective (e.g., "J-statistic")
    
    Returns:
        Formatted string representation
    """
    p = len(theta)
    lines = [f"{name}:"]
    for i in range(p):
        se_str = f"{se[i]:.4f}" if np.isfinite(se[i]) else "N/A"
        lines.append(f"  θ[{i}] = {theta[i]:.4f} (SE: {se_str})")
    lines.append(f"  {objective_name}: {objective:.6e}")
    lines.append(f"  Converged: {converged}")
    return "\n".join(lines)
