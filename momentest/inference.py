"""
Statistical inference for GMM/SMM estimation.

Provides J-test for overidentification, asymptotic standard errors,
and related diagnostic tests.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class JTestResult:
    """
    Result of Hansen-Sargan J-test for overidentifying restrictions.
    
    Attributes:
        J_statistic: The J-test statistic (n * g'Wg)
        df: Degrees of freedom (k - p)
        p_value: P-value from chi-squared distribution
        reject: Whether to reject at given significance level
        alpha: Significance level used
    """
    J_statistic: float
    df: int
    p_value: float
    reject: bool
    alpha: float
    
    def __repr__(self):
        status = "REJECT" if self.reject else "FAIL TO REJECT"
        return (
            f"J-Test for Overidentifying Restrictions:\n"
            f"  J-statistic: {self.J_statistic:.4f}\n"
            f"  Degrees of freedom: {self.df}\n"
            f"  P-value: {self.p_value:.4f}\n"
            f"  Decision at α={self.alpha}: {status} H0\n"
            f"  (H0: moment conditions are valid)"
        )


def j_test(
    objective: float,
    n: int,
    k: int,
    p: int,
    alpha: float = 0.05
) -> JTestResult:
    """
    Perform Hansen-Sargan J-test for overidentifying restrictions.
    
    The J-test checks whether the moment conditions are valid when the model
    is overidentified (k > p). Under the null hypothesis that the moment
    conditions are correctly specified, J ~ χ²(k-p).
    
    Args:
        objective: The minimized objective value g'Wg (NOT scaled by n)
        n: Sample size (for SMM: n_sim, for GMM: n_obs)
        k: Number of moment conditions
        p: Number of parameters
        alpha: Significance level (default: 0.05)
    
    Returns:
        JTestResult with J-statistic, degrees of freedom, p-value, and decision
    
    Raises:
        ValueError: If model is exactly identified (k == p) or underidentified (k < p)
    
    Example:
        >>> result = j_test(objective=0.05, n=1000, k=5, p=2)
        >>> print(result)
    """
    df = k - p
    
    if df <= 0:
        raise ValueError(
            f"J-test requires overidentification (k > p). "
            f"Got k={k}, p={p}, df={df}. "
            f"Cannot test exactly identified or underidentified models."
        )
    
    # J-statistic: n * g'Wg
    J = n * objective
    
    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(J, df)
    
    # Decision
    reject = p_value < alpha
    
    return JTestResult(
        J_statistic=J,
        df=df,
        p_value=p_value,
        reject=reject,
        alpha=alpha
    )


def sandwich_covariance(
    D: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
    n: int
) -> np.ndarray:
    """
    Compute sandwich covariance matrix for GMM/SMM estimator.
    
    The asymptotic variance of the GMM estimator is:
    Var(θ̂) = (1/n) * (D'WD)^{-1} D'W S W D (D'WD)^{-1}
    
    where:
    - D is the Jacobian of moments w.r.t. parameters (k x p)
    - W is the weighting matrix (k x k)
    - S is the covariance of moment conditions (k x k)
    - n is the sample size
    
    Args:
        D: Jacobian matrix of shape (k, p)
        W: Weighting matrix of shape (k, k)
        S: Moment covariance matrix of shape (k, k)
        n: Sample size
    
    Returns:
        Covariance matrix of shape (p, p)
    
    Raises:
        ValueError: If matrices have incompatible shapes
        Warning: If bread matrix is ill-conditioned
    """
    k, p = D.shape
    
    if W.shape != (k, k):
        raise ValueError(f"W must have shape ({k}, {k}), got {W.shape}")
    if S.shape != (k, k):
        raise ValueError(f"S must have shape ({k}, {k}), got {S.shape}")
    
    # Bread: (D'WD)^{-1}
    DtWD = D.T @ W @ D
    
    # Check condition number
    cond = np.linalg.cond(DtWD)
    if cond > 1e10:
        import warnings
        warnings.warn(
            f"Bread matrix (D'WD) is ill-conditioned (cond={cond:.2e}). "
            f"Standard errors may be imprecise.",
            RuntimeWarning
        )
    
    try:
        bread = np.linalg.inv(DtWD)
    except np.linalg.LinAlgError:
        return np.full((p, p), np.nan)
    
    # Meat: D'W S W D
    meat = D.T @ W @ S @ W @ D
    
    # Sandwich: bread * meat * bread / n
    sandwich = bread @ meat @ bread / n
    
    return sandwich


def asymptotic_se(
    D: np.ndarray,
    W: np.ndarray,
    S: np.ndarray,
    n: int
) -> np.ndarray:
    """
    Compute asymptotic standard errors for GMM/SMM estimator.
    
    Args:
        D: Jacobian matrix of shape (k, p)
        W: Weighting matrix of shape (k, k)
        S: Moment covariance matrix of shape (k, k)
        n: Sample size
    
    Returns:
        Standard errors of shape (p,)
    """
    cov = sandwich_covariance(D, W, S, n)
    var_diag = np.diag(cov)
    
    # Handle numerical issues
    var_diag = np.where(var_diag >= 0, var_diag, np.nan)
    
    return np.sqrt(var_diag)


def wald_test(
    theta: np.ndarray,
    se: np.ndarray,
    theta_0: Optional[np.ndarray] = None,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Wald tests for individual parameters.
    
    Tests H0: θ_i = θ_0_i for each parameter.
    
    Args:
        theta: Estimated parameters of shape (p,)
        se: Standard errors of shape (p,)
        theta_0: Null hypothesis values (default: zeros)
        alpha: Significance level
    
    Returns:
        Tuple of (t_statistics, p_values, reject) arrays
    """
    p = len(theta)
    
    if theta_0 is None:
        theta_0 = np.zeros(p)
    
    # t-statistics
    t_stats = (theta - theta_0) / se
    
    # Two-sided p-values (using normal approximation)
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    
    # Rejection decisions
    reject = p_values < alpha
    
    return t_stats, p_values, reject


def confidence_interval(
    theta: np.ndarray,
    se: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute asymptotic confidence intervals.
    
    Args:
        theta: Estimated parameters of shape (p,)
        se: Standard errors of shape (p,)
        alpha: Significance level (default: 0.05 for 95% CI)
    
    Returns:
        Tuple of (ci_lower, ci_upper) arrays
    """
    z = stats.norm.ppf(1 - alpha / 2)
    
    ci_lower = theta - z * se
    ci_upper = theta + z * se
    
    return ci_lower, ci_upper
