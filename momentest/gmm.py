"""
GMM (Generalized Method of Moments) engine for momentest.

Provides the GMMEngine class for computing sample moments from data
with analytical moment conditions, as an alternative to the simulation-based
SMM approach.

Requirements: 6.1, 6.2, 6.3
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .common import numerical_jacobian, format_estimate_result


@dataclass
class GMMEstimateResult:
    """
    Result container for GMM estimation.
    
    Attributes:
        theta: Estimated parameters
        se: Standard errors (asymptotic)
        objective: Final objective value (J-statistic)
        converged: Whether optimization converged
        sample_moments: Sample moments at theta
    """
    theta: np.ndarray
    se: np.ndarray
    objective: float
    converged: bool
    sample_moments: np.ndarray
    
    def __repr__(self):
        return format_estimate_result(
            "GMMEstimateResult", self.theta, self.se,
            self.objective, self.converged, "J-statistic"
        )


@dataclass
class GMMResult:
    """
    Result container for GMM moment computation.
    
    Attributes:
        g_bar: Sample mean of moment conditions (k,)
        S: Covariance matrix of moment conditions (k, k)
    """
    g_bar: np.ndarray
    S: np.ndarray


class GMMEngine:
    """
    GMM engine for computing sample moments from data.
    
    Unlike the SMM MomentEngine which simulates moments, GMMEngine computes
    sample moments ḡ(θ) = (1/n) Σᵢ g(zᵢ, θ) from observed data using
    user-provided moment condition functions.
    
    Attributes:
        data: Observed data (array or dictionary of arrays)
        k: Number of moment conditions
        p: Number of parameters
        moment_func: Function g(data, theta) -> (n, k) array of moment conditions
    
    Requirements: 6.1, 6.2, 6.3
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]],
        k: int,
        p: int,
        moment_func: Callable[[Any, np.ndarray], np.ndarray]
    ):
        """
        Initialize the GMM engine.
        
        Args:
            data: Observed data. Can be either:
                - numpy array of shape (n, d) or (n,) for univariate
                - dictionary of arrays {'var_name': array, ...} where all
                  arrays have the same length n
            k: Number of moment conditions
            p: Number of parameters to estimate
            moment_func: Function that computes moment conditions.
                Signature: g(data, theta) -> array of shape (n, k)
                where data is the stored data and theta is parameter vector.
        
        Raises:
            ValueError: If data is empty, k <= 0, or p <= 0
        
        Requirements: 6.2
        """
        # Validate inputs
        if data is None:
            raise ValueError("data cannot be empty")
        
        # Handle dictionary data (common in econometrics)
        if isinstance(data, dict):
            if len(data) == 0:
                raise ValueError("data dictionary cannot be empty")
            # Get sample size from first array
            first_key = next(iter(data))
            self._n = len(data[first_key])
            # Validate all arrays have same length
            for key, arr in data.items():
                if len(arr) != self._n:
                    raise ValueError(
                        f"All arrays in data dict must have same length. "
                        f"'{first_key}' has {self._n}, '{key}' has {len(arr)}"
                    )
            self._data = data
            self._is_dict = True
        else:
            # Handle array data
            if hasattr(data, '__len__') and len(data) == 0:
                raise ValueError("data cannot be empty")
            self._data = np.atleast_2d(np.asarray(data, dtype=np.float64))
            if self._data.shape[0] == 1 and len(data) > 1:
                # Handle 1D array case - should be (n, 1) not (1, n)
                self._data = self._data.T
            self._n = self._data.shape[0]
            self._is_dict = False
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        
        self._k = k
        self._p = p
        self._moment_func = moment_func
    
    @property
    def data(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Return the stored data (array or dictionary)."""
        return self._data
    
    @property
    def k(self) -> int:
        """Return the number of moment conditions."""
        return self._k
    
    @property
    def p(self) -> int:
        """Return the number of parameters."""
        return self._p
    
    @property
    def n(self) -> int:
        """Return the sample size."""
        return self._n
    
    def gbar(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sample moments and covariance at given parameter values.
        
        Computes:
        - ḡ(θ) = (1/n) Σᵢ g(zᵢ, θ)  (sample mean of moment conditions)
        - S = (1/n) Σᵢ (gᵢ - ḡ)(gᵢ - ḡ)'  (covariance of moment conditions)
        
        Args:
            theta: Parameter vector of length p
        
        Returns:
            Tuple of (g_bar, S) where:
            - g_bar: Sample mean of moment conditions, shape (k,)
            - S: Covariance matrix of moment conditions, shape (k, k)
        
        Raises:
            ValueError: If theta has wrong dimension or moment_func returns
                wrong shape
        
        Requirements: 6.1, 6.3
        """
        theta = np.asarray(theta, dtype=np.float64)
        
        # Validate theta dimension
        if theta.shape != (self._p,):
            raise ValueError(
                f"theta must have shape ({self._p},), got {theta.shape}"
            )
        
        # Compute moment conditions for all observations
        # g_i should have shape (n, k)
        g_i = self._moment_func(self._data, theta)
        g_i = np.asarray(g_i, dtype=np.float64)
        
        # Validate output shape
        if g_i.shape != (self._n, self._k):
            raise ValueError(
                f"moment_func must return array of shape ({self._n}, {self._k}), "
                f"got {g_i.shape}"
            )
        
        # Compute sample mean: ḡ = (1/n) Σᵢ gᵢ
        g_bar = np.mean(g_i, axis=0)
        
        # Compute covariance: S = (1/n) Σᵢ (gᵢ - ḡ)(gᵢ - ḡ)'
        # Using numpy's cov with rowvar=False and bias=True for 1/n normalization
        # Note: np.cov uses 1/(n-1) by default, we want 1/n
        g_centered = g_i - g_bar
        S = (g_centered.T @ g_centered) / self._n
        
        return g_bar, S
    
    def moments(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alias for gbar() to provide consistent interface with MomentEngine.
        
        This allows GMMEngine to be used interchangeably with MomentEngine
        in the estimation code.
        
        Args:
            theta: Parameter vector of length p
        
        Returns:
            Tuple of (g_bar, S) - same as gbar()
        
        Requirements: 6.1
        """
        return self.gbar(theta)
    
    def moments_jac(
        self,
        theta: np.ndarray,
        eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute moments, covariance, and numerical Jacobian.
        
        The Jacobian D is computed via central finite differences:
        D[i,j] = (g_bar(θ + εeⱼ) - g_bar(θ - εeⱼ)) / (2ε)
        
        Args:
            theta: Parameter vector of length p
            eps: Step size for numerical differentiation (default: 1e-6)
        
        Returns:
            Tuple of (g_bar, S, D) where:
            - g_bar: Sample mean of moment conditions, shape (k,)
            - S: Covariance matrix of moment conditions, shape (k, k)
            - D: Jacobian matrix ∂g_bar/∂θ, shape (k, p)
        
        Requirements: 6.1
        """
        theta = np.asarray(theta, dtype=np.float64)
        
        # Compute moments at theta
        g_bar, S = self.gbar(theta)
        
        # Compute Jacobian using shared utility
        D = numerical_jacobian(self.gbar, theta, self._k, self._p, eps)
        
        return g_bar, S, D


def gmm_estimate(
    data: np.ndarray,
    moment_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    bounds: List[Tuple[float, float]],
    k: int,
    weighting: str = "optimal",
    n_global: int = 50,
    seed: int = 42,
) -> GMMEstimateResult:
    """
    Estimate parameters using Generalized Method of Moments (GMM).
    
    This is the simple, high-level API. Just pass your data, moment function,
    and parameter bounds.
    
    Args:
        data: Observed data. Can be either:
            - numpy array of shape (n, ...) 
            - dictionary of arrays {'var_name': array, ...} where all
              arrays have the same length n (common in econometrics)
        
        moment_func: Moment condition function with signature:
            moment_func(data, theta) -> moments
            - data: the observed data (same format as input)
            - theta: parameter vector of shape (p,)
            - returns: moment conditions of shape (n, k)
        
        bounds: Parameter bounds as list of (lower, upper) tuples.
            Length determines number of parameters p.
        
        k: Number of moment conditions.
        
        weighting: Weighting matrix type (default: "optimal")
            - "identity": W = I (one-step)
            - "optimal": W = S^{-1} (two-step efficient)
        
        n_global: Number of global search points (default: 50)
        
        seed: Random seed for optimization (default: 42)
    
    Returns:
        GMMEstimateResult with theta, se, objective, converged, sample_moments
    
    Example:
        >>> # With dictionary data (common in econometrics)
        >>> def moment_func(data, theta):
        ...     beta0, beta1 = theta
        ...     residual = data['Y'] - beta0 - beta1 * data['X']
        ...     return np.column_stack([residual, residual * data['Z']])
        ...
        >>> result = gmm_estimate(
        ...     data={'Y': y, 'X': x, 'Z': z},
        ...     moment_func=moment_func,
        ...     bounds=[(-10, 10), (-10, 10)],
        ...     k=2
        ... )
        >>> print(result.theta)
    """
    # Import here to avoid circular imports
    from .estimation import EstimationSetup, estimate
    
    p = len(bounds)
    
    # Get sample size - handle both array and dict data
    if isinstance(data, dict):
        first_key = next(iter(data))
        n = len(data[first_key])
    else:
        n = len(data)
    
    # Create engine
    engine = GMMEngine(
        data=data,
        k=k,
        p=p,
        moment_func=moment_func
    )
    
    # Create setup
    setup = EstimationSetup(
        mode="GMM",
        model_name="gmm_model",
        moment_type="user_defined",
        k=k,
        p=p,
        n_sim=n,  # Not used for GMM but required
        shock_dim=1,
        seed=seed,
        weighting=weighting
    )
    
    # Target moments are zeros (GMM minimizes sample average of moment conditions)
    data_moments = np.zeros(k)
    
    # Run estimation
    result = estimate(
        setup,
        data_moments,
        bounds,
        n_global=n_global,
        data=data,
        moment_func=moment_func,
        engine=engine
    )
    
    return GMMEstimateResult(
        theta=result.theta_hat,
        se=result.se,
        objective=result.objective,
        converged=result.converged,
        sample_moments=result.m_bar
    )
