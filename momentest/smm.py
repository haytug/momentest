"""
SMM (Simulated Method of Moments) engine for user-defined simulations.

Provides the SMMEngine class for computing simulated moments using
user-provided simulation and moment functions, with Common Random Numbers
(CRN) support for smooth objective functions.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Union
import numpy as np

from .common import numerical_jacobian, format_estimate_result


@dataclass
class SMMResult:
    """
    Result container for SMM estimation.
    
    Attributes:
        theta: Estimated parameters
        se: Standard errors (asymptotic)
        objective: Final objective value
        converged: Whether optimization converged
        sim_moments: Simulated moments at theta
    """
    theta: np.ndarray
    se: np.ndarray
    objective: float
    converged: bool
    sim_moments: np.ndarray
    
    def __repr__(self):
        return format_estimate_result(
            "SMMResult", self.theta, self.se, 
            self.objective, self.converged, "Objective"
        )


class SMMEngine:
    """
    Python SMM engine for user-defined simulation functions.
    
    Provides CRN support by pre-drawing shocks and reusing them
    across parameter evaluations, ensuring smooth objective functions.
    
    Unlike the C++ MomentEngine which has built-in models, SMMEngine allows
    users to define their own simulation and moment computation functions
    in pure Python, making it easy to prototype and estimate custom
    structural models.
    
    Attributes:
        k: Number of moment conditions
        p: Number of parameters
        n_sim: Number of simulations
        shock_dim: Dimension of shocks per simulation
        sim_func: User-provided simulation function
        moment_func: User-provided moment function
        seed: Random seed for reproducibility
    
    Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6
    """
    
    def __init__(
        self,
        k: int,
        p: int,
        n_sim: int,
        shock_dim: int,
        sim_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        moment_func: Callable[[np.ndarray], np.ndarray],
        seed: int = 42,
        shocks: np.ndarray = None
    ):
        """
        Initialize the SMM engine with user-provided functions.
        
        Args:
            k: Number of moment conditions
            p: Number of parameters to estimate
            n_sim: Number of simulations per moment evaluation
            shock_dim: Dimension of shocks per simulation
            sim_func: Function that simulates data given parameters and shocks.
                Signature: sim_func(theta, shocks) -> simulated_data
                - theta: Parameter vector of shape (p,)
                - shocks: Pre-drawn shocks of shape (n_sim, shock_dim)
                - Returns: Simulated data of shape (n_sim, ...) where ... is
                  any shape that moment_func can process
            moment_func: Function that computes moments from simulated data.
                Signature: moment_func(simulated_data) -> moments
                - simulated_data: Output from sim_func
                - Returns: Moments of shape (n_sim, k) where each row is the
                  moment vector for one simulation
            seed: Random seed for pre-drawing shocks (default: 42)
            shocks: Optional pre-drawn shocks of shape (n_sim, shock_dim).
                If provided, these shocks are used instead of drawing new ones.
                This allows using the same random draws across different methods.
        
        Raises:
            ValueError: If k, p, n_sim, or shock_dim are non-positive
            TypeError: If sim_func or moment_func are not callable
        
        Requirements: 14.1, 14.2, 14.3
        """
        # Validate inputs
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        if n_sim <= 0:
            raise ValueError(f"n_sim must be positive, got {n_sim}")
        if shock_dim <= 0:
            raise ValueError(f"shock_dim must be positive, got {shock_dim}")
        
        if not callable(sim_func):
            raise TypeError("sim_func must be callable")
        if not callable(moment_func):
            raise TypeError("moment_func must be callable")
        
        self._k = k
        self._p = p
        self._n_sim = n_sim
        self._shock_dim = shock_dim
        self._sim_func = sim_func
        self._moment_func = moment_func
        self._seed = seed
        
        # Use provided shocks or pre-draw new ones
        if shocks is not None:
            shocks = np.asarray(shocks, dtype=np.float64)
            if shocks.shape != (n_sim, shock_dim):
                raise ValueError(
                    f"shocks must have shape ({n_sim}, {shock_dim}), "
                    f"got {shocks.shape}"
                )
            self._shocks = shocks.copy()
            self._rng = None  # No RNG needed when shocks are provided
        else:
            # Pre-draw shocks for CRN support
            self._rng = np.random.default_rng(seed)
            self._shocks = self._rng.standard_normal((n_sim, shock_dim))
    
    @property
    def k(self) -> int:
        """Return the number of moment conditions."""
        return self._k
    
    @property
    def p(self) -> int:
        """Return the number of parameters."""
        return self._p
    
    @property
    def n_sim(self) -> int:
        """Return the number of simulations."""
        return self._n_sim
    
    @property
    def shock_dim(self) -> int:
        """Return the shock dimension."""
        return self._shock_dim
    
    @property
    def seed(self) -> int:
        """Return the random seed."""
        return self._seed
    
    @property
    def shocks(self) -> np.ndarray:
        """Return the pre-drawn shocks (read-only copy)."""
        return self._shocks.copy()
    
    def _redraw_shocks(self, new_seed: int) -> None:
        """
        Re-draw shocks with a new seed.
        
        This is used internally for bootstrap replications where each
        replication needs different random shocks.
        
        Args:
            new_seed: New random seed for shock generation
        
        Requirements: 14.8
        """
        self._seed = new_seed
        self._rng = np.random.default_rng(new_seed)
        self._shocks = self._rng.standard_normal((self._n_sim, self._shock_dim))
    
    def moments(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute simulated moments using CRN.
        
        Simulates data using the user's sim_func with pre-drawn shocks,
        then computes moments using moment_func. Returns the mean moment
        vector and covariance matrix.
        
        Args:
            theta: Parameter vector of length p
        
        Returns:
            Tuple of (m_bar, S) where:
            - m_bar: Mean moment vector of shape (k,)
            - S: Covariance matrix of shape (k, k)
        
        Raises:
            ValueError: If theta has wrong dimension or moment_func returns
                wrong shape
        
        Requirements: 14.4, 14.5, 14.6
        """
        theta = np.asarray(theta, dtype=np.float64)
        
        # Validate theta dimension
        if theta.ndim != 1:
            raise ValueError(
                f"theta must be 1-dimensional, got {theta.ndim} dimensions"
            )
        if theta.shape[0] != self._p:
            raise ValueError(
                f"theta must have length {self._p}, got {theta.shape[0]}"
            )
        
        # Simulate data using pre-drawn shocks (CRN)
        simulated_data = self._sim_func(theta, self._shocks)
        
        # Compute moments from simulated data
        moments_array = self._moment_func(simulated_data)
        moments_array = np.asarray(moments_array, dtype=np.float64)
        
        # Validate output shape
        if moments_array.shape != (self._n_sim, self._k):
            raise ValueError(
                f"moment_func must return array of shape ({self._n_sim}, {self._k}), "
                f"got {moments_array.shape}"
            )
        
        # Compute mean moment vector: m̄ = (1/n_sim) Σᵢ mᵢ
        m_bar = np.mean(moments_array, axis=0)
        
        # Compute covariance: S = (1/n_sim) Σᵢ (mᵢ - m̄)(mᵢ - m̄)'
        m_centered = moments_array - m_bar
        S = (m_centered.T @ m_centered) / self._n_sim
        
        return m_bar, S
    
    def moments_jac(
        self,
        theta: np.ndarray,
        eps: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute moments, covariance, and numerical Jacobian.
        
        The Jacobian D is computed via central finite differences:
        D[i,j] = (m_bar(θ + εeⱼ) - m_bar(θ - εeⱼ)) / (2ε)
        
        Args:
            theta: Parameter vector of length p
            eps: Step size for numerical differentiation (default: 1e-6)
        
        Returns:
            Tuple of (m_bar, S, D) where:
            - m_bar: Mean moment vector of shape (k,)
            - S: Covariance matrix of shape (k, k)
            - D: Jacobian matrix ∂m_bar/∂θ of shape (k, p)
        
        Requirements: 14.4, 14.5
        """
        theta = np.asarray(theta, dtype=np.float64)
        
        # Compute moments at theta
        m_bar, S = self.moments(theta)
        
        # Compute Jacobian using shared utility
        D = numerical_jacobian(self.moments, theta, self._k, self._p, eps)
        
        return m_bar, S, D


def smm_estimate(
    sim_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    moment_func: Callable[[np.ndarray], np.ndarray],
    data_moments: Union[np.ndarray, List[float]],
    bounds: List[Tuple[float, float]],
    n_sim: int = 1000,
    shock_dim: int = 1,
    seed: int = 42,
    shocks: np.ndarray = None,
    weighting: str = "optimal",
    n_global: int = 50,
) -> SMMResult:
    """
    Estimate parameters using Simulated Method of Moments (SMM).
    
    This is the simple, high-level API. Just pass your simulation function,
    moment function, target moments, and parameter bounds.
    
    Args:
        sim_func: Simulation function with signature:
            sim_func(theta, shocks) -> simulated_data
            - theta: parameter vector of shape (p,)
            - shocks: random shocks of shape (n_sim, shock_dim)
            - returns: simulated data (any shape that moment_func accepts)
        
        moment_func: Moment function with signature:
            moment_func(simulated_data) -> moments
            - simulated_data: output from sim_func
            - returns: moments of shape (n_sim, k)
        
        data_moments: Target moments from data, length k.
            These are what you're trying to match.
        
        bounds: Parameter bounds as list of (lower, upper) tuples.
            Length determines number of parameters p.
        
        n_sim: Number of simulations (default: 1000)
        
        shock_dim: Dimension of shocks per simulation (default: 1)
        
        seed: Random seed for reproducibility (default: 42)
        
        shocks: Optional pre-drawn shocks of shape (n_sim, shock_dim).
            If provided, these shocks are used instead of drawing new ones.
            This allows using the same random draws across different methods.
        
        weighting: Weighting matrix type (default: "optimal")
            - "identity": W = I (one-step)
            - "optimal": W = S^{-1} (two-step efficient)
        
        n_global: Number of global search points (default: 50)
    
    Returns:
        SMMResult with theta, se, objective, converged, sim_moments
    
    Example:
        >>> def sim_func(theta, shocks):
        ...     mu, sigma = theta
        ...     return mu + sigma * shocks
        ...
        >>> def moment_func(sim_data):
        ...     return np.column_stack([sim_data, sim_data**2])
        ...
        >>> result = smm_estimate(
        ...     sim_func=sim_func,
        ...     moment_func=moment_func,
        ...     data_moments=[5.0, 26.0],  # mean=5, E[X^2]=26
        ...     bounds=[(0, 10), (0.1, 5)]
        ... )
        >>> print(result.theta)
    """
    # Import here to avoid circular imports
    from .estimation import EstimationSetup, estimate
    
    data_moments = np.asarray(data_moments)
    k = len(data_moments)
    p = len(bounds)
    
    # Create engine
    engine = SMMEngine(
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=shock_dim,
        sim_func=sim_func,
        moment_func=moment_func,
        seed=seed,
        shocks=shocks
    )
    
    # Create setup
    setup = EstimationSetup(
        mode="SMM",
        model_name="smm_model",
        moment_type="user_defined",
        k=k,
        p=p,
        n_sim=n_sim,
        shock_dim=shock_dim,
        seed=seed,
        weighting=weighting
    )
    
    # Run estimation
    result = estimate(
        setup,
        data_moments,
        bounds,
        n_global=n_global,
        engine=engine
    )
    
    return SMMResult(
        theta=result.theta_hat,
        se=result.se,
        objective=result.objective,
        converged=result.converged,
        sim_moments=result.m_bar
    )
