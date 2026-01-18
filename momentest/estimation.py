"""
Estimation module for momentest.

Provides the core estimation interface for SMM and GMM estimation,
including global search, local optimization, and result handling.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Callable
import numpy as np


@dataclass
class EstimationSetup:
    """
    Configuration for SMM/GMM estimation.
    
    Attributes:
        mode: Estimation mode - "SMM" for Simulated Method of Moments,
              "GMM" for Generalized Method of Moments
        model_name: Name identifier for the structural model
        moment_type: Description of moment conditions used
        k: Number of moment conditions
        p: Number of parameters to estimate
        n_sim: Number of simulations per moment evaluation (SMM only)
        shock_dim: Dimension of shocks per simulation
        seed: Random seed for reproducibility
        weighting: Weighting matrix type - "identity", "optimal", or "user"
        W_user: User-specified weighting matrix (required if weighting="user")
    
    Requirements: 3.1
    """
    mode: Literal["SMM", "GMM"]
    model_name: str
    moment_type: str
    k: int
    p: int
    n_sim: int = 2000
    shock_dim: int = 1
    seed: int = 42
    weighting: Literal["identity", "optimal", "user"] = "identity"
    W_user: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.p <= 0:
            raise ValueError(f"p must be positive, got {self.p}")
        if self.n_sim <= 0:
            raise ValueError(f"n_sim must be positive, got {self.n_sim}")
        if self.shock_dim <= 0:
            raise ValueError(f"shock_dim must be positive, got {self.shock_dim}")
        if self.weighting == "user" and self.W_user is None:
            raise ValueError("W_user must be provided when weighting='user'")
        if self.W_user is not None:
            if self.W_user.shape != (self.k, self.k):
                raise ValueError(
                    f"W_user must have shape ({self.k}, {self.k}), "
                    f"got {self.W_user.shape}"
                )


@dataclass
class EstimationResult:
    """
    Result container for SMM/GMM estimation.
    
    Attributes:
        theta_hat: Estimated parameter vector
        se: Standard errors (asymptotic or bootstrap)
        objective: Final objective function value
        converged: Whether optimization converged
        n_evals: Number of objective function evaluations
        history: List of (theta, objective) tuples from optimization
        m_bar: Simulated/sample moments at theta_hat
        S: Covariance matrix of moments at theta_hat
        W: Weighting matrix used in final estimation
    
    Requirements: 3.5
    """
    theta_hat: np.ndarray
    se: np.ndarray
    objective: float
    converged: bool
    n_evals: int
    history: List[Tuple[np.ndarray, float]]
    m_bar: np.ndarray
    S: np.ndarray
    W: np.ndarray


@dataclass
class BootstrapResult:
    """
    Result container for bootstrap inference.
    
    Attributes:
        theta_hat: Point estimate (from original estimation)
        se: Bootstrap standard errors (std of bootstrap estimates)
        ci_lower: Lower confidence interval bounds
        ci_upper: Upper confidence interval bounds
        alpha: Confidence level (e.g., 0.05 for 95% CI)
        bootstrap_estimates: Array of shape (n_boot, p) with all bootstrap estimates
        n_boot: Number of bootstrap replications
        n_converged: Number of replications that converged
    
    Requirements: 5.1, 5.3, 5.4
    """
    theta_hat: np.ndarray
    se: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    alpha: float
    bootstrap_estimates: np.ndarray
    n_boot: int
    n_converged: int


def compute_optimal_weighting(
    S: np.ndarray,
    regularization: float = 1e-8
) -> np.ndarray:
    """
    Compute the optimal weighting matrix W = S⁻¹.
    
    The optimal weighting matrix for GMM/SMM is the inverse of the moment
    covariance matrix S. This provides efficient estimation (minimum variance).
    
    Args:
        S: Moment covariance matrix of shape (k, k)
        regularization: Regularization parameter for singular matrices.
            Added to diagonal before inversion to ensure numerical stability.
    
    Returns:
        Optimal weighting matrix W = (S + regularization * I)⁻¹
    
    Raises:
        ValueError: If S is not square or has invalid shape
    
    Requirements: 3.4
    """
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError(f"S must be a square matrix, got shape {S.shape}")
    
    k = S.shape[0]
    
    # Check if S is singular or near-singular
    try:
        # Try direct inversion first
        cond_number = np.linalg.cond(S)
        if cond_number > 1e10:
            # Matrix is ill-conditioned, use regularization
            import warnings
            warnings.warn(
                f"Moment covariance matrix S is ill-conditioned (cond={cond_number:.2e}). "
                f"Using regularization with lambda={regularization}.",
                RuntimeWarning
            )
            S_reg = S + regularization * np.eye(k)
            W = np.linalg.inv(S_reg)
        else:
            W = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # Singular matrix - use regularized inverse
        import warnings
        warnings.warn(
            f"Moment covariance matrix S is singular. "
            f"Using regularization with lambda={regularization}.",
            RuntimeWarning
        )
        S_reg = S + regularization * np.eye(k)
        W = np.linalg.inv(S_reg)
    
    return W


def objective(
    theta: np.ndarray,
    engine,
    data_moments: np.ndarray,
    W: np.ndarray
) -> float:
    """
    Compute the SMM/GMM objective function.
    
    The objective is the quadratic form: (m̄ - m_data)' W (m̄ - m_data)
    
    Args:
        theta: Parameter vector of length p
        engine: MomentEngine instance for computing simulated moments
        data_moments: Target data moments of length k
        W: Weighting matrix of shape (k, k)
    
    Returns:
        Objective function value (scalar)
    
    Requirements: 3.3
    """
    # Compute simulated moments
    m_bar, _ = engine.moments(theta)
    
    # Compute moment deviation
    g = m_bar - data_moments
    
    # Compute quadratic form: g' W g
    obj = float(g @ W @ g)
    
    return obj


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _sample_sobol(
    n_samples: int,
    bounds: List[Tuple[float, float]],
    seed: int = 42
) -> np.ndarray:
    """
    Generate Sobol sequence samples within bounds.
    
    Args:
        n_samples: Number of samples to generate
        bounds: List of (lower, upper) bounds for each dimension
        seed: Random seed for scrambling
    
    Returns:
        Array of shape (n_samples, p) with samples in bounds
    """
    from scipy.stats import qmc
    
    p = len(bounds)
    sampler = qmc.Sobol(d=p, scramble=True, seed=seed)
    
    # Round up to power of 2 for optimal Sobol balance properties
    n_power2 = _next_power_of_2(n_samples)
    
    # Generate samples in [0, 1]^p (generate power of 2, then truncate)
    samples_unit = sampler.random(n_power2)[:n_samples]
    
    # Scale to bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    samples = lower + samples_unit * (upper - lower)
    
    return samples


def _sample_lhs(
    n_samples: int,
    bounds: List[Tuple[float, float]],
    seed: int = 42
) -> np.ndarray:
    """
    Generate Latin Hypercube samples within bounds.
    
    Args:
        n_samples: Number of samples to generate
        bounds: List of (lower, upper) bounds for each dimension
        seed: Random seed
    
    Returns:
        Array of shape (n_samples, p) with samples in bounds
    """
    from scipy.stats import qmc
    
    p = len(bounds)
    sampler = qmc.LatinHypercube(d=p, seed=seed)
    
    # Generate samples in [0, 1]^p
    samples_unit = sampler.random(n_samples)
    
    # Scale to bounds
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    samples = lower + samples_unit * (upper - lower)
    
    return samples


def global_search(
    engine,
    data_moments: np.ndarray,
    W: np.ndarray,
    bounds: List[Tuple[float, float]],
    n_global: int = 100,
    method: Literal["sobol", "lhs"] = "sobol",
    seed: int = 42,
    n_jobs: int = 1
) -> Tuple[np.ndarray, float, List[Tuple[np.ndarray, float]]]:
    """
    Perform global search over parameter space.
    
    Samples candidate points using Sobol or LHS and evaluates objectives
    in parallel to find a good starting point for local optimization.
    
    Args:
        engine: MomentEngine instance
        data_moments: Target data moments
        W: Weighting matrix
        bounds: List of (lower, upper) bounds for each parameter
        n_global: Number of candidate points to sample
        method: Sampling method - "sobol" or "lhs"
        seed: Random seed for sampling
        n_jobs: Number of parallel jobs (1 = sequential)
    
    Returns:
        Tuple of (best_theta, best_objective, history)
        where history is list of (theta, objective) for all candidates
    
    Requirements: 4.1, 4.2
    """
    from joblib import Parallel, delayed
    
    # Sample candidate points
    if method == "sobol":
        candidates = _sample_sobol(n_global, bounds, seed)
    else:
        candidates = _sample_lhs(n_global, bounds, seed)
    
    # Evaluate objectives
    def eval_candidate(theta):
        try:
            obj = objective(theta, engine, data_moments, W)
            return (theta.copy(), obj)
        except Exception:
            return (theta.copy(), np.inf)
    
    if n_jobs == 1:
        # Sequential evaluation
        results = [eval_candidate(candidates[i]) for i in range(n_global)]
    else:
        # Parallel evaluation
        results = Parallel(n_jobs=n_jobs)(
            delayed(eval_candidate)(candidates[i]) for i in range(n_global)
        )
    
    # Find best candidate
    history = results
    best_idx = np.argmin([r[1] for r in results])
    best_theta = results[best_idx][0]
    best_obj = results[best_idx][1]
    
    return best_theta, best_obj, history


def local_optimize(
    engine,
    data_moments: np.ndarray,
    W: np.ndarray,
    bounds: List[Tuple[float, float]],
    x0: np.ndarray,
    method: str = "L-BFGS-B",
    maxiter: int = 1000,
    tol: float = 1e-8
) -> Tuple[np.ndarray, float, bool, int, List[Tuple[np.ndarray, float]]]:
    """
    Perform local optimization from a starting point.
    
    Uses scipy.optimize.minimize with bounds enforcement.
    
    Args:
        engine: MomentEngine instance
        data_moments: Target data moments
        W: Weighting matrix
        bounds: List of (lower, upper) bounds for each parameter
        x0: Starting point for optimization
        method: Optimization method (default: L-BFGS-B)
        maxiter: Maximum iterations
        tol: Tolerance for convergence
    
    Returns:
        Tuple of (theta_hat, objective, converged, n_evals, history)
    
    Requirements: 4.4, 4.5
    """
    from scipy.optimize import minimize
    
    history = []
    n_evals = [0]  # Use list to allow modification in closure
    
    def obj_func(theta):
        n_evals[0] += 1
        obj = objective(theta, engine, data_moments, W)
        history.append((theta.copy(), obj))
        return obj
    
    # Convert bounds to scipy format
    scipy_bounds = [(b[0], b[1]) for b in bounds]
    
    # Run optimization
    result = minimize(
        obj_func,
        x0,
        method=method,
        bounds=scipy_bounds,
        options={"maxiter": maxiter, "ftol": tol, "gtol": tol}
    )
    
    theta_hat = result.x
    final_obj = result.fun
    converged = result.success
    
    # Ensure theta_hat is within bounds (clip if necessary due to numerical issues)
    for i, (lower, upper) in enumerate(bounds):
        theta_hat[i] = np.clip(theta_hat[i], lower, upper)
    
    return theta_hat, final_obj, converged, n_evals[0], history


def estimate(
    setup: EstimationSetup,
    data_moments: np.ndarray,
    bounds: List[Tuple[float, float]],
    n_global: int = 100,
    local_method: str = "L-BFGS-B",
    global_method: Literal["sobol", "lhs"] = "sobol",
    n_jobs: int = 1,
    regularization: float = 1e-8,
    data: Optional[np.ndarray] = None,
    moment_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    sim_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    engine: Optional[object] = None
) -> EstimationResult:
    """
    Main estimation entry point for SMM and GMM.
    
    Performs global search followed by local optimization to estimate
    structural parameters by matching simulated/sample moments to data moments.
    
    For SMM mode:
    - Pass sim_func and moment_func to create an SMMEngine automatically
    - Or pass a pre-constructed SMMEngine as the engine parameter
    - The SMMEngine uses Common Random Numbers (CRN) for smooth objectives
    
    For GMM mode:
    - Uses GMMEngine with analytical moment conditions
    - Requires data and moment_func parameters
    
    For optimal weighting (two-step estimation):
    1. First stage: Estimate with identity weighting matrix
    2. Compute optimal W = S⁻¹ at first-stage estimate
    3. Second stage: Re-estimate with optimal weighting matrix
    
    Args:
        setup: EstimationSetup configuration
        data_moments: Target data moments of length k (for SMM) or zeros (for GMM)
        bounds: List of (lower, upper) bounds for each parameter
        n_global: Number of global search candidates
        local_method: Local optimization method (default: L-BFGS-B)
        global_method: Global sampling method ("sobol" or "lhs")
        n_jobs: Number of parallel jobs for global search
        regularization: Regularization parameter for optimal weighting
            when S is singular or ill-conditioned
        data: Observed data array for GMM mode (required if mode="GMM")
        moment_func: Moment condition function for GMM or SMM.
            For GMM: g(data, theta) -> array of shape (n, k)
            For SMM: moment_func(simulated_data) -> array of shape (n_sim, k)
        sim_func: Simulation function for SMM mode.
            Signature: sim_func(theta, shocks) -> simulated_data
        engine: Optional pre-constructed engine (SMMEngine or GMMEngine).
            If provided, this engine is used directly instead of creating a new one.
    
    Returns:
        EstimationResult containing theta_hat, standard errors, and diagnostics
    
    Requirements: 3.2, 3.4, 3.5, 4.3, 6.1, 14.7
    """
    from .gmm import GMMEngine
    from .smm import SMMEngine
    
    # Validate inputs
    if len(data_moments) != setup.k:
        raise ValueError(
            f"data_moments length {len(data_moments)} != setup.k {setup.k}"
        )
    if len(bounds) != setup.p:
        raise ValueError(
            f"bounds length {len(bounds)} != setup.p {setup.p}"
        )
    
    # Create or use provided engine based on mode
    if engine is not None:
        # Use provided engine directly
        pass
    elif setup.mode == "GMM":
        # GMM mode: requires data and moment_func
        if data is None:
            raise ValueError("data is required for GMM mode")
        if moment_func is None:
            raise ValueError("moment_func is required for GMM mode")
        
        engine = GMMEngine(
            data=data,
            k=setup.k,
            p=setup.p,
            moment_func=moment_func
        )
    else:
        # SMM mode: requires sim_func and moment_func
        if sim_func is None:
            raise ValueError(
                "sim_func is required for SMM mode. "
                "Pass sim_func and moment_func, or provide a pre-constructed SMMEngine."
            )
        if moment_func is None:
            raise ValueError(
                "moment_func is required for SMM mode. "
                "Pass sim_func and moment_func, or provide a pre-constructed SMMEngine."
            )
        
        engine = SMMEngine(
            k=setup.k,
            p=setup.p,
            n_sim=setup.n_sim,
            shock_dim=setup.shock_dim,
            sim_func=sim_func,
            moment_func=moment_func,
            seed=setup.seed
        )
    
    # Determine initial weighting matrix
    if setup.weighting == "identity":
        W = np.eye(setup.k)
    elif setup.weighting == "user":
        W = setup.W_user
    else:
        # For optimal weighting, start with identity (first stage)
        W = np.eye(setup.k)
    
    # Global search
    best_theta, best_obj, global_history = global_search(
        engine, data_moments, W, bounds,
        n_global=n_global,
        method=global_method,
        seed=setup.seed,
        n_jobs=n_jobs
    )
    
    # Local optimization from best global point (first stage)
    theta_hat, final_obj, converged, n_evals, local_history = local_optimize(
        engine, data_moments, W, bounds, best_theta,
        method=local_method
    )
    
    # Combine histories
    full_history = global_history + local_history
    
    # Two-step optimal weighting
    if setup.weighting == "optimal":
        # Compute moments and covariance at first-stage estimate
        m_bar_stage1, S_stage1 = engine.moments(theta_hat)
        
        # Compute optimal weighting matrix W = S⁻¹
        W = compute_optimal_weighting(S_stage1, regularization=regularization)
        
        # Second stage: Re-optimize with optimal weighting
        # Start from first-stage estimate
        theta_hat, final_obj, converged_stage2, n_evals_stage2, local_history_stage2 = local_optimize(
            engine, data_moments, W, bounds, theta_hat,
            method=local_method
        )
        
        # Update convergence and evaluation count
        converged = converged and converged_stage2
        n_evals += n_evals_stage2
        full_history.extend(local_history_stage2)
    
    # Compute final moments and covariance
    m_bar, S = engine.moments(theta_hat)
    
    # Compute asymptotic standard errors
    # For GMM: SE = sqrt(diag((D'WD)^{-1})) / sqrt(n)
    # For SMM: SE = sqrt(diag((D'WD)^{-1} D'W S W D (D'WD)^{-1})) / sqrt(n_sim)
    try:
        _, _, D = engine.moments_jac(theta_hat)
        DtWD = D.T @ W @ D
        DtWD_inv = np.linalg.inv(DtWD)
        # Sandwich variance: (D'WD)^{-1} D'W S W D (D'WD)^{-1}
        var_matrix = DtWD_inv @ D.T @ W @ S @ W @ D @ DtWD_inv
        
        # Scale by sample size
        if setup.mode == "GMM" and hasattr(engine, 'n'):
            n_scale = engine.n
        else:
            n_scale = setup.n_sim
        
        var_diag = np.diag(var_matrix) / n_scale
        # Handle numerical issues: negative variances become NaN
        var_diag = np.where(var_diag >= 0, var_diag, np.nan)
        se = np.sqrt(var_diag)
    except np.linalg.LinAlgError:
        # If Jacobian is singular, return NaN for SE
        se = np.full(setup.p, np.nan)
    
    return EstimationResult(
        theta_hat=theta_hat,
        se=se,
        objective=final_obj,
        converged=converged,
        n_evals=n_evals + n_global,
        history=full_history,
        m_bar=m_bar,
        S=S,
        W=W
    )


def _single_bootstrap_replication(
    setup: EstimationSetup,
    data_moments: np.ndarray,
    bounds: List[Tuple[float, float]],
    boot_seed: int,
    n_global: int,
    local_method: str,
    global_method: str,
    regularization: float,
    data: Optional[np.ndarray] = None,
    moment_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    smm_engine_config: Optional[dict] = None
) -> Tuple[np.ndarray, bool]:
    """
    Run a single bootstrap replication with a different seed.
    
    Args:
        setup: EstimationSetup configuration (seed will be overridden)
        data_moments: Target data moments
        bounds: Parameter bounds
        boot_seed: Seed for this bootstrap replication
        n_global: Number of global search candidates
        local_method: Local optimization method
        global_method: Global sampling method
        regularization: Regularization parameter
        data: Observed data array for GMM mode
        moment_func: Moment condition function for GMM mode
        smm_engine_config: Configuration for SMMEngine (sim_func, moment_func, etc.)
            If provided, creates a new SMMEngine with the bootstrap seed.
    
    Returns:
        Tuple of (theta_hat, converged) for this replication
    
    Requirements: 5.1, 5.2, 14.8
    """
    from .smm import SMMEngine
    
    # Create a modified setup with the bootstrap seed
    boot_setup = EstimationSetup(
        mode=setup.mode,
        model_name=setup.model_name,
        moment_type=setup.moment_type,
        k=setup.k,
        p=setup.p,
        n_sim=setup.n_sim,
        shock_dim=setup.shock_dim,
        seed=boot_seed,  # Use different seed for each replication
        weighting=setup.weighting,
        W_user=setup.W_user
    )
    
    # Create engine for this bootstrap replication
    engine = None
    if smm_engine_config is not None:
        # Create SMMEngine with new seed for bootstrap
        engine = SMMEngine(
            k=smm_engine_config['k'],
            p=smm_engine_config['p'],
            n_sim=smm_engine_config['n_sim'],
            shock_dim=smm_engine_config['shock_dim'],
            sim_func=smm_engine_config['sim_func'],
            moment_func=smm_engine_config['moment_func'],
            seed=boot_seed  # Use bootstrap seed for new shocks
        )
    
    try:
        result = estimate(
            boot_setup,
            data_moments,
            bounds,
            n_global=n_global,
            local_method=local_method,
            global_method=global_method,
            n_jobs=1,  # No nested parallelism
            regularization=regularization,
            data=data,
            moment_func=moment_func,
            engine=engine
        )
        return result.theta_hat, result.converged
    except Exception:
        # If estimation fails, return NaN
        return np.full(setup.p, np.nan), False


def bootstrap(
    setup: EstimationSetup,
    data_moments: np.ndarray,
    bounds: List[Tuple[float, float]],
    n_boot: int = 200,
    alpha: float = 0.05,
    n_global: int = 50,
    local_method: str = "L-BFGS-B",
    global_method: Literal["sobol", "lhs"] = "sobol",
    n_jobs: int = -1,
    regularization: float = 1e-8,
    base_seed: Optional[int] = None,
    data: Optional[np.ndarray] = None,
    moment_func: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    engine: Optional[object] = None
) -> BootstrapResult:
    """
    Bootstrap inference for SMM/GMM estimation.
    
    Re-estimates the model with different random seeds for each bootstrap
    replication to quantify estimation uncertainty. Standard errors are
    computed as the standard deviation of bootstrap estimates, and
    confidence intervals are computed using percentiles.
    
    For SMMEngine: Each bootstrap replication creates a new SMMEngine with
    a different seed, which re-draws the shocks. This ensures proper
    bootstrap variability in the simulation-based estimation.
    
    Args:
        setup: EstimationSetup configuration
        data_moments: Target data moments of length k
        bounds: List of (lower, upper) bounds for each parameter
        n_boot: Number of bootstrap replications (default: 200)
        alpha: Significance level for confidence intervals (default: 0.05 for 95% CI)
        n_global: Number of global search candidates per replication
        local_method: Local optimization method (default: L-BFGS-B)
        global_method: Global sampling method ("sobol" or "lhs")
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = sequential)
        regularization: Regularization parameter for optimal weighting
        base_seed: Base seed for generating bootstrap seeds (default: setup.seed)
        data: Observed data array for GMM mode (required if mode="GMM")
        moment_func: Moment condition function for GMM mode (required if mode="GMM")
        engine: Optional pre-constructed engine (SMMEngine) for user-defined simulations.
            If provided, bootstrap will create new SMMEngines with different seeds
            for each replication, using the same sim_func and moment_func.
    
    Returns:
        BootstrapResult containing:
        - theta_hat: Point estimate from original estimation
        - se: Bootstrap standard errors
        - ci_lower, ci_upper: Percentile confidence intervals
        - bootstrap_estimates: All bootstrap estimates (n_boot, p)
        - n_boot: Number of replications
        - n_converged: Number of converged replications
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 14.8
    """
    from joblib import Parallel, delayed
    from .smm import SMMEngine
    
    # Validate inputs
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if len(data_moments) != setup.k:
        raise ValueError(
            f"data_moments length {len(data_moments)} != setup.k {setup.k}"
        )
    if len(bounds) != setup.p:
        raise ValueError(
            f"bounds length {len(bounds)} != setup.p {setup.p}"
        )
    
    # Validate GMM-specific inputs
    if setup.mode == "GMM":
        if data is None:
            raise ValueError("data is required for GMM mode")
        if moment_func is None:
            raise ValueError("moment_func is required for GMM mode")
    
    # Check if we're using SMMEngine and extract config for bootstrap
    smm_engine_config = None
    if isinstance(engine, SMMEngine):
        smm_engine_config = {
            'k': engine.k,
            'p': engine.p,
            'n_sim': engine.n_sim,
            'shock_dim': engine.shock_dim,
            'sim_func': engine._sim_func,
            'moment_func': engine._moment_func,
        }
    
    # First, run the original estimation to get the point estimate
    original_result = estimate(
        setup,
        data_moments,
        bounds,
        n_global=n_global,
        local_method=local_method,
        global_method=global_method,
        n_jobs=1,
        regularization=regularization,
        data=data,
        moment_func=moment_func,
        engine=engine
    )
    theta_hat = original_result.theta_hat
    
    # Generate seeds for bootstrap replications
    if base_seed is None:
        base_seed = setup.seed
    rng = np.random.default_rng(base_seed)
    boot_seeds = rng.integers(0, 2**31, size=n_boot)
    
    # Run bootstrap replications in parallel
    if n_jobs == 1:
        # Sequential execution
        results = [
            _single_bootstrap_replication(
                setup, data_moments, bounds, int(boot_seeds[i]),
                n_global, local_method, global_method, regularization,
                data, moment_func, smm_engine_config
            )
            for i in range(n_boot)
        ]
    else:
        # Parallel execution
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_bootstrap_replication)(
                setup, data_moments, bounds, int(boot_seeds[i]),
                n_global, local_method, global_method, regularization,
                data, moment_func, smm_engine_config
            )
            for i in range(n_boot)
        )
    
    # Extract estimates and convergence flags
    bootstrap_estimates = np.array([r[0] for r in results])
    converged_flags = np.array([r[1] for r in results])
    n_converged = int(np.sum(converged_flags))
    
    # Compute bootstrap standard errors (std of bootstrap estimates)
    # Only use converged replications for SE computation
    valid_mask = ~np.any(np.isnan(bootstrap_estimates), axis=1)
    if np.sum(valid_mask) > 1:
        se = np.std(bootstrap_estimates[valid_mask], axis=0, ddof=1)
    else:
        se = np.full(setup.p, np.nan)
    
    # Compute percentile confidence intervals
    # CI bounds at α/2 and 1-α/2 percentiles
    if np.sum(valid_mask) > 1:
        ci_lower = np.percentile(bootstrap_estimates[valid_mask], 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(bootstrap_estimates[valid_mask], 100 * (1 - alpha / 2), axis=0)
    else:
        ci_lower = np.full(setup.p, np.nan)
        ci_upper = np.full(setup.p, np.nan)
    
    return BootstrapResult(
        theta_hat=theta_hat,
        se=se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        alpha=alpha,
        bootstrap_estimates=bootstrap_estimates,
        n_boot=n_boot,
        n_converged=n_converged
    )
