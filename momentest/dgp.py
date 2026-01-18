"""
Data Generating Processes (DGPs) for learning and validation.

This module provides synthetic data generators with known true parameters,
allowing users to:
1. Learn SMM/GMM by recovering known parameters
2. Validate estimation code against ground truth
3. Conduct Monte Carlo experiments

Each DGP returns a DGPResult containing:
- data: Generated data as a dictionary of numpy arrays
- true_theta: True parameter values
- param_names: Names of parameters
- moment_function: Function to compute moment conditions
- description: Documentation of the DGP

DGPs:
-----
- linear_iv: Linear model with endogeneity and instruments (beginner)
- consumption_savings: Two-period consumption model (intermediate)
- dynamic_discrete_choice: Simple Rust-style DDC model (advanced)

Usage:
------
>>> from momentest.dgp import linear_iv, list_dgps
>>> dgp = linear_iv(n=1000, seed=42)
>>> print(dgp.true_theta)  # True parameters to recover
>>> print(dgp.data.keys())  # Available data
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Callable, Optional, Any


@dataclass
class DGPResult:
    """
    Result from a data generating process.
    
    Attributes:
        name: DGP identifier
        data: Dictionary of data arrays (e.g., {'Y': ..., 'X': ..., 'Z': ...})
        n: Number of observations
        true_theta: True parameter values as numpy array
        param_names: Names of parameters (same order as true_theta)
        moment_function: Function g(data, theta) -> (n, k) moment conditions
        k: Number of moment conditions
        p: Number of parameters
        description: Full description of the DGP
        difficulty: 'beginner', 'intermediate', or 'advanced'
    """
    name: str
    data: Dict[str, np.ndarray]
    n: int
    true_theta: np.ndarray
    param_names: List[str]
    moment_function: Callable[[Dict[str, np.ndarray], np.ndarray], np.ndarray]
    k: int
    p: int
    description: str
    difficulty: str
    
    def __repr__(self):
        return (f"DGPResult(name='{self.name}', n={self.n}, "
                f"k={self.k}, p={self.p}, difficulty='{self.difficulty}')")
    
    def info(self):
        """Print detailed information about the DGP."""
        print("=" * 70)
        print(f"DGP: {self.name}")
        print("=" * 70)
        print(f"\nDifficulty: {self.difficulty}")
        print(f"Observations: {self.n}")
        print(f"Parameters (p): {self.p}")
        print(f"Moments (k): {self.k}")
        print(f"\nTrue Parameters:")
        for name, val in zip(self.param_names, self.true_theta):
            print(f"  {name}: {val:.4f}")
        print(f"\nDescription:\n{self.description}")
        print("=" * 70)


def list_dgps() -> List[str]:
    """
    List all available DGPs.
    
    Returns:
        List of DGP names
    
    Example:
        >>> from momentest.dgp import list_dgps
        >>> print(list_dgps())
        ['linear_iv', 'consumption_savings', 'dynamic_discrete_choice']
    """
    return ['linear_iv', 'consumption_savings', 'dynamic_discrete_choice']


# =============================================================================
# DGP 1: Linear IV Model (Beginner)
# =============================================================================

def linear_iv(
    n: int = 1000,
    seed: int = 42,
    beta0: float = 1.0,
    beta1: float = 2.0,
    rho: float = 0.5,
) -> DGPResult:
    """
    Generate data from a linear IV model with endogeneity.
    
    Model:
        Y = β₀ + β₁X + ε
        X = π₀ + π₁Z + v
        Cov(ε, v) = ρ  (endogeneity)
    
    The instrument Z is correlated with X (relevance) but uncorrelated
    with ε (exclusion restriction).
    
    **Difficulty: Beginner**
    
    This is the classic IV setup. OLS is biased due to endogeneity,
    but GMM/IV using Z as an instrument recovers the true β₁.
    
    Args:
        n: Number of observations (default: 1000)
        seed: Random seed for reproducibility (default: 42)
        beta0: True intercept (default: 1.0)
        beta1: True slope on X (default: 2.0)
        rho: Correlation between ε and v (default: 0.5)
    
    Returns:
        DGPResult with data, true parameters, and moment function
    
    Example:
        >>> from momentest.dgp import linear_iv
        >>> dgp = linear_iv(n=1000, seed=42)
        >>> print(dgp.true_theta)  # [1.0, 2.0]
        >>> # OLS is biased, GMM with Z recovers truth
    """
    rng = np.random.default_rng(seed)
    
    # First-stage parameters (not estimated, just for DGP)
    pi0, pi1 = 0.5, 1.0
    
    # Generate instrument (exogenous)
    Z = rng.standard_normal(n)
    
    # Generate correlated errors
    # (ε, v) ~ N(0, Σ) where Σ = [[1, ρ], [ρ, 1]]
    cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov_matrix)
    errors = rng.standard_normal((n, 2)) @ L.T
    eps = errors[:, 0]
    v = errors[:, 1]
    
    # Generate endogenous X
    X = pi0 + pi1 * Z + v
    
    # Generate outcome Y
    Y = beta0 + beta1 * X + eps
    
    # Moment function for GMM
    def moment_func(data: Dict[str, np.ndarray], theta: np.ndarray) -> np.ndarray:
        """
        Moment conditions for linear IV.
        
        g₁(θ) = Y - β₀ - β₁X  (residual orthogonal to constant)
        g₂(θ) = Z(Y - β₀ - β₁X)  (residual orthogonal to instrument)
        
        Returns (n, k) array of moment conditions.
        """
        b0, b1 = theta
        residual = data['Y'] - b0 - b1 * data['X']
        
        moments = np.column_stack([
            residual,              # E[ε] = 0
            residual * data['Z'], # E[Zε] = 0 (exclusion restriction)
        ])
        return moments
    
    return DGPResult(
        name="linear_iv",
        data={
            'Y': Y,
            'X': X,
            'Z': Z,
            'constant': np.ones(n),
        },
        n=n,
        true_theta=np.array([beta0, beta1]),
        param_names=['beta0', 'beta1'],
        moment_function=moment_func,
        k=2,
        p=2,
        description="""
Linear IV Model with Endogeneity

Model:
    Y = β₀ + β₁X + ε
    X = π₀ + π₁Z + v
    Cov(ε, v) = ρ  (endogeneity)

The regressor X is endogenous because it's correlated with the error ε
through the common component v. This causes OLS to be biased.

The instrument Z satisfies:
1. Relevance: Cov(Z, X) ≠ 0 (Z predicts X)
2. Exclusion: Cov(Z, ε) = 0 (Z doesn't directly affect Y)

GMM Moment Conditions:
    E[ε] = 0           → E[Y - β₀ - β₁X] = 0
    E[Zε] = 0          → E[Z(Y - β₀ - β₁X)] = 0

This is exactly identified (k=2, p=2), so GMM = IV = 2SLS.

Learning objectives:
1. Understand why OLS is biased with endogeneity
2. See how instruments solve the endogeneity problem
3. Practice GMM estimation with moment conditions
""",
        difficulty="beginner",
    )


# =============================================================================
# DGP 2: Two-Period Consumption-Savings Model (Intermediate)
# =============================================================================

def consumption_savings(
    n: int = 1000,
    seed: int = 42,
    beta: float = 0.95,
    gamma: float = 2.0,
    r: float = 0.04,
    y1_mean: float = 100.0,
    y2_mean: float = 100.0,
) -> DGPResult:
    """
    Generate data from a two-period consumption-savings model.
    
    Model:
        max U(C₁) + β E[U(C₂)]
        s.t. C₁ + S = Y₁
             C₂ = Y₂ + (1+r)S
        
        U(C) = C^(1-γ) / (1-γ)  (CRRA utility)
    
    The Euler equation is:
        U'(C₁) = β(1+r) E[U'(C₂)]
        C₁^(-γ) = β(1+r) E[C₂^(-γ)]
    
    **Difficulty: Intermediate**
    
    This is a classic structural model. The Euler equation provides
    moment conditions for estimating (β, γ).
    
    Args:
        n: Number of households (default: 1000)
        seed: Random seed for reproducibility (default: 42)
        beta: Discount factor (default: 0.95)
        gamma: Risk aversion coefficient (default: 2.0)
        r: Interest rate (default: 0.04)
        y1_mean: Mean period-1 income (default: 100.0)
        y2_mean: Mean period-2 income (default: 100.0)
    
    Returns:
        DGPResult with data, true parameters, and moment function
    
    Example:
        >>> from momentest.dgp import consumption_savings
        >>> dgp = consumption_savings(n=1000, seed=42)
        >>> print(dgp.true_theta)  # [0.95, 2.0]
    """
    rng = np.random.default_rng(seed)
    
    # Generate income shocks (log-normal)
    y1 = y1_mean * np.exp(0.2 * rng.standard_normal(n))
    y2 = y2_mean * np.exp(0.3 * rng.standard_normal(n))
    
    # Solve for optimal consumption
    # From Euler equation: C₁^(-γ) = β(1+r) E[C₂^(-γ)]
    # With CRRA and budget constraint, optimal C₁ satisfies:
    # C₁ = (Y₁ + Y₂/(1+r)) / (1 + (β(1+r))^(1/γ))
    
    # Present value of lifetime income
    pv_income = y1 + y2 / (1 + r)
    
    # Optimal consumption (closed-form for two-period model)
    # This comes from solving the Euler equation
    factor = 1 + (beta * (1 + r)) ** (1 / gamma)
    c1 = pv_income / factor
    
    # Savings
    savings = y1 - c1
    
    # Period 2 consumption
    c2 = y2 + (1 + r) * savings
    
    # Add measurement error to make estimation non-trivial
    c1_obs = c1 * np.exp(0.05 * rng.standard_normal(n))
    c2_obs = c2 * np.exp(0.05 * rng.standard_normal(n))
    
    # Moment function for GMM/SMM
    def moment_func(data: Dict[str, np.ndarray], theta: np.ndarray) -> np.ndarray:
        """
        Moment conditions from Euler equation.
        
        The Euler equation implies:
            E[β(1+r)(C₂/C₁)^(-γ) - 1] = 0
        
        With instruments (constant, Y₁):
            E[(β(1+r)(C₂/C₁)^(-γ) - 1)] = 0
            E[(β(1+r)(C₂/C₁)^(-γ) - 1) * Y₁] = 0
        
        Returns (n, k) array of moment conditions.
        """
        b, g = theta
        
        # Euler equation error
        c_ratio = data['C2'] / data['C1']
        euler_error = b * (1 + data['r'][0]) * (c_ratio ** (-g)) - 1
        
        moments = np.column_stack([
            euler_error,                    # E[euler_error] = 0
            euler_error * data['Y1'],       # E[euler_error * Y1] = 0
            euler_error * data['Y2'],       # E[euler_error * Y2] = 0
        ])
        return moments
    
    return DGPResult(
        name="consumption_savings",
        data={
            'C1': c1_obs,
            'C2': c2_obs,
            'Y1': y1,
            'Y2': y2,
            'r': np.full(n, r),
            'constant': np.ones(n),
        },
        n=n,
        true_theta=np.array([beta, gamma]),
        param_names=['beta', 'gamma'],
        moment_function=moment_func,
        k=3,
        p=2,
        description=f"""
Two-Period Consumption-Savings Model

Model:
    max U(C₁) + β E[U(C₂)]
    s.t. C₁ + S = Y₁
         C₂ = Y₂ + (1+r)S
    
    U(C) = C^(1-γ) / (1-γ)  (CRRA utility)

Parameters:
    β = {beta} (discount factor)
    γ = {gamma} (relative risk aversion)
    r = {r} (interest rate, known)

The Euler equation is:
    C₁^(-γ) = β(1+r) E[C₂^(-γ)]

Or in terms of pricing errors:
    E[β(1+r)(C₂/C₁)^(-γ) - 1] = 0

GMM Moment Conditions (k=3, p=2, overidentified):
    E[euler_error] = 0
    E[euler_error * Y₁] = 0
    E[euler_error * Y₂] = 0

The model is overidentified, allowing a J-test of the model.

Learning objectives:
1. Understand structural estimation from economic theory
2. Practice deriving moment conditions from Euler equations
3. See how overidentification enables model testing
""",
        difficulty="intermediate",
    )


# =============================================================================
# DGP 3: Dynamic Discrete Choice Model (Advanced)
# =============================================================================

def dynamic_discrete_choice(
    n: int = 500,
    T: int = 20,
    seed: int = 42,
    theta_0: float = -1.0,
    theta_1: float = 0.5,
    beta: float = 0.9,
    RC: float = 5.0,
) -> DGPResult:
    """
    Generate data from a simple dynamic discrete choice model.
    
    This is a simplified version of Rust (1987) bus engine replacement.
    
    Model:
        State: x_t ∈ {0, 1, ..., X_max} (mileage/usage)
        Action: a_t ∈ {0, 1} (0=keep, 1=replace)
        
        Flow utility:
            u(x, a=0) = θ₀ + θ₁x + ε₀  (keep)
            u(x, a=1) = -RC + ε₁       (replace, reset x to 0)
        
        Transition:
            x' = 0 if a=1 (replace)
            x' = min(x + Δ, X_max) if a=0 (keep, usage increases)
    
    **Difficulty: Advanced**
    
    The agent solves a dynamic programming problem. Estimation uses
    conditional choice probabilities (CCPs) or nested fixed point.
    
    Args:
        n: Number of agents (default: 500)
        T: Number of time periods (default: 20)
        seed: Random seed for reproducibility (default: 42)
        theta_0: Constant in keep utility (default: -1.0)
        theta_1: Coefficient on state in keep utility (default: 0.5)
        beta: Discount factor (default: 0.9, known)
        RC: Replacement cost (default: 5.0, known)
    
    Returns:
        DGPResult with data, true parameters, and moment function
    
    Example:
        >>> from momentest.dgp import dynamic_discrete_choice
        >>> dgp = dynamic_discrete_choice(n=500, T=20, seed=42)
        >>> print(dgp.true_theta)  # [-1.0, 0.5]
    """
    rng = np.random.default_rng(seed)
    
    X_max = 10  # Maximum state
    
    # Solve the dynamic programming problem via value function iteration
    # V(x) = E[max_a {u(x,a) + β E[V(x')]}]
    
    # With Type-I extreme value errors, we have the logit formula:
    # P(a=1|x) = exp(v₁(x)) / (exp(v₀(x)) + exp(v₁(x)))
    # where v_a(x) = u(x,a) + β E[V(x'|a)]
    
    # Value function iteration
    V = np.zeros(X_max + 1)
    
    for _ in range(200):  # Iterate until convergence
        V_new = np.zeros(X_max + 1)
        
        for x in range(X_max + 1):
            # Choice-specific values (without ε)
            # Keep: utility + discounted continuation
            x_next_keep = min(x + 1, X_max)
            v_keep = theta_0 + theta_1 * x + beta * V[x_next_keep]
            
            # Replace: -RC + discounted continuation from x=0
            v_replace = -RC + beta * V[0]
            
            # Expected max with logit errors (log-sum-exp formula)
            V_new[x] = np.log(np.exp(v_keep) + np.exp(v_replace)) + 0.5772  # Euler constant
        
        if np.max(np.abs(V_new - V)) < 1e-8:
            break
        V = V_new
    
    # Compute choice probabilities
    P_replace = np.zeros(X_max + 1)
    for x in range(X_max + 1):
        x_next_keep = min(x + 1, X_max)
        v_keep = theta_0 + theta_1 * x + beta * V[x_next_keep]
        v_replace = -RC + beta * V[0]
        P_replace[x] = 1 / (1 + np.exp(v_keep - v_replace))
    
    # Simulate data
    states = []
    actions = []
    agent_ids = []
    time_ids = []
    
    for i in range(n):
        x = 0  # Start at state 0
        for t in range(T):
            states.append(x)
            agent_ids.append(i)
            time_ids.append(t)
            
            # Draw action
            a = 1 if rng.random() < P_replace[x] else 0
            actions.append(a)
            
            # Transition
            if a == 1:
                x = 0
            else:
                x = min(x + 1, X_max)
    
    states = np.array(states)
    actions = np.array(actions)
    agent_ids = np.array(agent_ids)
    time_ids = np.array(time_ids)
    
    # Moment function based on conditional choice probabilities
    def moment_func(data: Dict[str, np.ndarray], theta: np.ndarray) -> np.ndarray:
        """
        Moment conditions from CCP approach.
        
        The idea: observed choice frequencies should match model predictions.
        
        For each state x, we have:
            E[a - P(a=1|x; θ) | x] = 0
        
        We use state indicators as instruments.
        
        Returns (n_obs, k) array of moment conditions.
        """
        t0, t1 = theta
        
        # Solve for choice probabilities given theta
        # (This is a simplified version - full estimation would iterate)
        V_theta = np.zeros(X_max + 1)
        
        for _ in range(100):
            V_new = np.zeros(X_max + 1)
            for x in range(X_max + 1):
                x_next = min(x + 1, X_max)
                v_keep = t0 + t1 * x + beta * V_theta[x_next]
                v_replace = -RC + beta * V_theta[0]
                V_new[x] = np.log(np.exp(v_keep) + np.exp(v_replace)) + 0.5772
            if np.max(np.abs(V_new - V_theta)) < 1e-8:
                break
            V_theta = V_new
        
        # Choice probabilities
        P_theta = np.zeros(X_max + 1)
        for x in range(X_max + 1):
            x_next = min(x + 1, X_max)
            v_keep = t0 + t1 * x + beta * V_theta[x_next]
            v_replace = -RC + beta * V_theta[0]
            P_theta[x] = 1 / (1 + np.exp(v_keep - v_replace))
        
        # Moment conditions: a - P(a=1|x)
        x = data['state'].astype(int)
        a = data['action']
        
        # Prediction error
        pred_prob = P_theta[x]
        error = a - pred_prob
        
        # Moments: error interacted with state indicators
        moments = np.column_stack([
            error,                          # E[error] = 0
            error * data['state'],          # E[error * x] = 0
            error * (data['state'] ** 2),   # E[error * x²] = 0
        ])
        return moments
    
    return DGPResult(
        name="dynamic_discrete_choice",
        data={
            'state': states,
            'action': actions,
            'agent_id': agent_ids,
            'time': time_ids,
            'constant': np.ones(len(states)),
        },
        n=len(states),
        true_theta=np.array([theta_0, theta_1]),
        param_names=['theta_0', 'theta_1'],
        moment_function=moment_func,
        k=3,
        p=2,
        description=f"""
Dynamic Discrete Choice Model (Simplified Rust 1987)

Model:
    State: x_t ∈ {{0, 1, ..., {X_max}}} (usage/mileage)
    Action: a_t ∈ {{0, 1}} (0=keep, 1=replace)
    
    Flow utility:
        u(x, a=0) = θ₀ + θ₁x + ε₀  (keep)
        u(x, a=1) = -RC + ε₁       (replace)
    
    Transition:
        x' = 0 if a=1 (replace resets state)
        x' = min(x+1, {X_max}) if a=0 (usage increases)

Parameters:
    θ₀ = {theta_0} (constant in keep utility)
    θ₁ = {theta_1} (coefficient on state)
    β = {beta} (discount factor, known)
    RC = {RC} (replacement cost, known)

The agent solves a dynamic programming problem:
    V(x) = E[max_a {{u(x,a) + β E[V(x'|a)]}}]

With Type-I extreme value errors, choice probabilities are logit:
    P(a=1|x) = exp(v₁(x)) / (exp(v₀(x)) + exp(v₁(x)))

GMM Moment Conditions (CCP approach):
    E[a - P(a=1|x; θ)] = 0
    E[(a - P(a=1|x; θ)) * x] = 0
    E[(a - P(a=1|x; θ)) * x²] = 0

Learning objectives:
1. Understand dynamic discrete choice models
2. See how structural parameters affect behavior
3. Practice estimation with nested fixed point
""",
        difficulty="advanced",
    )


# =============================================================================
# Convenience function to load any DGP
# =============================================================================

def load_dgp(name: str, **kwargs) -> DGPResult:
    """
    Load a DGP by name.
    
    Args:
        name: DGP name (see list_dgps())
        **kwargs: Arguments passed to the DGP function
    
    Returns:
        DGPResult with data and true parameters
    
    Example:
        >>> from momentest.dgp import load_dgp, list_dgps
        >>> print(list_dgps())
        >>> dgp = load_dgp('linear_iv', n=500, seed=123)
        >>> print(dgp.true_theta)
    """
    dgps = {
        'linear_iv': linear_iv,
        'consumption_savings': consumption_savings,
        'dynamic_discrete_choice': dynamic_discrete_choice,
    }
    
    if name not in dgps:
        available = ', '.join(dgps.keys())
        raise ValueError(f"Unknown DGP: '{name}'. Available: {available}")
    
    return dgps[name](**kwargs)
