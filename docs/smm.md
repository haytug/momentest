# Simulated Method of Moments (SMM)

SMM estimates structural model parameters by matching simulated moments to data moments. It's particularly useful when the model's likelihood function is intractable but simulation is feasible.

## The Idea

You have:
- **Data** generating some observable moments (e.g., mean income, variance of consumption)
- **A structural model** that can simulate data given parameters θ

SMM finds θ such that simulated moments match data moments as closely as possible.

## Mathematical Framework

### Setup

Let:
- $y$ = observed data of size $n$
- $\theta \in \Theta \subset \mathbb{R}^p$ = parameter vector (p parameters)
- $m(y)$ = vector of k data moments
- $\tilde{m}(\theta, u)$ = simulated moments given parameters θ and random draws u

### The SMM Estimator

The SMM estimator minimizes the weighted distance between data and simulated moments:

$$\hat{\theta}_{SMM} = \arg\min_{\theta \in \Theta} \left[ m(y) - \bar{m}(\theta) \right]' W \left[ m(y) - \bar{m}(\theta) \right]$$

where:
- $\bar{m}(\theta) = \frac{1}{S} \sum_{s=1}^{S} \tilde{m}(\theta, u_s)$ is the average of S simulations
- $W$ is a positive definite weighting matrix

### Common Random Numbers (CRN)

A key technique in SMM is using the **same random draws** across all parameter evaluations:

```python
# Fix shocks ONCE before optimization
shocks = np.random.uniform(0, 1, size=(n_sim, n_obs))

# These shocks are reused for every θ evaluation
def simulate(theta, shocks):
    return model(theta, shocks)  # Same shocks, different θ
```

**Why CRN matters:**
- Without CRN: Objective function is noisy (different draws → different moments)
- With CRN: Objective function is smooth (same draws → comparable moments)
- Smooth objectives enable gradient-based optimization

### Weighting Matrix

The choice of W affects efficiency:

| Weighting | Matrix W | Properties |
|-----------|----------|------------|
| Identity | $W = I_k$ | Simple, consistent, inefficient |
| Optimal | $W = \hat{\Sigma}^{-1}$ | Efficient, requires first-step estimate |

**Two-step procedure:**
1. Estimate with $W = I$ to get $\hat{\theta}_1$
2. Compute $\hat{\Sigma}$ at $\hat{\theta}_1$
3. Re-estimate with $W = \hat{\Sigma}^{-1}$ to get $\hat{\theta}_{optimal}$

### Asymptotic Distribution

Under regularity conditions, the SMM estimator is asymptotically normal:

$$\sqrt{n}(\hat{\theta}_{SMM} - \theta_0) \xrightarrow{d} N(0, V)$$

where the asymptotic variance is:

$$V = (1 + 1/S)(G'WG)^{-1} G'W \Sigma W G (G'WG)^{-1}$$

- $G = \partial \bar{m}(\theta) / \partial \theta'$ is the Jacobian of moments
- $\Sigma$ is the variance-covariance matrix of moments
- The $(1 + 1/S)$ term accounts for simulation noise

With optimal weighting ($W = \Sigma^{-1}$), this simplifies to:

$$V = (1 + 1/S)(G'\Sigma^{-1}G)^{-1}$$

## Identification

For SMM to work, you need:

1. **Order condition**: $k \geq p$ (at least as many moments as parameters)
2. **Rank condition**: $G$ has full column rank (moments are informative about all parameters)

| Case | Condition | Implication |
|------|-----------|-------------|
| Under-identified | $k < p$ | Cannot estimate all parameters |
| Exactly identified | $k = p$ | Unique solution, no overidentification test |
| Over-identified | $k > p$ | Can test model specification (J-test) |

## J-Test for Overidentification

When $k > p$, you can test whether the model fits all moments:

$$J = n \cdot Q(\hat{\theta}) \xrightarrow{d} \chi^2(k-p)$$

where $Q(\hat{\theta})$ is the minimized objective value.

- **Reject**: Model doesn't fit the moments well
- **Fail to reject**: Model is consistent with the data

## Example: Truncated Normal

```python
import numpy as np
from momentest import smm_estimate, load_econ381

# Load data
dataset = load_econ381()
data = dataset['data']
cut_lb, cut_ub = dataset['bounds']

# Data moments
data_mean = data.mean()
data_var = data.var()

# Simulation function
def sim_func(theta, shocks):
    """Simulate truncated normal draws."""
    mu, sigma = theta
    # Inverse CDF method using uniform shocks
    from scipy import stats
    cut_lb_cdf = stats.norm.cdf(cut_lb, mu, sigma)
    cut_ub_cdf = stats.norm.cdf(cut_ub, mu, sigma)
    scaled = shocks * (cut_ub_cdf - cut_lb_cdf) + cut_lb_cdf
    return stats.norm.ppf(scaled, mu, sigma)

# Moment function
def moment_func(sim_data):
    """Compute mean and variance for each simulation."""
    means = sim_data.mean(axis=1)
    vars = sim_data.var(axis=1)
    return np.column_stack([means, vars])

# Estimate
result = smm_estimate(
    sim_func=sim_func,
    moment_func=moment_func,
    data_moments=[data_mean, data_var],
    bounds=[(0, 1000), (1, 500)],
    n_sim=300,
    shock_dim=len(data),
    weighting="optimal",
)

print(f"mu = {result.theta[0]:.2f}, sigma = {result.theta[1]:.2f}")
```

## When to Use SMM

**Use SMM when:**
- Likelihood is intractable (complex structural models)
- Model is easy to simulate
- You have clear moment conditions

**Consider alternatives when:**
- Likelihood is available → use MLE
- Moments have closed-form expressions → use GMM
- Very high-dimensional parameter space → may need different methods

## References

- McFadden, D. (1989). "A Method of Simulated Moments for Estimation of Discrete Response Models Without Numerical Integration"
- Pakes, A. & Pollard, D. (1989). "Simulation and the Asymptotics of Optimization Estimators"
- Duffie, D. & Singleton, K. (1993). "Simulated Moments Estimation of Markov Models of Asset Prices"
