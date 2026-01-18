# Generalized Method of Moments (GMM)

GMM estimates parameters by exploiting moment conditions—equations that hold in expectation at the true parameter values. It's one of the most widely used estimation methods in econometrics.

## The Idea

You have:
- **Data** $\{y_i\}_{i=1}^n$
- **Moment conditions** $E[g(y_i, \theta_0)] = 0$ that hold at the true parameter $\theta_0$

GMM finds θ that makes the sample analog of these conditions as close to zero as possible.

## Mathematical Framework

### Moment Conditions

A moment condition is a function $g(y, \theta)$ such that:

$$E[g(y_i, \theta_0)] = 0$$

**Examples:**
- Mean: $g(y, \mu) = y - \mu$ → $E[y - \mu] = 0$ implies $\mu = E[y]$
- Variance: $g(y, \sigma^2) = (y - \bar{y})^2 - \sigma^2$
- OLS: $g(y, x, \beta) = x(y - x'\beta)$ → orthogonality condition

### The GMM Estimator

The GMM estimator minimizes:

$$\hat{\theta}_{GMM} = \arg\min_{\theta \in \Theta} \bar{g}(\theta)' W \bar{g}(\theta)$$

where:
- $\bar{g}(\theta) = \frac{1}{n} \sum_{i=1}^{n} g(y_i, \theta)$ is the sample moment
- $W$ is a positive definite weighting matrix

### Weighting Matrix

| Weighting | Matrix W | Properties |
|-----------|----------|------------|
| Identity | $W = I_k$ | Simple, consistent, inefficient |
| Optimal | $W = \hat{S}^{-1}$ | Efficient, requires first-step |

where $S = Var(\sqrt{n}\bar{g}(\theta_0))$ is the long-run variance of the moments.

**Two-step efficient GMM:**
1. Estimate with $W = I$ → $\hat{\theta}_1$
2. Estimate $\hat{S}$ using $\hat{\theta}_1$
3. Re-estimate with $W = \hat{S}^{-1}$ → $\hat{\theta}_{efficient}$

### Asymptotic Distribution

Under regularity conditions:

$$\sqrt{n}(\hat{\theta}_{GMM} - \theta_0) \xrightarrow{d} N(0, V)$$

where:

$$V = (G'WG)^{-1} G'W S W G (G'WG)^{-1}$$

- $G = E[\partial g(y, \theta) / \partial \theta']$ is the Jacobian
- $S = Var(\sqrt{n}\bar{g})$ is the moment variance

With optimal weighting ($W = S^{-1}$):

$$V = (G'S^{-1}G)^{-1}$$

## GMM vs SMM

| Aspect | GMM | SMM |
|--------|-----|-----|
| Moments | Analytical (from data) | Simulated (from model) |
| Randomness | None | Simulation draws |
| Variance | $(G'S^{-1}G)^{-1}$ | $(1+1/S)(G'S^{-1}G)^{-1}$ |
| Use case | Closed-form moments | Complex models |

**Key insight:** GMM uses the data directly to compute moments, while SMM simulates from the model. If you can write down $E[g(y,\theta)]$ analytically, use GMM. If you need to simulate, use SMM.

## Identification

**Order condition:** $k \geq p$ (moments ≥ parameters)

**Rank condition:** $G = E[\partial g / \partial \theta']$ has full column rank

| Case | k vs p | Test available? |
|------|--------|-----------------|
| Under-identified | $k < p$ | No |
| Exactly identified | $k = p$ | No (J-test has 0 df) |
| Over-identified | $k > p$ | Yes (J-test) |

## J-Test (Hansen's Test)

When over-identified ($k > p$), test model specification:

$$J = n \cdot \bar{g}(\hat{\theta})' \hat{W} \bar{g}(\hat{\theta}) \xrightarrow{d} \chi^2(k-p)$$

- **Reject:** Moment conditions are inconsistent (model misspecification)
- **Fail to reject:** Model is consistent with all moment conditions

## Example: Truncated Normal with GMM

```python
import numpy as np
import scipy.integrate as intgr
from momentest import gmm_estimate, load_econ381

# Load data
dataset = load_econ381()
data = dataset['data']
cut_lb, cut_ub = dataset['bounds']

def trunc_norm_pdf(x, mu, sigma):
    """PDF of truncated normal."""
    from scipy import stats
    prob_in_bounds = stats.norm.cdf(cut_ub, mu, sigma) - stats.norm.cdf(cut_lb, mu, sigma)
    return stats.norm.pdf(x, mu, sigma) / prob_in_bounds

def model_moments(mu, sigma):
    """Analytical moments via integration."""
    xfx = lambda x: x * trunc_norm_pdf(x, mu, sigma)
    mean_model, _ = intgr.quad(xfx, cut_lb, cut_ub)
    
    x2fx = lambda x: (x - mean_model)**2 * trunc_norm_pdf(x, mu, sigma)
    var_model, _ = intgr.quad(x2fx, cut_lb, cut_ub)
    
    return mean_model, var_model

def moment_func(data, theta):
    """
    Per-observation moment conditions.
    
    Returns (n, k) array of moment conditions for each observation.
    """
    mu, sigma = theta
    mean_mod, var_mod = model_moments(mu, sigma)
    
    n = len(data)
    moments = np.zeros((n, 2))
    
    # Moment 1: deviation from mean
    moments[:, 0] = (data - mean_mod) / mean_mod
    
    # Moment 2: deviation from variance  
    data_mean = data.mean()
    moments[:, 1] = ((data - data_mean)**2 - var_mod) / var_mod
    
    return moments

# Estimate
result = gmm_estimate(
    data=data,
    moment_func=moment_func,
    bounds=[(1, 1000), (1, 500)],
    k=2,
    weighting="optimal",
)

print(f"mu = {result.theta[0]:.2f}, sigma = {result.theta[1]:.2f}")
```

## Moment Selection

Choosing moments involves trade-offs:

**More moments (larger k):**
- ✅ More information, potentially more efficient
- ❌ More chances for misspecification
- ❌ Weighting matrix harder to estimate

**Fewer moments:**
- ✅ More robust to misspecification
- ❌ Less efficient

**Guidelines:**
1. Start with moments that have clear economic interpretation
2. Ensure moments identify all parameters (check rank condition)
3. Use J-test to check if additional moments are consistent
4. Consider the bias-variance trade-off

## Common Applications

| Application | Moment Conditions |
|-------------|-------------------|
| Asset pricing | $E[m \cdot R - 1] = 0$ (Euler equations) |
| Labor economics | $E[x(y - x'\beta)] = 0$ (IV/2SLS) |
| IO | Supply/demand moments |
| Macro | Euler equations, policy functions |

## Relationship to Other Estimators

- **OLS** is GMM with $g(y,x,\beta) = x(y - x'\beta)$
- **IV/2SLS** is GMM with $g(y,x,z,\beta) = z(y - x'\beta)$
- **MLE** can be viewed as GMM with score as moment condition

## References

- Hansen, L.P. (1982). "Large Sample Properties of Generalized Method of Moments Estimators"
- Newey, W.K. & McFadden, D. (1994). "Large Sample Estimation and Hypothesis Testing" in Handbook of Econometrics
- Hall, A.R. (2005). "Generalized Method of Moments" (textbook)
