"""
Built-in datasets for momentest examples and learning.

This module provides educational datasets for learning GMM and SMM estimation.
Each dataset includes:
- Real data from classic econometric studies
- Documentation explaining the economic context
- Suggested moment conditions
- Benchmark estimates from the literature
- Exercises for self-study

Datasets:
---------
- econ381: Test scores (truncated normal) - good for learning basics
- consumption: FRED real consumption (Hall 1978) - classic GMM application  
- asset_pricing: Ken French returns + consumption (Hansen-Singleton 1982)

Usage:
------
>>> from momentest import load_consumption, list_datasets
>>> print(list_datasets())
>>> dataset = load_consumption()
>>> print(dataset['description'])
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class DatasetBundle:
    """
    Container for educational datasets.
    
    Attributes:
        name: Dataset identifier
        data: Dictionary of data arrays
        n: Number of observations
        description: Full description of the dataset and economic context
        moment_conditions: Description of suggested moment conditions
        benchmark_params: Benchmark estimates from the literature
        exercises: List of suggested exercises
        references: Academic references
        difficulty: 'beginner', 'intermediate', or 'advanced'
    """
    name: str
    data: Dict[str, np.ndarray]
    n: int
    description: str
    moment_conditions: str
    benchmark_params: Optional[Dict[str, float]]
    exercises: List[str]
    references: List[str]
    difficulty: str
    
    # Alias for compatibility
    @property
    def true_params(self):
        return self.benchmark_params
    
    def __repr__(self):
        return f"DatasetBundle(name='{self.name}', n={self.n}, difficulty='{self.difficulty}')"
    
    def info(self):
        """Print detailed information about the dataset."""
        print("=" * 70)
        print(f"Dataset: {self.name}")
        print("=" * 70)
        print(f"\nDifficulty: {self.difficulty}")
        print(f"Observations: {self.n}")
        print(f"\nDescription:\n{self.description}")
        print(f"\nMoment Conditions:\n{self.moment_conditions}")
        if self.benchmark_params:
            print(f"\nBenchmark Parameters:")
            for k, v in self.benchmark_params.items():
                print(f"  {k}: {v}")
        print(f"\nExercises:")
        for i, ex in enumerate(self.exercises, 1):
            print(f"  {i}. {ex}")
        print(f"\nReferences:")
        for ref in self.references:
            print(f"  - {ref}")
        print("=" * 70)


def list_datasets() -> List[str]:
    """
    List all available built-in datasets.
    
    Returns:
        List of dataset names
    
    Example:
        >>> from momentest import list_datasets
        >>> print(list_datasets())
        ['econ381', 'consumption', 'labor_supply', 'asset_pricing']
    """
    return ['econ381', 'consumption', 'labor_supply', 'asset_pricing']


def _get_data_path(filename: str) -> Path:
    """Get path to data file."""
    return Path(__file__).parent / "data" / filename


# =============================================================================
# Dataset: Econ 381 Test Scores (Truncated Normal)
# =============================================================================

def load_econ381() -> Dict[str, Any]:
    """
    Load the Econ 381 test scores dataset.
    
    This dataset contains 161 intermediate macroeconomics test scores
    from the OpenSourceEcon tutorial. Scores are bounded between 0 and 450,
    making it ideal for learning truncated normal estimation.
    
    **Difficulty: Beginner**
    
    Returns:
        dict with keys:
            - 'data': numpy array of test scores (n=161)
            - 'n': number of observations
            - 'bounds': (lower, upper) truncation bounds
            - 'mle_params': MLE estimates (mu, sigma) from the tutorial
            - 'description': dataset description
    
    Example:
        >>> from momentest import load_econ381
        >>> dataset = load_econ381()
        >>> data = dataset['data']
        >>> print(f"N={dataset['n']}, Mean={data.mean():.2f}")
        N=161, Mean=341.91
    """
    data_path = _get_data_path("Econ381totpts.txt")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please ensure the momentest package is installed correctly."
        )
    
    data = np.loadtxt(data_path)
    
    return {
        'data': data,
        'n': len(data),
        'bounds': (0.0, 450.0),
        'mle_params': {'mu': 622.16, 'sigma': 198.76},
        'description': (
            "Econ 381 intermediate macroeconomics test scores (2011-2012). "
            "Scores are bounded between 0 and 450. "
            "Source: OpenSourceEcon CompMethods tutorial."
        ),
    }


def load_econ381_bundle() -> DatasetBundle:
    """
    Load Econ 381 dataset as a full DatasetBundle with exercises.
    
    Returns:
        DatasetBundle with data, documentation, and exercises
    """
    raw = load_econ381()
    
    return DatasetBundle(
        name="econ381",
        data={'scores': raw['data']},
        n=raw['n'],
        description="""
Econ 381 Intermediate Macroeconomics Test Scores

This dataset contains 161 test scores from an intermediate macroeconomics
course. Scores are bounded between 0 and 450 points.

The data exhibits a truncated normal distribution - scores cannot go below
0 or above 450, but the underlying "ability" distribution is normal.

Economic interpretation: Students have latent ability drawn from N(μ, σ²),
but observed scores are truncated at the test bounds.

This is an ideal first dataset for learning moment estimation because:
1. The model (truncated normal) is simple and well-understood
2. MLE provides a benchmark for comparison
3. You can experiment with different moment conditions
""",
        moment_conditions="""
Suggested moment conditions:

1. Two moments (exactly identified):
   - E[x] = μ_truncated(μ, σ)  [mean]
   - E[(x - x̄)²] = σ²_truncated(μ, σ)  [variance]

2. Four moments (overidentified):
   - P(x < 220) = F_truncated(220; μ, σ)
   - P(220 ≤ x < 320) = F_truncated(320) - F_truncated(220)
   - P(320 ≤ x < 430) = F_truncated(430) - F_truncated(320)
   - P(x ≥ 430) = 1 - F_truncated(430)

The 4-moment specification allows testing model fit via J-test.
""",
        benchmark_params={'mu': 622.16, 'sigma': 198.76},
        exercises=[
            "Estimate (μ, σ) using GMM with mean and variance moments",
            "Compare GMM estimates to MLE - why are they similar?",
            "Add bin percentage moments and re-estimate with 4 moments",
            "Compute the J-test - does the truncated normal fit well?",
            "Try SMM: simulate from truncated normal and match moments",
            "Compare identity vs optimal weighting - which is more efficient?",
        ],
        references=[
            "OpenSourceEcon CompMethods: https://opensourceecon.github.io/CompMethods/",
        ],
        difficulty="beginner",
    )


# =============================================================================
# Dataset: Consumption (Hall 1978 - Random Walk Test)
# =============================================================================

def load_consumption() -> DatasetBundle:
    """
    Load real U.S. consumption data for GMM estimation.
    
    This dataset contains quarterly real personal consumption expenditures
    from FRED (1947-2025). It's used to test the Hall (1978) random walk
    hypothesis of consumption.
    
    **Difficulty: Intermediate**
    
    Model:
        ΔC_t = α + ε_t  (random walk with drift)
        
    Under the permanent income hypothesis with rational expectations,
    consumption changes should be unpredictable - lagged variables
    should have no predictive power.
    
    Source: FRED series PCECC96 (Real Personal Consumption Expenditures,
    Billions of Chained 2017 Dollars, Quarterly, Seasonally Adjusted)
    
    Returns:
        DatasetBundle with consumption data
    
    Example:
        >>> from momentest import load_consumption
        >>> dataset = load_consumption()
        >>> dataset.info()
    """
    data_path = _get_data_path("consumption_quarterly.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please ensure the momentest package is installed correctly."
        )
    
    # Load data
    raw = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
    dates = [row[0] for row in raw]
    consumption = np.array([float(row[1]) for row in raw])
    
    # Compute consumption growth (C_t / C_{t-1})
    c_growth = consumption[1:] / consumption[:-1]
    
    # Create lagged variables for GMM
    # We need: C_growth_t, C_growth_{t-1}, C_growth_{t-2}
    n = len(c_growth) - 2  # Drop first 2 to have lags
    
    c_growth_t = c_growth[2:]           # Current growth
    c_growth_lag1 = c_growth[1:-1]      # Lagged once
    c_growth_lag2 = c_growth[:-2]       # Lagged twice
    
    return DatasetBundle(
        name="consumption",
        data={
            'c_growth': c_growth_t,
            'c_growth_lag1': c_growth_lag1,
            'c_growth_lag2': c_growth_lag2,
            'constant': np.ones(n),
        },
        n=n,
        description="""
U.S. Real Personal Consumption Expenditures (1947-2025)

This dataset contains quarterly real consumption from FRED (series PCECC96).
It's used to test the Hall (1978) random walk hypothesis.

The permanent income hypothesis (PIH) with rational expectations implies:
    C_t = C_{t-1} + ε_t  (consumption is a random walk)

Or equivalently:
    ΔC_t = ε_t  (consumption changes are unpredictable)

This means lagged consumption growth should NOT predict current growth.
GMM can test this by checking if lagged variables are orthogonal to
the consumption innovation.

Key insight: If the PIH holds, regressing ΔC_t on ΔC_{t-1} should give
a coefficient of zero. GMM formalizes this as moment conditions.

Data source: FRED PCECC96
- Real Personal Consumption Expenditures
- Billions of Chained 2017 Dollars
- Quarterly, Seasonally Adjusted Annual Rate
- 1947Q1 to 2025Q3
""",
        moment_conditions="""
GMM Moment Conditions (Testing Random Walk):

Model: C_growth_t = α + ε_t

If consumption is a random walk, ε_t should be orthogonal to past info.

1. Basic test (k=2, p=1):
   E[ε_t] = 0
   E[ε_t * c_growth_{t-1}] = 0

   This tests if lagged growth predicts current growth.

2. Extended test (k=3, p=1):
   E[ε_t] = 0
   E[ε_t * c_growth_{t-1}] = 0
   E[ε_t * c_growth_{t-2}] = 0

Implementation:
    def moment_func(data, theta):
        alpha = theta[0]
        eps = data['c_growth'] - alpha
        
        moments = np.column_stack([
            eps,                        # E[ε] = 0
            eps * data['c_growth_lag1'], # E[ε * lag1] = 0
            eps * data['c_growth_lag2'], # E[ε * lag2] = 0
        ])
        return moments

Alternative: Test AR(1) model
    C_growth_t = α + β * C_growth_{t-1} + ε_t
    
    H0: β = 0 (random walk)
""",
        benchmark_params={'alpha': 1.008},  # ~0.8% quarterly growth
        exercises=[
            "Estimate mean consumption growth α using GMM",
            "Test if lagged growth predicts current growth (should be ~0)",
            "Estimate AR(1) model and test H0: β = 0",
            "Use J-test to check if all moment conditions hold",
            "Try different sample periods - does the random walk hold always?",
            "Compare pre-1980 vs post-1980 (Great Moderation)",
        ],
        references=[
            "Hall, R.E. (1978). Stochastic Implications of the Life Cycle-Permanent Income Hypothesis. JPE.",
            "FRED: https://fred.stlouisfed.org/series/PCECC96",
        ],
        difficulty="intermediate",
    )


# =============================================================================
# Dataset: Asset Pricing (Hansen-Singleton 1982)
# =============================================================================

def load_labor_supply() -> DatasetBundle:
    """
    Load PSID 1976 labor supply data for GMM estimation.
    
    This dataset is the classic Mroz (1987) extract from the Panel Study of
    Income Dynamics (PSID), containing 753 married women's labor supply data.
    It's widely used for labor supply estimation and selection models.
    
    **Difficulty: Intermediate**
    
    Model (Labor Supply):
        ln(hours) = α + γ * ln(wage) + β * X + ε
        
    where:
        - hours is annual hours worked
        - wage is hourly wage
        - X are demographic controls (age, education, kids, experience)
        - γ is the wage elasticity of labor supply
    
    Key identification challenge: Wages are endogenous (ability affects both).
    Solution: Use husband's characteristics as instruments.
    
    Source: Mroz (1987) extract from PSID, via R AER package
    
    Returns:
        DatasetBundle with labor supply data
    
    Example:
        >>> from momentest import load_labor_supply
        >>> dataset = load_labor_supply()
        >>> dataset.info()
    """
    data_path = _get_data_path("psid1976.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please ensure the momentest package is installed correctly."
        )
    
    # Load data
    raw = np.genfromtxt(data_path, delimiter=',', skip_header=1, 
                        dtype=None, encoding='utf-8')
    
    # Extract columns (see header for column order)
    # rownames,participation,hours,youngkids,oldkids,age,education,wage,...
    participation = np.array([1 if row[1] == 'yes' else 0 for row in raw])
    hours = np.array([float(row[2]) for row in raw])
    youngkids = np.array([float(row[3]) for row in raw])
    oldkids = np.array([float(row[4]) for row in raw])
    age = np.array([float(row[5]) for row in raw])
    education = np.array([float(row[6]) for row in raw])
    wage = np.array([float(row[7]) for row in raw])
    hhours = np.array([float(row[9]) for row in raw])  # husband's hours
    hage = np.array([float(row[10]) for row in raw])   # husband's age
    heducation = np.array([float(row[11]) for row in raw])  # husband's education
    hwage = np.array([float(row[12]) for row in raw])  # husband's wage
    fincome = np.array([float(row[13]) for row in raw])  # family income
    experience = np.array([float(row[19]) for row in raw])  # work experience
    
    # Filter to working women (positive hours and wages)
    working = (hours > 0) & (wage > 0)
    
    n = int(np.sum(working))
    
    # Create log variables for working sample
    log_hours = np.log(hours[working])
    log_wage = np.log(wage[working])
    log_hwage = np.log(hwage[working])
    
    return DatasetBundle(
        name="labor_supply",
        data={
            'log_hours': log_hours,
            'log_wage': log_wage,
            'hours': hours[working],
            'wage': wage[working],
            'age': age[working],
            'education': education[working],
            'experience': experience[working],
            'experience_sq': experience[working]**2,
            'youngkids': youngkids[working],
            'oldkids': oldkids[working],
            'heducation': heducation[working],
            'hage': hage[working],
            'log_hwage': log_hwage,
            'fincome': fincome[working],
            'constant': np.ones(n),
        },
        n=n,
        description="""
PSID 1976 Labor Supply Data (Mroz 1987)

This is the classic Mroz (1987) extract from the Panel Study of Income
Dynamics (PSID). It contains data on 753 married women, of which 428
were working (positive hours and wages).

The dataset is widely used for:
1. Labor supply estimation (wage elasticity)
2. Sample selection models (Heckman correction)
3. IV/GMM examples in econometrics courses

Variables:
- hours: Annual hours worked
- wage: Hourly wage ($/hour)
- age: Woman's age
- education: Years of schooling
- experience: Years of work experience
- youngkids: Number of children < 6 years old
- oldkids: Number of children 6-18 years old
- heducation: Husband's years of schooling
- hage: Husband's age
- hwage: Husband's hourly wage
- fincome: Family income ($)

The working sample (hours > 0, wage > 0) has 428 observations.

Key econometric issues:
1. Selection bias: We only observe wages for workers
2. Endogeneity: Wages may be correlated with unobserved ability
3. Measurement error: Self-reported hours and wages

Classic instruments: Husband's characteristics (heducation, hage, hwage)
are correlated with wife's wage (assortative mating) but arguably
uncorrelated with wife's labor supply error term.
""",
        moment_conditions="""
GMM Moment Conditions (IV Estimation):

Model: ln(hours) = α + γ * ln(wage) + β₁ * age + β₂ * education 
                   + β₃ * experience + β₄ * experience² + ε

The residual ε should be orthogonal to instruments Z.

1. Just identified (k=6, p=6):
   E[ε * 1] = 0              (constant)
   E[ε * heducation] = 0     (husband's education as IV for wage)
   E[ε * age] = 0            (exogenous)
   E[ε * education] = 0      (exogenous)
   E[ε * experience] = 0     (exogenous)
   E[ε * experience²] = 0    (exogenous)

2. Overidentified (k=7, p=6):
   Add: E[ε * hage] = 0      (husband's age)

Implementation:
    def moment_func(data, theta):
        alpha, gamma, b_age, b_edu, b_exp, b_exp2 = theta
        
        # Residual
        eps = (data['log_hours'] - alpha 
               - gamma * data['log_wage']
               - b_age * data['age']
               - b_edu * data['education']
               - b_exp * data['experience']
               - b_exp2 * data['experience_sq'])
        
        moments = np.column_stack([
            eps,                          # E[ε] = 0
            eps * data['heducation'],     # E[ε * hedu] = 0 (IV)
            eps * data['age'],            # E[ε * age] = 0
            eps * data['education'],      # E[ε * edu] = 0
            eps * data['experience'],     # E[ε * exp] = 0
            eps * data['experience_sq'],  # E[ε * exp²] = 0
        ])
        return moments

Note: This ignores selection (non-workers). For selection correction,
see Heckman (1979) two-step estimator.
""",
        benchmark_params={
            'gamma': 0.1,  # Typical wage elasticity estimate ~0.1-0.3
        },
        exercises=[
            "Estimate the wage elasticity γ using OLS - what do you get?",
            "Use husband's education as an IV for wife's wage - how does γ change?",
            "Add husband's age as an additional instrument and test overidentification",
            "Does the J-test reject? What might this mean about the instruments?",
            "Estimate separately for women with/without young children",
            "Compare identity vs optimal weighting - which is more efficient?",
        ],
        references=[
            "Mroz, T.A. (1987). The Sensitivity of an Empirical Model of Married Women's Hours of Work to Economic and Statistical Assumptions. Econometrica.",
            "Heckman, J.J. (1979). Sample Selection Bias as a Specification Error. Econometrica.",
            "Killingsworth, M.R. & Heckman, J.J. (1986). Female Labor Supply: A Survey. Handbook of Labor Economics.",
            "Data source: R AER package, https://vincentarelbundock.github.io/Rdatasets/",
        ],
        difficulty="intermediate",
    )


def load_asset_pricing() -> DatasetBundle:
    """
    Load real asset returns and consumption data for GMM estimation.
    
    This dataset combines:
    - Ken French market returns (1926-2025, monthly)
    - FRED consumption growth (1947-2025, quarterly)
    
    Used to estimate the Consumption CAPM (CCAPM) following
    Hansen and Singleton (1982).
    
    **Difficulty: Advanced**
    
    Model (CCAPM Euler equation):
        E[β * (C_{t+1}/C_t)^{-γ} * R_{t+1} - 1 | I_t] = 0
        
    where:
        - β is the discount factor
        - γ is relative risk aversion
        - R_{t+1} is the gross return
        - C_{t+1}/C_t is consumption growth
    
    Returns:
        DatasetBundle with returns and consumption data
    
    Example:
        >>> from momentest import load_asset_pricing
        >>> dataset = load_asset_pricing()
        >>> dataset.info()
    """
    # Load Ken French data
    ff_path = _get_data_path("ff_factors_monthly.csv")
    cons_path = _get_data_path("consumption_quarterly.csv")
    
    if not ff_path.exists() or not cons_path.exists():
        raise FileNotFoundError(
            "Data files not found. Please ensure momentest is installed correctly."
        )
    
    # Load monthly returns
    ff_raw = np.genfromtxt(ff_path, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
    ff_dates = [str(row[0]) for row in ff_raw]
    mkt_rf = np.array([float(row[1]) for row in ff_raw])  # Market excess return (%)
    rf = np.array([float(row[2]) for row in ff_raw])       # Risk-free rate (%)
    
    # Convert to gross returns (from percentage)
    gross_mkt = 1 + (mkt_rf + rf) / 100  # Market return
    gross_rf = 1 + rf / 100               # Risk-free return
    
    # Load quarterly consumption
    cons_raw = np.genfromtxt(cons_path, delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
    cons_dates = [row[0] for row in cons_raw]
    consumption = np.array([float(row[1]) for row in cons_raw])
    c_growth = consumption[1:] / consumption[:-1]
    
    # Convert monthly returns to quarterly (compound 3 months)
    # Start from 1947Q2 to match consumption growth
    # Find index for 194704 (April 1947)
    start_idx = None
    for i, d in enumerate(ff_dates):
        if d.startswith('1947') and int(d[4:6]) >= 4:
            start_idx = i
            break
    
    if start_idx is None:
        start_idx = 0
    
    # Compound monthly to quarterly
    n_quarters = min(len(c_growth), (len(ff_dates) - start_idx) // 3)
    
    quarterly_mkt = np.zeros(n_quarters)
    quarterly_rf = np.zeros(n_quarters)
    
    for q in range(n_quarters):
        m_start = start_idx + q * 3
        m_end = m_start + 3
        if m_end <= len(gross_mkt):
            quarterly_mkt[q] = np.prod(gross_mkt[m_start:m_end])
            quarterly_rf[q] = np.prod(gross_rf[m_start:m_end])
    
    # Align with consumption growth (both start 1947Q2)
    n = min(len(c_growth), n_quarters)
    c_growth = c_growth[:n]
    quarterly_mkt = quarterly_mkt[:n]
    quarterly_rf = quarterly_rf[:n]
    
    # Create lagged instruments
    n_final = n - 1  # Drop first obs for lags
    
    return DatasetBundle(
        name="asset_pricing",
        data={
            'consumption_growth': c_growth[1:],
            'gross_return': quarterly_mkt[1:],
            'gross_rf': quarterly_rf[1:],
            'consumption_growth_lag': c_growth[:-1],
            'return_lag': quarterly_mkt[:-1],
            'constant': np.ones(n_final),
        },
        n=n_final,
        description="""
U.S. Asset Returns and Consumption Growth (1947-2025)

This dataset combines:
1. Ken French market returns (compounded to quarterly)
2. FRED real consumption growth (quarterly)

Used to estimate the Consumption CAPM (CCAPM) via GMM, following
Hansen and Singleton (1982).

The CCAPM Euler equation states:
    E[β * (C_{t+1}/C_t)^{-γ} * R_{t+1} | I_t] = 1

where:
- β is the subjective discount factor (patience)
- γ is the coefficient of relative risk aversion
- C_{t+1}/C_t is consumption growth
- R_{t+1} is the gross return

Intuition: The stochastic discount factor m = β * (C/C')^{-γ} prices
all assets. High consumption growth → low marginal utility → low SDF →
assets that pay off in good times must offer higher returns.

The "equity premium puzzle" (Mehra-Prescott 1985): To match the high
equity premium (~6% annually), you need implausibly high γ (>10).

Data sources:
- Ken French Data Library: Market returns, risk-free rate
- FRED PCECC96: Real consumption
""",
        moment_conditions="""
GMM Moment Conditions (Euler Equations):

Define the pricing error for asset i:
    u_t(β, γ) = β * (C_{t+1}/C_t)^{-γ} * R_{i,t+1} - 1

Under rational expectations, E[u_t | I_t] = 0.

1. Basic (k=2, p=2) - just identified:
   E[u_t^{mkt}] = 0  (market return)
   E[u_t^{rf}] = 0   (risk-free rate)

2. With instruments (k=4, p=2) - overidentified:
   E[u_t^{mkt}] = 0
   E[u_t^{rf}] = 0
   E[u_t^{mkt} * c_growth_lag] = 0
   E[u_t^{mkt} * return_lag] = 0

Implementation:
    def moment_func(data, theta):
        beta, gamma = theta
        cg = data['consumption_growth']
        R_mkt = data['gross_return']
        R_rf = data['gross_rf']
        
        # Stochastic discount factor
        sdf = beta * cg**(-gamma)
        
        # Pricing errors
        u_mkt = sdf * R_mkt - 1
        u_rf = sdf * R_rf - 1
        
        moments = np.column_stack([
            u_mkt,
            u_rf,
            u_mkt * data['consumption_growth_lag'],
            u_mkt * data['return_lag'],
        ])
        return moments
""",
        benchmark_params={'beta': 0.99, 'gamma': 2.0},  # Standard values (but puzzle!)
        exercises=[
            "Estimate (β, γ) using the two Euler equations",
            "What γ do you need to match the equity premium? (Hint: very high)",
            "Add lagged instruments and test overidentification",
            "Try different sample periods - is γ stable?",
            "The J-test likely rejects - why? (model misspecification)",
            "Compare to Epstein-Zin preferences (separates risk aversion from EIS)",
        ],
        references=[
            "Hansen, L.P. & Singleton, K.J. (1982). GMM Estimation of Intertemporal Asset Pricing. Econometrica.",
            "Mehra, R. & Prescott, E.C. (1985). The Equity Premium: A Puzzle. JME.",
            "Ken French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
            "FRED: https://fred.stlouisfed.org/series/PCECC96",
        ],
        difficulty="advanced",
    )


# =============================================================================
# Convenience function to load any dataset
# =============================================================================

def load_dataset(name: str) -> DatasetBundle:
    """
    Load a dataset by name.
    
    Args:
        name: Dataset name (see list_datasets())
    
    Returns:
        DatasetBundle with data and documentation
    
    Example:
        >>> from momentest import load_dataset, list_datasets
        >>> print(list_datasets())
        >>> dataset = load_dataset('consumption')
        >>> dataset.info()
    """
    loaders = {
        'econ381': load_econ381_bundle,
        'consumption': load_consumption,
        'labor_supply': load_labor_supply,
        'asset_pricing': load_asset_pricing,
    }
    
    if name not in loaders:
        available = ', '.join(loaders.keys())
        raise ValueError(f"Unknown dataset: '{name}'. Available: {available}")
    
    return loaders[name]()
