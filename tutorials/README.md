# momentest Tutorials

Interactive Jupyter notebooks for learning GMM and SMM estimation with `momentest`.

## Tutorial Overview

| Tutorial | Topic | Difficulty | Time |
|----------|-------|------------|------|
| 01 | GMM Basics | Beginner | 30 min |
| 02 | SMM Basics | Beginner | 30 min |
| 03 | Optimal Weighting | Intermediate | 20 min |
| 04 | Bootstrap Inference | Intermediate | 25 min |
| 05 | Diagnostics | Intermediate | 25 min |
| 06 | Advanced Models | Advanced | 40 min |
| 07 | Real Data Applications | Advanced | 45 min |

## Getting Started

1. Install momentest:
   ```bash
   pip install momentest
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Open `01_gmm_basics.ipynb` and follow along!

## Tutorial Descriptions

### 01_gmm_basics.ipynb
Introduction to Generalized Method of Moments (GMM). Covers:
- What GMM is and when to use it
- Defining moment conditions
- Linear IV estimation example
- Identity vs optimal weighting
- J-test for overidentification

### 02_smm_basics.ipynb
Introduction to Simulated Method of Moments (SMM). Covers:
- When to use SMM vs GMM
- Simulation and moment functions
- Common Random Numbers (CRN)
- Truncated normal estimation example

### 03_optimal_weighting.ipynb
Deep dive into weighting matrices. Covers:
- Why weighting matters
- Identity vs optimal weighting
- Two-step estimation procedure
- Efficiency gains

### 04_bootstrap_inference.ipynb
Bootstrap methods for inference. Covers:
- Why bootstrap?
- Bootstrap standard errors
- Percentile confidence intervals
- Comparing asymptotic vs bootstrap SE

### 05_diagnostics.ipynb
Diagnostic tools and visualization. Covers:
- Objective function landscape
- Moment contributions
- Identification analysis
- J-test interpretation
- Convergence diagnostics

### 06_advanced_models.ipynb
Advanced structural models. Covers:
- Consumption-savings (Euler equations)
- Dynamic discrete choice (Rust-style)
- Tips for complex models

### 07_real_data_applications.ipynb
GMM and SMM with real datasets. Covers:
- Labor supply estimation with PSID data (Mroz 1987)
- Instrumental variables for wage elasticity
- Overidentification testing with J-test
- SMM for income dynamics with measurement error
- Subsample analysis and robustness checks

## Prerequisites

- Basic Python and NumPy
- Introductory econometrics (OLS, IV)
- For advanced tutorials: dynamic programming basics

## References

- Hansen (1982): GMM theory
- McFadden (1989): SMM foundations
- Rust (1987): Dynamic discrete choice
- Hall & Horowitz (1996): Bootstrap for GMM
