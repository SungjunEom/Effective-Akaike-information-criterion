# Modified Akaike Information Criterion for Quantifying Regularization in Nonlinear Regression
*This is a failed project.*


## Overview
This project introduces the Effective Akaike Information Criterion (EAIC), a modification of the Akaike Information Criterion (AIC) designed to account for the effects of regularization in model selection. While AIC has been a cornerstone in statistical modeling, it fails to consider the effective reduction in model complexity achieved through regularization techniques like LASSO, Ridge, and Dropout in deep learning. EAIC aims to fill this gap by incorporating the effective number of parameters into the calculation, offering a more robust tool for evaluating and comparing regularized models.

## Key Concepts

### Akaike Information Criterion (AIC)
AIC balances model goodness-of-fit with complexity, penalizing models with excessive parameters:
$$ \text{AIC} = 2k - 2\ln(\hat{L}) $$
where:
- $` k `$: Number of parameters
- $ \hat{L} $: Maximum likelihood value

AIC does not account for regularization, often leading to high scores for over-parameterized models that generalize well due to regularization.

### Effective Akaike Information Criterion (EAIC)
EAIC introduces the concept of the effective number of parameters ($  k_{\text{eff}} $), defined as those parameters exceeding a threshold $  h $:
\[ \mathrm{EAIC} = 2k_{\text{eff}} - 2\ln(\hat{L}) \]
\[ k_{\text{eff}} = \sum_{i=0}^k I(|\beta_i| > h) \]
where $  I(\cdot) $ is an indicator function.

### Regularization
Regularization techniques such as LASSO (L1) and Ridge (L2) constrain parameter magnitudes to prevent overfitting, effectively reducing the active dimensionality of models.

## Objectives
- Develop EAIC to better evaluate models incorporating regularization.
- Demonstrate EAIC’s utility across different nonlinear regression settings.
- Validate EAIC using datasets like Abalone and MNIST.

## Methodology
1. **Effective Number of Parameters**:
   - Replace raw parameter counts with $  k_{\text{eff}} $ in AIC to reflect only significant parameters.
   - Determine $  h $ as a fraction of parameter standard deviation (e.g., $  h = 0.1\sigma $).
2. **Datasets**:
   - **Abalone Dataset**: Predict abalone age using physical measurements.
   - **MNIST Dataset**: Evaluate nonlinear regression using digit classification.
3. **Evaluation**:
   - Compare AIC and EAIC across models like Linear Regression, LASSO, and Ridge.
   - Assess correlations between information criteria and generalization metrics like test Mean Squared Error (MSE).

## Results
### Abalone Dataset
- **Linear Models**: EAIC correlated slightly better with test MSE compared to AIC.
- **Regularized Models**: EAIC highlighted the impact of regularization more effectively.

### MNIST Dataset
- **Deep Learning Models**: EAIC demonstrated its potential to quantify effective complexity in over-parameterized settings.

## Conclusion
EAIC offers a promising extension to AIC by incorporating effective parameter counts, providing a nuanced measure of model complexity under regularization. While traditional AIC remains suitable for simpler models, EAIC excels in modern applications involving high-dimensional and regularized models. Further refinement and validation are needed to optimize its application across diverse scenarios.

