# Modified Akaike Information Criterion for Quantifying Regularization in Nonlinear Regression
*This is a failed project. I made this repo public for people not to repeat my mistake.*

[Paper version](paper.pdf)

## Overview
This project introduces the Effective Akaike Information Criterion (EAIC), a modification of the Akaike Information Criterion (AIC) designed to account for the effects of regularization in model selection. While AIC has been a cornerstone in statistical modeling, it fails to consider the effective reduction in model complexity achieved through regularization techniques like LASSO, Ridge, and Dropout in deep learning. EAIC aims to fill this gap by incorporating the effective number of parameters into the calculation, offering a more robust tool for evaluating and comparing regularized models.

## Key Concepts

### Akaike Information Criterion (AIC)
AIC balances model goodness-of-fit with complexity, penalizing models with excessive parameters[^1]:
[^1]:Hirotogu Akaike. Information theory and an extension of the maximum likelihood principle. In *Selected papers of hirotugu akaike*, pages 199–213. Springer, 1998.
```math
 \text{AIC} = 2k - 2\ln(\hat{L})
```
where:
- $` k `$: Number of parameters
- $` \hat{L} `$: Maximum likelihood value

AIC does not account for regularization, often leading to high scores for over-parameterized models that generalize well due to regularization.

### Effective Akaike Information Criterion (EAIC)
EAIC introduces the concept of the effective number of parameters ($`  k_{\text{eff}} `$), defined as those parameters exceeding a threshold $`  h `$:
```math
\mathrm{EAIC} = 2k_{\text{eff}} - 2\ln(\hat{L})
```
```math
k_{\text{eff}} = \sum_{i=0}^k I(|\beta_i| > h)
```
where $`  I(\cdot) `$ is an indicator function.

### Regularization
Regularization techniques such as LASSO (L1) and Ridge (L2) constrain parameter magnitudes to prevent overfitting, effectively reducing the active dimensionality of models.

## Objectives
- Develop EAIC to better evaluate models incorporating regularization.
- Demonstrate EAIC’s utility across different nonlinear regression settings.
- Validate EAIC using datasets like Abalone and MNIST.

## Methodology
1. **Effective Number of Parameters**:
   - Replace raw parameter counts with $`  k_{\text{eff}} `$ in AIC to reflect only significant parameters.
   - Determine $`  h `$ as a fraction of parameter standard deviation (e.g., $`  h = 0.1\sigma `$).
2. **Datasets**:
   - **Abalone Dataset**[^2]: Predict abalone age using physical measurements.
   [^2]:Nash, Warwick, Sellers, Tracy, Talbot, Simon, Cawthorn, Andrew, Ford, and Wes. Abalone. UCI Machine Learning Repository, 1995. DOI: https://doi.org/10.24432/C55C7W.
   - **MNIST Dataset**[^3]: Evaluate nonlinear regression using digit classification.
   [^3]:Li Deng. The mnist database of handwritten digit images for machine learning research. *IEEE Signal Processing Magazine*, 29(6):141–142, 2012.
3. **Evaluation**:
   - Compare AIC and EAIC across models like Linear Regression, LASSO, and Ridge.
   - Assess correlations between information criteria and generalization metrics like test Mean Squared Error (MSE).

## Results
### Abalone Dataset
- **Linear Models**: EAIC correlated slightly better with test MSE compared to AIC.
- **Regularized Models**: EAIC highlighted the impact of regularization more effectively.

| **Model**  | **AIC**   | **EAIC**  | **Test MSE** |
|------------|-----------|-----------|---------------|
| LASSO      | 15084.89  | 15076.89  | 5.7690        |
| Ridge      | 14956.28  | 14952.28  | 5.6247        |
| Linear     | 14825.98  | 14823.98  | 5.3125        |



| **Metric**                                | **Correlation(%)** |
|-------------------------------------------|---------------------|
| Correlation between AIC and Test MSE      | 97.8983            |
| Correlation between EAIC and Test MSE     | 97.9921            |

### MNIST Dataset
- **Deep Learning Models**: Both EAIC and AIC demonstrated its potential to quantify effective complexity in over-parameterized settings.

| **Model**   | **Test Accuracy (%)** | **AIC**      | **EAIC**      | **Total Params** | **Effective Params** |
|-------------|------------------------|--------------|---------------|-------------------|-----------------------|
| SNN         | 95.17                 | 222926.91    | 204022.91     | 101770            | 92318                 |
| DNN         | 97.04                 | 493637.43    | 453629.43     | 242762            | 222758                |
| DNN-Reg     | 96.99                 | 494638.56    | 451640.56     | 242762            | 221263                |


| **Metric**                                | **Correlation (%)** |
|-------------------------------------------|----------------------|
| Correlation between AIC and Test Accuracy | 99.96               |
| Correlation between EAIC and Test Accuracy| 99.99               |


## Conclusion
EAIC offers a promising extension to AIC by incorporating effective parameter counts, providing a nuanced measure of model complexity under regularization. While traditional AIC remains suitable for simpler models, EAIC excels in modern applications involving high-dimensional and regularized models. Further refinement and validation are needed to optimize its application across diverse scenarios.

## BibTeX
```
@software{eom_2025_software,
  author       = {Sungjun Eom},
  title        = {Modified Akaike Information Criterion for Quantifying Regularization in Nonlinear Regression},
  year         = {2025},
  url          = {https://github.com/SungjunEom/eaic_dnn},
  note         = {Released under the MIT License},
}

@article{eom_2025_article,
  author       = {Sungjun Eom},
  title        = {Modified Akaike Information Criterion for Quantifying Regularization in Nonlinear Regression},
  year         = {2025},
  url          = {https://github.com/SungjunEom/eaic_dnn},
  note         = {Paper version.}
}
```