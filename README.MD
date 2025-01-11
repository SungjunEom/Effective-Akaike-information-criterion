# Modified Akaike Information Criterion for Quantifying Regularization in Nonlinear Regression
*This is a failed project.*

## Introduction
Model selection is a crucial step in statistical modeling. It is about balancing the trade-off between model complexity and goodness-of-fit. One of the most widely used criteria is the Akaike Information Criterion (AIC). It calculates the likelihood penalized by the number of parameters to discourage overfitting, or over-parameterization. Smaller AIC value is assumed to be better. On the other hand, in modern applications such as deep learning, the models often contain a large number of parameters so that models constructed by deep learning are always considered over-parameterized. Although they are over-parameterized, the deep learning models generalize well because of the regularization techniques, such as dropout, L1 (LASSO) and L2 (ridge) regularization. If one tries to calculate the AIC for these highly over-parameterized models, it always gives a high score, meaning that the models are over-parameterized. In this respect, the AIC is not a suitable measure for models which are over-parameterized but generalize well. AIC does not inherently account for regularization, as it penlizes models based purely on the number of parameters rather than their effective complexity after regularization.

This research aims to address this gap by proposing a modification to AIC that incorporates the effects of regularization. Specifically, we introduced the effective number of parameters whose absolute values exceed a certain threshold. This approach reflects the fact that regularization can reduce the number of active parameters in a model. By introducing the effective number of parameters into AIC, we want to improve its applicability in regularized models. We named this modified AIC as Effective Akaike Information Criterion (EAIC) meaning that it reflects the "effective" number of parameters.

The objective of this study is to validate the theory of EAIC through empirical analysis. We will investigate the properties of EAIC with various nonlinear regression techniques, such as polynomial regression, domain-knowledge based nonlinear regression, and deep learning. By incorporating regularization into AIC, this research seeks to provide a more robust tool for model selection while considering regularization.


## Related Work
### Akaike Information Criterion
The AIC has long been a pivotal tool in model selection, particularly in contexts where statistical inference and predictive accuracy are crucial. Developed by Akaike \cite{akaike1998information}, AIC provides a method for balancing the trade-off between model complexity and goodness-of-fit. It operates under the principle of parsimony, where the aim is to select a model that adequately explains the data with the fewest parameters. AIC is calculated as a combination of the likelihood of the model and a penalty term based on the number of estimated parameters \eqref{eq:aic}.

$$\text{AIC} &= 2k - 2\ln(\hat{L})$$

$\hat{L}$ denotes the maximized value of the likelihood function and $k$ denotes the number of parameters.
This formulation effectively discourages over-fitting by penalizing models with excessive complexity, thus aligning well with parsimony. Over the years, AIC has been extended and adapted for various modeling approaches. Adjustments such as AICc, a corrected version of AIC, have been developed to address biases in small smaple sizes \cite{hurvich1989regression}. These extensions highlight the flexibility of AIC in different statistical environments.

In highly parameterized models, such as deep learning, traditional AIC may fall short. This is because AIC inherently assumes that model complexity is directly correlated with the number of parameters, which does not account for the regularization effects that can significantly alter the active dimensional complexity of a model.

### Regularization
Regularization techniques have become essential in modern statistical modeling and machine learning when dealing with high-dimensional data. These methods aim to prevent overfitting by introducing additional constraints on the model parameters. Two famous forms of regularization are L1 regularization (LASSO) and L2 regularization (ridge regression). \cite{montgomery2021introduction} These constraints are incorporated into objective with a Lagrangian framework.
In LASSO, the penalty term is equal to the absolute value of the coefficients, and the optimization problem can be expressed as below.
\begin{equation*}
    \mathcal{L}(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|
\end{equation*}
In this case, the Lagrangian incorporates the regularization constraint which forces sparsity in the parameter estimates. The presence of the absolute value penalty effectively pushes some coefficients exactly to zero, making LASSO particularly useful for feature selection.

Ridge regression introduces a penalty based on the squared values of the coefficients.
\begin{equation*}
    \mathcal{L}(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2
\end{equation*}
In this formulation, the Lagrangian framework forces the coefficients to have small values. L2 regularization does not produce exact zeros but shrinks all parameter estimates toward zero, leading to a smoother model. This approach is particularly advantageous in situations where multicollinearity exists among the predictors, as it helps distribute the influence of correlated variables more evenly.

### Effective Number of Parameters
In over-parameterized models including neural networks, since not all parameters contribute to the model's predictability, there has been attempts to count the effective number of parameters. Recent work by \cite{curth2024u} addresses this concept by examining the phenomenon called Double Descent. Traditional statistical learning has long been guided by a U-shaped curve relating model's complexity and generalization error. However, in over-parameterized models, a second descent in test error has been observed as parameter counts continue to increase. The authors of \cite{curth2024u} examine this by redefining the complexity through the effective number of parameters, not the true number of parameters. Specifically, they demonstrate that while raw parameter counts grow, not all parameters increase model's complexity. They propose Generalized Effective Parameter measure for model complexity in over-parameterized models. Their definition is based on the variance of parameters with some weights called smoother. 

From a theoretical perspective, \cite{morucci2024model} expand on the concept of effective parameters by introducing a general framework for Mean Integrated Squared Error (MISE) in large neural networks. They identify two distinct behaviors—double descent and monotonic MISE reduction—depending on model design and regularization strategies. Additionally, \cite{zhang2024manipulating} focus on sparse double descent, where L1 regularization and sparsity induce a nuanced complexity-generalization relationship. These studies collectively underscore the importance of redefining complexity metrics to better reflect the impact of regularization and parameter sparsity.

Building on these foundations, our work introduces the Effective Akaike Information Criterion (EAIC), a practical adaptation that explicitly incorporates effective parameter counts. By defining a threshold for parameter significance, we hope that EAIC will bridge theoretical insights with model selection considering regularization, providing a practical tool for evaluating models across diverse settings, including highly parameterized neural networks and traditional regression models.

## Methodology
### Effective Akaike Information Criterion
In traditional AIC, the complexity is purely a function of the total number of parameters denoted as $k$. However, in regularized models, not all parameters contribute equally to the model's predictive ability. Regularization techniques such as L1 and L2 induce sparsity or shrink the value of parameters to reduce model's complexity. Therefore, the true "effective" complexity of the model is less than or equal to the number of total parameters.

EAIC is derived from an additive combination of AIC and a negative value of measure $l$ which quantifies how much regularized the model is.

\begin{align}
    \mathrm{EAIC} &:= \mathrm{AIC} - 2l(g)\\
    &= 2(k - l(g)) - 2\ln(\hat{L})\label{eq:eaic-derv}\\
    &= 2\keff - 2\ln(\hat{L})\label{eq:eaic}
\end{align}
$l$ is a function that gives the number of regularized parameters below a threshold $h$ as an output where $g$ is an approximating model. The first two terms in \eqref{eq:eaic-derv} can be combined together then become $\keff$ in \eqref{eq:eaic} which refers to the effective number of parameters. Then EAIC is reformulated by replacing $k$ in AIC with $\keff$, representing only those parameters that have an absolute value exceeding a predefined threshold $h$. These are considered to be the parameters that are still meaningful to the model. This reflects the number of parameters whose estimated values significantly contribute to the model after regularization has been applied, distinguishing between active and inactive parameters.

A formal definition of $\keff$ is below,
\begin{equation} \label{eq:keff}
    \keff = \sum_{i=0}^k I(|\beta_i|>h)
\end{equation}
where $I(\cdot)$ denotes an indicator function. This approach ensures that the model complexity, as measured by EAIC, better reflects the true number of effective parameters after regularization, taking into account their contribution to the model's performance.

The choice of the threshold $h$ is a crucial component of the EAIC. One may consider the scale of independent variables for determining $h$, since the unit of independent variables determines the magnitude of slope with the analogy of simple linear regression. When the independent variables are many, meaning that the dimensions of input are high, the data often normalized to have values between -1 and 1 or similar. To consider the scale-variant nature of parameters shown here, the threshold $h$ is determined as a fraction of or a multiplication $r$ to the empirical deviation $\sigma$ of parameters, $h=r\sigma$, resulting in the number of statistically significant parameters. Typical choices for $r$ are listed below.
\begin{itemize}
    \item $h=0.1\sigma$: This would be a relatively strict threshold, meaning only parameters that deviate significantly from zero.
    \item $h=1.96\sigma$: 95\% confidence interval as a threshold implying that how significant a single parameter is when assumed for it to follow Gaussian distribution.
\end{itemize}

The introduction of effective parameters enables for statisticians to compare models with same parameter space but with different values of parameters. This happens in the nonlinear regression having a non-convex objective, leading multiple optima.

In the demonstration of EAIC with empirical analysis that follows, we will apply the EAIC to various regularized regression models that the Regression Analysis 2 (회귀분석2) covered so far.

## Dataset
### Abalone Dataset
The Abalone dataset from UCI Machine Learning Repository \cite{miscabalone1} is a well-known dataset used primarily for regression analysis. The goal is to predict the age of abalone from physical measurements. This dataset has been used in predictive modeling research. The properties of this dataset are follows.
\begin{itemize}
    \item \textbf{Sex (Nominal)}: The gender of the abalone (M: Male, F: Female, I: Infant).
    \item \textbf{Length (Continuous)}: The longest shell measurement in millimeters.
    \item \textbf{Diameter (Continuous)}: The diameter of the shell, perpendicular to the length.
    \item \textbf{Height (Continuous)}: The height of the shell, with meat inside, in millimeters.
    \item \textbf{Whole weight (Continuous)}: Total weight of the abalone, in grams.
    \item \textbf{Shucked weight (Continuous)}: Weight of the abalone's meat, in grams.
    \item \textbf{Viscera weight (Continuous)}: Gut weight after bleeding, in grams.
    \item \textbf{Shell weight (Continuous)}: Weight of the shell after being dried, in grams.
    \item \textbf{Rings (Integer)}: The number of rings, which serves as the target variable for prediction. The age of an abalone is usually estimated as \texttt{Rings + 1.5}.
\end{itemize}

### MNIST
In addtion to the Abalone dataset, we will utilize the MNIST dataset \cite{deng2012mnist} which is widely recognized benchmark for image classification tasks. We demonstrate the effectiveness of the EAIC in the context of highly nonlinear regression models with non-convex multiple optima objectives. The MNIST dataset consists of images of handwritten digits (0-9), providing a rich and complex challenge for classification.

The objective is to classify each image into one of the ten digit categories. By leveraging this dataset, we aim to illustrate how EAIC can quantify the generalizability of deep neural networks, which are often characterized by complex architectures and a large number of parameters. Evaluating the models using EAIC will allow us to gain insights into their effective complexity and predictive accuracy, particularly in the face of the challenges posed by nonlinear relationships and potential overfitting in deep learning scenarios.

## Results
In this section, we demonstrate our proposed evaluation metric for model selection with two different cases, which are abalone dataset and MNIST. Each case represents two different situations in regression analysis. The first case, the abalone dataset, is typically considered to be suitable for traditional linear regression analysis. On the other hand, MNIST is recently developed and suitable for nonlinear regression because of its high nonlinearity.
### Case 1: Ablaone
\subsubsection{Linear regression with and without regularization}
In this section, we present a comparison between the Akaike Information Criterion (AIC) and the Effective Akaike Information Criterion (EAIC) in selecting regression models with better generalizability using the Abalone dataset. The goal is to evaluate whether EAIC provides a better measure for model selection in regularized models compared to the traditional AIC.

We conducted an analysis using three different models:
\begin{itemize}
    \item \textbf{LASSO Regression} Linear regression with L1 regularization.
    \item \textbf{Ridge Regression} Linear regression with L2 regularization.
    \item \textbf{Linear Regression} Standard linear regression without regularization.
\end{itemize}

The Abalone dataset was used for this analysis, excluding the nominal variable \texttt{Sex} to avoid issues with non-numeric data. The dataset was split into training and test sets, with 80\% of the data used for training and the remaining 20\% for testing. The threshold parameter \( r \) for EAIC was set to 0.1 and the lambda parameter $r$ for regularization was also set to 0.1

The results of the analysis are summarized in Table 1, which presents the AIC, EAIC, and test Mean Squared Error (MSE) for each model.

\begin{table}[h!]
\centering
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Model} & \textbf{AIC} & \textbf{EAIC} & \textbf{Test MSE} \\ \midrule
LASSO  & 15084.89 & 15076.89 & 5.7690 \\
Ridge  & 14956.28 & 14952.28 & 5.6247 \\
Linear & 14825.98 & 14823.98 & 5.3125 \\ \bottomrule
\end{tabular}
\caption{AIC and EAIC values for different models}
\label{tab:linear-regression-aic-eaic}
\end{table}
From the results, the Linear Regression model achieved the lowest AIC, EAIC, and test MSE, indicating that it is the best model according to both information criteria and actual performance on unseen data. This outcome is expected due to the simplicity of the dataset and the absence of high-dimensional features that might necessitate regularization.

To assess which criterion better predicts generalizability, we computed the correlation between the information criteria and the test MSE. 
\begin{table}[h!]
\centering
\begin{tabular}{@{}lc@{}}
\toprule
\textbf{Metric} & \textbf{Correlation(\%)} \\ \midrule
Correlation between AIC and Test MSE & 97.8983 \\
Correlation between EAIC and Test MSE & 97.9921 \\ \bottomrule
\end{tabular}
\caption{AIC and EAIC values for different models}
\label{tab:linear-regression-corr}
\end{table}

In Table \ref{tab:linear-regression-corr}, the slightly higher correlation between EAIC and test MSE suggests that EAIC may be marginally better at representing generalizability. This is anticipated since EAIC adjusts for the effective number of parameters, potentially providing a more accurate penalty for model complexity in regularized models.

\subsubsection{Nonlinear regression with and without regularization}


### Case 2: MNIST

## Conclusion
In this paper, we introduced the EAIC as a modification to the traditional AIC for models incorporating regularization. While the AIC has proven to be a valuable tool for model selection, it does not account for the effects of regularization techniques that can reduce the true complexity of a model by shrinking or sparsifying parameters. The EAIC addresses this limitation by introducing the concept of the effective number of parameters $\keff$ which reflects only those parameters that have a significant contribution to the model after regularization. By adjusting the threshold $h$, the EAIC can balance model complexity and goodness-of-fit more efficiently than traditional AIC, especially in over-parameterized and regularized models such as deep neural networks. However, we also found it out that in the traditional regression where over-parameterization does not occur, the AIC still be considerable.