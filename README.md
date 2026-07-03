# Ames Housing Price Prediction: An Architectural Comparison Study

This repository evaluates the efficacy of divergent machine learning paradigms—specifically parametric linear models, gradient-boosted tree ensembles, and deep neural networks—when applied to high-cardinality, small-sample tabular regression tasks. 

The primary scope of the study focuses on feature engineering pipelines, optimization stability under regularization constraints, and empirical limits of structural complexity given small sample sizes.

## Performance Evaluation

| Model Architecture | Preprocessing Pipeline | Test RMSE | Leaderboard Rank |
| :--- | :--- | :--- | :--- |
| **Ridge Linear Regression ($L_2$)** | Median Imputation + Log Transform + Standardization + OHE | 0.15190 | 3,292 |
| **Multi-Layer Perceptron (NN)** | Standardization + Full Feature Encoding + Dropout | 0.17245 | ~3,500 |
| **Tuned XGBoost Ensemble** | Median Imputation + Log Transform + OHE | **0.12966** | **2,135** |

---

## Architecture Insights & Optimization Analysis

### 1. Ridge Regression Baseline
* **Methodology:** Categorical features were processed via One-Hot Encoding (OHE). A log transformation log(1+x) was applied to the target variable (`SalePrice`) to normalize right-skewness.
* **Analysis:** While computationally efficient, the parametric assumptions of linear regression failed to capture non-linear feature interactions, such as variations in depreciation across disparate neighborhoods.

### 2. Gradient-Boosted Decision Trees (XGBoost)
* **Methodology:** Hyperparameter optimization was conducted on a gradient-boosted tree framework. Minimal geometric scaling was required due to tree-based invariance to feature distributions.
* **Analysis:** As the top-performing independent model (0.12966 RMSE), XGBoost demonstrated the structural superiority of decision trees on tabular data. The model natively maps the discrete, jagged decision boundaries common to spreadsheet features, yielding high sample efficiency.

### 3. Deep Learning Challenge: Multi-Layer Perceptron (MLP)
* **Methodology:** A dense, feed-forward neural network was constructed using TensorFlow/Keras, utilizing an initial `StandardScaler` pipeline to stabilize gradient descent.
* **Optimization Diagnostics:** Initial iterations exhibited optimization stagnation, plateauing at 0.5052 validation RMSE. Diagnostic analysis revealed a parameter conflict: an aggressive $L_2$ regularization penalty ($\lambda = 0.01$) effectively neutralized a conservative learning rate ($\eta = 0.001$). This backpropagation friction collapsed the hidden layer weights toward zero, forcing the network to predict the global sample mean.
* **Resolution:** Adjusting the learning rate to 0.015 and implementing a 0.3 Dropout rate allowed the optimizer to overcome the regularizer's weight-decay threshold. This settled at a local validation RMSE of 0.1386 and generalized to 0.17245 on the unseen test set.
* **Conclusion:** The MLP proved highly susceptible to overfitting due to severe data scarcity (1,168 training observations), confirming that tree ensembles remain the mathematically optimal choice for sparse tabular structures.

---

## Repository Structure

* `/notebooks/housing-prices-regularised-linear-regression.ipynb` — Log transforms, parametric modeling, and baseline execution.
* `/notebooks/housing-price-predictor-decision-trees.ipynb` — Hyperparameter tuning and feature importance analysis.
* `/notebooks/neural-network-mlp.ipynb` — Keras MLP architecture, learning curve diagnostics, and regularization tuning.
* `/data/` — Local dataset references and validation splits.
