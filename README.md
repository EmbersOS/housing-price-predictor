# Ames Housing Price Prediction: An Architectural Comparison Study

This repository compares three machine learning approaches—regularized linear models, gradient-boosted tree ensembles, and deep neural networks—on the Ames Housing Price Prediction dataset. The study focuses on how different model architectures perform on small tabular datasets with mixed numerical and categorical features.
In addition to comparing predictive performance, the project investigates preprocessing strategies, regularization techniques, and the effect of model complexity on generalization.

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
* **Analysis:** Ridge Regression provided a strong baseline but was limited in modeling complex nonlinear relationships between housing features and sale price, resulting in lower predictive performance than tree-based models.

### 2. Gradient-Boosted Decision Trees (XGBoost)
* **Methodology:** Hyperparameter optimization was conducted on a gradient-boosted tree framework. Minimal geometric scaling was required due to tree-based invariance to feature distributions.
* **Analysis:** As the top-performing independent model (0.12966 RMSE), XGBoost achieved the lowest RMSE among the evaluated models. Its ability to model nonlinear feature interactions while handling heterogeneous tabular data made it particularly effective for this dataset. The model natively maps the discrete, jagged decision boundaries common to spreadsheet features, yielding high sample efficiency.

### 3. Deep Learning Challenge: Multi-Layer Perceptron (MLP)
* **Methodology:** A dense, feed-forward neural network was constructed using TensorFlow/Keras, utilizing an initial `StandardScaler` pipeline to stabilize gradient descent.
* **Optimization Diagnostics:** Initial iterations exhibited optimization stagnation, plateauing at 0.5052 validation RMSE. Diagnostic analysis revealed a parameter conflict: an aggressive $L_2$ regularization penalty ($\lambda = 0.01$) effectively neutralized a conservative learning rate ($\eta = 0.001$). The combination of a conservative learning rate and strong L2 regularization prevented effective weight updates, leading to underfitting during early training.
* **Resolution:** Adjusting the learning rate to 0.015 and implementing a 0.3 Dropout rate allowed the optimizer to overcome the regularizer's weight-decay threshold. This settled at a local validation RMSE of 0.1386 and generalized to 0.17245 on the unseen test set.
* **Conclusion:** These experiments highlight the practical challenges of applying neural networks to relatively small tabular datasets.

---

## Repository Structure

* `/notebooks/housing-prices-regularised-linear-regression.ipynb` — Log transforms, parametric modeling, and baseline execution.
* `/notebooks/housing-price-predictor-decision-trees.ipynb` — Hyperparameter tuning and feature importance analysis.
* `/notebooks/house-price-predictor-neural-network.ipynb` — Keras MLP architecture, learning curve diagnostics, and regularization tuning.
* `/data/` — Local dataset references and validation splits.
* 
## Future Scope: Model Blending
The next evolution of this project involves developing a weighted ensemble blend (e.g., 70% XGBoost + 30% Neural Network). Since both models approach spatial boundaries entirely differently, ensembling their predictions will smooth individual model variance and increase accuracy to a level greater than either model
