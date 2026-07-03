# Ames Housing Price Prediction: A Multi-Model Benchmark Study

An end-to-end machine learning study evaluating and comparing different algorithmic approaches—Parametric Regularized Regression, Tree-Based Ensembles, and Deep Neural Networks—to solve a high-cardinality regression problem using the Kaggle Ames Housing dataset.

##  Project Architecture & Model Leaderboard
Rather than relying on a single algorithm, this repository serves as a comparative benchmark to analyze how different model architectures handle non-linear relationships, multi-collinearity, and the bias-variance tradeoff.

| Model Architecture | Preprocessing Pipeline | Validation RMSE | Kaggle Rank | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Ridge Linear Regression ($L_2$)** | Median Imputation + Log Transform + Scaling + OHE | 0.1554 | 3,292 |
| **Random Forest Regressor** | Minimal (No scaling or OHE tracking needed) | 0.1296 | 2135 |
| **Multi-Layer Perceptron (NN)** | Strict Scaling + Full Feature Encoding | *TBD* | *TBD* |

---

##  Engineering Pipeline & Technical Decisions

### 1. Target Transformation & Skewness Mitigation
Housing prices inherently exhibit a severe right-skewed distribution due to a small percentage of high-value outliers. Training a linear model on raw prices heavily penalizes errors on expensive homes, warping the line of best fit. 
- Applied a log transformation using `np.log1p` to compress the target variance and normalize the distribution.
- Utilized `np.expm1` on final model inferences to reverse the transformation back to human-readable U.S. Dollar values for evaluation.

### 2. High-Cardinality Categorical Encoding
To process non-numeric features (e.g., `Neighborhood`, `RoofStyle`), the pipeline implements **One-Hot Encoding** (`pd.get_dummies`).
- **Dummy Variable Trap Mitigation:** Enabled `drop_first=True` to intentionally drop the baseline category column. This prevents perfect multicollinearity, ensuring that the feature matrix remains stable during matrix inversion in the Ordinary Least Squares calculation.
- **Data Alignment:** Synchronized the categorical schemas between the training and testing streams prior to encoding to prevent shape mismatch runtime errors.

### 3. Hyperparameter Optimization & Bias-Variance Analysis
Upon expanding the feature space from 36 numeric metrics to 200+ encoded dimensions, the baseline OLS model experienced severe **high variance (overfitting)**, with training error dropping to 0.0916 while validation error exploded to 0.1737.
- Implemented **Ridge Regression** to introduce a weight-shrinking $L_2$ regularization penalty ($\alpha \sum w_j^2$).
- Executed a programmatic grid sweep over a spectrum of $\alpha$ parameters to plot the **Bias-Variance Tradeoff Curve**, successfully isolating the mathematical "sweet spot" at $\alpha = 200$ to stabilize validation performance.

---

##  Repository Structure
```text
├── .gitignore         # Filters out Pycache, checkpoints, and heavy data files
├── LICENSE            # MIT Open Source License
├── README.md          # Project documentation and benchmark study
└── house-prices-ridge-regression.ipynb  # Phase 1: Regularized Linear Model
