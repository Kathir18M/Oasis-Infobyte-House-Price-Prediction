# Oasis Infobyte Data Analytics Internship – Project 4
# House Price Prediction (End-to-End Machine Learning System)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)]()
[![Internship](https://img.shields.io/badge/Oasis%20Infobyte-Data%20Analytics-blueviolet.svg)]()

---

## 📌 Executive Summary
This repository contains a complete, production-grade Machine Learning solution developed for **Oasis Infobyte Data Analytics Internship Project 4: House Price Prediction**. 

The goal of this project is to accurately predict residential property sale prices using the **Ames Housing dataset** (79 explanatory features describing residential homes in Ames, Iowa). The system incorporates domain-driven feature engineering, data leakage prevention via Scikit-Learn `Pipeline` and `ColumnTransformer`, target variable normalization via `log1p`/`expm1`, multi-model benchmarking (Linear Regression, Ridge, and Lasso), residual diagnostics, and an independent inference module.

---

## 🏗️ Repository Architecture

```text
Oasis_Infobyte_Project_4/
├── data/
│   ├── data_description.txt        # Full description of all 79 raw features
│   ├── sample_submission.csv       # Kaggle benchmark submission format
│   ├── test.csv                    # Test set (1,459 rows × 80 features)
│   └── train.csv                   # Labeled training set (1,460 rows × 81 columns)
├── models/
│   └── best_house_price_model.pkl  # Serialized Scikit-Learn pipeline (Joblib)
├── notebooks/
│   └── Oasis_Infobyte_Project_4_House_Price_Prediction.ipynb # Interactive EDA & experimentation
├── outputs/
│   ├── figures/                    # 10 publication-quality diagnostic & EDA plots
│   │   ├── 01_target_distribution.png
│   │   ├── 02_missing_values_bar.png
│   │   ├── 03_correlation_heatmap.png
│   │   ├── 04_overall_qual_boxplot.png
│   │   ├── 05_scatter_livarea_vs_price.png
│   │   ├── 06_totalsf_vs_price.png
│   │   ├── 07_house_age_scatter.png
│   │   ├── actual_vs_predicted.png
│   │   ├── model_comparison.png
│   │   └── residual_plot.png
│   ├── model_comparison.csv        # Validation metric summary across models
│   └── submission.csv              # Final test predictions (Id, SalePrice)
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── eda.py                      # Automated EDA & visualization generation
│   ├── feature_engineering.py      # Reusable domain feature transformer
│   └── train_model.py              # Modular training, validation & submission script
├── predict.py                      # Standalone prediction script (NO RETRAINING)
├── requirements.txt                # Project dependency specifications
├── README.md                       # Comprehensive project documentation
└── .gitignore                      # Environment, checkpoint & cache exclusions
```

---

## 🔬 End-to-End Methodology

### 1. Data Understanding & Validation
- **Training Data**: 1,460 records across 79 explanatory features + `Id` + `SalePrice`.
- **Test Data**: 1,459 unlabelled records across 79 explanatory features + `Id`.
- **Target Analysis**: Raw `SalePrice` exhibits noticeable positive skewness (mean: \$180,921, median: \$163,000, max: \$755,000). A natural log transformation $\ln(1 + \text{SalePrice})$ is applied to stabilize variance and satisfy homoscedasticity assumptions for linear estimators.

### 2. Missing Value & Outlier Strategy
- Numeric features (e.g., `LotFrontage`, `GarageYrBlt`, `MasVnrArea`) are imputed using **median** statistics learned strictly on the training partition.
- Categorical features with missing values (e.g., `Alley`, `PoolQC`, `Fence`, `FireplaceQu`) are handled using **most frequent / mode** replacement, and encoded using `OneHotEncoder(handle_unknown="ignore")`.
- Ground truth test labels are never inspected or used during imputation, scaling, or training.

### 3. Domain Feature Engineering
In `src/feature_engineering.py`, domain knowledge is used to construct high-signal compound features:
- **`TotalSF`**: Aggregates basement, 1st floor, and 2nd floor area (`TotalBsmtSF + 1stFlrSF + 2ndFlrSF`).
- **`TotalBathrooms`**: Weighted sum of all bathrooms (`FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath`).
- **`HouseAge`**: Age of property at the transaction year (`YrSold - YearBuilt`, bounded $\ge 0$).
- **`YearsSinceRemodel`**: Elapsed time since property was remodeled (`YrSold - YearRemodAdd`, bounded $\ge 0$).
- **`TotalPorchSF`**: Aggregate outdoor living space (`WoodDeckSF + OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch`).
- **Binary Amenity Flags**: `HasPool`, `Has2ndFloor`, `HasGarage`, `HasBsmt`.

### 4. Leakage-Free Preprocessing Pipeline
All transformations are encapsulated inside Scikit-Learn `ColumnTransformer` and `Pipeline`:
```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])
```

### 5. Model Training & Comparison
The dataset was partitioned into an **80% training set (1,168 samples)** and a **20% holdout validation set (292 samples)** with `random_state=42`. Three regularized and unregularized linear architectures were benchmarked:
1. **Linear Regression (OLS)**
2. **Ridge Regression** ($L_2$ penalty, $\alpha = 10.0$)
3. **Lasso Regression** ($L_1$ penalty, $\alpha = 0.001$, $\text{max\_iter} = 20,000$)

All models were evaluated on the validation set using:
- **Log RMSE**: $\sqrt{\frac{1}{n} \sum (\log(1 + y) - \log(1 + \hat{y}))^2}$
- **Dollar RMSE**: $\sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$
- **MAE**: $\frac{1}{n} \sum |y - \hat{y}|$
- **Coefficient of Determination ($R^2$)**

---

## 📊 Actual Experimental Results

The models achieved the following performance on the holdout validation set (saved in `outputs/model_comparison.csv`):

| Model | Log RMSE ↓ | Validation RMSE ($) ↓ | Validation MAE ($) ↓ | Validation $R^2$ ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | **0.13347** | **\$23,186.47** | **\$15,367.02** | **0.9299 (92.99%)** |
| **Ridge Regression** ($\alpha=10.0$) | 0.13570 | \$24,284.25 | \$16,034.68 | 0.9231 (92.31%) |
| **Lasso Regression** ($\alpha=0.001$) | 0.13917 | \$24,889.93 | \$16,250.93 | 0.9192 (91.92%) |

> **Selection**: **Linear Regression** demonstrated superior generalization on the holdout partition with an $R^2$ score of **92.99%** and Log RMSE of **0.13347**. The model was subsequently retrained on 100% of labeled training data (1,460 rows) to maximize sample efficiency before generating test predictions.

---

## 📈 Visualizations & Diagnostics

All figures are automatically exported to `outputs/figures/`:

| Artifact | Description |
| :--- | :--- |
| **`01_target_distribution.png`** | Bimodal/right-skewed raw price vs normalized bell curve under `log1p`. |
| **`02_missing_values_bar.png`** | Ranked missingness percentage across all raw features in `train.csv`. |
| **`03_correlation_heatmap.png`** | Correlation matrix of top predictive features (`OverallQual`, `GrLivArea`, `TotalSF`, `GarageCars`). |
| **`04_overall_qual_boxplot.png`** | Monotonic relationship between property build quality rating (1–10) and sale price. |
| **`05_scatter_livarea_vs_price.png`** | Ground living area vs price with quality heat colormap highlighting potential outliers. |
| **`06_totalsf_vs_price.png`** | Linear relationship and regression slope between engineered `TotalSF` and price. |
| **`07_house_age_scatter.png`** | Depreciation curve showing relationship between building age and market value. |
| **`model_comparison.png`** | Bar chart comparing validation Log RMSE across Linear, Ridge, and Lasso. |
| **`actual_vs_predicted.png`** | Validation scatter plot comparing actual vs predicted prices against ideal $y = x$. |
| **`residual_plot.png`** | Residual errors vs fitted values and residual distribution centered at 0. |

---

## 🚀 How to Run the Project

### 1. Clone & Set Up Environment
```powershell
# Clone the repository
git clone https://github.com/<your-username>/Oasis_Infobyte_Project_4.git
cd Oasis_Infobyte_Project_4

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Exploratory Visualizations
```powershell
python src/eda.py
```
*Outputs 7 comprehensive EDA figures to `outputs/figures/`.*

### 3. Train & Evaluate Models
```powershell
python src/train_model.py
```
*Splits training data, evaluates all candidate models, exports `outputs/model_comparison.csv`, creates diagnostic charts, retrains on all training records, generates `outputs/submission.csv`, and serializes `models/best_house_price_model.pkl`.*

### 4. Run Standalone Inference (Without Retraining)
```powershell
python predict.py
```
*Loads the serialized pipeline and runs instant inference (< 2 seconds) on test samples without retraining.*

### 5. Interactive Notebook
```powershell
jupyter notebook notebooks/Oasis_Infobyte_Project_4_House_Price_Prediction.ipynb
```

---

## 🛡️ Best Practices & Quality Assurance
- **Zero Data Leakage**: Standard scaling and categorical encoding are fitted solely within Scikit-Learn pipelines.
- **Robust Categorical Handling**: `OneHotEncoder(handle_unknown='ignore')` prevents test-time crashes on novel categories.
- **Inverse Target Transformation**: All target calculations use `expm1` with boundary clipping `np.maximum(0, ...)` to ensure real-world, non-negative valuations.
- **No Model Retraining on Inference**: `predict.py` strictly loads the pre-trained Joblib object and executes inference in milliseconds.

---

## 👨‍💻 Author & Internship Details
- **Project**: Oasis Infobyte Data Analytics Internship (Project 4)
- **Role**: Data Analytics / Machine Learning Intern
- **Domain**: Real Estate Valuation & Regression Modeling
