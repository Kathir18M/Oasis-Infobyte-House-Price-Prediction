"""
Model Training and Evaluation Pipeline for Ames Housing Price Prediction.

This module orchestrates:
1. Data loading and feature engineering (via src.feature_engineering).
2. Construction of Scikit-Learn ColumnTransformer and Pipelines (preventing data leakage).
3. Train/validation splitting (80/20, random_state=42).
4. Training Linear Regression, Ridge, and Lasso regressors on log1p-transformed target.
5. Multi-metric evaluation (RMSE_Log, RMSE, MAE, R²).
6. Performance comparison and metric visualization.
7. Model selection and full-retraining on all labeled training data.
8. Generating test predictions and exporting outputs/submission.csv.
9. Serializing the final trained pipeline to models/best_house_price_model.pkl.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.feature_engineering import create_features


def load_and_preprocess_data(train_path="data/train.csv", test_path="data/test.csv"):
    """
    Loads train and test sets and applies consistent feature engineering.
    """
    print(f"[Data] Loading train from '{train_path}' and test from '{test_path}'...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    test_ids = test["Id"]

    # Extract target and drop Id
    X = train.drop(columns=["SalePrice", "Id"], errors="ignore")
    y = train["SalePrice"]
    X_test = test.drop(columns=["Id"], errors="ignore")

    # Apply domain feature engineering
    X = create_features(X)
    X_test = create_features(X_test)

    return X, y, X_test, test_ids


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer handling numeric imputation/scaling
    and categorical imputation/one-hot encoding.
    """
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    print(f"[Preprocessing] Numeric features: {len(numeric_features)}, Categorical features: {len(categorical_features)}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor


def evaluate_models(X, y, preprocessor):
    """
    Splits data, trains candidate models on log1p(y), evaluates metrics,
    and returns results DataFrame and trained pipelines.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=42),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=20000, random_state=42)
    }

    results = []
    trained_pipelines = {}

    print("\n" + "=" * 60)
    print("           MODEL TRAINING AND VALIDATION EVALUATION")
    print("=" * 60)

    for name, model in candidate_models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Train on log-transformed target
        pipe.fit(X_train, y_train_log)

        # Predict log prices on validation set
        val_pred_log = pipe.predict(X_val)

        # Transform back to actual dollar amounts and clip at 0
        val_pred = np.maximum(0, np.expm1(val_pred_log))

        # Calculate metrics
        rmse_log = np.sqrt(mean_squared_error(y_val_log, val_pred_log))
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)

        results.append({
            "Model": name,
            "RMSE_Log": rmse_log,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })
        trained_pipelines[name] = pipe

        print(f"\nModel: {name}")
        print(f"  • Log RMSE : {rmse_log:.5f}")
        print(f"  • RMSE ($) : ${rmse:,.2f}")
        print(f"  • MAE ($)  : ${mae:,.2f}")
        print(f"  • R² Score : {r2:.4f}")

    results_df = pd.DataFrame(results).sort_values(by="RMSE_Log").reset_index(drop=True)
    return results_df, trained_pipelines, X_val, y_val, candidate_models


def save_evaluation_artifacts(results_df, best_pipeline, X_val, y_val, figures_dir="outputs/figures"):
    """
    Saves model comparison metrics CSV and generates evaluation diagnostic plots.
    """
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Save CSV
    results_df.to_csv("outputs/model_comparison.csv", index=False)
    print("\n[Outputs] Saved 'outputs/model_comparison.csv'")

    # Set visualization theme
    sns.set_theme(style="whitegrid")

    # 2. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    bar_plot = sns.barplot(data=results_df, x="Model", y="RMSE_Log", hue="Model", palette="Blues_r", legend=False)
    plt.title("Model Comparison - Log RMSE (Lower is Better)", fontsize=14, fontweight="bold")
    plt.ylabel("Root Mean Squared Error (Log Scale)", fontsize=11)
    plt.xlabel("Model Architecture", fontsize=11)
    for p in bar_plot.patches:
        height = p.get_height()
        bar_plot.annotate(f"{height:.4f}",
                          (p.get_x() + p.get_width() / 2., height),
                          ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                          textcoords='offset points')
    plt.ylim(0, results_df["RMSE_Log"].max() * 1.25)
    plt.savefig(os.path.join(figures_dir, "model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Outputs] Saved '{figures_dir}/model_comparison.png'")

    # 3. Actual vs Predicted Plot
    val_preds_log = best_pipeline.predict(X_val)
    val_preds = np.maximum(0, np.expm1(val_preds_log))

    plt.figure(figsize=(8, 7))
    plt.scatter(y_val, val_preds, alpha=0.6, color="#1f77b4", edgecolor="k", linewidth=0.5, label="Validation Predictions")
    max_val = max(y_val.max(), val_preds.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Ideal Fit (y = x)")
    plt.xlabel("Actual Sale Price ($)", fontsize=12)
    plt.ylabel("Predicted Sale Price ($)", fontsize=12)
    plt.title("Actual vs. Predicted House Prices", fontsize=14, fontweight="bold")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend(frameon=True)
    plt.savefig(os.path.join(figures_dir, "actual_vs_predicted.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Outputs] Saved '{figures_dir}/actual_vs_predicted.png'")

    # 4. Residual Diagnostics Plot
    residuals = y_val - val_preds
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter of Residuals vs Predicted
    axes[0].scatter(val_preds, residuals, alpha=0.6, color="#2b5c8f", edgecolor="w")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.8)
    axes[0].set_xlabel("Predicted Sale Price ($)", fontsize=11)
    axes[0].set_ylabel("Residuals (Actual - Predicted) ($)", fontsize=11)
    axes[0].set_title("Residuals vs. Fitted Values", fontsize=13, fontweight="bold")

    # Histogram / KDE of Residuals
    sns.histplot(residuals, kde=True, color="#e377c2", ax=axes[1], bins=30)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual Error ($)", fontsize=11)
    axes[1].set_ylabel("Density / Count", fontsize=11)
    axes[1].set_title("Distribution of Residuals", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "residual_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Outputs] Saved '{figures_dir}/residual_plot.png'")


def retrain_and_generate_submission(best_model_name, candidate_models, preprocessor, X, y, X_test, test_ids):
    """
    Retrains the selected top-performing model on all labeled training data (X, y)
    and predicts on the test set, exporting submission.csv and serialized model.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  RETRAINING SELECTED MODEL [{best_model_name}] ON FULL DATASET")
    print("=" * 60)

    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", candidate_models[best_model_name])
    ])

    # Fit on 100% of labeled training data
    final_pipeline.fit(X, np.log1p(y))

    # Inference on unlabelled test data
    test_preds_log = final_pipeline.predict(X_test)
    test_preds = np.maximum(0, np.expm1(test_preds_log))

    # Export Kaggle-compatible submission
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": test_preds
    })
    submission.to_csv("outputs/submission.csv", index=False)
    print(f"[Submission] Successfully generated 'outputs/submission.csv' ({len(submission)} rows)")

    # Serialize trained pipeline with Joblib
    model_path = "models/best_house_price_model.pkl"
    joblib.dump(final_pipeline, model_path)
    print(f"[Model] Serialized final model pipeline to '{model_path}'")

    return final_pipeline


def main():
    """Main execution entry point."""
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X, y, X_test, test_ids = load_and_preprocess_data()
    preprocessor = build_preprocessor(X)
    results_df, trained_pipelines, X_val, y_val, candidate_models = evaluate_models(X, y, preprocessor)

    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\n" + "=" * 60)
    print(f"Top-Performing Model Selected: {best_model_name}")
    print("=" * 60)

    save_evaluation_artifacts(results_df, best_pipeline, X_val, y_val)
    retrain_and_generate_submission(best_model_name, candidate_models, preprocessor, X, y, X_test, test_ids)

    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
