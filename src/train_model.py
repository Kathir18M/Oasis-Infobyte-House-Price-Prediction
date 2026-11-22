import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# 1. Create output directories

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)


# 2. Load data

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

test_ids = test["Id"]

X = train.drop(columns=["SalePrice", "Id"])
y = train["SalePrice"]

X_test = test.drop(columns=["Id"])


# 3. Feature engineering

def create_features(df):

    df = df.copy()

    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    df["TotalBathrooms"] = (
        df["FullBath"].fillna(0)
        + df["HalfBath"].fillna(0) * 0.5
        + df["BsmtFullBath"].fillna(0)
        + df["BsmtHalfBath"].fillna(0) * 0.5
    )

    df["HouseAge"] = (
        df["YrSold"] - df["YearBuilt"]
    )

    df["YearsSinceRemodel"] = (
        df["YrSold"] - df["YearRemodAdd"]
    )

    df["TotalPorchSF"] = (
        df["WoodDeckSF"].fillna(0)
        + df["OpenPorchSF"].fillna(0)
        + df["EnclosedPorch"].fillna(0)
        + df["3SsnPorch"].fillna(0)
        + df["ScreenPorch"].fillna(0)
    )

    return df


X = create_features(X)
X_test = create_features(X_test)


# 4. Identify column types

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


# 5. Preprocessing pipelines

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 6. Train validation split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 7. Define models

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=10.0),

    "Lasso Regression": Lasso(
        alpha=0.001,
        max_iter=20000
    )
}


# 8. Model training and evaluation

results = []
trained_models = {}

y_train_log = np.log1p(y_train)

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train_log)

    predictions_log = pipeline.predict(X_val)

    predictions = np.expm1(predictions_log)

    predictions = np.maximum(predictions, 0)

    rmse_log = np.sqrt(
        mean_squared_error(
            np.log1p(y_val),
            predictions_log
        )
    )

    rmse = np.sqrt(
        mean_squared_error(y_val, predictions)
    )

    mae = mean_absolute_error(
        y_val,
        predictions
    )

    r2 = r2_score(
        y_val,
        predictions
    )

    results.append({
        "Model": name,
        "RMSE_Log": rmse_log,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    trained_models[name] = pipeline

    print("\n", name)
    print("Log RMSE:", rmse_log)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R2:", r2)


# 9. Compare models

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE_Log"
)

print("\nMODEL COMPARISON")
print(results_df)

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# 10. Model comparison visualization

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="RMSE_Log"
)

plt.title("Model Comparison - Log RMSE")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    "outputs/figures/model_comparison.png"
)

plt.close()


# 11. Select best model

best_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_name]

print("\nBest Model:", best_name)


# 12. Actual vs predicted visualization

val_predictions = np.maximum(
    0,
    np.expm1(best_model.predict(X_val))
)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_val,
    val_predictions,
    alpha=0.6
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted House Prices")

plt.savefig(
    "outputs/figures/actual_vs_predicted.png"
)

plt.close()


# 13. Residual analysis

residuals = y_val - val_predictions

plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=val_predictions,
    y=residuals
)

plt.axhline(0, linestyle="--")

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")

plt.title("Residual Analysis")

plt.savefig(
    "outputs/figures/residual_plot.png"
)

plt.close()


# 14. Train final model using all labeled data

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", models[best_name])
])

final_model.fit(X, np.log1p(y))


# 15. Generate test predictions

test_predictions = np.expm1(
    final_model.predict(X_test)
)

test_predictions = np.maximum(
    test_predictions, 0
)


# 16. Create submission file

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_predictions
})

submission.to_csv(
    "outputs/submission.csv",
    index=False
)


# 17. Save final model

joblib.dump(
    final_model,
    "models/best_house_price_model.pkl"
)


print("\nPROJECT COMPLETED SUCCESSFULLY!")

print("Best Model:", best_name)

print("Submission saved!")

print("Model saved!")