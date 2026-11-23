"""
Standalone Model Inference Script for House Price Prediction.

Demonstrates loading the serialized Scikit-learn Pipeline and predicting
residential house prices for new samples WITHOUT retraining the model.
"""

import time
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import create_features

MODEL_PATH = "models/best_house_price_model.pkl"
TEST_DATA_PATH = "data/test.csv"


def load_model(model_path=MODEL_PATH):
    """Loads the pre-trained Scikit-Learn Pipeline from disk."""
    print(f"[Inference] Loading trained model from '{model_path}'...")
    return joblib.load(model_path)


def predict_sample(sample_df: pd.DataFrame, model) -> float:
    """
    Applies feature engineering and infers sale price for a given sample.
    """
    # Apply feature engineering
    processed_sample = create_features(sample_df)

    # Predict in log scale and invert with expm1
    pred_log = model.predict(processed_sample)
    prediction = np.maximum(0, np.expm1(pred_log))[0]
    return float(prediction)


def main():
    start_time = time.time()
    model = load_model()

    # Load first 5 samples from test set for demonstration
    test_df = pd.read_csv(TEST_DATA_PATH)
    sample_houses = test_df.head(5)

    print("\n" + "=" * 65)
    print("        INDEPENDENT INFERENCE DEMO (NO RETRAINING)")
    print("=" * 65)

    for i in range(len(sample_houses)):
        house = sample_houses.iloc[[i]]
        house_id = house["Id"].values[0]
        feature_subset = house.drop(columns=["Id"], errors="ignore")

        pred_price = predict_sample(feature_subset, model)

        overall_qual = house["OverallQual"].values[0] if "OverallQual" in house.columns else "N/A"
        gr_liv_area = house["GrLivArea"].values[0] if "GrLivArea" in house.columns else "N/A"
        year_built = house["YearBuilt"].values[0] if "YearBuilt" in house.columns else "N/A"

        print(f"House ID: {house_id}")
        print(f"  • Overall Quality : {overall_qual} / 10")
        print(f"  • GrLivArea       : {gr_liv_area:,} sq ft")
        print(f"  • Year Built      : {year_built}")
        print(f"  • Predicted Price : ${pred_price:,.2f}")
        print("-" * 65)

    elapsed = (time.time() - start_time) * 1000
    print(f"\nInference completed in {elapsed:.2f} ms.")


if __name__ == "__main__":
    main()
