import joblib
import pandas as pd
import numpy as np

model = joblib.load(
    "models/best_house_price_model.pkl"
)

new_house = pd.read_csv("data/test.csv").head(1)

house_id = new_house["Id"].iloc[0]

new_house = new_house.drop(columns=["Id"])

# Apply the same feature engineering

from src.train_model import create_features

new_house = create_features(new_house)

prediction = np.expm1(
    model.predict(new_house)
)[0]

print("House ID:", house_id)

print(
    "Predicted House Price:",
    round(prediction, 2)
)