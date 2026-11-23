"""
Feature Engineering Module for Ames Housing Price Prediction.

This module provides reusable feature engineering functions designed
to enhance tabular signals without causing data leakage.
"""

import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates engineered domain features from the Ames housing dataset.
    
    Features engineered:
    - TotalSF: Total square footage combining basement, 1st floor, and 2nd floor.
    - TotalBathrooms: Combined full and half baths (including basement).
    - HouseAge: Age of the house at the time of sale.
    - YearsSinceRemodel: Years between remodel and sale date.
    - TotalPorchSF: Aggregate porch and deck square footage.
    - HasPool: Binary flag indicating presence of a pool.
    - Has2ndFloor: Binary flag indicating presence of a second floor.
    - HasGarage: Binary flag indicating presence of a garage.
    - HasBsmt: Binary flag indicating presence of a basement.

    Parameters
    ----------
    df : pd.DataFrame
        Input housing dataset containing raw Ames features.

    Returns
    -------
    pd.DataFrame
        DataFrame with new engineered features appended.
    """
    df = df.copy()

    # Total square footage across all finished levels
    total_bsmt = df["TotalBsmtSF"].fillna(0) if "TotalBsmtSF" in df.columns else 0
    first_flr = df["1stFlrSF"].fillna(0) if "1stFlrSF" in df.columns else 0
    second_flr = df["2ndFlrSF"].fillna(0) if "2ndFlrSF" in df.columns else 0
    df["TotalSF"] = total_bsmt + first_flr + second_flr

    # Comprehensive bathroom count
    full_bath = df["FullBath"].fillna(0) if "FullBath" in df.columns else 0
    half_bath = df["HalfBath"].fillna(0) if "HalfBath" in df.columns else 0
    bsmt_full_bath = df["BsmtFullBath"].fillna(0) if "BsmtFullBath" in df.columns else 0
    bsmt_half_bath = df["BsmtHalfBath"].fillna(0) if "BsmtHalfBath" in df.columns else 0
    df["TotalBathrooms"] = full_bath + (0.5 * half_bath) + bsmt_full_bath + (0.5 * bsmt_half_bath)

    # Temporal / age-based features (clipped at 0 to avoid negative values)
    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = np.maximum(0, df["YrSold"] - df["YearBuilt"])
    if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
        df["YearsSinceRemodel"] = np.maximum(0, df["YrSold"] - df["YearRemodAdd"])

    # Total porch and deck square footage
    wood_deck = df["WoodDeckSF"].fillna(0) if "WoodDeckSF" in df.columns else 0
    open_porch = df["OpenPorchSF"].fillna(0) if "OpenPorchSF" in df.columns else 0
    enclosed_porch = df["EnclosedPorch"].fillna(0) if "EnclosedPorch" in df.columns else 0
    three_ssn_porch = df["3SsnPorch"].fillna(0) if "3SsnPorch" in df.columns else 0
    screen_porch = df["ScreenPorch"].fillna(0) if "ScreenPorch" in df.columns else 0
    df["TotalPorchSF"] = wood_deck + open_porch + enclosed_porch + three_ssn_porch + screen_porch

    # Amenity indicators
    if "PoolArea" in df.columns:
        df["HasPool"] = (df["PoolArea"].fillna(0) > 0).astype(int)
    if "2ndFlrSF" in df.columns:
        df["Has2ndFloor"] = (df["2ndFlrSF"].fillna(0) > 0).astype(int)
    if "GarageArea" in df.columns:
        df["HasGarage"] = (df["GarageArea"].fillna(0) > 0).astype(int)
    if "TotalBsmtSF" in df.columns:
        df["HasBsmt"] = (df["TotalBsmtSF"].fillna(0) > 0).astype(int)

    return df
