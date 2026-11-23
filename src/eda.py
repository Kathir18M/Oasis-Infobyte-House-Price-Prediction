"""
Exploratory Data Analysis (EDA) Module for Ames Housing Price Prediction.

Generates and saves professional visualizations for:
- Target variable distribution (Raw vs Log-transformed)
- Correlation heatmap of top predictive features
- Missing value percentages
- Feature relationships (GrLivArea, TotalSF, HouseAge, OverallQual)
- Outlier detection plots
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.feature_engineering import create_features


def set_plotting_style():
    """Sets a clean, modern aesthetic for all project figures."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 16,
        "figure.autolayout": True
    })


def run_eda(data_path="data/train.csv", output_dir="outputs/figures"):
    """
    Performs full exploratory data analysis and exports visualizations to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_plotting_style()

    print(f"[EDA] Loading training data from {data_path}...")
    df = pd.read_csv(data_path)
    df_feat = create_features(df)

    # 1. Target Distribution (Raw vs Log1p)
    print("[EDA] Generating 01_target_distribution.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df["SalePrice"], kde=True, color="#2b5c8f", ax=axes[0], bins=35)
    axes[0].set_title("Original SalePrice Distribution (Right-Skewed)")
    axes[0].set_xlabel("Sale Price ($)")
    axes[0].set_ylabel("Count")

    sns.histplot(np.log1p(df["SalePrice"]), kde=True, color="#2ca02c", ax=axes[1], bins=35)
    axes[1].set_title("Log-Transformed log1p(SalePrice) (Normal Distribution)")
    axes[1].set_xlabel("log1p(Sale Price)")
    axes[1].set_ylabel("Count")

    plt.savefig(os.path.join(output_dir, "01_target_distribution.png"), dpi=300)
    plt.close()

    # 2. Missing Value Analysis
    print("[EDA] Generating 02_missing_values_bar.png...")
    missing = df.isnull().sum()
    missing_pct = (missing[missing > 0] / len(df)) * 100
    missing_pct = missing_pct.sort_values(ascending=False)

    if not missing_pct.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_pct.values, y=missing_pct.index, hue=missing_pct.index, palette="mako", legend=False)
        plt.title("Percentage of Missing Values by Feature (Train Dataset)")
        plt.xlabel("Missing Percentage (%)")
        plt.ylabel("Features")
        for idx, value in enumerate(missing_pct.values):
            plt.text(value + 0.5, idx, f"{value:.1f}%", va="center", fontsize=9)
        plt.xlim(0, 105)
        plt.savefig(os.path.join(output_dir, "02_missing_values_bar.png"), dpi=300)
        plt.close()

    # 3. Correlation Heatmap (Top 12 numeric features with SalePrice)
    print("[EDA] Generating 03_correlation_heatmap.png...")
    numeric_df = df_feat.select_dtypes(include=np.number).drop(columns=["Id"], errors="ignore")
    corr = numeric_df.corr()
    top_corr_features = corr["SalePrice"].abs().sort_values(ascending=False).head(13).index
    top_corr_matrix = numeric_df[top_corr_features].corr()

    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(top_corr_matrix, dtype=bool))
    sns.heatmap(
        top_corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Top Feature Correlations with SalePrice")
    plt.savefig(os.path.join(output_dir, "03_correlation_heatmap.png"), dpi=300)
    plt.close()

    # 4. OverallQual vs SalePrice Boxplot
    print("[EDA] Generating 04_overall_qual_boxplot.png...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="OverallQual", y="SalePrice", data=df, hue="OverallQual", palette="viridis", legend=False)
    plt.title("Sale Price vs Overall Quality Rating (1 - 10)")
    plt.xlabel("Overall Quality Rating")
    plt.ylabel("Sale Price ($)")
    plt.savefig(os.path.join(output_dir, "04_overall_qual_boxplot.png"), dpi=300)
    plt.close()

    # 5. GrLivArea vs SalePrice (Outlier Identification)
    print("[EDA] Generating 05_scatter_livarea_vs_price.png...")
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x="GrLivArea",
        y="SalePrice",
        hue="OverallQual",
        palette="Spectral",
        data=df,
        alpha=0.75,
        edgecolor="w",
        s=60
    )
    plt.title("Above Ground Living Area vs Sale Price (with Outlier Context)")
    plt.xlabel("Above Ground Living Area (GrLivArea in sq ft)")
    plt.ylabel("Sale Price ($)")
    plt.savefig(os.path.join(output_dir, "05_scatter_livarea_vs_price.png"), dpi=300)
    plt.close()

    # 6. Engineered TotalSF vs SalePrice
    print("[EDA] Generating 06_totalsf_vs_price.png...")
    plt.figure(figsize=(9, 6))
    sns.regplot(
        x="TotalSF",
        y="SalePrice",
        data=df_feat,
        scatter_kws={"alpha": 0.5, "color": "#1f77b4"},
        line_kws={"color": "#d62728", "linewidth": 2}
    )
    plt.title("Engineered Total Square Footage (TotalSF) vs Sale Price")
    plt.xlabel("Total Square Footage (Basement + 1st + 2nd Floor)")
    plt.ylabel("Sale Price ($)")
    plt.savefig(os.path.join(output_dir, "06_totalsf_vs_price.png"), dpi=300)
    plt.close()

    # 7. House Age vs Sale Price
    print("[EDA] Generating 07_house_age_scatter.png...")
    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        x="HouseAge",
        y="SalePrice",
        hue="OverallQual",
        palette="coolwarm",
        data=df_feat,
        alpha=0.7,
        s=55
    )
    plt.title("House Age at Sale Date vs Sale Price")
    plt.xlabel("House Age (Years)")
    plt.ylabel("Sale Price ($)")
    plt.savefig(os.path.join(output_dir, "07_house_age_scatter.png"), dpi=300)
    plt.close()

    print(f"[EDA] All exploratory visualizations successfully generated in {output_dir}/")


if __name__ == "__main__":
    run_eda()
