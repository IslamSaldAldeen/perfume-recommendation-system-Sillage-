import numpy as np
import pandas as pd

from .config import COLUMNS_TO_KEEP, ACCORD_COLS, TEXT_COLS, CLUSTER_NAMES


def select_needed_columns(perfume_data: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed for the recommendation system."""
    missing_cols = [col for col in COLUMNS_TO_KEEP if col not in perfume_data.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    return perfume_data[COLUMNS_TO_KEEP].copy()


def fix_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Convert rating columns to numbers and fix rating scale if needed."""
    data = data.copy()

    data["Rating Value"] = pd.to_numeric(data["Rating Value"], errors="coerce")
    data["Rating Count"] = pd.to_numeric(data["Rating Count"], errors="coerce")

    # Original values may appear like 393 instead of 3.93
    data["Rating Value"] = np.where(
        data["Rating Value"] > 5,
        data["Rating Value"] / 100,
        data["Rating Value"]
    )

    return data


def add_weighted_rating(data: pd.DataFrame) -> pd.DataFrame:
    """Create Weighted Rating using rating value and rating count."""
    data = data.copy()

    c = data["Rating Value"].mean()
    m = data["Rating Count"].quantile(0.75)
    v = data["Rating Count"]
    r = data["Rating Value"]

    data["Weighted Rating"] = (v / (v + m)) * r + (m / (v + m)) * c

    return data


def clean_text_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize text columns and fill missing accord values row-wise."""
    data = data.copy()

    # These columns are used as text, so standardize them
    for col in TEXT_COLS:
        data[col] = data[col].fillna("").astype(str).str.lower().str.strip()

    # Clean accord columns while preserving NaN first
    for col in ACCORD_COLS:
        data[col] = data[col].astype("string").str.lower().str.strip()

    # Fill missing mainaccord2-mainaccord5 from the previous accord in the same row only
    data[ACCORD_COLS] = data[ACCORD_COLS].T.ffill().T

    return data


def create_scent_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Combine notes and accords into one text column for TF-IDF."""
    data = data.copy()

    profile_cols = ["Top", "Middle", "Base"] + ACCORD_COLS

    data["scent_profile"] = (
        data[profile_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
    )

    return data


def build_final_data(data: pd.DataFrame) -> pd.DataFrame:
    """Select final columns used by the app and recommendation system."""
    final_columns = [
        "URL",
        "Perfume",
        "Brand",
        "Gender",
        "Rating Value",
        "Rating Count",
        "Weighted Rating",
        "Top",
        "Middle",
        "Base",
        "mainaccord1",
        "mainaccord2",
        "mainaccord3",
        "mainaccord4",
        "mainaccord5",
        "scent_profile",
        "cluster",
        "cluster_name",
    ]

    return data[final_columns].copy()


def preprocess_perfume_data(perfume_data: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline before TF-IDF and clustering."""
    data = select_needed_columns(perfume_data)
    data = fix_numeric_columns(data)
    data = add_weighted_rating(data)
    data = clean_text_columns(data)
    data = create_scent_profile(data)
    return data


def add_cluster_names(data: pd.DataFrame) -> pd.DataFrame:
    """Map numeric cluster labels to human-readable cluster names."""
    data = data.copy()
    data["cluster_name"] = data["cluster"].map(CLUSTER_NAMES)
    return data
