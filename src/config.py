from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_PATH = DATA_DIR / "perfumes.xlsx"

TFIDF_MODEL_PATH = MODEL_DIR / "tfidf_model.pkl"
KMEANS_MODEL_PATH = MODEL_DIR / "kmeans_model.pkl"
TFIDF_MATRIX_PATH = MODEL_DIR / "tfidf_matrix.pkl"
ASSOCIATION_RULES_PATH = MODEL_DIR / "association_rules.pkl"

FINAL_DATA_PATH = OUTPUT_DIR / "final_perfume_data.csv"

COLUMNS_TO_KEEP = [
    "URL", "Perfume", "Brand", "Gender",
    "Rating Value", "Rating Count",
    "Top", "Middle", "Base",
    "mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5"
]

ACCORD_COLS = ["mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5"]

TEXT_COLS = ["Perfume", "Brand", "Gender", "Top", "Middle", "Base"]

TRANSACTION_COLS = [
    "Top", "Middle", "Base",
    "mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5"
]

CLUSTER_NAMES = {
    0: "Floral Yellow White Flowers",
    1: "Warm Spicy Woody",
    2: "Oud Leather Oriental",
    3: "Powdery Iris Musky",
    4: "Sweet Vanilla Gourmand",
    5: "Fresh Green Aromatic",
    6: "Rose Fruity Floral",
    7: "Dark Leather Tobacco",
    8: "White Floral Citrus",
    9: "Fresh Lavender Aromatic",
}
