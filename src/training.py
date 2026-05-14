import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import (
    TFIDF_MODEL_PATH,
    KMEANS_MODEL_PATH,
    TFIDF_MATRIX_PATH,
    ASSOCIATION_RULES_PATH,
    FINAL_DATA_PATH,
)
from .preprocessing import add_cluster_names, build_final_data
from .association_rules import build_association_rules


def train_tfidf(data):
    """Train TF-IDF vectorizer on scent_profile."""
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["scent_profile"])
    return tfidf, tfidf_matrix


def train_kmeans(tfidf_matrix, n_clusters=10):
    """Train KMeans clustering model."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    kmeans.fit(tfidf_matrix)
    return kmeans


def train_and_save_models(cleaned_data):
    """Train TF-IDF, KMeans, association rules, and save all project artifacts."""
    tfidf, tfidf_matrix = train_tfidf(cleaned_data)
    kmeans = train_kmeans(tfidf_matrix, n_clusters=10)

    cleaned_data = cleaned_data.copy()
    cleaned_data["cluster"] = kmeans.labels_
    cleaned_data = add_cluster_names(cleaned_data)

    final_data = build_final_data(cleaned_data)
    strong_rules = build_association_rules(cleaned_data)

    TFIDF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(tfidf, TFIDF_MODEL_PATH)
    joblib.dump(kmeans, KMEANS_MODEL_PATH)
    joblib.dump(tfidf_matrix, TFIDF_MATRIX_PATH)
    joblib.dump(strong_rules, ASSOCIATION_RULES_PATH)

    final_data.to_csv(FINAL_DATA_PATH, index=False)

    return {
        "tfidf": tfidf,
        "kmeans": kmeans,
        "tfidf_matrix": tfidf_matrix,
        "strong_rules": strong_rules,
        "final_data": final_data,
    }


def load_artifacts():
    """Load saved files used by the app."""
    import pandas as pd

    tfidf = joblib.load(TFIDF_MODEL_PATH)
    kmeans = joblib.load(KMEANS_MODEL_PATH)
    tfidf_matrix = joblib.load(TFIDF_MATRIX_PATH)
    strong_rules = joblib.load(ASSOCIATION_RULES_PATH)
    final_data = pd.read_csv(FINAL_DATA_PATH)

    return tfidf, kmeans, tfidf_matrix, strong_rules, final_data
