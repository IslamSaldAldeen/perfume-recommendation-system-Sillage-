import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


DATA_PATH = "outputs/final_perfume_data.csv"
OUTPUT_PATH = "outputs/cluster_visualization.png"


def create_text_features(df):
    """
    Combine important perfume columns into one text column for TF-IDF.
    The function checks if each column exists before using it.
    """

    possible_columns = [
        "Perfume",
        "Brand",
        "Gender",
        "Accord 1",
        "Accord 2",
        "Accord 3",
        "Accord 4",
        "Accord 5",
        "Notes",
        "cluster_name",
    ]

    existing_columns = [col for col in possible_columns if col in df.columns]

    if not existing_columns:
        raise ValueError("No suitable text columns found for visualization.")

    df["visual_text"] = df[existing_columns].fillna("").astype(str).agg(" ".join, axis=1)

    return df


def visualize_clusters():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Run train.py first to generate final_perfume_data.csv."
        )

    df = pd.read_csv(DATA_PATH)

    if "cluster_name" not in df.columns:
        raise ValueError("The dataset must contain a 'cluster_name' column.")

    df = create_text_features(df)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["visual_text"])

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())

    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    plt.figure(figsize=(12, 8))

    for cluster_name in df["cluster_name"].unique():
        cluster_data = df[df["cluster_name"] == cluster_name]

        plt.scatter(
            cluster_data["PCA1"],
            cluster_data["PCA2"],
            label=cluster_name,
            alpha=0.7,
            s=40,
        )

    plt.title("Perfume Clusters Visualization using PCA", fontsize=16)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Fragrance Family", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.show()

    print(f"Cluster visualization saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    visualize_clusters()