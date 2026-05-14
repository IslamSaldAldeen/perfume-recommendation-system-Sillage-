from src.data_loader import load_perfume_data
from src.preprocessing import preprocess_perfume_data
from src.training import train_and_save_models
from src.config import DATA_PATH


def main():
    print(f"Loading dataset from: {DATA_PATH}")

    perfume_data = load_perfume_data(DATA_PATH)
    print(f"Original dataset shape: {perfume_data.shape}")

    cleaned_data = preprocess_perfume_data(perfume_data)
    print(f"Cleaned dataset shape: {cleaned_data.shape}")

    artifacts = train_and_save_models(cleaned_data)

    print("\nSaved successfully:")
    print("- models/tfidf_model.pkl")
    print("- models/kmeans_model.pkl")
    print("- models/tfidf_matrix.pkl")
    print("- models/association_rules.pkl")
    print("- outputs/final_perfume_data.csv")
    print(f"Final data shape: {artifacts['final_data'].shape}")
    print(f"Association rules shape: {artifacts['strong_rules'].shape}")


if __name__ == "__main__":
    main()
