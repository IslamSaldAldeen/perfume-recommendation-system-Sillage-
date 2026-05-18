# Perfume Recommendation System

A practical Python-based perfume recommendation system with an interactive Streamlit interface.

The system recommends perfumes based on the user’s preferred notes and accords,
applies data mining techniques such as TF-IDF,
K-Means Clustering, and Association Rule Mining,
and displays results in a clean luxury-style frontend.

---

## Project Overview

This project helps users discover perfumes that match their scent preferences.

The user can choose:

- notes they love
- notes they dislike
- preferred accords
- gender preference
- minimum weighted rating
- number of recommendations

The system then returns:

- main perfume recommendations
- “Also You May Like” suggestions
- fragrance clusters
- association rules
- dataset insights and visualizations

---

## Folder Structure

```text
perfume_project_with_luxury_frontend/
│
├── app.py                  # Streamlit user interface
├── train.py                # Run once to preprocess data and save models/files
├── requirements.txt        # Required Python libraries
├── README.md               # Project documentation
│
├── data/
│   └── perfumes.xlsx       # Original perfume dataset
│
├── models/
│   ├── tfidf_model.pkl
│   ├── kmeans_model.pkl
│   ├── tfidf_matrix.pkl
│   └── association_rules.pkl
│
├── outputs/
│   └── final_perfume_data.csv
│
└── src/
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── training.py
    ├── association_rules.py
    └── recommender.py

## To Run the Project

1. `pip install -r requirements.txt`  
   Installs all the required libraries needed to run the project.

2. `python train.py`  
   Loads the dataset, preprocesses the data, trains the models, generates the association rules, and saves the required files inside the `models/` and `outputs/` folders.

3. `streamlit run app.py`  
   Starts the Streamlit web application so the user can interact with the perfume recommendation system.
