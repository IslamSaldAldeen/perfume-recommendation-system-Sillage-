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
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── data/
│   └── perfumes.xlsx
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
```

## To Run the Project

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Train/preprocess the system:

```bash
python train.py
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```
   Starts the Streamlit web application so the user can interact with the perfume recommendation system.
