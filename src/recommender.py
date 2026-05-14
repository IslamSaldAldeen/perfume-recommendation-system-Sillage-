import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from .association_rules import get_associated_items, split_items
from .config import TRANSACTION_COLS


def normalize_list(items):
    """Normalize user input lists."""
    if items is None:
        return []

    return [
        str(item).lower().strip()
        for item in items
        if str(item).strip() != ""
    ]


def generate_reason(row, preferred_notes, preferred_accords=None, gender=None, minimum_rating=None):
    preferred_accords = preferred_accords or []
    user_terms = preferred_notes + preferred_accords

    matched_terms = [
        term for term in user_terms
        if term.lower() in str(row["scent_profile"]).lower()
    ]

    parts = []

    if matched_terms:
        parts.append(f"matches your selected scent preferences: {', '.join(matched_terms)}")
    else:
        parts.append("has a scent profile close to your selected preferences")

    parts.append(f"belongs to the '{row['cluster_name']}' fragrance family")

    if gender and str(gender).lower() != "no preference":
        parts.append(f"matches your gender preference ({gender}) or is unisex")

    if minimum_rating:
        parts.append(f"passes your minimum rating filter of {minimum_rating}")

    parts.append(f"has a weighted rating of {row['Weighted Rating']:.2f}/5")
    parts.append(f"has a scent similarity score of {row['similarity_score'] * 100:.1f}%")

    return "Recommended because it " + ", ".join(parts) + "."


def recommend_perfumes(
    final_data,
    tfidf,
    tfidf_matrix,
    preferred_notes,
    preferred_accords=None,
    disliked_notes=None,
    gender=None,
    minimum_rating=None,
    top_n=5
):
    """Recommend perfumes based on user preferences."""
    preferred_notes = normalize_list(preferred_notes)
    preferred_accords = normalize_list(preferred_accords)
    disliked_notes = normalize_list(disliked_notes)

    if not preferred_notes:
        raise ValueError("preferred_notes is required")

    user_profile = " ".join(preferred_notes + preferred_accords)

    user_vector = tfidf.transform([user_profile])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    results = final_data.copy()
    results["similarity_score"] = similarity_scores

    results["scent_profile"] = (
        results["scent_profile"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    if disliked_notes:
        disliked_pattern = "|".join(map(re.escape, disliked_notes))
        results = results[
            ~results["scent_profile"].str.contains(
                disliked_pattern,
                case=False,
                na=False,
                regex=True
            )
        ]

    if gender is not None and str(gender).strip() != "" and str(gender).lower().strip() != "no preference":
        gender_clean = str(gender).lower().strip()
        results = results[
            (results["Gender"].astype(str).str.lower() == gender_clean) |
            (results["Gender"].astype(str).str.lower() == "unisex")
        ]

    if minimum_rating is not None:
        minimum_rating = float(minimum_rating)
        results = results[results["Weighted Rating"] >= minimum_rating]

    empty_columns = [
        "URL", "Perfume", "Brand", "Gender", "cluster_name",
        "Weighted Rating", "similarity_score", "final_score", "reason"
    ]

    if results.empty:
        return pd.DataFrame(columns=empty_columns)

    scaler = MinMaxScaler()
    results["rating_score"] = scaler.fit_transform(results[["Weighted Rating"]])

    results["final_score"] = (
        0.7 * results["similarity_score"] +
        0.3 * results["rating_score"]
    )

    recommendations = results.sort_values(
        by="final_score",
        ascending=False
    ).head(top_n).copy()

    recommendations["reason"] = recommendations.apply(
        lambda row: generate_reason(
            row,
            preferred_notes,
            preferred_accords,
            gender,
            minimum_rating
        ),
        axis=1
    )

    return recommendations[empty_columns]


def get_also_you_may_like_perfumes(
    associated_items_df,
    data,
    disliked_notes=None,
    gender=None,
    minimum_rating=None,
    exclude_urls=None,
    top_n=5
):
    """Recommend perfumes for the Also You May Like section using association rules."""
    also_columns = [
        "URL", "Perfume", "Brand", "Gender", "Weighted Rating",
        "cluster_name", "matched_associated_items", "association_score"
    ]

    disliked_notes = normalize_list(disliked_notes)
    exclude_urls = normalize_list(exclude_urls)

    if associated_items_df is None or associated_items_df.empty:
        return pd.DataFrame(columns=also_columns)

    if "associated_item" not in associated_items_df.columns:
        return pd.DataFrame(columns=also_columns)

    associated_items = normalize_list(associated_items_df["associated_item"].tolist())

    if not associated_items:
        return pd.DataFrame(columns=also_columns)

    results = data.copy()

    if "URL" in results.columns and len(exclude_urls) > 0:
        results = results[~results["URL"].astype(str).isin(exclude_urls)]

    if results.empty:
        return pd.DataFrame(columns=also_columns)

    if gender is not None and str(gender).strip() != "" and str(gender).lower().strip() != "no preference":
        gender_clean = str(gender).lower().strip()
        results = results[
            (results["Gender"].astype(str).str.lower() == gender_clean) |
            (results["Gender"].astype(str).str.lower() == "unisex")
        ]

    if minimum_rating is not None:
        minimum_rating = float(minimum_rating)
        results = results[results["Weighted Rating"] >= minimum_rating]

    if results.empty:
        return pd.DataFrame(columns=also_columns)

    def get_perfume_items(row):
        perfume_items = []
        for col in TRANSACTION_COLS:
            if col in row.index:
                perfume_items.extend(split_items(row[col]))
        return set(perfume_items)

    results["perfume_items_set"] = results.apply(get_perfume_items, axis=1)

    if disliked_notes:
        results = results[
            ~results["perfume_items_set"].apply(
                lambda items: any(disliked in items for disliked in disliked_notes)
            )
        ]

    if results.empty:
        return pd.DataFrame(columns=also_columns)

    def exact_association_score(items):
        matched_items = [item for item in associated_items if item in items]
        return len(matched_items), matched_items

    scores = results["perfume_items_set"].apply(exact_association_score)

    results["association_score"] = scores.apply(lambda x: x[0])
    results["matched_associated_items"] = scores.apply(lambda x: x[1])

    results = results[results["association_score"] > 0]

    if results.empty:
        return pd.DataFrame(columns=also_columns)

    results = results.sort_values(
        by=["association_score", "Weighted Rating"],
        ascending=False
    )

    return results[also_columns].head(top_n)


def perfume_recommendation_system(
    final_data,
    tfidf,
    tfidf_matrix,
    strong_rules,
    liked_notes=None,
    liked_accords=None,
    disliked_notes=None,
    gender=None,
    minimum_rating=None,
    top_n_main=5,
    top_n_also=5
):
    """Return main recommendations, associated items, and Also You May Like perfumes."""
    liked_notes = normalize_list(liked_notes)
    liked_accords = normalize_list(liked_accords)
    disliked_notes = normalize_list(disliked_notes)

    main_recommendations = recommend_perfumes(
        final_data=final_data,
        tfidf=tfidf,
        tfidf_matrix=tfidf_matrix,
        preferred_notes=liked_notes,
        preferred_accords=liked_accords,
        disliked_notes=disliked_notes,
        gender=gender,
        minimum_rating=minimum_rating,
        top_n=top_n_main
    )

    associated_items = get_associated_items(
        strong_rules=strong_rules,
        liked_notes=liked_notes,
        liked_accords=liked_accords,
        disliked_notes=disliked_notes,
        top_n=10
    )

    main_urls = []
    if not main_recommendations.empty and "URL" in main_recommendations.columns:
        main_urls = main_recommendations["URL"].astype(str).tolist()

    also_you_may_like = get_also_you_may_like_perfumes(
        associated_items_df=associated_items,
        data=final_data,
        disliked_notes=disliked_notes,
        gender=gender,
        minimum_rating=minimum_rating,
        exclude_urls=main_urls,
        top_n=top_n_also
    )

    return main_recommendations, associated_items, also_you_may_like
