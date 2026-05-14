import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from .config import TRANSACTION_COLS


def split_items(text):
    """
    Split perfume notes/accords into clean individual items.
    Example: 'vanilla, amber, musk' -> ['vanilla', 'amber', 'musk']
    """
    if pd.isna(text):
        return []

    text = str(text).lower().strip()
    items = [item.strip() for item in text.split(",")]
    return [item for item in items if item != ""]


def create_transactions(data):
    """Convert perfume rows into transaction lists."""
    transactions = []

    for _, row in data.iterrows():
        perfume_items = []

        for col in TRANSACTION_COLS:
            if col in row.index:
                perfume_items.extend(split_items(row[col]))

        perfume_items = list(set(perfume_items))
        transactions.append(perfume_items)

    return transactions


def build_association_rules(data, min_support=0.02, min_confidence=0.3):
    """Build strong association rules from notes and accords."""
    transactions = create_transactions(data)

    te = TransactionEncoder()
    transaction_array = te.fit(transactions).transform(transactions)

    transactions_encoded = pd.DataFrame(
        transaction_array,
        columns=te.columns_
    )

    frequent_itemsets = apriori(
        transactions_encoded,
        min_support=min_support,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        return pd.DataFrame(
            columns=["antecedents", "consequents", "support", "confidence", "lift"]
        )

    rules_simple = rules[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ].copy()

    strong_rules = rules_simple[
        (rules_simple["confidence"] >= 0.4) &
        (rules_simple["lift"] >= 1.2)
    ].copy()

    strong_rules = strong_rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False
    )

    return strong_rules


def get_associated_items(strong_rules, liked_notes=None, liked_accords=None, disliked_notes=None, top_n=10):
    """Find related perfume notes/accords using association rules."""
    liked_notes = liked_notes or []
    liked_accords = liked_accords or []
    disliked_notes = disliked_notes or []

    liked_notes = [
        str(item).lower().strip()
        for item in liked_notes
        if str(item).strip() != ""
    ]

    liked_accords = [
        str(item).lower().strip()
        for item in liked_accords
        if str(item).strip() != ""
    ]

    disliked_notes = [
        str(item).lower().strip()
        for item in disliked_notes
        if str(item).strip() != ""
    ]

    user_likes = set(liked_notes + liked_accords)
    user_dislikes = set(disliked_notes)

    associated_results = []

    if strong_rules is None or strong_rules.empty:
        return pd.DataFrame(
            columns=[
                "associated_item",
                "matched_input",
                "rule_antecedents",
                "confidence",
                "lift",
                "support",
            ]
        )

    for _, rule in strong_rules.iterrows():
        antecedents = set(rule["antecedents"])
        consequents = set(rule["consequents"])

        if user_dislikes.intersection(antecedents) or user_dislikes.intersection(consequents):
            continue

        if len(user_likes.intersection(antecedents)) > 0:
            for item in consequents:
                if item in user_likes or item in user_dislikes:
                    continue

                associated_results.append({
                    "associated_item": item,
                    "matched_input": list(user_likes.intersection(antecedents)),
                    "rule_antecedents": list(antecedents),
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                    "support": rule["support"],
                })

    associated_df = pd.DataFrame(associated_results)

    if associated_df.empty:
        return pd.DataFrame(
            columns=[
                "associated_item",
                "matched_input",
                "rule_antecedents",
                "confidence",
                "lift",
                "support",
            ]
        )

    associated_df = associated_df.sort_values(
        by=["lift", "confidence", "support"],
        ascending=False
    )

    associated_df = associated_df.drop_duplicates(
        subset=["associated_item"],
        keep="first"
    )

    return associated_df.head(top_n)
