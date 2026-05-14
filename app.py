import ast
import html
import re
from collections import Counter

import pandas as pd
import streamlit as st
import altair as alt

from src.training import load_artifacts
from src.recommender import perfume_recommendation_system


st.set_page_config(
    page_title="Sillage — Perfume Intelligence System",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Styling inspired by website.html ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600;700&family=DM+Mono:wght@300;400&display=swap');

    :root {
        --bg: #fff7fb;
        --bg2: #fffafd;
        --bg3: #fff1f7;
        --panel: #ffffff;
        --border: #f2cddd;
        --border2: #e7aec6;
        --gold: #b88a52;
        --gold2: #d6b16b;
        --cream: #33202a;
        --muted: #8e6b78;
        --text: #3f2a34;
        --text2: #624653;
        --rose: #b97a95;
        --rose2: #d9aec1;
        --lavender: #9b8ac4;
        --mint: #89a99b;
        --shadow: rgba(118, 82, 98, 0.13);
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 8%, rgba(241,166,197,0.30), transparent 28%),
            radial-gradient(circle at 92% 10%, rgba(217,168,95,0.20), transparent 26%),
            linear-gradient(180deg, #fff7fb 0%, #fffaf5 48%, #fff7fb 100%) !important;
        color: var(--text);
        font-size: 22px !important;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff1f7 0%, #fffaf5 100%);
        border-right: 1px solid var(--border);
    }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        color: var(--cream) !important;
        font-weight: 600 !important;
        letter-spacing: 0.035em;
    }

    h2 { font-size: 2.65rem !important; }
    h3 { font-size: 2.22rem !important; }

    p, label, div, span {
        font-family: 'Cormorant Garamond', Georgia, serif;
    }

    label, .stMarkdown, .stText, .stSelectbox, .stMultiSelect {
        font-size: 1.28rem !important;
    }

    .mono, .stat-label, .chip, .small-muted, .stButton button, .stSelectbox label,
    .stTextInput label, .stSlider label, .stMultiSelect label, .stNumberInput label {
        font-family: 'DM Mono', monospace !important;
    }

    .hero {
        text-align: center;
        padding: 3.8rem 1.5rem 2.6rem;
        border: 1px solid var(--border);
        border-radius: 30px;
        margin-bottom: 1.5rem;
        background:
            radial-gradient(circle at 50% 0%, rgba(255,255,255,0.88), transparent 54%),
            linear-gradient(135deg, rgba(255,241,247,0.95), rgba(255,250,245,0.96));
        box-shadow: 0 18px 55px var(--shadow);
        position: relative;
        overflow: hidden;
    }
    .hero:before, .hero:after {
        content: "";
        position: absolute;
        border-radius: 999px;
        filter: blur(1px);
        opacity: 0.75;
    }
    .hero:before {
        width: 180px; height: 180px;
        background: rgba(241,166,197,0.22);
        top: -70px; left: -50px;
    }
    .hero:after {
        width: 220px; height: 220px;
        background: rgba(217,168,95,0.15);
        bottom: -90px; right: -60px;
    }
    .hero .logo {
        font-size: 2rem;
        color: var(--rose);
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.32em;
        text-transform: uppercase;
        margin-bottom: 0.9rem;
        position: relative;
        z-index: 1;
    }
    .hero h1 {
        font-size: clamp(3.8rem, 6.5vw, 6.6rem) !important;
        margin: 0;
        line-height: 0.98;
        position: relative;
        z-index: 1;
    }
    .hero em {
        color: var(--rose);
        font-style: italic;
        text-shadow: 0 6px 24px rgba(216,111,155,0.18);
    }
    .hero p {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 1.02rem;
        margin-top: 1.2rem;
        position: relative;
        z-index: 1;
    }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 1.6rem 0 1.9rem;
    }
    .stat-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.4rem 1rem;
        text-align: center;
        box-shadow: 0 12px 30px var(--shadow);
    }
    .stat-number {
        display: block;
        color: var(--rose);
        font-size: 2.65rem;
        line-height: 1;
        font-weight: 700;
    }
    .stat-label {
        display: block;
        color: var(--muted);
        font-size: 0.82rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    .section-box {
        background: rgba(255,255,255,0.78);
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.35rem;
        margin-bottom: 1rem;
        min-height: 180px;
        box-shadow: 0 12px 30px var(--shadow);
    }
    .section-box h3 {
        color: var(--rose) !important;
        margin-top: 0.35rem;
    }
    .explain-box {
        background: rgba(255,255,255,0.76);
        border: 1px solid var(--border);
        border-left: 5px solid var(--rose);
        border-radius: 18px;
        padding: 1.15rem 1.3rem;
        margin: 1rem 0;
        color: var(--text2);
        line-height: 1.8;
        font-size: 1.32rem;
        box-shadow: 0 10px 28px var(--shadow);
    }

    .perfume-card {
        background:
            linear-gradient(135deg, rgba(255,255,255,0.90), rgba(255,241,247,0.78));
        border: 1px solid var(--border);
        border-left: 6px solid var(--rose);
        border-radius: 24px;
        padding: 1.3rem 1.45rem;
        margin-bottom: 1rem;
        transition: 0.2s ease;
        box-shadow: 0 12px 32px var(--shadow);
    }
    .perfume-card:hover {
        border-color: var(--rose2);
        transform: translateY(-2px);
        box-shadow: 0 18px 44px rgba(135,79,105,0.20);
    }
    .perfume-card-inner {
        display: flex;
        gap: 1.15rem;
        align-items: flex-start;
    }
    .perfume-img-wrap {
        flex: 0 0 118px;
        min-height: 148px;
        background: linear-gradient(135deg, #fffdfb, #f8eef3);
        border: 1px solid var(--border);
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.90), 0 8px 20px rgba(118,82,98,0.10);
    }
    .perfume-img {
        width: 100%;
        height: 148px;
        object-fit: contain;
        padding: 0.45rem;
        mix-blend-mode: multiply;
    }
    .perfume-img-placeholder {
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.07em;
        text-align: center;
        padding: 0.7rem;
        text-transform: uppercase;
    }
    .perfume-card-content {
        flex: 1;
        min-width: 0;
    }
    .perfume-name {
        font-size: 2.05rem;
        color: var(--cream);
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 0.1rem;
    }
    .perfume-brand {
        font-family: 'DM Mono', monospace;
        font-size: 0.9rem;
        color: var(--rose);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.7rem;
    }
    .small-muted {
        color: var(--muted);
        font-size: 1.02rem;
        letter-spacing: 0.06em;
    }
    .chip {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border: 1px solid var(--border2);
        background: #fff7fb;
        color: var(--text2);
        font-size: 0.96rem;
        margin: 0.22rem 0.25rem 0.16rem 0;
        border-radius: 999px;
    }
    .score {
        color: var(--gold);
        font-family: 'DM Mono', monospace;
        font-size: 0.96rem;
        font-weight: 600;
    }
    .why {
        color: var(--text2);
        margin-top: 0.7rem;
        font-size: 1.28rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 10px 28px var(--shadow);
    }
    div[data-testid="stMetricValue"] {
        color: var(--rose);
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 2rem;
    }

    .stButton button {
        background: linear-gradient(135deg, var(--rose), var(--gold2)) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 1.02rem;
        padding: 0.9rem 1.4rem;
        box-shadow: 0 12px 30px rgba(216,111,155,0.28);
    }
    .stButton button:hover {
        filter: brightness(1.04);
        box-shadow: 0 16px 40px rgba(216,111,155,0.35);
        transform: translateY(-1px);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.55rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.45rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.76);
        border: 1px solid var(--border);
        border-radius: 999px;
        color: var(--muted);
        font-family: 'DM Mono', monospace;
        font-size: 0.98rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.55rem 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #fff1f7, #fff7e9) !important;
        color: var(--rose) !important;
        border-color: var(--rose2) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"] input,
    textarea,
    .stSlider {
        border-radius: 16px !important;
    }



    /* Pretty form container */
    div[data-testid="stForm"] {
        background: rgba(255,255,255,0.82);
        border: 1px solid #dbc4cf;
        border-radius: 26px;
        padding: 1.25rem 1.25rem 1.4rem;
        box-shadow: 0 18px 45px rgba(118,82,98,0.12);
    }

    /* Selectbox / multiselect main boxes */
    div[data-baseweb="select"] > div {
        background: linear-gradient(135deg, #fffdfb, #f7edf2) !important;
        border: 1px solid #d8bdca !important;
        border-radius: 18px !important;
        min-height: 52px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.85), 0 8px 20px rgba(118,82,98,0.07) !important;
    }
    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="select"] > div:focus-within {
        border-color: #b97a95 !important;
        box-shadow: 0 0 0 4px rgba(185,122,149,0.12), 0 10px 24px rgba(118,82,98,0.10) !important;
    }
    div[data-baseweb="select"] input {
        color: #3f2a34 !important;
        font-size: 1.12rem !important;
        font-family: 'Cormorant Garamond', Georgia, serif !important;
    }
    div[data-baseweb="select"] svg {
        color: #8e6b78 !important;
        fill: #8e6b78 !important;
    }

    /* Multiselect selected pills */
    span[data-baseweb="tag"] {
        background: linear-gradient(135deg, #efe0e8, #f8eef3) !important;
        color: #3f2a34 !important;
        border: 1px solid #d8bdca !important;
        border-radius: 999px !important;
        min-height: 34px !important;
        padding-left: 0.55rem !important;
        padding-right: 0.45rem !important;
        box-shadow: 0 4px 12px rgba(118,82,98,0.10) !important;
    }
    span[data-baseweb="tag"] span {
        color: #3f2a34 !important;
        font-size: 1.02rem !important;
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        font-weight: 600 !important;
    }
    span[data-baseweb="tag"] svg {
        color: #624653 !important;
        fill: #624653 !important;
    }

    /* Dropdown menu */
    div[data-baseweb="popover"] ul,
    ul[role="listbox"] {
        background: #fffdfb !important;
        border: 1px solid #d8bdca !important;
        border-radius: 18px !important;
        box-shadow: 0 18px 45px rgba(118,82,98,0.18) !important;
        padding: 0.35rem !important;
    }
    li[role="option"] {
        color: #3f2a34 !important;
        font-size: 1.08rem !important;
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        border-radius: 14px !important;
        margin: 0.15rem 0 !important;
    }
    li[role="option"]:hover,
    li[aria-selected="true"] {
        background: #f2e5ec !important;
        color: #33202a !important;
    }

    /* Larger labels and cleaner slider colors */
    .stMultiSelect label, .stSelectbox label, .stSlider label {
        color: #624653 !important;
        font-size: 1.17rem !important;
        font-weight: 600 !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #b97a95 !important;
        border-color: #b97a95 !important;
        box-shadow: 0 0 0 6px rgba(185,122,149,0.12) !important;
    }
    .stSlider [data-baseweb="slider"] > div {
        color: #b97a95 !important;
    }

    /* Main button softer */
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #b97a95, #d6b16b) !important;
        color: #fffdfb !important;
        font-size: 1.12rem !important;
        padding: 0.95rem 1.6rem !important;
        border-radius: 999px !important;
        text-transform: none !important;
        letter-spacing: 0.04em !important;
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        font-weight: 700 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 28px var(--shadow);
    }

    @media (max-width: 900px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .stat-grid { grid-template-columns: repeat(2, 1fr); }
        .hero h1 { font-size: 3rem !important; }
        .perfume-card-inner { gap: 0.85rem; }
        .perfume-img-wrap { flex-basis: 92px; min-height: 118px; border-radius: 16px; }
        .perfume-img { height: 118px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_load_artifacts():
    return load_artifacts()


def parse_items(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip().lower() for x in value if str(x).strip()]
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [str(x).strip().lower() for x in parsed if str(x).strip()]
    except Exception:
        pass
    parts = re.split(r",|;|\||/", value)
    return [p.strip().lower() for p in parts if p.strip()]


def pretty(value):
    return str(value).replace("-", " ").title()


def get_all_notes(data):
    notes = []
    for col in ["Top", "Middle", "Base"]:
        if col in data.columns:
            for value in data[col].dropna().head(20000):
                notes.extend(parse_items(value))
    return sorted(set(notes))


def get_all_accords(data):
    accord_cols = [c for c in data.columns if c.lower().startswith("mainaccord")]
    accords = []
    for col in accord_cols:
        accords.extend(data[col].dropna().astype(str).str.strip().str.lower().tolist())
    return sorted(set([a for a in accords if a and a != "nan"]))


def item_counter(data, cols, limit=15):
    counter = Counter()
    for col in cols:
        if col in data.columns:
            for value in data[col].dropna():
                counter.update(parse_items(value))
    return pd.DataFrame(counter.most_common(limit), columns=["Item", "Count"])


def accord_counter(data, limit=15):
    accord_cols = [c for c in data.columns if c.lower().startswith("mainaccord")]
    values = []
    for col in accord_cols:
        values.extend(data[col].dropna().astype(str).str.strip().str.lower().tolist())
    values = [v for v in values if v and v != "nan"]
    return pd.DataFrame(Counter(values).most_common(limit), columns=["Accord", "Count"])



def render_pretty_bar_chart(df, x_col, y_col, title=""):
    if df.empty:
        st.info("No data available.")
        return
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X(f"{x_col}:N", sort="-y", axis=alt.Axis(labelAngle=-35, labelFontSize=17, title=None)),
            y=alt.Y(f"{y_col}:Q", axis=alt.Axis(labelFontSize=17, title=None)),
            color=alt.value("#b97a95"),
            tooltip=[x_col, y_col],
        )
        .properties(height=330)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="#ead9e2", domainColor="#d8bdca", tickColor="#d8bdca")
        .configure_title(fontSize=22, color="#3b2430", font="Cormorant Garamond")
    )
    st.altair_chart(chart, use_container_width=True)



def get_first_available_url(row):
    """Return the first usable URL/link-like value from a perfume row."""
    if row is None:
        return None

    preferred_columns = [
        "url", "URL", "Url", "link", "Link", "perfume_url", "Perfume URL",
        "Fragrantica URL", "fragrantica_url", "href", "page_url", "Page URL",
        "Image URL", "image_url", "img_url", "image", "Image",
    ]

    for col in preferred_columns:
        try:
            value = row.get(col, None)
        except Exception:
            value = None
        if value is not None and pd.notna(value) and str(value).strip():
            return str(value).strip()

    try:
        items = row.items()
    except Exception:
        return None

    for col, value in items:
        col_name = str(col).lower()
        if any(key in col_name for key in ["url", "link", "image", "img"]):
            if value is not None and pd.notna(value) and str(value).strip():
                return str(value).strip()

    return None


def get_fragrantica_image_url(perfume_page_url):
    """
    Convert a Fragrantica perfume page URL into the matching fimgs image URL.
    Also supports direct image URLs if the dataset already contains them.
    """
    if perfume_page_url is None or pd.isna(perfume_page_url):
        return None

    url = str(perfume_page_url).strip()
    if not url or url.lower() == "nan":
        return None

    if re.search(r"\.(jpg|jpeg|png|webp)(\?.*)?$", url, flags=re.IGNORECASE):
        return url

    match = re.search(r"-(\d+)\.html", url)
    if not match:
        return None

    perfume_id = match.group(1)
    return f"https://fimgs.net/mdimg/perfume/375x500.{perfume_id}.jpg"


def build_perfume_image_html(row):
    page_or_image_url = get_first_available_url(row)
    image_url = get_fragrantica_image_url(page_or_image_url)

    if not image_url:
        return '<div class="perfume-img-wrap"><div class="perfume-img-placeholder">No image</div></div>'

    safe_image_url = html.escape(image_url, quote=True)
    return (
        '<div class="perfume-img-wrap">'
        f'<img class="perfume-img" src="{safe_image_url}" alt="Perfume bottle image" referrerpolicy="no-referrer" />'
        '</div>'
    )


def render_perfume_card(row, rank=None, also=False):
    name = html.escape(str(row.get("Perfume", "Unknown perfume")))
    brand = html.escape(str(row.get("Brand", "Unknown brand")))
    gender = html.escape(str(row.get("Gender", "—")))
    family = html.escape(str(row.get("cluster_name", "—")))
    rating = row.get("Weighted Rating", None)
    final_score = row.get("final_score", row.get("association_score", None))
    reason = html.escape(str(row.get("reason", "")))
    image_html = build_perfume_image_html(row)

    chips = ""
    if also and "matched_associated_items" in row:
        matched = row.get("matched_associated_items", [])
        if isinstance(matched, str):
            matched = parse_items(matched)
        chips = "".join([f'<span class="chip">{html.escape(pretty(x))}</span>' for x in matched[:8]])

    rank_html = f'<span class="small-muted">#{rank}</span>' if rank else ""
    rating_html = f'<span class="score">Rating {float(rating):.2f}/5</span>' if pd.notna(rating) else ""
    score_html = f'<span class="score"> · Score {float(final_score):.3f}</span>' if pd.notna(final_score) else ""

    st.markdown(
        f"""
        <div class="perfume-card">
            <div class="perfume-card-inner">
                {image_html}
                <div class="perfume-card-content">
                    {rank_html}
                    <div class="perfume-name">{name}</div>
                    <div class="perfume-brand">{brand}</div>
                    <div class="small-muted">Gender: {gender} · Family: {family}</div>
                    <div style="margin-top:0.45rem;">{rating_html}{score_html}</div>
                    <div>{chips}</div>
                    <div class="why">{reason}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


try:
    tfidf, kmeans, tfidf_matrix, strong_rules, final_data = cached_load_artifacts()
except Exception as e:
    st.error("Models/data are not ready yet. Run this command first:")
    st.code("python train.py")
    st.exception(e)
    st.stop()

all_notes = get_all_notes(final_data)
all_accords = get_all_accords(final_data)

st.markdown(
    """
    <div class="hero">
        <div class="logo">sillage</div>
        <h1>Perfume <em>Intelligence</em><br/>Recommendation System</h1>
        <p>Soft Luxury Design · K-Means Clustering · TF-IDF · Content-Based Recommender · Association Rule Mining</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="stat-grid">
        <div class="stat-card"><span class="stat-number">{len(final_data):,}</span><span class="stat-label">Perfumes</span></div>
        <div class="stat-card"><span class="stat-number">{final_data['Brand'].nunique() if 'Brand' in final_data.columns else 0:,}</span><span class="stat-label">Brands</span></div>
        <div class="stat-card"><span class="stat-number">{len(all_notes):,}</span><span class="stat-label">Unique Notes</span></div>
        <div class="stat-card"><span class="stat-number">{final_data['cluster_name'].nunique() if 'cluster_name' in final_data.columns else 0}</span><span class="stat-label">Clusters</span></div>
        <div class="stat-card"><span class="stat-number">{len(strong_rules):,}</span><span class="stat-label">Assoc. Rules</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, rec_tab, clusters_tab, rules_tab, insights_tab = st.tabs(
    ["✦ Overview", "💗 Recommender", "🌸 Clusters", "✨ Associations", "📊 Insights"]
)

with overview_tab:
    st.markdown("### Project Pipeline")
    cols = st.columns(5)
    steps = [
        ("01", "Data Cleaning", "Clean ratings, notes, accords, and missing values."),
        ("02", "Feature Engineering", "Build scent_profile from notes and accords."),
        ("03", "TF-IDF", "Transform perfume text profiles into numeric vectors."),
        ("04", "K-Means", "Group perfumes into fragrance families."),
        ("05", "Recommendations", "Rank perfumes using similarity, rating, and rules."),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class="section-box">
                    <div class="small-muted">STEP {num}</div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Top Notes")
        notes_df = item_counter(final_data, ["Top", "Middle", "Base"], limit=15)
        if not notes_df.empty:
            render_pretty_bar_chart(notes_df, "Item", "Count")
    with c2:
        st.markdown("### Top Accords")
        accords_df = accord_counter(final_data, limit=15)
        if not accords_df.empty:
            render_pretty_bar_chart(accords_df, "Accord", "Count")

with rec_tab:
    st.markdown("### Find My Perfume")
    st.markdown(
        """
        <div class="explain-box">
        Choose notes and accords you like, add anything you dislike, then the system recommends perfumes using your trained model files.
        The <strong>Also You May Like</strong> section is powered by association rules.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("recommendation_form"):
        c1, c2 = st.columns(2)
        with c1:
            liked_notes = st.multiselect(
                "Notes I love",
                options=all_notes,
                default=[x for x in ["vanilla", "rose", "amber"] if x in all_notes],
            )
            disliked_notes = st.multiselect("Notes I dislike", options=all_notes)
        with c2:
            liked_accords = st.multiselect(
                "Preferred accords",
                options=all_accords,
                default=[x for x in ["sweet"] if x in all_accords],
            )
            gender = st.selectbox("Gender preference", ["No Preference", "women", "men", "unisex"])

        c3, c4, c5 = st.columns(3)
        with c3:
            minimum_rating = st.slider("Minimum weighted rating", 0.0, 5.0, 3.5, 0.1)
        with c4:
            top_n_main = st.slider("Main recommendations", 1, 12, 6)
        with c5:
            top_n_also = st.slider("Also You May Like", 1, 12, 6)

        submitted = st.form_submit_button("✦ Find My Perfumes", type="primary")

    if submitted:
        if not liked_notes and not liked_accords:
            st.warning("Choose at least one note or accord.")
        else:
            main_recommendations, associated_items, also_you_may_like = perfume_recommendation_system(
                final_data=final_data,
                tfidf=tfidf,
                tfidf_matrix=tfidf_matrix,
                strong_rules=strong_rules,
                liked_notes=liked_notes if liked_notes else liked_accords,
                liked_accords=liked_accords,
                disliked_notes=disliked_notes,
                gender=gender,
                minimum_rating=minimum_rating,
                top_n_main=top_n_main,
                top_n_also=top_n_also,
            )

            st.markdown("## ✦ Recommendations")
            if main_recommendations.empty:
                st.info("No perfumes matched your filters. Try lowering the minimum rating or removing dislikes.")
            else:
                for idx, (_, row) in enumerate(main_recommendations.iterrows(), start=1):
                    render_perfume_card(row, rank=idx)

            st.markdown("## Also You May Like")
            if also_you_may_like.empty:
                st.info("No association-rule suggestions found for these choices.")
            else:
                for idx, (_, row) in enumerate(also_you_may_like.iterrows(), start=1):
                    render_perfume_card(row, rank=idx, also=True)

            with st.expander("View associated notes / accords"):
                st.dataframe(associated_items, use_container_width=True)

with clusters_tab:
    st.markdown("### Fragrance Clusters")
    if "cluster_name" in final_data.columns:
        cluster_summary = (
            final_data.groupby("cluster_name")
            .agg(
                perfumes=("Perfume", "count"),
                avg_rating=("Weighted Rating", "mean"),
            )
            .reset_index()
            .sort_values("perfumes", ascending=False)
        )
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

        selected_cluster = st.selectbox("Explore cluster", cluster_summary["cluster_name"].tolist())
        samples = final_data[final_data["cluster_name"] == selected_cluster].sort_values("Weighted Rating", ascending=False).head(10)
        for idx, (_, row) in enumerate(samples.iterrows(), start=1):
            render_perfume_card(row, rank=idx)
    else:
        st.info("cluster_name column was not found in final data.")

with rules_tab:
    st.markdown("### Association Rules")
    st.markdown(
        """
        <div class="explain-box">
        Association rules discover scent items that appear together more often than random chance. 
        High lift means the pair is more meaningful.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(strong_rules.head(50), use_container_width=True, hide_index=True)

with insights_tab:
    st.markdown("### Data Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        if "Gender" in final_data.columns:
            st.markdown("#### Gender Distribution")
            st.dataframe(final_data["Gender"].value_counts().reset_index(), hide_index=True)
    with c2:
        if "Weighted Rating" in final_data.columns:
            st.markdown("#### Rating Summary")
            st.dataframe(final_data["Weighted Rating"].describe().reset_index(), hide_index=True)
    with c3:
        st.markdown("#### Data Shape")
        st.metric("Rows", f"{final_data.shape[0]:,}")
        st.metric("Columns", f"{final_data.shape[1]:,}")

    st.markdown("#### Preview Final Dataset")
    st.dataframe(final_data.head(30), use_container_width=True)