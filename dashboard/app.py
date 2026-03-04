import glob
import json
import os

import altair as alt
import pandas as pd
import streamlit as st


RAW_PATH = "/opt/spark-data/raw/dataset.csv"
CLEANED_DIR = "/opt/spark-data/processed/cleaned"
AGG_DIR = "/opt/spark-data/processed/agg"
TEXT_FEATURES_DIR = "/opt/spark-data/processed/text_features"
KEYWORDS_DIR = "/opt/spark-data/processed/keywords"
BIGRAMS_DIR = "/opt/spark-data/processed/bigrams"
TFIDF_DIR = "/opt/spark-data/processed/tfidf"
SENTIMENT_DIR = "/opt/spark-data/processed/sentiment"
TOPIC_CLUSTERS_DIR = "/opt/spark-data/processed/topic_clusters"
SIMILARITY_DIR = "/opt/spark-data/processed/similarity_pairs"
METRICS_PATH = "/opt/spark-data/processed/metrics.json"
QUALITY_REPORT_PATH = "/opt/spark-data/processed/quality_report.json"


def parquet_files(directory: str) -> list[str]:
    return sorted(
        path
        for path in glob.glob(os.path.join(directory, "*.parquet"))
        if os.path.basename(path).endswith(".parquet")
    )


@st.cache_data(ttl=30)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_data(ttl=30)
def load_parquet(directory: str) -> pd.DataFrame:
    files = parquet_files(directory)
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files)


@st.cache_data(ttl=30)
def load_metrics() -> dict[str, object]:
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


@st.cache_data(ttl=30)
def load_quality_report() -> dict[str, object]:
    if not os.path.exists(QUALITY_REPORT_PATH):
        return {}
    with open(QUALITY_REPORT_PATH, "r", encoding="utf-8") as qr_file:
        return json.load(qr_file)


def render_metric_card(title: str, value: str, delta: str | None = None) -> None:
    delta_html = f"<p class='metric-delta'>{delta}</p>" if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
          <p class="metric-label">{title}</p>
          <p class="metric-value">{value}</p>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Spark Transformation Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(222, 110, 70, 0.18), transparent 26%),
          linear-gradient(180deg, #f6f0e7 0%, #fffaf4 52%, #f4efe8 100%);
      }
      .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2.5rem;
      }
      html, body, [class*="css"] {
        font-family: Georgia, "Times New Roman", serif;
      }
      .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(47, 79, 79, 0.96), rgba(143, 65, 43, 0.92));
        color: #fff7ef;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 20px 50px rgba(62, 38, 29, 0.12);
        margin-bottom: 1rem;
      }
      .hero h1 {
        margin: 0;
        font-size: 2.2rem;
      }
      .hero p {
        margin: 0.45rem 0 0 0;
        max-width: 52rem;
        line-height: 1.5;
      }
      .metric-card {
        border-radius: 18px;
        padding: 1rem 1rem 0.85rem 1rem;
        background: rgba(255, 252, 247, 0.9);
        border: 1px solid rgba(110, 75, 58, 0.14);
        box-shadow: 0 10px 30px rgba(80, 53, 40, 0.08);
      }
      .metric-label {
        margin: 0;
        font-size: 0.95rem;
        color: #6f5847;
      }
      .metric-value {
        margin: 0.2rem 0 0 0;
        font-size: 2rem;
        color: #2e2a24;
      }
      .metric-delta {
        margin: 0.3rem 0 0 0;
        color: #8f412b;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

header_col, action_col = st.columns([6, 1])
with header_col:
    st.markdown(
        """
        <div class="hero">
          <h1>Spark Transformation Dashboard</h1>
          <p>Track the CSV pipeline from raw records to cleaned output, then inspect the text-only RDD chain that normalizes, filters, Snowball-stems, scores sentiment, builds TF-IDF, clusters topics, and finds similar titles from embeddings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with action_col:
    st.write("")
    st.write("")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if not os.path.exists(RAW_PATH):
    st.error("Raw dataset missing. Place the Kaggle CSV at data/raw/dataset.csv and rerun the page.")
    st.stop()

raw_df = load_csv(RAW_PATH)
cleaned_df = load_parquet(CLEANED_DIR)
agg_df = load_parquet(AGG_DIR)
text_features_df = load_parquet(TEXT_FEATURES_DIR)
keywords_df = load_parquet(KEYWORDS_DIR)
bigrams_df = load_parquet(BIGRAMS_DIR)
tfidf_df = load_parquet(TFIDF_DIR)
sentiment_df = load_parquet(SENTIMENT_DIR)
topic_clusters_df = load_parquet(TOPIC_CLUSTERS_DIR)
similarity_df = load_parquet(SIMILARITY_DIR)
metrics = load_metrics()

raw_rows = int(metrics.get("raw_row_count", len(raw_df)))
cleaned_rows = int(metrics.get("cleaned_row_count", len(cleaned_df))) if not cleaned_df.empty else 0
agg_rows = int(metrics.get("aggregated_row_count", len(agg_df))) if not agg_df.empty else 0
dropped_rows = int(metrics.get("dropped_row_count", max(raw_rows - cleaned_rows, 0))) if cleaned_rows else 0
retained_pct = (cleaned_rows / raw_rows * 100) if raw_rows else 0.0

text_rows = int(metrics.get("text_feature_row_count", len(text_features_df))) if not text_features_df.empty else 0
keyword_rows = int(metrics.get("keyword_row_count", len(keywords_df))) if not keywords_df.empty else 0
bigram_rows = int(metrics.get("bigram_row_count", len(bigrams_df))) if not bigrams_df.empty else 0
tfidf_rows = int(metrics.get("tfidf_row_count", len(tfidf_df))) if not tfidf_df.empty else 0
sentiment_rows = int(metrics.get("sentiment_row_count", len(sentiment_df))) if not sentiment_df.empty else 0
topic_cluster_rows = int(metrics.get("topic_cluster_count", topic_clusters_df["topic_cluster"].nunique())) if not topic_clusters_df.empty else 0
similarity_rows = int(metrics.get("similarity_pair_count", len(similarity_df))) if not similarity_df.empty else 0

positive_docs = int(metrics.get("positive_document_count", 0))
neutral_docs = int(metrics.get("neutral_document_count", 0))
negative_docs = int(metrics.get("negative_document_count", 0))

metrics_cols = st.columns(4)
with metrics_cols[0]:
    render_metric_card("Raw rows", f"{raw_rows:,}")
with metrics_cols[1]:
    render_metric_card("Cleaned rows", f"{cleaned_rows:,}", f"{retained_pct:.1f}% retained" if cleaned_rows else None)
with metrics_cols[2]:
    render_metric_card("Dropped rows", f"{dropped_rows:,}")
with metrics_cols[3]:
    render_metric_card("Aggregated groups", f"{agg_rows:,}")

text_metric_cols = st.columns(5)
with text_metric_cols[0]:
    render_metric_card("Text profiles", f"{text_rows:,}", f"{metrics.get('stemmed_document_count', 0):,} stemmed docs")
with text_metric_cols[1]:
    render_metric_card("Keywords", f"{keyword_rows:,}", str(metrics.get("top_keyword") or ""))
with text_metric_cols[2]:
    render_metric_card("Bigrams", f"{bigram_rows:,}", str(metrics.get("top_bigram") or ""))
with text_metric_cols[3]:
    render_metric_card("TF-IDF terms", f"{tfidf_rows:,}", str(metrics.get("top_tfidf_term") or ""))
with text_metric_cols[4]:
    render_metric_card("Topic clusters", f"{topic_cluster_rows:,}", f"{similarity_rows:,} similarity pairs")

sentiment_metric_cols = st.columns(3)
with sentiment_metric_cols[0]:
    render_metric_card("Sentiment docs", f"{sentiment_rows:,}")
with sentiment_metric_cols[1]:
    render_metric_card("Positive / Neutral", f"{positive_docs:,}", f"{neutral_docs:,} neutral")
with sentiment_metric_cols[2]:
    render_metric_card("Negative docs", f"{negative_docs:,}")

if metrics.get("top_similarity_pair"):
    st.caption(f"Top similarity pair: {metrics['top_similarity_pair']}")

generated_at = metrics.get("generated_at_utc")
if generated_at:
    st.caption(f"Latest Spark run: {generated_at}")

if cleaned_df.empty or agg_df.empty:
    st.warning("Processed outputs are not available yet. Run spark-submit to generate cleaned, aggregated, text, TF-IDF, sentiment, and metrics files.")

stage_counts = pd.DataFrame(
    [
        {"stage": "Raw", "rows": raw_rows},
        {"stage": "Cleaned", "rows": cleaned_rows},
    ]
)

type_frames = []
if "type" in raw_df.columns:
    raw_type_counts = raw_df["type"].fillna("Unknown").value_counts().rename_axis("content_type").reset_index(name="rows")
    raw_type_counts["stage"] = "Raw"
    type_frames.append(raw_type_counts)
if not cleaned_df.empty and "content_type" in cleaned_df.columns:
    cleaned_type_counts = (
        cleaned_df["content_type"].fillna("Unknown").value_counts().rename_axis("content_type").reset_index(name="rows")
    )
    cleaned_type_counts["stage"] = "Cleaned"
    type_frames.append(cleaned_type_counts)

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Pipeline Retention")
    st.altair_chart(
        alt.Chart(stage_counts)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("stage:N", title="Stage"),
            y=alt.Y("rows:Q", title="Rows"),
            color=alt.Color("stage:N", scale=alt.Scale(range=["#d68452", "#355c5b"]), legend=None),
            tooltip=["stage", "rows"],
        )
        .properties(height=320),
        use_container_width=True,
    )

with chart_right:
    st.subheader("Content Type Mix")
    if type_frames:
        type_mix = pd.concat(type_frames, ignore_index=True)
        st.altair_chart(
            alt.Chart(type_mix)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("stage:N", title="Stage"),
                y=alt.Y("rows:Q", title="Rows"),
                color=alt.Color(
                    "content_type:N",
                    title="Content type",
                    scale=alt.Scale(range=["#355c5b", "#d68452", "#c6a96f", "#8f412b"]),
                ),
                tooltip=["stage", "content_type", "rows"],
            )
            .properties(height=320),
            use_container_width=True,
        )
    else:
        st.info("Content type columns are not available.")

detail_left, detail_right = st.columns(2)

with detail_left:
    st.subheader("Top Countries After Cleaning")
    if not agg_df.empty:
        top_countries = agg_df.sort_values("title_count", ascending=False).head(15)
        st.altair_chart(
            alt.Chart(top_countries)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("title_count:Q", title="Titles"),
                y=alt.Y("country:N", sort="-x", title="Country"),
                color=alt.Color(
                    "content_type:N",
                    title="Content type",
                    scale=alt.Scale(range=["#355c5b", "#d68452"]),
                ),
                tooltip=[
                    "country",
                    "content_type",
                    "title_count",
                    "avg_release_year",
                    "min_release_year",
                    "max_release_year",
                ],
            )
            .properties(height=420),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate aggregated output.")

with detail_right:
    st.subheader("Release Year Trend")
    if not cleaned_df.empty and {"release_year", "content_type"}.issubset(cleaned_df.columns):
        release_trend = (
            cleaned_df.groupby(["release_year", "content_type"]).size().reset_index(name="titles").sort_values("release_year")
        )
        st.altair_chart(
            alt.Chart(release_trend)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("release_year:Q", title="Release year"),
                y=alt.Y("titles:Q", title="Titles"),
                color=alt.Color(
                    "content_type:N",
                    title="Content type",
                    scale=alt.Scale(range=["#355c5b", "#d68452"]),
                ),
                tooltip=["release_year", "content_type", "titles"],
            )
            .properties(height=420),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate cleaned output.")

derived_col_left, derived_col_right = st.columns(2)

with derived_col_left:
    st.subheader("Recent Content Flag")
    if not cleaned_df.empty and {"content_type", "is_recent"}.issubset(cleaned_df.columns):
        recent_mix = (
            cleaned_df.assign(
                recent_bucket=cleaned_df["is_recent"].map({True: "Recent (>= 2020)", False: "Back catalog"})
            )
            .groupby(["content_type", "recent_bucket"])
            .size()
            .reset_index(name="titles")
        )
        st.altair_chart(
            alt.Chart(recent_mix)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("content_type:N", title="Content type"),
                y=alt.Y("titles:Q", title="Titles"),
                color=alt.Color(
                    "recent_bucket:N",
                    title="Derived flag",
                    scale=alt.Scale(range=["#c6a96f", "#8f412b"]),
                ),
                tooltip=["content_type", "recent_bucket", "titles"],
            )
            .properties(height=320),
            use_container_width=True,
        )
    else:
        st.info("The derived is_recent flag is not available yet.")

with derived_col_right:
    st.subheader("Average Release Year by Top Markets")
    if not agg_df.empty:
        top_markets = agg_df.sort_values("title_count", ascending=False).head(12)
        st.altair_chart(
            alt.Chart(top_markets)
            .mark_circle(size=180)
            .encode(
                x=alt.X("avg_release_year:Q", title="Average release year"),
                y=alt.Y("country:N", sort="-x", title="Country"),
                size=alt.Size("title_count:Q", title="Titles"),
                color=alt.Color(
                    "content_type:N",
                    title="Content type",
                    scale=alt.Scale(range=["#355c5b", "#d68452"]),
                ),
                tooltip=["country", "content_type", "title_count", "avg_release_year"],
            )
            .properties(height=320),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate aggregated output.")

text_left, text_right = st.columns(2)

with text_left:
    st.subheader("Top Keywords From RDD Text Pipeline")
    if not keywords_df.empty:
        top_keywords = keywords_df.sort_values("title_mentions", ascending=False).head(20)
        st.altair_chart(
            alt.Chart(top_keywords)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("title_mentions:Q", title="Titles containing keyword"),
                y=alt.Y("keyword:N", sort="-x", title="Keyword"),
                color=alt.Color(
                    "content_type:N",
                    title="Content type",
                    scale=alt.Scale(range=["#355c5b", "#d68452"]),
                ),
                tooltip=["keyword", "content_type", "title_mentions"],
            )
            .properties(height=420),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate the keyword output.")

with text_right:
    st.subheader("Top Bigrams From RDD Text Pipeline")
    if not bigrams_df.empty:
        top_bigrams = bigrams_df.sort_values("title_mentions", ascending=False).head(20)
        st.altair_chart(
            alt.Chart(top_bigrams)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("title_mentions:Q", title="Titles containing bigram"),
                y=alt.Y("bigram:N", sort="-x", title="Bigram"),
                color=alt.Color(
                    "content_type:N",
                    title="Content type",
                    scale=alt.Scale(range=["#355c5b", "#d68452"]),
                ),
                tooltip=["bigram", "content_type", "title_mentions"],
            )
            .properties(height=420),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate the bigram output.")

analysis_left, analysis_right = st.columns(2)

with analysis_left:
    st.subheader("Top TF-IDF Terms")
    if not tfidf_df.empty:
        top_tfidf = tfidf_df.sort_values(["avg_tfidf", "max_tfidf"], ascending=False).head(20)
        st.altair_chart(
            alt.Chart(top_tfidf)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("avg_tfidf:Q", title="Average TF-IDF"),
                y=alt.Y("term:N", sort="-x", title="Stemmed term"),
                color=alt.Color(
                    "content_type:N",
                    title="Content type",
                    scale=alt.Scale(range=["#355c5b", "#d68452"]),
                ),
                tooltip=["term", "content_type", "avg_tfidf", "max_tfidf", "document_count"],
            )
            .properties(height=420),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate the TF-IDF output.")

with analysis_right:
    st.subheader("Sentiment Distribution")
    if not sentiment_df.empty:
        sentiment_mix = (
            sentiment_df.groupby(["content_type", "sentiment_label"]).size().reset_index(name="titles")
        )
        st.altair_chart(
            alt.Chart(sentiment_mix)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("content_type:N", title="Content type"),
                y=alt.Y("titles:Q", title="Titles"),
                color=alt.Color(
                    "sentiment_label:N",
                    title="Sentiment",
                    scale=alt.Scale(range=["#355c5b", "#c6a96f", "#8f412b"]),
                ),
                tooltip=["content_type", "sentiment_label", "titles"],
            )
            .properties(height=420),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate the sentiment output.")

cluster_left, cluster_right = st.columns(2)

with cluster_left:
    st.subheader("Topic Cluster Distribution")
    if not topic_clusters_df.empty:
        cluster_mix = (
            topic_clusters_df.groupby(["topic_cluster", "content_type"]).size().reset_index(name="titles")
        )
        st.altair_chart(
            alt.Chart(cluster_mix)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("topic_cluster:O", title="Topic cluster"),
                y=alt.Y("titles:Q", title="Titles"),
                color=alt.Color(
                    "content_type:N",
                    title="Content type",
                    scale=alt.Scale(range=["#355c5b", "#d68452"]),
                ),
                tooltip=["topic_cluster", "content_type", "titles"],
            )
            .properties(height=360),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate topic cluster output.")

with cluster_right:
    st.subheader("Most Distinct Cluster Terms")
    if not topic_clusters_df.empty and "top_tfidf_term" in topic_clusters_df.columns:
        cluster_terms = (
            topic_clusters_df.dropna(subset=["top_tfidf_term"])
            .groupby(["topic_cluster", "top_tfidf_term"])
            .size()
            .reset_index(name="titles")
            .sort_values("titles", ascending=False)
            .head(20)
        )
        st.altair_chart(
            alt.Chart(cluster_terms)
            .mark_circle(size=180)
            .encode(
                x=alt.X("titles:Q", title="Titles"),
                y=alt.Y("top_tfidf_term:N", sort="-x", title="Cluster term"),
                color=alt.Color(
                    "topic_cluster:O",
                    title="Cluster",
                    scale=alt.Scale(range=["#355c5b", "#d68452", "#c6a96f", "#8f412b", "#7a8f3c", "#b1573b", "#457b9d", "#c98a46"]),
                ),
                tooltip=["topic_cluster", "top_tfidf_term", "titles"],
            )
            .properties(height=360),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate topic cluster output.")

text_metric_left, text_metric_right = st.columns(2)

with text_metric_left:
    st.subheader("Average Text Density")
    if not text_features_df.empty:
        text_density = (
            text_features_df.groupby("content_type")[["token_count", "unique_token_count"]]
            .mean()
            .round(2)
            .reset_index()
            .melt(id_vars="content_type", var_name="metric", value_name="value")
        )
        st.altair_chart(
            alt.Chart(text_density)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("content_type:N", title="Content type"),
                y=alt.Y("value:Q", title="Average count"),
                color=alt.Color(
                    "metric:N",
                    title="Metric",
                    scale=alt.Scale(range=["#c6a96f", "#8f412b"]),
                ),
                tooltip=["content_type", "metric", "value"],
            )
            .properties(height=320),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate text feature output.")

with text_metric_right:
    st.subheader("Dominant Terms by Content Type")
    if not text_features_df.empty and {"content_type", "dominant_term"}.issubset(text_features_df.columns):
        dominant_terms = (
            text_features_df.dropna(subset=["dominant_term"])
            .groupby(["content_type", "dominant_term"])
            .size()
            .reset_index(name="titles")
            .sort_values("titles", ascending=False)
            .head(15)
        )
        st.altair_chart(
            alt.Chart(dominant_terms)
            .mark_circle(size=180)
            .encode(
                x=alt.X("titles:Q", title="Titles"),
                y=alt.Y("dominant_term:N", sort="-x", title="Dominant term"),
                color=alt.Color(
                    "content_type:N",
                    title="Content type",
                    scale=alt.Scale(range=["#355c5b", "#d68452"]),
                ),
                tooltip=["content_type", "dominant_term", "titles"],
            )
            .properties(height=320),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate text feature output.")

similarity_left, similarity_right = st.columns(2)

with similarity_left:
    st.subheader("Top Similar Titles")
    if not similarity_df.empty:
        top_pairs = similarity_df.sort_values("cosine_similarity", ascending=False).head(20)
        top_pairs["pair_label"] = top_pairs["left_title"] + " <> " + top_pairs["right_title"]
        st.altair_chart(
            alt.Chart(top_pairs)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("cosine_similarity:Q", title="Cosine similarity"),
                y=alt.Y("pair_label:N", sort="-x", title="Title pair"),
                color=alt.Color(
                    "left_content_type:N",
                    title="Left content type",
                    scale=alt.Scale(range=["#355c5b", "#d68452"]),
                ),
                tooltip=[
                    "left_title",
                    "right_title",
                    "left_topic_cluster",
                    "right_topic_cluster",
                    "cosine_similarity",
                    "embedding_distance",
                ],
            )
            .properties(height=420),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate similarity output.")

with similarity_right:
    st.subheader("Similarity by Cluster Pair")
    if not similarity_df.empty:
        similarity_mix = (
            similarity_df.groupby(["left_topic_cluster", "right_topic_cluster"])["cosine_similarity"]
            .mean()
            .reset_index()
            .rename(columns={"cosine_similarity": "avg_similarity"})
            .sort_values("avg_similarity", ascending=False)
            .head(20)
        )
        similarity_mix["cluster_pair"] = (
            similarity_mix["left_topic_cluster"].astype(str)
            + " -> "
            + similarity_mix["right_topic_cluster"].astype(str)
        )
        st.altair_chart(
            alt.Chart(similarity_mix)
            .mark_circle(size=220)
            .encode(
                x=alt.X("avg_similarity:Q", title="Average cosine similarity"),
                y=alt.Y("cluster_pair:N", sort="-x", title="Cluster pair"),
                color=alt.Color(
                    "avg_similarity:Q",
                    title="Avg similarity",
                    scale=alt.Scale(scheme="oranges"),
                ),
                tooltip=["cluster_pair", "avg_similarity"],
            )
            .properties(height=420),
            use_container_width=True,
        )
    else:
        st.info("Run the Spark job to populate similarity output.")

raw_tab, cleaned_tab, agg_tab, text_tab, keyword_tab, bigram_tab, tfidf_tab, sentiment_tab, cluster_tab, similarity_tab, quality_tab, explorer_tab = st.tabs(
    ["Raw sample", "Cleaned sample", "Aggregated sample", "Text features", "Keywords", "Bigrams", "TF-IDF", "Sentiment", "Topic clusters", "Similarity", "Data Quality", "Topic Explorer"]
)

with raw_tab:
    st.dataframe(raw_df.head(10), use_container_width=True, hide_index=True)

with cleaned_tab:
    if cleaned_df.empty:
        st.info("Run the Spark job to populate cleaned output.")
    else:
        st.dataframe(cleaned_df.head(10), use_container_width=True, hide_index=True)

with agg_tab:
    if agg_df.empty:
        st.info("Run the Spark job to populate aggregated output.")
    else:
        st.dataframe(agg_df.head(25), use_container_width=True, hide_index=True)

with text_tab:
    if text_features_df.empty:
        st.info("Run the Spark job to populate text feature output.")
    else:
        st.dataframe(text_features_df.head(25), use_container_width=True, hide_index=True)

with keyword_tab:
    if keywords_df.empty:
        st.info("Run the Spark job to populate keyword output.")
    else:
        st.dataframe(keywords_df.head(25), use_container_width=True, hide_index=True)

with bigram_tab:
    if bigrams_df.empty:
        st.info("Run the Spark job to populate bigram output.")
    else:
        st.dataframe(bigrams_df.head(25), use_container_width=True, hide_index=True)

with tfidf_tab:
    if tfidf_df.empty:
        st.info("Run the Spark job to populate TF-IDF output.")
    else:
        st.dataframe(tfidf_df.head(25), use_container_width=True, hide_index=True)

with sentiment_tab:
    if sentiment_df.empty:
        st.info("Run the Spark job to populate sentiment output.")
    else:
        st.dataframe(sentiment_df.head(25), use_container_width=True, hide_index=True)

with cluster_tab:
    if topic_clusters_df.empty:
        st.info("Run the Spark job to populate topic cluster output.")
    else:
        st.dataframe(topic_clusters_df.head(25), use_container_width=True, hide_index=True)

with similarity_tab:
    if similarity_df.empty:
        st.info("Run the Spark job to populate similarity output.")
    else:
        st.dataframe(similarity_df.head(25), use_container_width=True, hide_index=True)

with quality_tab:
    qr = load_quality_report()
    if not qr:
        st.info("Run the quality report job to populate this tab: `make quality`")
    else:
        score = qr.get("quality_score", 0)
        score_color = "#355c5b" if score >= 0.9 else ("#c6a96f" if score >= 0.7 else "#8f412b")
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 6px solid {score_color};">
              <p class="metric-label">Overall Quality Score</p>
              <p class="metric-value" style="color: {score_color};">{score:.2%}</p>
              <p class="metric-delta">{qr.get('total_rows', 0):,} rows &middot; {qr.get('column_count', 0)} columns</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        col_stats = qr.get("columns", [])
        if col_stats:
            st.subheader("Column Null Percentage")
            null_data = pd.DataFrame(col_stats)
            st.altair_chart(
                alt.Chart(null_data)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("null_pct:Q", title="Null %"),
                    y=alt.Y("column:N", sort="-x", title="Column"),
                    color=alt.condition(
                        alt.datum.null_pct > 5,
                        alt.value("#8f412b"),
                        alt.value("#355c5b"),
                    ),
                    tooltip=["column", "null_count", "null_pct", "distinct_count"],
                )
                .properties(height=max(len(col_stats) * 30, 200)),
                use_container_width=True,
            )

            st.subheader("Distinct Value Counts")
            st.altair_chart(
                alt.Chart(null_data)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("distinct_count:Q", title="Distinct values"),
                    y=alt.Y("column:N", sort="-x", title="Column"),
                    color=alt.value("#c6a96f"),
                    tooltip=["column", "distinct_count", "null_count"],
                )
                .properties(height=max(len(col_stats) * 30, 200)),
                use_container_width=True,
            )

        dup_info = qr.get("duplicates", {})
        if dup_info:
            dup_cols = st.columns(3)
            with dup_cols[0]:
                render_metric_card("Total Rows", f"{dup_info.get('total_rows', 0):,}")
            with dup_cols[1]:
                render_metric_card("Distinct Rows", f"{dup_info.get('distinct_rows', 0):,}")
            with dup_cols[2]:
                render_metric_card(
                    "Duplicates",
                    f"{dup_info.get('duplicate_rows', 0):,}",
                    f"{dup_info.get('duplicate_pct', 0):.2f}%",
                )

        st.caption(f"Report generated: {qr.get('generated_at_utc', 'N/A')}")

with explorer_tab:
    if topic_clusters_df.empty:
        st.info("Run the Spark job to populate topic cluster output.")
    else:
        cluster_ids = sorted(topic_clusters_df["topic_cluster"].unique())
        selected_cluster = st.selectbox("Select a cluster", cluster_ids, key="cluster_explorer_select")

        cluster_slice = topic_clusters_df[topic_clusters_df["topic_cluster"] == selected_cluster]
        st.metric("Cluster size", f"{len(cluster_slice):,} titles")

        # Top keywords in this cluster (from top_tfidf_term column)
        if "top_tfidf_term" in cluster_slice.columns:
            term_counts = (
                cluster_slice.dropna(subset=["top_tfidf_term"])
                .groupby("top_tfidf_term")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(10)
            )
            if not term_counts.empty:
                st.subheader("Top 10 Keywords in Cluster")
                st.altair_chart(
                    alt.Chart(term_counts)
                    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                    .encode(
                        x=alt.X("count:Q", title="Titles"),
                        y=alt.Y("top_tfidf_term:N", sort="-x", title="Keyword"),
                        color=alt.value("#355c5b"),
                        tooltip=["top_tfidf_term", "count"],
                    )
                    .properties(height=300),
                    use_container_width=True,
                )

        st.subheader("Sample Titles")
        display_cols = ["content_id", "title", "content_type", "country"]
        if "sentiment_label" in cluster_slice.columns:
            display_cols.append("sentiment_label")
        st.dataframe(
            cluster_slice[display_cols].head(5),
            use_container_width=True,
            hide_index=True,
        )

        # Cluster size distribution across all clusters
        st.subheader("Cluster Size Distribution")
        cluster_sizes = (
            topic_clusters_df.groupby("topic_cluster")
            .size()
            .reset_index(name="titles")
        )
        st.altair_chart(
            alt.Chart(cluster_sizes)
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("topic_cluster:O", title="Cluster"),
                y=alt.Y("titles:Q", title="Titles"),
                color=alt.condition(
                    alt.datum.topic_cluster == selected_cluster,
                    alt.value("#8f412b"),
                    alt.value("#355c5b"),
                ),
                tooltip=["topic_cluster", "titles"],
            )
            .properties(height=300),
            use_container_width=True,
        )
