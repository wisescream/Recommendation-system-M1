import glob
import json
import math
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone

from nltk.stem.snowball import SnowballStemmer
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import BucketedRandomProjectionLSH, Normalizer, Word2Vec
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "spark-master")
INPUT_PATH = "/opt/spark-data/raw/dataset.csv"
PROCESSED_ROOT = "/opt/spark-data/processed"
CLEANED_PATH = f"{PROCESSED_ROOT}/cleaned"
AGG_PATH = f"{PROCESSED_ROOT}/agg"
PREVIEW_PATH = f"{PROCESSED_ROOT}/preview.csv"
METRICS_PATH = f"{PROCESSED_ROOT}/metrics.json"
TEXT_FEATURES_PATH = f"{PROCESSED_ROOT}/text_features"
KEYWORDS_PATH = f"{PROCESSED_ROOT}/keywords"
BIGRAMS_PATH = f"{PROCESSED_ROOT}/bigrams"
TFIDF_PATH = f"{PROCESSED_ROOT}/tfidf"
SENTIMENT_PATH = f"{PROCESSED_ROOT}/sentiment"
TOPIC_CLUSTERS_PATH = f"{PROCESSED_ROOT}/topic_clusters"
SIMILARITY_PATH = f"{PROCESSED_ROOT}/similarity_pairs"
TOPIC_CLUSTER_COUNT = 8
SIMILARITY_DISTANCE_THRESHOLD = 0.75
SIMILARITY_NEIGHBOR_LIMIT = 3

STOPWORDS = {
    "about",
    "after",
    "all",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "between",
    "but",
    "can",
    "down",
    "for",
    "from",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "into",
    "its",
    "just",
    "like",
    "more",
    "most",
    "new",
    "not",
    "now",
    "off",
    "one",
    "only",
    "out",
    "over",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "you",
    "your",
}

POSITIVE_TERMS = {
    "adventure",
    "best",
    "brave",
    "celebrate",
    "charm",
    "comfort",
    "courage",
    "delight",
    "dream",
    "family",
    "friend",
    "fun",
    "glory",
    "good",
    "great",
    "happy",
    "hero",
    "hope",
    "inspire",
    "joy",
    "kind",
    "laugh",
    "love",
    "magic",
    "peace",
    "rescue",
    "smile",
    "success",
    "warm",
    "wonder",
}

NEGATIVE_TERMS = {
    "abuse",
    "battle",
    "betray",
    "blood",
    "crime",
    "danger",
    "dark",
    "dead",
    "death",
    "enemy",
    "evil",
    "fear",
    "fight",
    "kill",
    "loss",
    "murder",
    "pain",
    "revenge",
    "risk",
    "sad",
    "scandal",
    "terror",
    "tragic",
    "violence",
    "war",
    "wound",
}

STEMMER = SnowballStemmer("english")


def ensure_input_exists() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Expected CSV at {INPUT_PATH}. Download the Kaggle dataset or place it there manually."
        )


def write_single_csv(dataframe, target_file: str) -> None:
    temp_dir = f"{target_file}_tmp"

    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    if os.path.isfile(target_file):
        os.remove(target_file)

    dataframe.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)

    part_files = glob.glob(os.path.join(temp_dir, "part-*.csv"))
    if not part_files:
        raise FileNotFoundError(f"No CSV part file was generated in {temp_dir}")

    os.replace(part_files[0], target_file)
    shutil.rmtree(temp_dir)


def write_metrics(payload: dict[str, object], target_file: str) -> None:
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    with open(target_file, "w", encoding="utf-8") as metrics_file:
        json.dump(payload, metrics_file, indent=2)


def build_text_document(row) -> tuple[str, str, str, str, str]:
    title = row["title"] or ""
    listed_in = row["listed_in"] or ""
    description = row["description"] or ""
    combined_text = " ".join(part for part in (title, listed_in, description) if part)
    return row["content_id"], row["content_type"], row["country"], title, combined_text


def normalize_document_text(document: tuple[str, str, str, str, str]) -> tuple[str, str, str, str, str]:
    content_id, content_type, country, title, text = document
    normalized_text = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    return content_id, content_type, country, title, normalized_text


def tokenize_document_text(document: tuple[str, str, str, str, str]) -> tuple[str, str, str, str, list[str]]:
    content_id, content_type, country, title, normalized_text = document
    tokens = normalized_text.split() if normalized_text else []
    return content_id, content_type, country, title, tokens


def filter_document_tokens(document: tuple[str, str, str, str, list[str]]) -> tuple[str, str, str, str, list[str]]:
    content_id, content_type, country, title, tokens = document
    filtered_tokens = [
        token
        for token in tokens
        if len(token) >= 3 and not token.isdigit() and token not in STOPWORDS
    ]
    return content_id, content_type, country, title, filtered_tokens


def stem_token(token: str) -> str:
    return STEMMER.stem(token)


POSITIVE_STEMS = {stem_token(term) for term in POSITIVE_TERMS}
NEGATIVE_STEMS = {stem_token(term) for term in NEGATIVE_TERMS}


def stem_document_tokens(document: tuple[str, str, str, str, list[str]]) -> tuple[str, str, str, str, list[str]]:
    content_id, content_type, country, title, tokens = document
    stemmed_tokens = [stem_token(token) for token in tokens]
    stemmed_tokens = [token for token in stemmed_tokens if len(token) >= 3]
    return content_id, content_type, country, title, stemmed_tokens


def score_document_sentiment(
    document: tuple[str, str, str, str, list[str]],
) -> tuple[str, str, str, str, list[str], int, int, float, str]:
    content_id, content_type, country, title, stemmed_tokens = document
    positive_hits = sum(1 for token in stemmed_tokens if token in POSITIVE_STEMS)
    negative_hits = sum(1 for token in stemmed_tokens if token in NEGATIVE_STEMS)
    sentiment_score = round(
        (positive_hits - negative_hits) / len(stemmed_tokens),
        4,
    ) if stemmed_tokens else 0.0

    if sentiment_score > 0.03:
        sentiment_label = "positive"
    elif sentiment_score < -0.03:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"

    return (
        content_id,
        content_type,
        country,
        title,
        stemmed_tokens,
        positive_hits,
        negative_hits,
        sentiment_score,
        sentiment_label,
    )


def build_text_feature_base(
    document: tuple[str, str, str, str, list[str], int, int, float, str],
) -> tuple[str, str, str, str, int, int, float, str | None, str, int, int, float, str]:
    (
        content_id,
        content_type,
        country,
        title,
        stemmed_tokens,
        positive_hits,
        negative_hits,
        sentiment_score,
        sentiment_label,
    ) = document
    term_counter = Counter(stemmed_tokens)
    token_count = len(stemmed_tokens)
    unique_token_count = len(term_counter)
    avg_token_length = (
        round(sum(len(token) for token in stemmed_tokens) / token_count, 2)
        if token_count
        else 0.0
    )
    dominant_term = term_counter.most_common(1)[0][0] if term_counter else None
    term_preview = ", ".join(term for term, _ in term_counter.most_common(5))
    return (
        content_id,
        content_type,
        country,
        title,
        token_count,
        unique_token_count,
        avg_token_length,
        dominant_term,
        term_preview,
        positive_hits,
        negative_hits,
        sentiment_score,
        sentiment_label,
    )


def merge_text_feature_rows(
    item: tuple[str, tuple[tuple[str, ...], tuple[str, float] | None]],
) -> tuple[str, str, str, str, int, int, float, str | None, str, int, int, float, str, str | None, float | None]:
    _, (base_row, top_tfidf_info) = item
    top_tfidf_term, top_tfidf_score = top_tfidf_info if top_tfidf_info else (None, None)
    return (*base_row, top_tfidf_term, top_tfidf_score)


def build_keyword_pairs(
    document: tuple[str, str, str, str, list[str], int, int, float, str],
) -> list[tuple[tuple[str, str], int]]:
    _, content_type, _, _, stemmed_tokens, *_ = document
    return [((content_type, token), 1) for token in sorted(set(stemmed_tokens))]


def build_bigram_pairs(
    document: tuple[str, str, str, str, list[str], int, int, float, str],
) -> list[tuple[tuple[str, str], int]]:
    _, content_type, _, _, stemmed_tokens, *_ = document
    bigrams = {" ".join(pair) for pair in zip(stemmed_tokens, stemmed_tokens[1:])}
    return [((content_type, bigram), 1) for bigram in sorted(bigrams)]


def build_term_frequency_records(
    document: tuple[str, str, str, str, list[str], int, int, float, str],
) -> list[tuple[str, tuple[str, str, str, str, int, int]]]:
    content_id, content_type, country, title, stemmed_tokens, *_ = document
    token_count = len(stemmed_tokens)
    term_counter = Counter(stemmed_tokens)
    return [
        (term, (content_id, content_type, country, title, term_count, token_count))
        for term, term_count in term_counter.items()
    ]


def compute_tfidf_record(
    term_entry: tuple[str, tuple[tuple[str, str, str, str, int, int], int]],
    document_count: int,
) -> tuple[str, str, str, str, str, int, int, int, float]:
    term, (term_payload, document_frequency) = term_entry
    content_id, content_type, country, title, term_count, token_count = term_payload
    term_frequency = term_count / token_count if token_count else 0.0
    inverse_document_frequency = math.log((document_count + 1) / (document_frequency + 1)) + 1.0
    tfidf_score = round(term_frequency * inverse_document_frequency, 4)
    return (
        content_id,
        content_type,
        country,
        title,
        term,
        term_count,
        token_count,
        document_frequency,
        tfidf_score,
    )


def pick_higher_tfidf(
    left: tuple[str, str, str, str, str, int, int, int, float],
    right: tuple[str, str, str, str, str, int, int, int, float],
) -> tuple[str, str, str, str, str, int, int, int, float]:
    if right[8] > left[8]:
        return right
    if right[8] == left[8] and right[4] < left[4]:
        return right
    return left


def combine_tfidf_summary(
    left: tuple[float, float, int],
    right: tuple[float, float, int],
) -> tuple[float, float, int]:
    return left[0] + right[0], max(left[1], right[1]), left[2] + right[2]


def build_sentiment_row(
    document: tuple[str, str, str, str, list[str], int, int, float, str],
) -> tuple[str, str, str, str, int, int, int, float, str]:
    (
        content_id,
        content_type,
        country,
        title,
        stemmed_tokens,
        positive_hits,
        negative_hits,
        sentiment_score,
        sentiment_label,
    ) = document
    return (
        content_id,
        content_type,
        country,
        title,
        len(stemmed_tokens),
        positive_hits,
        negative_hits,
        sentiment_score,
        sentiment_label,
    )


def build_session() -> SparkSession:
    return (
        SparkSession.builder.appName("kaggle-csv-transform")
        .master(MASTER_URL)
        .config("spark.driver.host", DRIVER_HOST)
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def main() -> None:
    ensure_input_exists()
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    raw_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("mode", "PERMISSIVE")
        .csv(INPUT_PATH)
    )

    print("Input schema:")
    raw_df.printSchema()

    raw_count = raw_df.count()
    print(f"Raw row count: {raw_count}")
    print("Raw sample:")
    raw_df.show(10, truncate=False)

    release_year_col = F.expr("try_cast(release_year as int)")
    duration_digits = F.regexp_extract(F.col("duration"), r"(\d+)", 1)

    cleaned_df = (
        raw_df.select(
            F.col("show_id").alias("content_id"),
            F.col("title").alias("title"),
            F.col("type").alias("content_type"),
            F.trim(F.split(F.col("country"), ",").getItem(0)).alias("country"),
            release_year_col.alias("release_year"),
            F.col("rating").alias("rating"),
            F.to_date(F.trim(F.col("date_added")), "MMMM d, yyyy").alias("date_added"),
            F.when(duration_digits != "", duration_digits.cast("int")).alias("duration_value"),
            F.col("listed_in").alias("listed_in"),
            F.col("description").alias("description"),
        )
        .filter(
            F.col("country").isNotNull()
            & (F.col("country") != "")
            & F.col("release_year").isNotNull()
            & (F.col("release_year") >= 2015)
        )
        .withColumn("added_year", F.year("date_added"))
        .withColumn("is_recent", F.col("release_year") >= F.lit(2020))
    )

    cleaned_count = cleaned_df.count()
    print(f"Cleaned row count: {cleaned_count}")
    print("Cleaned sample:")
    cleaned_df.show(10, truncate=False)

    text_documents_rdd = (
        cleaned_df.select("content_id", "content_type", "country", "title", "listed_in", "description")
        .rdd.map(build_text_document)
    )
    normalized_text_rdd = text_documents_rdd.map(normalize_document_text)
    tokenized_text_rdd = normalized_text_rdd.map(tokenize_document_text).filter(lambda document: len(document[4]) > 0).cache()
    filtered_tokens_rdd = tokenized_text_rdd.map(filter_document_tokens).filter(lambda document: len(document[4]) > 0).cache()
    stemmed_tokens_rdd = filtered_tokens_rdd.map(stem_document_tokens).filter(lambda document: len(document[4]) > 0).cache()
    sentiment_scored_rdd = stemmed_tokens_rdd.map(score_document_sentiment).cache()

    tokenized_count = tokenized_text_rdd.count()
    filtered_count = filtered_tokens_rdd.count()
    stemmed_count = sentiment_scored_rdd.count()

    print(f"Tokenized document count: {tokenized_count}")
    print("Normalized text RDD sample:")
    for content_id, text in normalized_text_rdd.map(lambda document: (document[0], document[4][:90])).take(5):
        print(f"{content_id}: {text}")

    print("Filtered token RDD sample:")
    for content_id, tokens in filtered_tokens_rdd.map(lambda document: (document[0], document[4][:8])).take(5):
        print(f"{content_id}: {tokens}")

    print("Stemmed token RDD sample:")
    for content_id, stems in stemmed_tokens_rdd.map(lambda document: (document[0], document[4][:8])).take(5):
        print(f"{content_id}: {stems}")

    print("Sentiment RDD sample:")
    for content_id, score, label in sentiment_scored_rdd.map(
        lambda document: (document[0], document[7], document[8])
    ).take(5):
        print(f"{content_id}: score={score}, label={label}")

    keyword_counts_rdd = sentiment_scored_rdd.flatMap(build_keyword_pairs).reduceByKey(lambda left, right: left + right)
    bigram_counts_rdd = sentiment_scored_rdd.flatMap(build_bigram_pairs).reduceByKey(lambda left, right: left + right)
    term_frequency_rdd = sentiment_scored_rdd.flatMap(build_term_frequency_records)
    document_frequency_rdd = term_frequency_rdd.map(lambda item: (item[0], 1)).reduceByKey(lambda left, right: left + right)
    tfidf_rdd = term_frequency_rdd.join(document_frequency_rdd).map(
        lambda item: compute_tfidf_record(item, stemmed_count)
    ).cache()
    top_tfidf_by_title_rdd = tfidf_rdd.map(lambda row: (row[0], row)).reduceByKey(pick_higher_tfidf).map(
        lambda item: item[1]
    )
    tfidf_summary_rdd = (
        tfidf_rdd.map(lambda row: ((row[1], row[4]), (row[8], row[8], 1)))
        .reduceByKey(combine_tfidf_summary)
        .map(
            lambda item: (
                item[0][0],
                item[0][1],
                round(item[1][0] / item[1][2], 4),
                round(item[1][1], 4),
                item[1][2],
            )
        )
    )
    sentiment_label_counts = sentiment_scored_rdd.map(lambda document: (document[8], 1)).reduceByKey(
        lambda left, right: left + right
    ).collectAsMap()
    document_ml_df = spark.createDataFrame(
        sentiment_scored_rdd,
        [
            "content_id",
            "content_type",
            "country",
            "title",
            "stemmed_tokens",
            "positive_hits",
            "negative_hits",
            "sentiment_score",
            "sentiment_label",
        ],
    )
    word2vec_model = Word2Vec(
        vectorSize=64,
        minCount=2,
        windowSize=5,
        inputCol="stemmed_tokens",
        outputCol="embedding",
        seed=42,
    ).fit(document_ml_df)
    embedded_docs_df = word2vec_model.transform(document_ml_df)
    normalized_embedding_df = Normalizer(
        inputCol="embedding",
        outputCol="normalized_embedding",
        p=2.0,
    ).transform(embedded_docs_df)
    clustered_df = KMeans(
        k=TOPIC_CLUSTER_COUNT,
        seed=42,
        maxIter=30,
        featuresCol="embedding",
        predictionCol="topic_cluster",
    ).fit(embedded_docs_df).transform(normalized_embedding_df)

    similarity_model = BucketedRandomProjectionLSH(
        inputCol="normalized_embedding",
        outputCol="hashes",
        bucketLength=1.5,
        numHashTables=3,
    ).fit(clustered_df)
    similarity_candidates_df = (
        similarity_model.approxSimilarityJoin(
            clustered_df.alias("left"),
            clustered_df.alias("right"),
            SIMILARITY_DISTANCE_THRESHOLD,
            distCol="distance",
        )
        .filter(F.col("datasetA.content_id") < F.col("datasetB.content_id"))
        .select(
            F.col("datasetA.content_id").alias("left_content_id"),
            F.col("datasetA.title").alias("left_title"),
            F.col("datasetA.content_type").alias("left_content_type"),
            F.col("datasetA.topic_cluster").alias("left_topic_cluster"),
            F.col("datasetB.content_id").alias("right_content_id"),
            F.col("datasetB.title").alias("right_title"),
            F.col("datasetB.content_type").alias("right_content_type"),
            F.col("datasetB.topic_cluster").alias("right_topic_cluster"),
            F.round(1 - ((F.col("distance") * F.col("distance")) / F.lit(2.0)), 4).alias("cosine_similarity"),
            F.round(F.col("distance"), 4).alias("embedding_distance"),
        )
        .filter(F.col("cosine_similarity") >= 0.7)
    )
    similarity_output_df = (
        similarity_candidates_df.withColumn(
            "pair_rank",
            F.row_number().over(
                Window.partitionBy("left_content_id").orderBy(
                    F.desc("cosine_similarity"),
                    F.asc("right_content_id"),
                )
            ),
        )
        .filter(F.col("pair_rank") <= SIMILARITY_NEIGHBOR_LIMIT)
        .drop("pair_rank")
    )

    print("Keyword RDD sample:")
    for (content_type, keyword), title_mentions in keyword_counts_rdd.takeOrdered(
        10,
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        print(f"{content_type} | {keyword} -> {title_mentions}")

    print("Bigram RDD sample:")
    for (content_type, bigram), title_mentions in bigram_counts_rdd.takeOrdered(
        10,
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        print(f"{content_type} | {bigram} -> {title_mentions}")

    print("TF-IDF RDD sample:")
    for content_id, content_type, _, _, term, _, _, document_frequency, tfidf_score in tfidf_rdd.takeOrdered(
        10,
        key=lambda item: (-item[8], item[1], item[4], item[0]),
    ):
        print(
            f"{content_id} | {content_type} | {term} -> tfidf={tfidf_score}, document_frequency={document_frequency}"
        )

    text_feature_base_rdd = sentiment_scored_rdd.map(build_text_feature_base).map(lambda row: (row[0], row))
    top_tfidf_lookup_rdd = top_tfidf_by_title_rdd.map(lambda row: (row[0], (row[4], row[8])))
    text_features_rdd = text_feature_base_rdd.leftOuterJoin(top_tfidf_lookup_rdd).map(merge_text_feature_rows)

    text_features_df = spark.createDataFrame(
        text_features_rdd,
        [
            "content_id",
            "content_type",
            "country",
            "title",
            "token_count",
            "unique_token_count",
            "avg_token_length",
            "dominant_term",
            "term_preview",
            "positive_hits",
            "negative_hits",
            "sentiment_score",
            "sentiment_label",
            "top_tfidf_term",
            "top_tfidf_score",
        ],
    )
    keyword_df = spark.createDataFrame(
        keyword_counts_rdd.map(lambda item: (item[0][0], item[0][1], item[1])),
        ["content_type", "keyword", "title_mentions"],
    )
    bigram_df = spark.createDataFrame(
        bigram_counts_rdd.map(lambda item: (item[0][0], item[0][1], item[1])),
        ["content_type", "bigram", "title_mentions"],
    )
    tfidf_df = spark.createDataFrame(
        tfidf_summary_rdd,
        ["content_type", "term", "avg_tfidf", "max_tfidf", "document_count"],
    )
    sentiment_df = spark.createDataFrame(
        sentiment_scored_rdd.map(build_sentiment_row),
        [
            "content_id",
            "content_type",
            "country",
            "title",
            "token_count",
            "positive_hits",
            "negative_hits",
            "sentiment_score",
            "sentiment_label",
        ],
    )
    topic_clusters_df = (
        clustered_df.select(
            "content_id",
            "content_type",
            "country",
            "title",
            "topic_cluster",
        )
        .join(
            text_features_df.select(
                "content_id",
                "dominant_term",
                "top_tfidf_term",
                "top_tfidf_score",
                "sentiment_label",
                "sentiment_score",
            ),
            "content_id",
            "left",
        )
    )

    text_features_output_df = text_features_df.orderBy(F.desc("token_count"), F.asc("content_id"))
    keyword_output_df = keyword_df.orderBy(F.desc("title_mentions"), F.asc("content_type"), F.asc("keyword"))
    bigram_output_df = bigram_df.orderBy(F.desc("title_mentions"), F.asc("content_type"), F.asc("bigram"))
    tfidf_output_df = tfidf_df.orderBy(
        F.desc("avg_tfidf"),
        F.desc("max_tfidf"),
        F.asc("content_type"),
        F.asc("term"),
    )
    sentiment_output_df = sentiment_df.orderBy(
        F.desc(F.abs(F.col("sentiment_score"))),
        F.asc("content_id"),
    )
    topic_clusters_output_df = topic_clusters_df.orderBy(
        F.asc("topic_cluster"),
        F.desc("top_tfidf_score"),
        F.asc("content_id"),
    )
    similarity_output_df = similarity_output_df.orderBy(
        F.desc("cosine_similarity"),
        F.asc("left_content_id"),
        F.asc("right_content_id"),
    )

    text_feature_count = text_features_output_df.count()
    keyword_count = keyword_output_df.count()
    bigram_count = bigram_output_df.count()
    tfidf_count = tfidf_output_df.count()
    sentiment_count = sentiment_output_df.count()
    topic_cluster_count = topic_clusters_output_df.select("topic_cluster").distinct().count()
    similarity_pair_count = similarity_output_df.count()

    print(f"Text feature row count: {text_feature_count}")
    print("Text feature sample:")
    text_features_output_df.show(10, truncate=False)

    print(f"Keyword row count: {keyword_count}")
    print("Keyword sample:")
    keyword_output_df.show(10, truncate=False)

    print(f"Bigram row count: {bigram_count}")
    print("Bigram sample:")
    bigram_output_df.show(10, truncate=False)

    print(f"TF-IDF row count: {tfidf_count}")
    print("TF-IDF sample:")
    tfidf_output_df.show(10, truncate=False)

    print(f"Sentiment row count: {sentiment_count}")
    print("Sentiment sample:")
    sentiment_output_df.show(10, truncate=False)

    print(f"Topic cluster count: {topic_cluster_count}")
    print("Topic cluster sample:")
    topic_clusters_output_df.show(10, truncate=False)

    print(f"Similarity pair count: {similarity_pair_count}")
    print("Similarity pair sample:")
    similarity_output_df.show(10, truncate=False)

    agg_df = (
        cleaned_df.groupBy("content_type", "country")
        .agg(
            F.count("*").alias("title_count"),
            F.round(F.avg("release_year"), 2).alias("avg_release_year"),
            F.min("release_year").alias("min_release_year"),
            F.max("release_year").alias("max_release_year"),
        )
        .orderBy(F.desc("title_count"), F.desc("avg_release_year"))
    )

    agg_count = agg_df.count()
    print(f"Aggregated row count: {agg_count}")
    print("Aggregated sample:")
    agg_df.show(10, truncate=False)

    cleaned_df.write.mode("overwrite").parquet(CLEANED_PATH)
    agg_df.write.mode("overwrite").parquet(AGG_PATH)
    text_features_output_df.write.mode("overwrite").parquet(TEXT_FEATURES_PATH)
    keyword_output_df.write.mode("overwrite").parquet(KEYWORDS_PATH)
    bigram_output_df.write.mode("overwrite").parquet(BIGRAMS_PATH)
    tfidf_output_df.write.mode("overwrite").parquet(TFIDF_PATH)
    sentiment_output_df.write.mode("overwrite").parquet(SENTIMENT_PATH)
    topic_clusters_output_df.write.mode("overwrite").parquet(TOPIC_CLUSTERS_PATH)
    similarity_output_df.write.mode("overwrite").parquet(SIMILARITY_PATH)
    write_single_csv(agg_df.limit(50), PREVIEW_PATH)

    top_keyword_row = keyword_output_df.first()
    top_bigram_row = bigram_output_df.first()
    top_tfidf_row = tfidf_output_df.first()
    top_similarity_row = similarity_output_df.first()

    write_metrics(
        {
            "source_file": INPUT_PATH,
            "raw_row_count": raw_count,
            "cleaned_row_count": cleaned_count,
            "aggregated_row_count": agg_count,
            "dropped_row_count": raw_count - cleaned_count,
            "tokenized_document_count": tokenized_count,
            "filtered_document_count": filtered_count,
            "stemmed_document_count": stemmed_count,
            "text_feature_row_count": text_feature_count,
            "keyword_row_count": keyword_count,
            "bigram_row_count": bigram_count,
            "tfidf_row_count": tfidf_count,
            "sentiment_row_count": sentiment_count,
            "topic_cluster_count": topic_cluster_count,
            "similarity_pair_count": similarity_pair_count,
            "positive_document_count": sentiment_label_counts.get("positive", 0),
            "neutral_document_count": sentiment_label_counts.get("neutral", 0),
            "negative_document_count": sentiment_label_counts.get("negative", 0),
            "top_keyword": (
                f"{top_keyword_row['keyword']} ({top_keyword_row['content_type']}, {top_keyword_row['title_mentions']})"
                if top_keyword_row
                else None
            ),
            "top_bigram": (
                f"{top_bigram_row['bigram']} ({top_bigram_row['content_type']}, {top_bigram_row['title_mentions']})"
                if top_bigram_row
                else None
            ),
            "top_tfidf_term": (
                f"{top_tfidf_row['term']} ({top_tfidf_row['content_type']}, avg={top_tfidf_row['avg_tfidf']})"
                if top_tfidf_row
                else None
            ),
            "top_similarity_pair": (
                f"{top_similarity_row['left_title']} <-> {top_similarity_row['right_title']} ({top_similarity_row['cosine_similarity']})"
                if top_similarity_row
                else None
            ),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        METRICS_PATH,
    )

    print(f"Wrote cleaned data to {CLEANED_PATH}")
    print(f"Wrote aggregated data to {AGG_PATH}")
    print(f"Wrote text feature data to {TEXT_FEATURES_PATH}")
    print(f"Wrote keyword data to {KEYWORDS_PATH}")
    print(f"Wrote bigram data to {BIGRAMS_PATH}")
    print(f"Wrote TF-IDF data to {TFIDF_PATH}")
    print(f"Wrote sentiment data to {SENTIMENT_PATH}")
    print(f"Wrote topic cluster data to {TOPIC_CLUSTERS_PATH}")
    print(f"Wrote similarity pair data to {SIMILARITY_PATH}")
    print(f"Wrote preview file to {PREVIEW_PATH}")
    print(f"Wrote metrics file to {METRICS_PATH}")

    tokenized_text_rdd.unpersist()
    filtered_tokens_rdd.unpersist()
    stemmed_tokens_rdd.unpersist()
    sentiment_scored_rdd.unpersist()
    tfidf_rdd.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
