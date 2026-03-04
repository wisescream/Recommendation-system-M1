"""Unit and integration tests for the PySpark transformation pipeline."""

from __future__ import annotations

import json
import os
import sys

import pytest

# Make jobs/ and helpers/ importable when running from the project root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jobs.transform import (
    NEGATIVE_STEMS,
    POSITIVE_STEMS,
    STOPWORDS,
    build_bigram_pairs,
    build_keyword_pairs,
    build_text_document,
    build_text_feature_base,
    compute_tfidf_record,
    filter_document_tokens,
    normalize_document_text,
    score_document_sentiment,
    stem_document_tokens,
    stem_token,
    tokenize_document_text,
    write_metrics,
)
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Pure-function unit tests (no Spark session required)
# ---------------------------------------------------------------------------

class TestNormalizeDocumentText:
    def test_lowercases_and_strips_punctuation(self):
        doc = ("id1", "Movie", "US", "Title", "Hello, World! 123")
        result = normalize_document_text(doc)
        assert result[4] == "hello world 123"

    def test_collapses_whitespace(self):
        doc = ("id1", "Movie", "US", "T", "too   many   spaces")
        assert "  " not in normalize_document_text(doc)[4]

    def test_empty_text(self):
        doc = ("id1", "Movie", "US", "T", "")
        assert normalize_document_text(doc)[4] == ""


class TestTokenizeDocumentText:
    def test_splits_on_whitespace(self):
        doc = ("id1", "Movie", "US", "T", "hello world test")
        result = tokenize_document_text(doc)
        assert result[4] == ["hello", "world", "test"]

    def test_empty_text_gives_empty_tokens(self):
        doc = ("id1", "Movie", "US", "T", "")
        assert tokenize_document_text(doc)[4] == []


class TestFilterDocumentTokens:
    def test_removes_stopwords(self):
        tokens = list(STOPWORDS)[:3] + ["adventure"]
        doc = ("id1", "Movie", "US", "T", tokens)
        result = filter_document_tokens(doc)
        assert "adventure" in result[4]
        for sw in list(STOPWORDS)[:3]:
            assert sw not in result[4]

    def test_removes_short_tokens(self):
        doc = ("id1", "Movie", "US", "T", ["ab", "ok", "adventure"])
        result = filter_document_tokens(doc)
        assert result[4] == ["adventure"]

    def test_removes_digits(self):
        doc = ("id1", "Movie", "US", "T", ["123", "adventure"])
        result = filter_document_tokens(doc)
        assert result[4] == ["adventure"]


class TestStemToken:
    def test_known_stem(self):
        assert stem_token("running") == "run"

    def test_already_stemmed(self):
        stemmed = stem_token("brave")
        assert isinstance(stemmed, str)
        assert len(stemmed) > 0


class TestStemDocumentTokens:
    def test_stems_all_tokens(self):
        doc = ("id1", "Movie", "US", "T", ["running", "adventures", "braving"])
        result = stem_document_tokens(doc)
        assert all(isinstance(t, str) for t in result[4])

    def test_removes_too_short_stems(self):
        doc = ("id1", "Movie", "US", "T", ["an", "is", "adventure"])
        result = stem_document_tokens(doc)
        for token in result[4]:
            assert len(token) >= 3


class TestScoreDocumentSentiment:
    def test_positive_document(self):
        positive_tokens = list(POSITIVE_STEMS)[:5]
        doc = ("id1", "Movie", "US", "T", positive_tokens, 0, 0, 0.0, "")
        # Build the correct input tuple for score_document_sentiment
        input_doc = ("id1", "Movie", "US", "T", positive_tokens)
        result = score_document_sentiment(input_doc)
        assert result[7] > 0  # sentiment_score > 0
        assert result[8] == "positive"

    def test_negative_document(self):
        negative_tokens = list(NEGATIVE_STEMS)[:5]
        input_doc = ("id1", "Movie", "US", "T", negative_tokens)
        result = score_document_sentiment(input_doc)
        assert result[7] < 0
        assert result[8] == "negative"

    def test_neutral_document(self):
        neutral_tokens = ["random", "unknown", "token", "words", "stuff"]
        input_doc = ("id1", "Movie", "US", "T", neutral_tokens)
        result = score_document_sentiment(input_doc)
        assert result[8] == "neutral"

    def test_empty_tokens(self):
        input_doc = ("id1", "Movie", "US", "T", [])
        result = score_document_sentiment(input_doc)
        assert result[7] == 0.0
        assert result[8] == "neutral"

    def test_sentiment_range(self):
        """Sentiment score should be in [-1, 1]."""
        tokens = list(POSITIVE_STEMS)[:3] + list(NEGATIVE_STEMS)[:3] + ["misc"]
        input_doc = ("id1", "Movie", "US", "T", tokens)
        result = score_document_sentiment(input_doc)
        assert -1.0 <= result[7] <= 1.0


class TestBuildKeywordPairs:
    def test_unique_keywords(self):
        doc = ("id1", "Movie", "US", "T", ["hero", "brave", "hero"], 1, 0, 0.2, "positive")
        pairs = build_keyword_pairs(doc)
        keywords = [kw for (_, kw), _ in pairs]
        assert keywords == sorted(set(keywords))

    def test_content_type_in_key(self):
        doc = ("id1", "TV Show", "US", "T", ["hero"], 0, 0, 0.0, "neutral")
        pairs = build_keyword_pairs(doc)
        assert pairs[0][0][0] == "TV Show"


class TestBuildBigramPairs:
    def test_bigram_format(self):
        doc = ("id1", "Movie", "US", "T", ["brave", "hero", "fight"], 1, 0, 0.1, "positive")
        pairs = build_bigram_pairs(doc)
        for (_, bigram), _ in pairs:
            parts = bigram.split(" ")
            assert len(parts) == 2

    def test_empty_tokens_no_bigrams(self):
        doc = ("id1", "Movie", "US", "T", [], 0, 0, 0.0, "neutral")
        assert build_bigram_pairs(doc) == []

    def test_single_token_no_bigrams(self):
        doc = ("id1", "Movie", "US", "T", ["hero"], 0, 0, 0.0, "neutral")
        assert build_bigram_pairs(doc) == []


class TestComputeTfidfRecord:
    def test_tfidf_non_negative(self):
        term_entry = ("hero", (("s1", "Movie", "US", "Hero", 3, 10), 5))
        result = compute_tfidf_record(term_entry, 100)
        tfidf_score = result[8]
        assert tfidf_score >= 0


class TestWriteMetrics:
    def test_writes_valid_json(self, tmp_output_dir):
        target = os.path.join(tmp_output_dir, "metrics.json")
        payload = {"key1": 42, "key2": "value"}
        write_metrics(payload, target)
        with open(target, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded == payload


# ---------------------------------------------------------------------------
# Spark integration tests (use session-scoped SparkSession + sample_df)
# ---------------------------------------------------------------------------

class TestCleaningRemovesNulls:
    def test_null_country_removed(self, spark, sample_df):
        release_year_col = F.expr("try_cast(release_year as int)")
        duration_digits = F.regexp_extract(F.col("duration"), r"(\d+)", 1)

        cleaned = (
            sample_df.select(
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
        )

        # Rows with null country (s41), null release_year (s42),
        # empty country (s43), and pre-2015 release_year (s44, s45) should be removed
        ids = [row["content_id"] for row in cleaned.select("content_id").collect()]
        for bad_id in ("s41", "s42", "s43", "s44", "s45"):
            assert bad_id not in ids, f"{bad_id} should have been filtered out"

    def test_retains_valid_rows(self, spark, sample_df):
        release_year_col = F.expr("try_cast(release_year as int)")
        cleaned = (
            sample_df.select(
                F.col("show_id").alias("content_id"),
                F.trim(F.split(F.col("country"), ",").getItem(0)).alias("country"),
                release_year_col.alias("release_year"),
            )
            .filter(
                F.col("country").isNotNull()
                & (F.col("country") != "")
                & F.col("release_year").isNotNull()
                & (F.col("release_year") >= 2015)
            )
        )
        # We have 40 good rows + 5 edge-case short text rows (s46-s50) = 45
        # minus s44 (2010) and s45 (2000) which are pre-2015 → 43
        # Actually: s41 (null country), s42 (null year), s43 (empty country),
        # s44 (2010), s45 (2000) → 5 dropped, 45 remain
        assert cleaned.count() == 45


class TestTextRDDPipeline:
    def test_build_text_document(self, spark):
        row = {
            "content_id": "s1",
            "content_type": "Movie",
            "country": "US",
            "title": "Title",
            "listed_in": "Drama",
            "description": "A story about life.",
        }
        result = build_text_document(row)
        assert result[0] == "s1"
        assert "Title" in result[4]
        assert "Drama" in result[4]
        assert "A story about life." in result[4]

    def test_end_to_end_rdd_pipeline(self, spark, sample_df):
        """Run the RDD chain on the sample and verify counts remain sane."""
        release_year_col = F.expr("try_cast(release_year as int)")
        cleaned = (
            sample_df.select(
                F.col("show_id").alias("content_id"),
                F.col("title"),
                F.col("type").alias("content_type"),
                F.trim(F.split(F.col("country"), ",").getItem(0)).alias("country"),
                release_year_col.alias("release_year"),
                F.col("listed_in"),
                F.col("description"),
            )
            .filter(
                F.col("country").isNotNull()
                & (F.col("country") != "")
                & F.col("release_year").isNotNull()
                & (F.col("release_year") >= 2015)
            )
        )
        rdd = (
            cleaned.select("content_id", "content_type", "country", "title", "listed_in", "description")
            .rdd.map(build_text_document)
        )
        normalized = rdd.map(normalize_document_text)
        tokenized = normalized.map(tokenize_document_text).filter(lambda d: len(d[4]) > 0)
        filtered = tokenized.map(filter_document_tokens).filter(lambda d: len(d[4]) > 0)
        stemmed = filtered.map(stem_document_tokens).filter(lambda d: len(d[4]) > 0)
        sentiment = stemmed.map(score_document_sentiment)

        assert tokenized.count() > 0
        assert filtered.count() > 0
        assert stemmed.count() > 0
        assert sentiment.count() > 0

        # Sentiment scores are in [-1, 1]
        for doc in sentiment.collect():
            assert -1.0 <= doc[7] <= 1.0


class TestMetricsKeys:
    def test_all_expected_keys(self, tmp_output_dir):
        """Verify the expected metric keys from a mock payload."""
        expected_keys = {
            "source_file",
            "raw_row_count",
            "cleaned_row_count",
            "aggregated_row_count",
            "dropped_row_count",
            "tokenized_document_count",
            "filtered_document_count",
            "stemmed_document_count",
            "text_feature_row_count",
            "keyword_row_count",
            "bigram_row_count",
            "tfidf_row_count",
            "sentiment_row_count",
            "topic_cluster_count",
            "similarity_pair_count",
            "positive_document_count",
            "neutral_document_count",
            "negative_document_count",
            "top_keyword",
            "top_bigram",
            "top_tfidf_term",
            "top_similarity_pair",
            "generated_at_utc",
        }
        payload = {k: 0 for k in expected_keys}
        target = os.path.join(tmp_output_dir, "metrics.json")
        write_metrics(payload, target)
        with open(target, "r", encoding="utf-8") as fh:
            assert set(json.load(fh).keys()) == expected_keys
