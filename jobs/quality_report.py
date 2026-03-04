"""Generate a data-quality report from the cleaned parquet output.

This script reads the cleaned dataset written by ``transform.py`` and produces
``data/processed/quality_report.json`` with per-column statistics, duplicate
detection, and an overall quality score.

Usage inside Docker
-------------------
```
docker compose exec spark-master spark-submit \
  --master spark://spark-master:7077 /opt/spark-apps/quality_report.py
```

Or locally with PySpark on the PATH:
```
spark-submit jobs/quality_report.py
```
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://spark-master:7077")
DRIVER_HOST = os.getenv("SPARK_DRIVER_HOST", "spark-master")
PROCESSED_ROOT = "/opt/spark-data/processed"
CLEANED_PATH = f"{PROCESSED_ROOT}/cleaned"
REPORT_PATH = f"{PROCESSED_ROOT}/quality_report.json"


def build_session() -> SparkSession:
    return (
        SparkSession.builder.appName("quality-report")
        .master(MASTER_URL)
        .config("spark.driver.host", DRIVER_HOST)
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


def column_stats(df, col_name: str, total: int) -> dict:
    """Compute per-column statistics."""
    stats: dict = {"column": col_name}

    null_count = df.filter(F.col(col_name).isNull() | (F.trim(F.col(col_name).cast("string")) == "")).count()
    stats["null_count"] = null_count
    stats["null_pct"] = round(null_count / total * 100, 2) if total else 0.0
    stats["distinct_count"] = df.select(col_name).distinct().count()

    # Numeric statistics (if applicable)
    dtype = dict(df.dtypes).get(col_name, "string")
    if dtype in ("int", "bigint", "double", "float", "long", "short"):
        agg_result = df.agg(
            F.min(col_name).alias("min_val"),
            F.max(col_name).alias("max_val"),
            F.round(F.avg(col_name), 4).alias("mean_val"),
        ).first()
        stats["min"] = agg_result["min_val"]
        stats["max"] = agg_result["max_val"]
        stats["mean"] = float(agg_result["mean_val"]) if agg_result["mean_val"] is not None else None
    else:
        stats["min"] = None
        stats["max"] = None
        stats["mean"] = None

    return stats


def detect_duplicates(df, key_cols: list[str]) -> dict:
    """Count exact duplicates by the given key columns."""
    total = df.count()
    distinct = df.select(key_cols).distinct().count()
    dup_count = total - distinct
    return {
        "key_columns": key_cols,
        "total_rows": total,
        "distinct_rows": distinct,
        "duplicate_rows": dup_count,
        "duplicate_pct": round(dup_count / total * 100, 2) if total else 0.0,
    }


def compute_quality_score(column_stats_list: list[dict], dup_info: dict) -> float:
    """quality_score = 1.0 - (avg_null_pct/100 * 0.5 + dup_pct/100 * 0.5)"""
    if not column_stats_list:
        return 0.0

    avg_null_pct = sum(c["null_pct"] for c in column_stats_list) / len(column_stats_list)
    dup_pct = dup_info.get("duplicate_pct", 0.0)
    score = 1.0 - (avg_null_pct / 100.0 * 0.5 + dup_pct / 100.0 * 0.5)
    return round(max(0.0, min(1.0, score)), 4)


def main() -> None:
    spark = build_session()
    spark.sparkContext.setLogLevel("WARN")

    if not os.path.isdir(CLEANED_PATH):
        raise FileNotFoundError(
            f"Cleaned data not found at {CLEANED_PATH}. Run transform.py first."
        )

    df = spark.read.parquet(CLEANED_PATH)
    total = df.count()
    print(f"Loaded {total} rows from {CLEANED_PATH}")

    # Per-column statistics
    col_stats = [column_stats(df, col, total) for col in df.columns]

    # Duplicate detection on content_id (should be unique primary key)
    key_cols = ["content_id"]
    if "content_id" not in df.columns:
        key_cols = [df.columns[0]]
    dup_info = detect_duplicates(df, key_cols)

    quality_score = compute_quality_score(col_stats, dup_info)

    report = {
        "total_rows": total,
        "column_count": len(df.columns),
        "columns": col_stats,
        "duplicates": dup_info,
        "quality_score": quality_score,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"Quality score: {quality_score}")
    print(f"Report written to {REPORT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
