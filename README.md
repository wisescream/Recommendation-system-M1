# Dockerized Spark CSV Pipeline

![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)

This project runs a small local Spark standalone cluster with Docker Compose and processes a Kaggle CSV dataset with PySpark. The default dataset is `shivamb/netflix-shows`, which is normalized to `./data/raw/dataset.csv` before the job runs.

## Project layout

```text
.
|-- docker-compose.yml
|-- Makefile
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- dashboard/
|   |-- app.py
|   |-- Dockerfile
|   `-- requirements.txt
|-- spark/
|   |-- Dockerfile
|   `-- requirements.txt
|-- helpers/
|   `-- download_kaggle.py
|-- jobs/
|   |-- transform.py
|   `-- quality_report.py
|-- tests/
|   |-- conftest.py
|   |-- test_transform.py
|   |-- test_helpers.py
|   `-- test_dashboard.py
`-- data/
    |-- raw/
    `-- processed/
```

Container mounts:

- `./data/raw/dataset.csv` -> `/opt/spark-data/raw/dataset.csv`
- `./data/processed` -> `/opt/spark-data/processed`
- `./jobs` -> `/opt/spark-apps`

## Prerequisites

- Docker Desktop or Docker Engine with `docker compose`
- Kaggle account only if you want the helper container to download the dataset for you

The Spark master and worker use a custom Docker image built from `spark:4.0.1-python3` with `nltk` installed so the job can use a proper Snowball stemmer inside the cluster.

## Start Spark

Start the cluster:

```bash
docker compose up -d
```

If you want to force a rebuild of the custom Spark image after code or dependency changes:

```bash
docker compose up -d --build
```

Check the UIs:

- Spark Master UI: http://localhost:8080
- Spark Worker UI: http://localhost:8081
- Dashboard UI: http://localhost:8501

The Spark master URL used by the job is `spark://spark-master:7077`.

## Get the dataset

### Option 1: automatic download with Kaggle CLI in Docker

PowerShell:

```powershell
$env:KAGGLE_USERNAME="your_kaggle_username"
$env:KAGGLE_KEY="your_kaggle_key"
docker compose --profile tools run --rm kaggle-downloader
```

Bash:

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_key"
docker compose --profile tools run --rm kaggle-downloader
```

By default the helper downloads `shivamb/netflix-shows`, extracts the largest CSV from the archive, and saves it as `./data/raw/dataset.csv`.

To use a different Kaggle dataset, also set `KAGGLE_DATASET`, for example:

```bash
export KAGGLE_DATASET="shivamb/netflix-shows"
```

### Option 2: manual download on the host

If you do not want to provide `KAGGLE_USERNAME` and `KAGGLE_KEY`, download the dataset manually from Kaggle on the host:

1. Open `https://www.kaggle.com/datasets/shivamb/netflix-shows`
2. Download and extract the archive
3. Copy the CSV to `./data/raw/dataset.csv`

The Spark job expects the file to exist exactly at `./data/raw/dataset.csv`.

## Submit the Spark job

Run the transformation from the Spark master container:

```bash
docker compose exec -T spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --deploy-mode client --conf spark.driver.host=spark-master --conf spark.driver.bindAddress=0.0.0.0 --conf spark.pyspark.python=python3 /opt/spark-apps/transform.py
```

The job prints:

- input schema
- raw row count
- `show(10)` for the raw dataframe
- cleaned row count
- `show(10)` for the cleaned dataframe
- normalized text RDD samples
- filtered token RDD samples
- stemmed token RDD samples
- sentiment RDD samples
- top keyword and bigram samples from the RDD pipeline
- top TF-IDF samples from the RDD pipeline
- topic cluster samples
- embedding-based similarity samples
- `show(10)` for the text feature dataframe
- `show(10)` for the keyword dataframe
- `show(10)` for the bigram dataframe
- `show(10)` for the TF-IDF dataframe
- `show(10)` for the sentiment dataframe
- `show(10)` for the topic cluster dataframe
- `show(10)` for the similarity dataframe
- aggregated row count
- `show(10)` for the aggregated dataframe

## Text Treatments With RDD

The Spark job now includes an explicit text-processing chain built from one RDD to another. The treatments are applied only on text derived from `title`, `listed_in`, and `description`.

1. Convert cleaned rows into a document-text RDD
2. Normalize the raw text in a new RDD
3. Tokenize that normalized text into another RDD
4. Filter short tokens, digits, and stop words in another RDD
5. Stem the filtered tokens in another RDD with `nltk` Snowball stemming
6. Score sentiment from the stemmed-token RDD
7. Derive keyword counts, bigram counts, and TF-IDF summaries from downstream RDDs
8. Build document embeddings from the stemmed tokens
9. Create topic clusters and similarity pairs from those embeddings

These RDD results are converted back to DataFrames and written to disk.

## Dashboard

The dashboard runs as part of `docker compose up -d` and reads the same mounted `./data` directory as Spark.

Open it at:

- http://localhost:8501

It visualizes:

- raw versus cleaned row counts
- the number of rows dropped by the cleaning filters
- content type mix before and after transformation
- top countries by title count from the aggregated dataset
- release year trends and the derived `is_recent` flag
- top keywords and bigrams from the RDD text pipeline
- top TF-IDF terms from the RDD text pipeline
- sentiment distribution from the RDD text pipeline
- topic cluster distribution and representative cluster terms
- embedding-based title similarity pairs
- per-title text feature summaries such as token counts, dominant terms, and top TF-IDF terms
- raw, cleaned, and aggregated data samples

If you open the dashboard before running the Spark job, it shows the raw dataset and a message telling you to run `spark-submit` first.

## Outputs

After a successful run you should have:

- `./data/processed/cleaned/` with parquet files
- `./data/processed/agg/` with parquet files
- `./data/processed/text_features/` with parquet files
- `./data/processed/keywords/` with parquet files
- `./data/processed/bigrams/` with parquet files
- `./data/processed/tfidf/` with parquet files
- `./data/processed/sentiment/` with parquet files
- `./data/processed/topic_clusters/` with parquet files
- `./data/processed/similarity_pairs/` with parquet files
- `./data/processed/preview.csv` as a single easy-to-open CSV preview
- `./data/processed/metrics.json` with dashboard summary metrics

## Acceptance checks

Expected host-side results:

- `./data/processed/cleaned/` exists and contains Spark output files
- `./data/processed/agg/` exists and contains Spark output files
- `./data/processed/text_features/` exists and contains Spark output files
- `./data/processed/keywords/` exists and contains Spark output files
- `./data/processed/bigrams/` exists and contains Spark output files
- `./data/processed/tfidf/` exists and contains Spark output files
- `./data/processed/sentiment/` exists and contains Spark output files
- `./data/processed/topic_clusters/` exists and contains Spark output files
- `./data/processed/similarity_pairs/` exists and contains Spark output files
- `./data/processed/preview.csv` exists
- `http://localhost:8501` loads the dashboard
- the job console output shows schema, counts, and dataframe samples

## Troubleshooting

- If `docker compose exec` fails, wait until the master UI is reachable on `http://localhost:8080`.
- If ports `8080` or `8081` are already in use, change the host-side ports in `docker-compose.yml`.
- If port `8501` is already in use, change the dashboard host-side port in `docker-compose.yml`.
- If the job says `Expected CSV at /opt/spark-data/raw/dataset.csv`, the dataset was not downloaded or renamed correctly.
- If the dashboard says processed outputs are missing, run the Spark job again so `cleaned`, `agg`, `text_features`, `keywords`, `bigrams`, `tfidf`, `sentiment`, `topic_clusters`, `similarity_pairs`, and `metrics.json` are regenerated.
- If the Spark containers fail on startup after dependency changes, rebuild them with `docker compose up -d --build`.
- The first `kaggle-downloader` run is slower because it installs the Kaggle CLI inside the helper container before downloading.
- To stop the cluster, run `docker compose down`.

## Data Quality Report

After running the main transform, generate a quality report:

```bash
make quality
```

This produces `./data/processed/quality_report.json` with:

- per-column null counts, null percentages, and distinct value counts
- numeric min/max/mean where applicable
- duplicate detection on `content_id`
- an overall quality score: `1.0 - (avg_null_pct * 0.5 + dup_pct * 0.5)`

The dashboard **Data Quality** tab visualizes this report automatically.

## Makefile Targets

| Target    | Description                                    |
|-----------|------------------------------------------------|
| `up`      | Start the Spark cluster, worker, and dashboard |
| `down`    | Stop all containers                            |
| `run`     | Submit the main Spark transform job            |
| `quality` | Generate the data-quality report               |
| `test`    | Run the pytest test suite locally              |
| `logs`    | Tail container logs                            |
| `clean`   | Remove all processed outputs (keeps raw data)  |

## Testing

Run the test suite locally (requires `pyspark`, `nltk`, `streamlit`, and `pytest` installed):

```bash
make test
```

Or directly:

```bash
python -m pytest tests/ -v --tb=short
```

Tests cover:

- Pure-function unit tests for every RDD transformation helper
- Spark integration tests using a 50-row synthetic fixture
- Credential validation for the Kaggle download helper
- Dashboard helper smoke tests (metrics loading, parquet discovery)

CI runs automatically on push/PR to `main` via GitHub Actions.
