.PHONY: up down run quality test logs clean

## Start the Spark cluster, worker, and dashboard
up:
	docker compose up -d --build

## Stop all containers
down:
	docker compose down

## Submit the main Spark transform job
run:
	docker compose exec -T spark-master /opt/spark/bin/spark-submit \
		--master spark://spark-master:7077 \
		--deploy-mode client \
		--conf spark.driver.host=spark-master \
		--conf spark.driver.bindAddress=0.0.0.0 \
		--conf spark.pyspark.python=python3 \
		/opt/spark-apps/transform.py

## Generate the data-quality report
quality:
	docker compose exec -T spark-master /opt/spark/bin/spark-submit \
		--master spark://spark-master:7077 \
		--deploy-mode client \
		--conf spark.driver.host=spark-master \
		--conf spark.driver.bindAddress=0.0.0.0 \
		--conf spark.pyspark.python=python3 \
		/opt/spark-apps/quality_report.py

## Run the test suite locally (requires pyspark + nltk on the host)
test:
	python -m pytest tests/ -v --tb=short

## Tail container logs
logs:
	docker compose logs -f --tail=80

## Remove all processed outputs (keeps raw data)
clean:
	rm -rf data/processed/cleaned data/processed/agg data/processed/text_features \
		data/processed/keywords data/processed/bigrams data/processed/tfidf \
		data/processed/sentiment data/processed/topic_clusters data/processed/similarity_pairs \
		data/processed/preview.csv data/processed/metrics.json data/processed/quality_report.json
