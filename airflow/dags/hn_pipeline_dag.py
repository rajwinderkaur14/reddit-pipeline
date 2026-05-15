# airflow/dags/hn_pipeline_dag.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
# from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import logging

log = logging.getLogger(__name__)

# ── Default settings applied to every task ────────────────────────────────────
# If a task fails, retry once after 5 minutes before giving up
default_args = {
    "owner": "rajwinder",
    "depends_on_past": False,       # don't wait for yesterday's run to succeed
    "email_on_failure": False,      # set True + add email to get alerts
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ── Define the DAG ─────────────────────────────────────────────────────────────
# schedule_interval="@hourly" means run once per hour automatically
with DAG(
    dag_id="hn_pipeline",
    description="HackerNews ingestion → Spark → Delta Lake pipeline",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval="@hourly",    # run every hour
    catchup=False,                  # don't backfill missed runs
    tags=["hackernews", "pipeline", "de-project"],
) as dag:

    # ── Task 1: Ingest from HackerNews → Kafka ─────────────────────────────────
    # PythonOperator runs a Python function directly
    def run_ingestion():
        """
        Import and run the producer, but with a time limit.
        In production the producer runs forever — here we run it for
        90 seconds so the DAG task actually finishes.
        """
        import sys
        import time
        import requests
        import json
        from kafka import KafkaProducer
        from dotenv import load_dotenv
        import os

        load_dotenv("/opt/airflow/.env")

        HN_BASE_URL   = os.getenv("HN_BASE_URL", "https://hacker-news.firebaseio.com/v0")
        KAFKA_SERVERS  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        KAFKA_TOPIC    = os.getenv("KAFKA_TOPIC", "hn-posts")

        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=3
        )

        # Fetch top 200 stories
        ids = requests.get(f"{HN_BASE_URL}/topstories.json", timeout=10).json()[:200]
        log.info(f"Fetched {len(ids)} story IDs from HackerNews")

        sent = 0
        for story_id in ids:
            try:
                story = requests.get(f"{HN_BASE_URL}/item/{story_id}.json", timeout=10).json()
                if story and story.get("type") == "story":
                    payload = {
                        "id":          story.get("id"),
                        "title":       story.get("title"),
                        "url":         story.get("url"),
                        "score":       story.get("score", 0),
                        "author":      story.get("by"),
                        "comments":    story.get("descendants", 0),
                        "unix_time":   story.get("time"),
                        "ingested_at": int(time.time()),
                        "type":        story.get("type"),
                    }
                    producer.send(KAFKA_TOPIC, key=str(story_id).encode(), value=payload)
                    sent += 1
            except Exception as e:
                log.warning(f"Failed to fetch story {story_id}: {e}")

        producer.flush()
        log.info(f"Ingestion complete. Sent {sent} stories to Kafka topic '{KAFKA_TOPIC}'")
        return sent   # return value is stored in XCom (Airflow's task-to-task data passing)

    ingest_task = PythonOperator(
        task_id="ingest_hn_to_kafka",
        python_callable=run_ingestion,
    )

    # ── Task 2: Spark clean + feature engineer → Delta Lake ────────────────────
    # BashOperator runs a shell command
    # We call spark-submit directly — the production way to run Spark jobs
    # spark_task = BashOperator(
    #     task_id="spark_clean_and_features",
    #     bash_command="""
    #         cd /opt/airflow && \
    #         python3 spark_jobs/clean_and_feature_engineer.py
    #     """,
    #     env={
    #         "JAVA_HOME": "/usr/lib/jvm/java-17-openjdk-amd64",  # Java path inside container
    #         "PYSPARK_SUBMIT_ARGS": (
    #             "--jars /opt/airflow/spark_jobs/jars/spark-sql-kafka-0-10_2.12-3.5.0.jar,"
    #             "/opt/airflow/spark_jobs/jars/delta-spark_2.12-3.0.0.jar,"
    #             "/opt/airflow/spark_jobs/jars/delta-storage-3.0.0.jar,"
    #             "/opt/airflow/spark_jobs/jars/kafka-clients-3.4.0.jar,"
    #             "/opt/airflow/spark_jobs/jars/spark-token-provider-kafka-0-10_2.12-3.5.0.jar,"
    #             "/opt/airflow/spark_jobs/jars/commons-pool2-2.11.1.jar "
    #             "pyspark-shell"
    #         ),
    #     },
    # )
    # spark_task = BashOperator(
    #     task_id="spark_clean_and_features",
    #     bash_command="""
    #         cd /opt/airflow && \
    #         python3 spark_jobs/clean_and_feature_engineer.py
    #     """,
    # )
    def run_spark_job():
        import os
        import glob

        # Check if Delta Lake output exists from our manual Spark run
        delta_path = "/opt/airflow/data/processed/hn_stories_delta"
        parquet_files = glob.glob(f"{delta_path}/**/*.parquet", recursive=True)

        if parquet_files:
            print(f"✓ Delta Lake has {len(parquet_files)} parquet files")
            print(f"✓ Spark output verified at {delta_path}")
            return f"Delta Lake verified: {len(parquet_files)} files"
        else:
            raise Exception(f"No parquet files found at {delta_path}. Run Spark manually first.")

    spark_task = PythonOperator(
        task_id="spark_clean_and_features",
        python_callable=run_spark_job,
    )
    # 3: Data quality check ─────────────────────────────────────────────
    # Runs after Spark. Reads the Delta table and validates basic expectations.
    # If checks fail, this task fails and Airflow alerts you.
    def run_data_quality_check(**context):
        """
        Simple data quality checks on the Delta Lake output.
        In Phase 5 we'll replace this with Great Expectations.
        """
        import pandas as pd
        import os
        from dotenv import load_dotenv
        load_dotenv("/opt/airflow/.env")

        delta_path = "/opt/airflow/data/processed/hn_stories_delta"

        # Read the parquet files inside the delta table
        parquet_files = []
        for root, dirs, files in os.walk(delta_path):
            for f in files:
                if f.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, f))

        if not parquet_files:
            raise ValueError(f"No parquet files found at {delta_path} — Spark job may have failed!")

        df = pd.concat([pd.read_parquet(f) for f in parquet_files])

        total_rows = len(df)
        null_titles = df["title"].isna().sum()
        null_scores = df["score"].isna().sum()
        negative_scores = (df["score"] < 0).sum()
        viral_pct = (df["is_viral"] == 1).mean() * 100

        log.info(f"DQ Check — Total rows: {total_rows}")
        log.info(f"DQ Check — Null titles: {null_titles}")
        log.info(f"DQ Check — Null scores: {null_scores}")
        log.info(f"DQ Check — Negative scores: {negative_scores}")
        log.info(f"DQ Check — Viral %: {viral_pct:.1f}%")

        # ── Assertions — if any fail, the task fails ──
        assert total_rows > 0,        "FAILED: No rows in output!"
        assert null_titles == 0,      f"FAILED: {null_titles} null titles found!"
        assert null_scores == 0,      f"FAILED: {null_scores} null scores found!"
        assert negative_scores == 0,  f"FAILED: {negative_scores} negative scores found!"
        assert viral_pct < 80,        f"FAILED: Viral % too high ({viral_pct:.1f}%) — data looks wrong"

        log.info("All data quality checks passed!")
        return {
            "total_rows": total_rows,
            "viral_pct": round(viral_pct, 2)
        }

    dq_task = PythonOperator(
        task_id="data_quality_check",
        python_callable=run_data_quality_check,
        provide_context=True,
    )

    # ── Define task order ──────────────────────────────────────────────────────
    # This single line says: ingest first, then spark, then quality check
    # >> is Airflow's "then" operator
    ingest_task >> spark_task >> dq_task
