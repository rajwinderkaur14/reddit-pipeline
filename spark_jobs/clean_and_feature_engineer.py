# spark_jobs/clean_and_feature_engineer.py

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, LongType
)
from dotenv import load_dotenv

load_dotenv()

# ── paths ──────────────────────────────────────────────────────────────────────
KAFKA_SERVERS  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC    = os.getenv("KAFKA_TOPIC", "hn-posts")
OUTPUT_PATH    = "data/processed/hn_stories_delta"   # local Delta Lake table
CHECKPOINT_DIR = "data/checkpoints/hn_clean"         # Spark needs this to track progress


# ── 1. Create Spark session ────────────────────────────────────────────────────
# SparkSession is the entry point to everything in Spark
# Think of it like the "connection" to your Spark engine
# def create_spark_session() -> SparkSession:
#     jars = ",".join([
#         "spark_jobs/jars/spark-sql-kafka-0-10_2.12-3.5.0.jar",
#         "spark_jobs/jars/delta-spark_2.12-3.0.0.jar",
#         "spark_jobs/jars/delta-storage-3.0.0.jar",
#     ])

#     return (
#         SparkSession.builder
#         .appName("HN-Clean-And-Feature-Engineer")
#         .master("local[*]")               # use all CPU cores on your machine
#         .config("spark.jars", jars)
#         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
#         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
#         .config("spark.sql.shuffle.partitions", "4")  # keep low for local dev
#         .getOrCreate()
#     )

def create_spark_session():
    return (
        SparkSession.builder
        .appName("HN-Clean-And-Feature-Engineer")
        .master("local[*]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
# ── 2. Define the schema of our Kafka messages ─────────────────────────────────
# Kafka sends everything as raw bytes. We tell Spark what shape the JSON is.
# This is like telling Spark "expect these columns with these types"
RAW_SCHEMA = StructType([
    StructField("id",          LongType(),    True),
    StructField("title",       StringType(),  True),
    StructField("url",         StringType(),  True),
    StructField("score",       IntegerType(), True),
    StructField("author",      StringType(),  True),
    StructField("comments",    IntegerType(), True),
    StructField("unix_time",   LongType(),    True),
    StructField("ingested_at", LongType(),    True),
    StructField("type",        StringType(),  True),
])


# ── 3. Read from Kafka ─────────────────────────────────────────────────────────
def read_from_kafka(spark: SparkSession):
    return (
        spark.read                             # batch read (not streaming)
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest") # read ALL messages from beginning
        .load()
    )


# ── 4. Parse raw Kafka bytes → structured columns ──────────────────────────────
def parse_kafka_messages(raw_df):
    # Kafka gives us: key (bytes), value (bytes), topic, partition, offset, timestamp
    # The actual story JSON is in the 'value' column as bytes
    # We cast it to string, then parse the JSON using our schema
    return (
        raw_df
        .select(
            F.col("value").cast("string").alias("json_str"),
            F.col("timestamp").alias("kafka_timestamp")   # when Kafka received it
        )
        .select(
            F.from_json("json_str", RAW_SCHEMA).alias("data"),
            "kafka_timestamp"
        )
        .select("data.*", "kafka_timestamp")   # flatten nested struct into columns
    )


# ── 5. Clean the data ──────────────────────────────────────────────────────────
# Real data is messy. This step removes garbage rows and fixes types.
def clean(df):
    return (
        df
        # Drop rows where essential fields are missing
        .filter(F.col("id").isNotNull())
        .filter(F.col("title").isNotNull())
        .filter(F.col("score").isNotNull())

        # Drop rows with nonsense values
        .filter(F.col("score") >= 0)
        .filter(F.col("comments") >= 0)
        .filter(F.length(F.col("title")) > 3)    # title must be more than 3 chars

        # Remove duplicate stories (same id appearing twice)
        .dropDuplicates(["id"])

        # Trim whitespace from text fields
        .withColumn("title",  F.trim(F.col("title")))
        .withColumn("author", F.trim(F.col("author")))

        # Fill nulls with sensible defaults
        .fillna({"url": "unknown", "author": "unknown", "comments": 0})
    )


# ── 6. Engineer features ───────────────────────────────────────────────────────
# Features = the columns the ML model will use to make predictions
# We're predicting virality, so we create signals that might indicate virality
def engineer_features(df):
    return (
        df
        # Convert unix timestamp → proper datetime columns
        .withColumn("posted_at", F.to_timestamp(F.col("unix_time")))
        .withColumn("hour_of_day", F.hour("posted_at"))        # 0-23
        .withColumn("day_of_week", F.dayofweek("posted_at"))   # 1=Sun, 7=Sat
        .withColumn("is_weekend",  (F.col("day_of_week").isin(1, 7)).cast("int"))

        # Text-based features
        .withColumn("title_length",    F.length("title"))
        .withColumn("title_word_count", F.size(F.split("title", " ")))
        .withColumn("has_url",         F.when(F.col("url") == "unknown", 0).otherwise(1))

        # Is it a "Show HN" or "Ask HN" post? (these tend to perform differently)
        .withColumn("is_show_hn", F.when(F.col("title").startswith("Show HN"), 1).otherwise(0))
        .withColumn("is_ask_hn",  F.when(F.col("title").startswith("Ask HN"),  1).otherwise(0))

        # Engagement ratio — comments per score point (how discussable is it?)
        .withColumn("comment_score_ratio",
            F.when(F.col("score") > 0,
                F.col("comments") / F.col("score")
            ).otherwise(0.0)
        )

        # TARGET LABEL for ML: 1 if story is "viral" (score > 100), else 0
        # This is what our model will learn to predict
        .withColumn("is_viral", F.when(F.col("score") >= 100, 1).otherwise(0))

        # Drop columns we no longer need
        .drop("unix_time", "type", "kafka_timestamp")
    )


# ── 7. Write to Delta Lake ─────────────────────────────────────────────────────
def write_to_delta(df, path: str):
    os.makedirs(path, exist_ok=True)
    (
        df.write
        .format("delta")
        .mode("overwrite")     # replace table each run (use "append" in production)
        .option("overwriteSchema", "true")
        .save(path)
    )
    print(f"✓ Written to Delta Lake at: {path}")


# ── 8. Show a summary of what we processed ────────────────────────────────────
def print_summary(df):
    total    = df.count()
    viral    = df.filter(F.col("is_viral") == 1).count()
    show_hn  = df.filter(F.col("is_show_hn") == 1).count()
    ask_hn   = df.filter(F.col("is_ask_hn") == 1).count()

    print("\n" + "="*50)
    print("         PIPELINE SUMMARY")
    print("="*50)
    print(f"  Total stories processed : {total}")
    print(f"  Viral stories (score≥100): {viral}  ({100*viral//total if total else 0}%)")
    print(f"  Show HN posts           : {show_hn}")
    print(f"  Ask HN posts            : {ask_hn}")
    print("="*50 + "\n")

    print("Sample rows:")
    df.select("id", "title", "score", "comments", "is_viral",
              "hour_of_day", "title_word_count", "is_show_hn").show(10, truncate=50)


# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Spark job...")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")   # suppress noisy Spark INFO logs

    print("Reading from Kafka...")
    raw_df = read_from_kafka(spark)

    print("Parsing messages...")
    parsed_df = parse_kafka_messages(raw_df)

    print("Cleaning data...")
    cleaned_df = clean(parsed_df)

    print("Engineering features...")
    featured_df = engineer_features(cleaned_df)

    # Cache the dataframe — we'll use it twice (summary + write)
    # Without cache, Spark would recompute everything twice
    featured_df.cache()

    print_summary(featured_df)

    print("Writing to Delta Lake...")
    write_to_delta(featured_df, OUTPUT_PATH)

    spark.stop()
    print("Spark job complete.")
