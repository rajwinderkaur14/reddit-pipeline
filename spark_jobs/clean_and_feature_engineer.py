import os
import time
import requests
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

OUTPUT_PATH = "data/processed/hn_stories_delta"

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

def fetch_from_hn_api(spark):
    print("Fetching stories from HackerNews API...")
    ids = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10).json()[:200]
    stories = []
    for sid in ids:
        try:
            s = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=5).json()
            if s and s.get("type") == "story" and s.get("title"):
                stories.append({
                    "id":          int(s.get("id", 0)),
                    "title":       str(s.get("title", "")),
                    "url":         str(s.get("url", "unknown")),
                    "score":       int(s.get("score", 0)),
                    "author":      str(s.get("by", "unknown")),
                    "comments":    int(s.get("descendants", 0)),
                    "unix_time":   int(s.get("time", 0)),
                    "ingested_at": int(time.time()),
                })
        except:
            continue
    print(f"Fetched {len(stories)} stories")
    return spark.createDataFrame(stories)

def clean(df):
    return (
        df
        .filter(F.col("id").isNotNull())
        .filter(F.col("title").isNotNull())
        .filter(F.col("score") >= 0)
        .filter(F.col("comments") >= 0)
        .filter(F.length(F.col("title")) > 3)
        .dropDuplicates(["id"])
        .withColumn("title",  F.trim(F.col("title")))
        .withColumn("author", F.trim(F.col("author")))
        .fillna({"url": "unknown", "author": "unknown", "comments": 0})
    )

def engineer_features(df):
    return (
        df
        .withColumn("posted_at",        F.to_timestamp(F.col("unix_time")))
        .withColumn("hour_of_day",      F.hour("posted_at"))
        .withColumn("day_of_week",      F.dayofweek("posted_at"))
        .withColumn("is_weekend",       (F.col("day_of_week").isin(1, 7)).cast("int"))
        .withColumn("title_length",     F.length("title"))
        .withColumn("title_word_count", F.size(F.split("title", " ")))
        .withColumn("has_url",          F.when(F.col("url") == "unknown", 0).otherwise(1))
        .withColumn("is_show_hn",       F.when(F.col("title").startswith("Show HN"), 1).otherwise(0))
        .withColumn("is_ask_hn",        F.when(F.col("title").startswith("Ask HN"), 1).otherwise(0))
        .withColumn("comment_score_ratio",
            F.when(F.col("score") > 0, F.col("comments") / F.col("score")).otherwise(0.0))
        .withColumn("is_viral",         F.when(F.col("score") >= 100, 1).otherwise(0))
        .drop("unix_time", "ingested_at")
    )

def write_to_delta(df, path):
    os.makedirs(path, exist_ok=True)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(path)
    print(f"Written to Delta Lake at: {path}")

def print_summary(df):
    total   = df.count()
    viral   = df.filter(F.col("is_viral") == 1).count()
    show_hn = df.filter(F.col("is_show_hn") == 1).count()
    ask_hn  = df.filter(F.col("is_ask_hn") == 1).count()
    print("\n" + "="*50)
    print("         PIPELINE SUMMARY")
    print("="*50)
    print(f"  Total stories : {total}")
    print(f"  Viral (>=100) : {viral} ({100*viral//total if total else 0}%)")
    print(f"  Show HN       : {show_hn}")
    print(f"  Ask HN        : {ask_hn}")
    print("="*50 + "\n")
    df.select("id","title","score","comments","is_viral","hour_of_day","is_show_hn").show(10, truncate=50)

if __name__ == "__main__":
    print("Starting Spark job...")
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    raw_df      = fetch_from_hn_api(spark)
    cleaned_df  = clean(raw_df)
    featured_df = engineer_features(cleaned_df)
    featured_df.cache()
    print_summary(featured_df)
    write_to_delta(featured_df, OUTPUT_PATH)
    spark.stop()
    print("Spark job complete.")
