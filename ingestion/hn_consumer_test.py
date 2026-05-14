# ingestion/hn_consumer_test.py
# This is just a verification script — not part of the final pipeline

import json
from kafka import KafkaConsumer
from dotenv import load_dotenv
import os

load_dotenv()

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC", "hn-posts")

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_SERVERS,
    auto_offset_reset="earliest",      # read from the very beginning
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

print(f"Listening to Kafka topic '{KAFKA_TOPIC}'... (Ctrl+C to stop)\n")

for message in consumer:
    story = message.value
    print(f"[{story['id']}] {story['title']}")
    print(f"  Score: {story['score']}  |  Comments: {story['comments']}  |  Author: {story['author']}")
    print()
