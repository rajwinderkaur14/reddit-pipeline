# ingestion/hn_producer.py

import json
import time
import logging
import requests
from kafka import KafkaProducer
from dotenv import load_dotenv
import os

# Load secrets from .env file
load_dotenv()

# Set up logging — prints timestamped messages to terminal so you know what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# --- CONFIG ---
HN_BASE_URL = os.getenv("HN_BASE_URL", "https://hacker-news.firebaseio.com/v0")
KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC", "hn-posts")
FETCH_INTERVAL_SECONDS = 60   # how often we poll HackerNews
TOP_N_STORIES = 200           # how many top stories to fetch each round


def get_top_story_ids():
    """Fetch the current top story IDs from HackerNews."""
    url = f"{HN_BASE_URL}/topstories.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()          # throws error if request failed
    return response.json()[:TOP_N_STORIES]


def get_story_details(story_id: int):
    """Fetch full details of a single story by its ID."""
    url = f"{HN_BASE_URL}/item/{story_id}.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    story = response.json()

    # Only return stories (not comments/jobs/polls)
    if story and story.get("type") == "story":
        return story
    return None


def create_kafka_producer() -> KafkaProducer:
    """Create and return a Kafka producer client."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        # Serialize Python dict → JSON bytes before sending to Kafka
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        # Retry up to 3 times if send fails
        retries=3
    )


def enrich_story(story: dict) -> dict:
    """Add extra fields useful for our pipeline."""
    return {
        "id":          story.get("id"),
        "title":       story.get("title"),
        "url":         story.get("url"),
        "score":       story.get("score", 0),
        "author":      story.get("by"),
        "comments":    story.get("descendants", 0),
        "unix_time":   story.get("time"),               # original post timestamp
        "ingested_at": int(time.time()),                # when WE fetched it
        "type":        story.get("type"),
    }


def run():
    """Main loop — keep fetching and publishing forever."""
    log.info("Starting HackerNews Kafka producer...")
    producer = create_kafka_producer()
    log.info(f"Connected to Kafka at {KAFKA_SERVERS}, topic: {KAFKA_TOPIC}")

    seen_ids = set()   # track already-sent stories to avoid duplicates

    while True:
        try:
            log.info("Fetching top story IDs from HackerNews...")
            story_ids = get_top_story_ids()

            # Only process stories we haven't sent yet
            new_ids = [sid for sid in story_ids if sid not in seen_ids]
            log.info(f"Found {len(story_ids)} top stories, {len(new_ids)} are new")

            for story_id in new_ids:
                story = get_story_details(story_id)
                if story:
                    enriched = enrich_story(story)
                    # Send to Kafka — key is story ID so same story always
                    # goes to the same Kafka partition (ordering guarantee)
                    producer.send(
                        KAFKA_TOPIC,
                        key=str(story_id).encode("utf-8"),
                        value=enriched
                    )
                    seen_ids.add(story_id)
                    log.info(f"  Sent → [{story_id}] {enriched['title'][:60]}... (score: {enriched['score']})")

            producer.flush()   # make sure all messages are actually sent
            log.info(f"Done. Waiting {FETCH_INTERVAL_SECONDS}s before next fetch...\n")

        except requests.RequestException as e:
            log.error(f"Network error fetching from HackerNews: {e}")
        except Exception as e:
            log.error(f"Unexpected error: {e}")

        time.sleep(FETCH_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
