"""SQLite-backed prediction logging and analytics queries."""

import re
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = Path(os.getenv("TRUTHLENS_DB_PATH", str(PROJECT_ROOT / "data" / "truthlens.db")))

STOP_WORDS = {
    "about", "after", "again", "against", "among", "because", "before", "being", "between",
    "could", "during", "from", "have", "into", "more", "most", "other", "over", "said",
    "some", "such", "than", "that", "their", "there", "these", "they", "this", "through",
    "under", "were", "which", "while", "with", "would", "your", "will", "what", "when",
}


def _connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.execute("PRAGMA busy_timeout = 10000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_text TEXT NOT NULL,
                domain TEXT,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                model_used TEXT NOT NULL
            )
            """
        )
        connection.commit()


def log_prediction(text: str, label: str, confidence: float, model_used: str, domain: str | None = None) -> None:
    for attempt in range(3):
        try:
            with _connect() as connection:
                connection.execute(
                    "INSERT INTO predictions (timestamp, input_text, domain, label, confidence, model_used) VALUES (?, ?, ?, ?, ?, ?)",
                    (datetime.now(timezone.utc).isoformat(), text, domain, label, confidence, model_used),
                )
                connection.commit()
            return
        except sqlite3.OperationalError:
            if attempt == 2:
                raise
            time.sleep(0.05 * (attempt + 1))


def get_analytics() -> dict:
    initialize_database()
    with _connect() as connection:
        total = connection.execute("SELECT COUNT(*) AS count FROM predictions").fetchone()["count"]
        labels = connection.execute(
            "SELECT label, COUNT(*) AS count FROM predictions GROUP BY label"
        ).fetchall()
        daily = connection.execute(
            "SELECT substr(timestamp, 1, 10) AS date, COUNT(*) AS total, SUM(label = 'Fake') AS fake, SUM(label = 'Real') AS real FROM predictions GROUP BY date ORDER BY date DESC LIMIT 30"
        ).fetchall()
        models = connection.execute(
            "SELECT model_used AS model, COUNT(*) AS count FROM predictions GROUP BY model_used ORDER BY count DESC"
        ).fetchall()
        fake_texts = connection.execute(
            "SELECT input_text FROM predictions WHERE label = 'Fake'"
        ).fetchall()

    label_counts = {row["label"]: row["count"] for row in labels}
    keyword_counts: dict[str, int] = {}
    for row in fake_texts:
        words = re.findall(r"[a-zA-Z]{4,}", row["input_text"].lower())
        for word in set(words):
            if word not in STOP_WORDS:
                keyword_counts[word] = keyword_counts.get(word, 0) + 1

    top_keywords = [
        {"keyword": keyword, "count": count}
        for keyword, count in sorted(keyword_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    ]
    over_time = [
        {"date": row["date"], "total": row["total"], "fake": row["fake"] or 0, "real": row["real"] or 0}
        for row in reversed(daily)
    ]
    return {
        "total_checks": total,
        "fake_count": label_counts.get("Fake", 0),
        "real_count": label_counts.get("Real", 0),
        "over_time": over_time,
        "top_keywords": top_keywords,
        "model_usage": [{"model": row["model"], "count": row["count"]} for row in models],
    }


initialize_database()
