import sqlite3
import os
from pathlib import Path

# Save in the data/processed directory inside the workspace
DB_PATH = Path("data") / "processed" / "corpora.db"

def init_db():
    """Initializes the SQLite database and creates the status table."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS corpus_status (
            corpus_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            chunk_count INTEGER,
            file_count INTEGER,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def set_corpus_status(corpus_id: str, status: str, chunk_count: int = None, file_count: int = None):
    """Sets or updates the status of a corpus. Thread-safe by opening/closing on each call."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    # Using SQL ON CONFLICT to perform insert-or-update
    cursor.execute("""
        INSERT INTO corpus_status (corpus_id, status, chunk_count, file_count, created_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        ON CONFLICT(corpus_id) DO UPDATE SET
            status=excluded.status,
            chunk_count=coalesce(excluded.chunk_count, chunk_count),
            file_count=coalesce(excluded.file_count, file_count)
    """, (corpus_id, status, chunk_count, file_count))
    conn.commit()
    conn.close()

def get_corpus_status(corpus_id: str) -> dict | None:
    """Returns the status dict for corpus_id, or None if not found."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT status, chunk_count, file_count FROM corpus_status WHERE corpus_id=?", (corpus_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "status": row[0],
            "chunk_count": row[1],
            "file_count": row[2]
        }
    return None

def delete_corpus_status(corpus_id: str):
    """Deletes the corpus status entry from SQLite."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("DELETE FROM corpus_status WHERE corpus_id=?", (corpus_id,))
    conn.commit()
    conn.close()

def get_expired_corpora(hours: int = 24) -> list[str]:
    """Returns a list of corpus IDs created more than `hours` ago."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    # Find records where created_at is older than `hours`
    cursor.execute(
        "SELECT corpus_id FROM corpus_status WHERE datetime(created_at) < datetime('now', '-' || ? || ' hours')",
        (hours,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]

def get_total_active_corpora() -> int:
    """Returns the total count of active corpora in the database."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM corpus_status")
    count = cursor.fetchone()[0]
    conn.close()
    return count

