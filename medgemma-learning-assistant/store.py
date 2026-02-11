# ============================================================
# SQLite store for saved notes and flashcards.
# ============================================================

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "learning_data.db"


def get_conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn=None):
    close = False
    if conn is None:
        conn = get_conn()
        close = True
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            front TEXT NOT NULL,
            back TEXT NOT NULL,
            topic TEXT,
            next_review TEXT,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    if close:
        conn.close()


# ----- Notes -----

def save_note(question: str, answer: str, sources: list[str] | None = None) -> int:
    conn = get_conn()
    init_db(conn)
    sources_json = json.dumps(sources) if sources else None
    cur = conn.execute(
        "INSERT INTO notes (question, answer, sources_json, created_at) VALUES (?, ?, ?, ?)",
        (question, answer, sources_json, datetime.utcnow().isoformat()),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_all_notes() -> list[dict]:
    conn = get_conn()
    init_db(conn)
    rows = conn.execute("SELECT id, question, answer, sources_json, created_at FROM notes ORDER BY created_at DESC").fetchall()
    conn.close()
    out = []
    for r in rows:
        sources = json.loads(r["sources_json"]) if r["sources_json"] else None
        out.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "sources": sources,
            "created_at": r["created_at"],
        })
    return out


# ----- Flashcards -----

def save_flashcard(front: str, back: str, topic: str | None = None) -> int:
    conn = get_conn()
    init_db(conn)
    cur = conn.execute(
        "INSERT INTO flashcards (front, back, topic, next_review, created_at) VALUES (?, ?, ?, ?, ?)",
        (front, back, topic or "", datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def save_flashcards(cards: list[dict], topic: str | None = None) -> int:
    """cards = [{"front": "...", "back": "..."}, ...]"""
    conn = get_conn()
    init_db(conn)
    now = datetime.utcnow().isoformat()
    for c in cards:
        conn.execute(
            "INSERT INTO flashcards (front, back, topic, next_review, created_at) VALUES (?, ?, ?, ?, ?)",
            (c.get("front", ""), c.get("back", ""), topic or "", now, now),
        )
    conn.commit()
    count = len(cards)
    conn.close()
    return count


def get_all_flashcards(topic: str | None = None) -> list[dict]:
    conn = get_conn()
    init_db(conn)
    if topic:
        rows = conn.execute(
            "SELECT id, front, back, topic, next_review, created_at FROM flashcards WHERE topic = ? OR topic = '' ORDER BY id",
            (topic,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT id, front, back, topic, next_review, created_at FROM flashcards ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_flashcards_for_review(limit: int = 20) -> list[dict]:
    """Return cards due for review (next_review <= now or null)."""
    conn = get_conn()
    init_db(conn)
    now = datetime.utcnow().isoformat()
    rows = conn.execute(
        "SELECT id, front, back, topic FROM flashcards WHERE next_review IS NULL OR next_review <= ? ORDER BY id LIMIT ?",
        (now, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_flashcard_review(card_id: int, next_review_iso: str) -> None:
    conn = get_conn()
    conn.execute("UPDATE flashcards SET next_review = ? WHERE id = ?", (next_review_iso, card_id))
    conn.commit()
    conn.close()
