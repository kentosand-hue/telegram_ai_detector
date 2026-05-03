from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable


@dataclass(frozen=True)
class CheckRecord:
    file_type: str
    file_name: str
    probability: float
    label: str
    created_at: str


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT,
                file_type TEXT NOT NULL,
                file_name TEXT NOT NULL,
                probability REAL NOT NULL,
                label TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_checks_user_id ON checks(user_id, created_at DESC);"
        )
        conn.commit()


def save_check(
    db_path: Path,
    *,
    user_id: int,
    username: str | None,
    file_type: str,
    file_name: str,
    probability: float,
    label: str,
    created_at: str,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO checks (user_id, username, file_type, file_name, probability, label, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (user_id, username, file_type, file_name, probability, label, created_at),
        )
        conn.commit()


def get_history(db_path: Path, *, user_id: int, limit: int) -> list[CheckRecord]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT file_type, file_name, probability, label, created_at
            FROM checks
            WHERE user_id = ?
            ORDER BY datetime(created_at) DESC
            LIMIT ?;
            """,
            (user_id, limit),
        ).fetchall()

    return [
        CheckRecord(
            file_type=row[0],
            file_name=row[1],
            probability=row[2],
            label=row[3],
            created_at=row[4],
        )
        for row in rows
    ]
