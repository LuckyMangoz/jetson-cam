#!/usr/bin/env python3
"""SQLite storage for detection history"""

import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = "data/detections.db"
BATCH_SIZE = 50

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT,
    model       TEXT    NOT NULL,
    resolution  TEXT    NOT NULL,
    threshold   REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL,
    frame_number  INTEGER NOT NULL,
    wall_time     TEXT    NOT NULL,
    label         TEXT    NOT NULL,
    confidence    REAL    NOT NULL,
    x             INTEGER NOT NULL,
    y             INTEGER NOT NULL,
    width         INTEGER NOT NULL,
    height        INTEGER NOT NULL,
    target_x      REAL,
    target_y      REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_det_session ON detections(session_id);
CREATE INDEX IF NOT EXISTS idx_det_time    ON detections(wall_time);
CREATE INDEX IF NOT EXISTS idx_det_label   ON detections(label);
"""

INSERT_DETECTION = (
    "INSERT INTO detections (session_id, frame_number, wall_time, "
    "label, confidence, x, y, width, height, target_x, target_y) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")


def now_iso():
    """Current UTC time as a sortable string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_db(path=DB_PATH):
    """Open the database, creating the file and tables if needed."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.commit()
    return connection


def start_session(connection, model, resolution, threshold):
    """Record a new run. Returns the session id."""
    cursor = connection.execute(
        "INSERT INTO sessions (started_at, model, resolution, threshold) "
        "VALUES (?, ?, ?, ?)",
        (now_iso(), model, resolution, threshold))
    connection.commit()
    return cursor.lastrowid


def make_row(session_id, frame_number, detection,
             target_x=None, target_y=None):
    """Build one detection row. Column order must match INSERT_DETECTION."""
    return (session_id, frame_number, now_iso(),
            detection.label, detection.confidence,
            detection.x, detection.y, detection.width, detection.height,
            target_x, target_y)


def flush(connection, pending):
    """Write every queued row in one transaction, then clear the list."""
    if not pending:
        return 0

    connection.executemany(INSERT_DETECTION, pending)
    connection.commit()

    written = len(pending)
    pending.clear()
    return written


def end_session(connection, session_id, pending):
    """Flush anything left and stamp the session's end time."""
    flush(connection, pending)
    connection.execute(
        "UPDATE sessions SET ended_at = ? WHERE id = ?",
        (now_iso(), session_id))
    connection.commit()