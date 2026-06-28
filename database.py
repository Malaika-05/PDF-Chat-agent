import sqlite3
import json
from datetime import datetime

DB_PATH = "chat_history.db"

def init_db():
    """Create tables if they don't exist — runs once at startup"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sessions table — one row per PDF
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_name TEXT NOT NULL,
            chunk_count INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    
    # Messages table — every Q&A pair
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    conn.commit()
    conn.close()

def create_session(pdf_name: str, chunk_count: int) -> int:
    """Create a new session when PDF is uploaded, return session_id"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (pdf_name, chunk_count) VALUES (?, ?)",
        (pdf_name, chunk_count)
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id

def save_message(session_id: int, role: str, content: str, sources: list = None):
    """Save a single message to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, sources) VALUES (?, ?, ?, ?)",
        (session_id, role, content, json.dumps(sources or []))
    )
    conn.commit()
    conn.close()

def get_sessions() -> list:
    """Get all sessions for the sidebar list"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.id, s.pdf_name, s.chunk_count, s.created_at,
               COUNT(m.id) as message_count
        FROM sessions s
        LEFT JOIN messages m ON s.id = m.session_id AND m.role = 'user'
        GROUP BY s.id
        ORDER BY s.created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "pdf_name": r[1],
            "chunk_count": r[2],
            "created_at": r[3],
            "message_count": r[4]
        }
        for r in rows
    ]

def get_messages(session_id: int) -> list:
    """Load all messages for a session"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, sources, created_at FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "role": r[0],
            "content": r[1],
            "sources": json.loads(r[2] or "[]"),
            "created_at": r[3]
        }
        for r in rows
    ]

def delete_session(session_id: int):
    """Delete a session and all its messages"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()