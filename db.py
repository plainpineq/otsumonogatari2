# db.py
import sqlite3

# =========================
# User DB (認証・アカウント)
# =========================

USER_DB_PATH = "users.db"


def get_user_conn():
    conn = sqlite3.connect(USER_DB_PATH)
    conn.row_factory = sqlite3.Row  # ★ ここが重要
    return conn


def init_user_db():
    with get_user_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        );
        """)
        conn.commit()


# =========================
# Writing DB (創作データ)
# =========================

WRITING_DB_PATH = "writing.db"


def get_conn():
    return sqlite3.connect(WRITING_DB_PATH)


def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS story (
            id TEXT PRIMARY KEY,
            title TEXT,
            synopsis TEXT,
            doc_type TEXT DEFAULT 'novel'
        );

        CREATE TABLE IF NOT EXISTS scene (
            id TEXT PRIMARY KEY,
            story_id TEXT,
            title TEXT,
            summary TEXT,
            order_no INTEGER,
            time_start INTEGER,
            time_end INTEGER,
            location TEXT
        );

        CREATE TABLE IF NOT EXISTS character (
            id TEXT PRIMARY KEY,
            story_id TEXT,
            name TEXT,
            role TEXT,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS author_context (
            story_id TEXT PRIMARY KEY,
            genre TEXT,
            theme_or_claim TEXT,
            values TEXT,
            constraints TEXT
        );
        """)
        conn.commit()
