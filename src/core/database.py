import sqlite3

from pathlib import Path

from src.schemas.database import DB_SCHEMA, REQUIRED_TABLES
from src.resources.utils import get_connection
from src.env import PERSISTENT_DATA



def _database_has_required_tables(conn: sqlite3.Connection) -> bool:
    """Check if all required tables exist in the database."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = {row[0] for row in cursor.fetchall()}
    return REQUIRED_TABLES.issubset(existing_tables)


def initialize_database():
    """Initializes the SQLite database with the required schema."""
    db_path = Path(PERSISTENT_DATA)
    db_exists = db_path.exists()
    initialize = True

    if db_exists:
        try:
            with sqlite3.connect(db_path) as conn:
                if _database_has_required_tables(conn):
                    print(f"Database {db_path} already initialized")
                    initialize = False
                else:
                    print("Database incomplete. Recreating the database with the schema...")
        except sqlite3.DatabaseError:
            print("Fichier existant invalide. Il ne s'agit pas d'une base SQLite valide.")

    if initialize:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(DB_SCHEMA)
        print("Database initialized successfully.")


def get_all_threads() -> list[tuple]:
    """Retrieve all threads from the database."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT thread_id, thread_name FROM threads ORDER BY thread_id ASC")
        return [(row[0], row[1]) for row in cursor.fetchall()]


def new_thread(thread_id: int, thread_name: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO threads (thread_id, thread_name) VALUES (?, ?)", (thread_id, thread_name)
        )


def delete_thread(thread_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))