import sqlite3
import os

from pathlib import Path

from src.schemas.database import DB_SCHEMA, REQUIRED_TABLES
from src.utils.utils import get_connection
from src.env import PERSISTENT_DIR

from src.utils.logging import get_logger
databaseLogger = get_logger("database")

def _database_has_required_tables(conn: sqlite3.Connection) -> bool:
    """Check if all required tables exist in the database."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    existing_tables = {row[0] for row in cursor.fetchall()}
    return REQUIRED_TABLES.issubset(existing_tables)


def initialize_database():
    """Initializes the SQLite database with the required schema."""
    db_path = Path(PERSISTENT_DIR)
    os.makedirs(db_path.stem, exist_ok=True)
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
        databaseLogger.info("Database initialized successfully.")


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


def edit_thread_name(thread_id: int, new_thread_name: str) -> bool:
    """
    Edit the thread name for a given thread ID.
    
    Args:
        thread_id (int): The thread ID to update (primary key)
        new_thread_name (str): The new name for the thread
    
    Returns:
        bool: True if the thread was updated successfully, False if no thread was found
    """
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE threads SET thread_name = ? WHERE thread_id = ?", 
            (new_thread_name, thread_id)
        )
        # Return True if at least one row was affected, False otherwise
        return cursor.rowcount > 0


def delete_thread(thread_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))