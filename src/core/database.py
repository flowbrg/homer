#DEPRECATED


import sqlite3
import os

from pathlib import Path

from src.schemas.database import DB_SCHEMA, REQUIRED_TABLES
from src.resources.utils import get_connection


class DatabaseWrapper:
    def __init__(self):
        self.db_path = Path(os.getenv("DB_PATH"))
        self._initialize_database()

    def _database_has_required_tables(self,conn: sqlite3.Connection) -> bool:
        """Check if all required tables exist in the database."""
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = {row[0] for row in cursor.fetchall()}
        return REQUIRED_TABLES.issubset(existing_tables)

    def _initialize_database(self):
        """Initializes the SQLite database with the required schema."""
        db_exists = self.db_path.exists()
        initialize = True

        if db_exists:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    if self._database_has_required_tables(conn):
                        print(f"Database {self.db_path} already initialized")
                        initialize = False
                    else:
                        print("Base de données incomplète. Réinitialisation du schéma...")
            except sqlite3.DatabaseError:
                print("Fichier existant invalide. Il ne s'agit pas d'une base SQLite valide.")
        
        if initialize:
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(DB_SCHEMA)
            print("Base de données initialisée avec succès.")



def get_all_threads() -> list[dict]:
    """Retrieve all threads from the database."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT thread_id, thread_name FROM threads")
        return [(row[0], row[1]) for row in cursor.fetchall()]

def new_thread(thread_id: int, thread_name: str):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chats (thread_id, name) VALUES (?, ?)", (thread_id, thread_name)
        )

def delete_thread(thread_id: int):
    with get_connection() as conn:
        conn.execute("DELETE FROM chats WHERE thread_id = ?", (thread_id,))
