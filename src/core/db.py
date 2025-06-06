import sqlite3
import os

from pathlib import Path
from datetime import datetime

from src.schemas.db import DB_SCHEMA, REQUIRED_TABLES
from src.resources.utils import get_connection


class DatabaseWrapper:
    def __init__(self):
        self.db_path = Path(os.getenv("DB_PATH"))
        self.conn = get_connection()
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

    def get_all_chats(self):
        with get_connection() as conn:
            cursor = conn.execute("SELECT chat_id, chat_name FROM chats ORDER BY chat_id DESC")
            return cursor.fetchall()

    def get_messages_for_chat(self,chat_id: int):
        with get_connection() as conn:
            cursor = conn.execute("""
                SELECT message_type, message FROM messages
                WHERE chat_id = ? ORDER BY timestamp ASC
            """, (chat_id,))
            return cursor.fetchall()

    def create_chat(self,chat_name: str) -> int:
        with get_connection() as conn:
            cursor = conn.execute("INSERT INTO chats (chat_name) VALUES (?)", (chat_name,))
            if cursor.lastrowid is None:
                raise RuntimeError("Failed to retrieve lastrowid after inserting chat.")
            return cursor.lastrowid

    def add_message(self,chat_id: int, msg_type: str, message: str):
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO messages (chat_id, message_type, message, timestamp)
                VALUES (?, ?, ?, ?)
            """, (chat_id, msg_type, message, datetime.now()))

    def delete_chat(self,chat_id: int):
        with get_connection() as conn:
            conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
