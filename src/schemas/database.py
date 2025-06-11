DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    thread_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    PRIMARY KEY (thread_id)
);
"""
REQUIRED_TABLES = {'chats'}