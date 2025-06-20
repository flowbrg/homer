DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS threads (
    thread_id INTEGER NOT NULL,
    thread_name TEXT NOT NULL,
    PRIMARY KEY (thread_id)
);
"""
REQUIRED_TABLES = {'threads'}