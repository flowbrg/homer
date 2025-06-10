DB_SCHEMA = """

CREATE TABLE IF NOT EXISTS threads (
    thread_id INTEGER NOT NULL,
    thread_name TEXT NOT NULL,
    PRIMARY KEY (thread_id)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id INTEGER NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint TEXT,
    metadata TEXT,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id),
    FOREIGN KEY (thread_id)
        REFERENCES threads (thread_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS writes (
    thread_id INTEGER NOT NULL,
    checkpoint_ns TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT,
    type TEXT,
    value TEXT,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx),
    FOREIGN KEY (thread_id, checkpoint_ns, checkpoint_id)
        REFERENCES checkpoints (thread_id, checkpoint_ns, checkpoint_id)
        ON DELETE CASCADE
);
"""
REQUIRED_TABLES = {'threads', 'checkpoints','writes'}