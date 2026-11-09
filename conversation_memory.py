"""SQLite-backed bounded conversation history for the RAG API."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str


class ConversationMemoryError(RuntimeError):
    """Raised when conversation history cannot be read or written."""


class SQLiteConversationMemory:
    """Persist recent chat turns across API requests and process restarts.

    Each method opens its own SQLite connection, making the store suitable for
    multiple worker threads and multiple local API workers sharing one file.
    For horizontally distributed deployments, use a shared database/service.
    """

    def __init__(self, database_path: str | Path = "./conversation_memory.sqlite3", *, max_messages: int = 12) -> None:
        if max_messages < 2:
            raise ValueError("max_messages must be at least 2")
        self.database_path = Path(database_path).expanduser()
        self.max_messages = max_messages
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS conversation_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )"""
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversation_id_id "
                    "ON conversation_messages(conversation_id, id)"
                )
        except sqlite3.Error as exc:
            raise ConversationMemoryError(f"Could not initialize conversation storage: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path), timeout=15)
        connection.row_factory = sqlite3.Row
        return connection

    def get_history(self, conversation_id: str) -> list[ChatMessage]:
        """Return up to ``max_messages`` recent messages in chronological order."""
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """SELECT role, content FROM (
                           SELECT id, role, content FROM conversation_messages
                           WHERE conversation_id = ? ORDER BY id DESC LIMIT ?
                       ) ORDER BY id ASC""",
                    (conversation_id, self.max_messages),
                ).fetchall()
            return [ChatMessage(role=str(row["role"]), content=str(row["content"])) for row in rows]
        except sqlite3.Error as exc:
            raise ConversationMemoryError(f"Could not read conversation history: {exc}") from exc

    def add_exchange(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        """Atomically append a successful user/assistant exchange and prune old messages."""
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany(
                    "INSERT INTO conversation_messages (conversation_id, role, content) VALUES (?, ?, ?)",
                    [
                        (conversation_id, "user", user_message),
                        (conversation_id, "assistant", assistant_message),
                    ],
                )
                connection.execute(
                    """DELETE FROM conversation_messages
                       WHERE conversation_id = ? AND id NOT IN (
                           SELECT id FROM conversation_messages
                           WHERE conversation_id = ? ORDER BY id DESC LIMIT ?
                       )""",
                    (conversation_id, conversation_id, self.max_messages),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise ConversationMemoryError(f"Could not save conversation exchange: {exc}") from exc
