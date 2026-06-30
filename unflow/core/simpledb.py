# create a simple database for graph files
import sqlite3
from pathlib import Path

from unflow.core.constants import DB_PATH, INSERT_QUERY, SELECT_QUERY, TABLE_CREATE_QUERY


class DB:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute(TABLE_CREATE_QUERY)

    def save_graph(self, name: str, graph_data: bytes):
        with self.conn:
            self.conn.execute(INSERT_QUERY, (name, graph_data))

    def load_graph(self, name: str) -> bytes | None:
        cursor = self.conn.cursor()
        cursor.execute(SELECT_QUERY, (name,))
        row = cursor.fetchone()
        return row[0] if row else None

    def close(self):
        self.conn.close()
