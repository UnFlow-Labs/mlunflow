# create a simple database for graph files
import sqlite3
from pathlib import Path

from unflow.core.constants import (
    DB_PATH,
    EXECUTION_RECORD_TABLE_CREATE_QUERY,
    INSERT_EXECUTION_RECORD_QUERY,
    INSERT_OUTCOME_QUERY,
    INSERT_QUERY,
    OUTCOME_TABLE_CREATE_QUERY,
    SELECT_EXECUTION_RECORD_QUERY,
    SELECT_OUTCOME_BY_GRAPH_QUERY,
    SELECT_OUTCOME_QUERY,
    SELECT_QUERY,
    TABLE_CREATE_QUERY,
)


class DB:
    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        with self.conn:
            self.conn.execute(TABLE_CREATE_QUERY)
            self.conn.execute(OUTCOME_TABLE_CREATE_QUERY)
            self.conn.execute(EXECUTION_RECORD_TABLE_CREATE_QUERY)

    def save_graph(self, name: str, path: str, graph_data: bytes):
        with self.conn:
            self.conn.execute(INSERT_QUERY, (name, path, graph_data))

    def load_graph(self, name: str, path: str) -> bytes | None:
        cursor = self.conn.cursor()
        cursor.execute(SELECT_QUERY, (name, path))
        row = cursor.fetchone()
        return row[0] if row else None

    def save_outcome(self, state_name: str, graph_path: str, outcome_data: bytes):
        with self.conn:
            # get the graph_id for the state_name
            graph_id = self.conn.execute("SELECT id FROM graphs WHERE path = ?", (graph_path,)).fetchone()
            graph_id = graph_id[0] if graph_id else None
            # Now, insert or replace the outcome data
            self.conn.execute(INSERT_OUTCOME_QUERY, (state_name, graph_id, outcome_data))

    def load_outcome(self, state_name: str, graph_path: str | None = None) -> bytes | None:
        cursor = self.conn.cursor()
        if graph_path is not None:
            graph_row = self.conn.execute("SELECT id FROM graphs WHERE path = ?", (graph_path,)).fetchone()
            if graph_row is None:
                return None
            cursor.execute(SELECT_OUTCOME_BY_GRAPH_QUERY, (state_name, graph_row[0]))
        else:
            cursor.execute(SELECT_OUTCOME_QUERY, (state_name,))
        row = cursor.fetchone()
        return row[0] if row else None

    def save_execution_record(
        self, graph_path: str, state_name: str, status: str, start_time: float, end_time: float, error: str | None
    ):
        with self.conn:
            # get the graph_id for the state_name
            graph_id = self.conn.execute("SELECT id FROM graphs WHERE path = ?", (graph_path,)).fetchone()
            graph_id = graph_id[0] if graph_id else None
            self.conn.execute(
                INSERT_EXECUTION_RECORD_QUERY, (graph_id, state_name, status, start_time, end_time, error)
            )

    def load_execution_record(self, graph_path: str, state_name: str) -> tuple[str, float, float, str | None] | None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM graphs WHERE path = ?", (graph_path,))
        graph_id = cursor.fetchone()
        if graph_id is None:
            return None
        graph_id = graph_id[0]
        cursor.execute(SELECT_EXECUTION_RECORD_QUERY, (graph_id, state_name))
        row = cursor.fetchone()
        return row if row else None

    def clear_graph(self, name: str, path: str):
        with self.conn:
            self.conn.execute("DELETE FROM graphs WHERE name = ? AND path = ?", (name, path))

    def close(self):
        self.conn.close()
