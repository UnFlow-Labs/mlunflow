import os
import pathlib

APP_ROOT = pathlib.Path(__file__).parent.parent
HOME_PATH = pathlib.Path.home()
TMP_PATH = HOME_PATH / ".tmp" / ".unflow"
DB_PATH = TMP_PATH / ".graph.db"

PICKLE_PATH = TMP_PATH / "pickles"
os.makedirs(PICKLE_PATH, exist_ok=True)
"""
graphs stores only the graph structure, not the outcomes. Outcomes are stored in a separate table.
execution_records table stores the execution records for each state
"""
TABLE_NAME = "graphs"
TABLE_CREATE_QUERY = f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    path TEXT UNIQUE,
                    graph BLOB
                )
            """

INSERT_QUERY = f"""
                INSERT INTO {TABLE_NAME} (name, path, graph)
                VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    name = excluded.name,
                    graph = excluded.graph
            """  # noqa: S608
SELECT_QUERY = f"""
                SELECT graph FROM {TABLE_NAME} WHERE name =  ? AND path = ?
                """  # noqa: S608

#####Outcome table and queries#####

OUTCOME_TABLE_NAME = "outcomes"
OUTCOME_TABLE_CREATE_QUERY = f"""
                CREATE TABLE IF NOT EXISTS {OUTCOME_TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_id TEXT,
                    graph_id INTEGER REFERENCES {TABLE_NAME}(id),
                    outcome BLOB
                )
            """
INSERT_OUTCOME_QUERY = f"""
                INSERT OR REPLACE INTO {OUTCOME_TABLE_NAME} (state_id, graph_id, outcome) VALUES (?, ?, ?)
            """  # noqa: S608
SELECT_OUTCOME_QUERY = f"""
                SELECT outcome FROM {OUTCOME_TABLE_NAME} WHERE state_id = ?
                """  # noqa: S608

SELECT_OUTCOME_BY_GRAPH_QUERY = f"""
                SELECT outcome FROM {OUTCOME_TABLE_NAME}
                WHERE state_id = ? AND graph_id = ?
                ORDER BY id DESC
                LIMIT 1
                """  # noqa: S608

####Execution record table and queries#####
EXECUTION_RECORD_TABLE_NAME = "execution_records"
EXECUTION_RECORD_TABLE_CREATE_QUERY = f"""
                CREATE TABLE IF NOT EXISTS {EXECUTION_RECORD_TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    graph_id INTEGER REFERENCES {TABLE_NAME}(id),
                    state_name TEXT,
                    status TEXT,
                    start_time REAL,
                    end_time REAL,
                    error TEXT
                )
            """
INSERT_EXECUTION_RECORD_QUERY = f"""
                INSERT OR REPLACE INTO {EXECUTION_RECORD_TABLE_NAME} 
                (graph_id, state_name, status, start_time, end_time, error) 
                VALUES (?, ?, ?, ?, ?, ?)
            """  # noqa: S608
SELECT_EXECUTION_RECORD_QUERY = f"""
                SELECT state_name, status, start_time, end_time, error FROM {EXECUTION_RECORD_TABLE_NAME} 
                WHERE graph_id = ?
                """  # noqa: S608
