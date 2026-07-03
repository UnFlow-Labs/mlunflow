import os
import pathlib

APP_ROOT = pathlib.Path(__file__).parent.parent
HOME_PATH = pathlib.Path.home()
TMP_PATH = HOME_PATH / ".tmp" / ".unflow"
DB_PATH = TMP_PATH / ".graph.db"

PICKLE_PATH = TMP_PATH / "pickles"
os.makedirs(PICKLE_PATH, exist_ok=True)

TABLE_NAME = "graphs"
TABLE_CREATE_QUERY = f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    graph BLOB
                )
            """
INSERT_QUERY = f"""
                INSERT OR REPLACE INTO {TABLE_NAME} (name, graph) VALUES (?, ?)
            """  # noqa: S608
SELECT_QUERY = f"""
                SELECT graph FROM {TABLE_NAME} WHERE name = ?
                """  # noqa: S608
