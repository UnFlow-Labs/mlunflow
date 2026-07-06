from unflow.core.simpledb import DB


class TestDB:
    PATH = "unflow_test_db"

    def test_save_and_load_graph(self, tmp_db: DB):
        data = b"test graph data"
        tmp_db.save_graph("test_graph", self.PATH, data)
        loaded = tmp_db.load_graph("test_graph", self.PATH)
        assert loaded == data

    def test_load_missing_graph(self, tmp_db: DB):
        loaded = tmp_db.load_graph("nonexistent", self.PATH)
        assert loaded is None

    def test_overwrite_graph(self, tmp_db: DB):
        tmp_db.save_graph("dup", self.PATH, b"original")
        tmp_db.save_graph("dup", self.PATH, b"overwritten")
        loaded = tmp_db.load_graph("dup", self.PATH)
        assert loaded == b"overwritten"

    def test_clear_graph(self, tmp_db: DB):
        tmp_db.save_graph("to_clear", self.PATH, b"data")
        tmp_db.clear_graph("to_clear", self.PATH)
        loaded = tmp_db.load_graph("to_clear", self.PATH)
        assert loaded is None

    def test_clear_nonexistent_graph(self, tmp_db: DB):
        tmp_db.clear_graph("does_not_exist", self.PATH)  # should not raise

    def test_multiple_graphs_independent(self, tmp_db: DB):
        path1 = "unflow_test_db/g1"
        path2 = "unflow_test_db/g2"
        tmp_db.save_graph("g1", path1, b"data1")
        tmp_db.save_graph("g2", path2, b"data2")
        assert tmp_db.load_graph("g1", path1) == b"data1"
        assert tmp_db.load_graph("g2", path2) == b"data2"

    def test_create_table_on_init(self, tmp_path):
        db = DB(tmp_path / "new.db")
        try:
            db.save_graph("test", self.PATH, b"data")
            assert db.load_graph("test", self.PATH) == b"data"
        finally:
            db.close()

    def test_close(self, tmp_path):
        db = DB(tmp_path / "close_test.db")
        db.save_graph("x", self.PATH, b"y")
        db.close()
        # should not crash
