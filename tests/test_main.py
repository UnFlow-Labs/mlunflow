from unflow.main import greet


def test_greet():
    assert greet("Cambridge") == "Hello, Cambridge!"
