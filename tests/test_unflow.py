"""Tests for unflow."""

import pytest

from unflow.unflow import main


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the main entry point prints a greeting."""
    main()
    captured = capsys.readouterr()
    assert "Hello, World!" in captured.out
