from pathlib import Path

from unflow.core.code.context_files import (
    ContextData,
    ContextFileTracker,
    build_context_change_prompt,
    build_hashes,
    format_context_changes_for_llm,
    get_context_changes,
    read_context_files,
)


def test_read_context_files_builds_hashes(tmp_path: Path):
    file_path = tmp_path / "context.txt"
    file_path.write_text("hello")

    context_data = read_context_files([file_path])

    assert context_data.data[file_path.name] == "hello"
    assert file_path.name in context_data.hashes


def test_get_context_changes_returns_content_diff():
    old_context = ContextData(
        data={"context.txt": "line1\nline2\n"},
        context_files=[],
        hashes={"context.txt": "old"},
    )
    new_context = ContextData(
        data={"context.txt": "line1\nline3\n"},
        context_files=[],
        hashes={"context.txt": "new"},
    )

    changes = get_context_changes(old_context, new_context)

    assert "context.txt" in changes
    assert changes["context.txt"]["old_hash"] == "old"
    assert changes["context.txt"]["new_hash"] == "new"
    assert changes["context.txt"]["content_changes"]


def test_build_hashes_uses_context_data_content():
    context_data = ContextData(data={"a.txt": "abc"}, context_files=[], hashes={})

    hashes = build_hashes(context_data)

    assert hashes == context_data.hashes
    assert "a.txt" in hashes


def test_context_file_tracker_update_refreshes_snapshot(tmp_path: Path):
    file_path = tmp_path / "ctx.txt"
    file_path.write_text("v1")

    tracker = ContextFileTracker([file_path])
    snap1 = tracker.snapshot()
    assert snap1.data[file_path.name] == "v1"

    file_path.write_text("v2")
    tracker.update({file_path.resolve()})
    snap2 = tracker.snapshot()

    assert snap2.data[file_path.name] == "v2"
    assert snap1.hashes[file_path.name] != snap2.hashes[file_path.name]


def test_format_context_changes_for_llm_includes_summary_and_patch():
    changes = {
        "lenet.py": {
            "old_hash": "old-h",
            "new_hash": "new-h",
            "content_changes": {
                1: "--- previous",
                2: "+++ current",
                3: "@@ -1,2 +1,2 @@",
                4: "-x = 1",
                5: "+x = 2",
            },
        }
    }

    formatted = format_context_changes_for_llm(changes)

    assert len(formatted) == 1
    item = formatted[0]
    assert item["path"] == "lenet.py"
    assert item["change_type"] == "modified"
    assert item["added_lines"] == 1
    assert item["removed_lines"] == 1
    assert item["patch"]
    assert item["hunk_headers"] == ["@@ -1,2 +1,2 @@"]


def test_build_context_change_prompt_returns_json_string():
    changes = {
        "lenet.py": {
            "old_hash": "a",
            "new_hash": "b",
            "content_changes": {1: "--- previous", 2: "+++ current", 3: "@@ -1 +1 @@", 4: "-a", 5: "+b"},
        }
    }

    prompt = build_context_change_prompt(changes)

    assert '"task"' in prompt
    assert '"changes"' in prompt
    assert "lenet.py" in prompt
