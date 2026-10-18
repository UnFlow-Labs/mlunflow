
import hashlib
from pathlib import Path
from typing import Any
from .context_files import ContextData, ContextFiles, read_context_files, _to_key, _safe_read_text
from .diff_code import get_procedure_changes





def get_context_changes(old_context: ContextData, new_context: ContextData) -> dict[str, dict[str, str | dict[int, str] | None]]:
    """
    Compare two ContextData instances and return a dictionary of changes per file.
    """
    changes: dict[str, dict[str, str | dict[int, str] | None]] = {}
    for file_name in set(old_context.hashes.keys()).union(new_context.hashes.keys()):
        old_hash = old_context.hashes.get(file_name)
        new_hash = new_context.hashes.get(file_name)
        if old_hash != new_hash:
            old_content = old_context.data.get(file_name, "")
            new_content = new_context.data.get(file_name, "")
            changes[file_name] = {
                "old_hash": old_hash,
                "new_hash": new_hash,
                "content_changes": get_procedure_changes(old_content, new_content),
            }
    return changes


def _ordered_diff_lines(content_changes: dict[int, str]) -> list[str]:
    return [content_changes[index] for index in sorted(content_changes)]


def _count_patch_lines(patch_lines: list[str]) -> tuple[int, int]:
    added_lines = 0
    removed_lines = 0
    for line in patch_lines:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added_lines += 1
        elif line.startswith("-"):
            removed_lines += 1
    return added_lines, removed_lines


def format_context_changes_for_llm(
    context_changes: dict[str, dict[str, Any]],
    include_patch: bool = True,
    max_patch_chars: int = 4000,
) -> list[dict[str, Any]]:
    """
    Convert raw context change objects into an LLM-friendly representation.
    """
    formatted: list[dict[str, Any]] = []

    for file_path, payload in context_changes.items():
        content_changes = payload.get("content_changes") or {}
        patch_lines = _ordered_diff_lines(content_changes)
        added_lines, removed_lines = _count_patch_lines(patch_lines)
        hunk_headers = [line for line in patch_lines if line.startswith("@@")]
        patch = "\n".join(patch_lines)
        if max_patch_chars > 0 and len(patch) > max_patch_chars:
            patch = f"{patch[:max_patch_chars]}\n... [truncated]"

        old_hash = payload.get("old_hash")
        new_hash = payload.get("new_hash")
        if old_hash is None and new_hash is not None:
            change_type = "added"
        elif old_hash is not None and new_hash is None:
            change_type = "deleted"
        else:
            change_type = "modified"

        summary = (
            f"{change_type} file {file_path}; "
            f"{added_lines} additions, {removed_lines} deletions, {len(hunk_headers)} hunks"
        )

        item = {
            "path": file_path,
            "change_type": change_type,
            "old_hash": old_hash,
            "new_hash": new_hash,
            "summary": summary,
            "added_lines": added_lines,
            "removed_lines": removed_lines,
            "hunk_headers": hunk_headers,
        }
        if include_patch:
            item["patch"] = patch
        formatted.append(item)

    return formatted
def format_procedure_changes_for_llm(
    procedure_changes: dict[int, str],
    include_patch: bool = True,
    max_patch_chars: int = 4000,
) -> dict[str, Any]:
    """
    Convert raw procedure changes into an LLM-friendly representation.
    """
    patch_lines = _ordered_diff_lines(procedure_changes)
    added_lines, removed_lines = _count_patch_lines(patch_lines)
    hunk_headers = [line for line in patch_lines if line.startswith("@@")]
    patch = "\n".join(patch_lines)
    if max_patch_chars > 0 and len(patch) > max_patch_chars:
        patch = f"{patch[:max_patch_chars]}\n... [truncated]"

    summary = (
        f"procedure modified; {added_lines} additions, "
        f"{removed_lines} deletions, {len(hunk_headers)} hunks"
    )

    payload: dict[str, Any] = {
        "change_type": "modified" if procedure_changes else "unchanged",
        "summary": summary,
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "hunk_headers": hunk_headers,
    }
    if include_patch:
        payload["patch"] = patch
    return payload

def get_all_context_str(context_data: ContextData| None = None) -> str:
    """
    Concatenate the content of all context files into a single string.
    """
    text = ""
    if context_data is None or not context_data.data:
        return text
    for file_name, content in context_data.data.items():
        text += f"File: {file_name}\n{content}\n\n"
    return text.strip()