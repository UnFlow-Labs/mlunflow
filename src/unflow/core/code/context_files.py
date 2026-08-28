import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_watch: Callable[..., Any] | None = None
try:
    from watchfiles import watch as _watch_impl  # type: ignore[import-not-found]

    _watch = _watch_impl
except ModuleNotFoundError:
    pass

ContextFiles = Path | list[Path] | None


def _to_key(file_path: Path) -> str:
    return file_path.name


def _safe_read_text(file_path: Path) -> str | None:
    if not file_path.exists() or not file_path.is_file():
        return None
    return file_path.read_text()


def load_context_files(context_files: ContextFiles) -> list[Path]:
    if context_files is None:
        return []
    if isinstance(context_files, Path):
        return [context_files.resolve()]
    if isinstance(context_files, list):
        return [path.resolve() for path in context_files]
    raise ValueError(f"Invalid context_files: {context_files}")


def build_hashes(context_data: 'ContextData') -> dict[str, str]:
    """
    Build a dictionary of hashes for the context files.
    """
    hashes = {}
    for file_name, file_content in context_data.data.items():
        file_hash = hashlib.sha256(file_content.encode("utf-8")).hexdigest()
        hashes[file_name] = file_hash
    context_data.hashes = hashes
    return hashes


@dataclass
class ContextData:
    """
    A class to hold the context data read from context files.
    """
    data: dict[str, str]
    context_files: list[Path]
    hashes: dict[str, str]
    base_path: Path | None = None
    def __dict__(self):
        return {
            "data": self.data,
            "context_files": [str(path) for path in self.context_files],
            "hashes": self.hashes,
            "base_path": str(self.base_path) if self.base_path else None,
        }


def read_context_files(context_files: ContextFiles, strict: bool = True) -> ContextData:
    """
    Read the content of context files and return a ContextData instance.
    """
    context_files_list = load_context_files(context_files)
    context_data = ContextData(data={}, base_path=None, context_files=context_files_list, hashes={})
    for file_path in context_files_list:
        content = _safe_read_text(file_path)
        if content is None and strict:
            raise FileNotFoundError(f"Context file not found: {file_path}")
        if content is not None:
            context_data.data[_to_key(file_path)] = content
    build_hashes(context_data)
    return context_data






class ContextFileTracker:
    def __init__(self, context_files: ContextFiles):
        self._lock = threading.Lock()
        self._context_data = read_context_files(context_files, strict=False)
        self._watched_paths = set(self._context_data.context_files)
        self._watch_directories = sorted({path.parent for path in self._watched_paths})
        self._stop_event = threading.Event()
        self._watch_thread: threading.Thread | None = None

    def start(self, debounce: int = 300):
        if self._watch_thread is not None and self._watch_thread.is_alive():
            return

        def _run_watch_loop():
            if _watch is None:
                return

            for changes in _watch(*self._watch_directories, stop_event=self._stop_event, debounce=debounce):
                changed_paths = {Path(path).resolve() for _, path in changes}
                relevant_paths = changed_paths.intersection(self._watched_paths)
                if relevant_paths:
                    self.update(relevant_paths)

        if not self._watch_directories:
            return

        self._stop_event.clear()
        self._watch_thread = threading.Thread(target=_run_watch_loop, daemon=True)
        self._watch_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._watch_thread is not None:
            self._watch_thread.join(timeout=1)

    def update(self, changed_paths: set[Path] | None = None):
        with self._lock:
            paths_to_update = changed_paths if changed_paths is not None else self._watched_paths
            for path in paths_to_update:
                key = _to_key(path)
                content = _safe_read_text(path)
                if content is None:
                    self._context_data.data.pop(key, None)
                    self._context_data.hashes.pop(key, None)
                    continue
                self._context_data.data[key] = content
                self._context_data.hashes[key] = hashlib.sha256(content.encode("utf-8")).hexdigest()

    def snapshot(self) -> ContextData:
        with self._lock:
            return ContextData(
                data=dict(self._context_data.data),
                context_files=list(self._context_data.context_files),
                hashes=dict(self._context_data.hashes),
                base_path=self._context_data.base_path,
            )