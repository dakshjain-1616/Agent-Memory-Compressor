"""Persistence layer for MemoryStore: save/load via JSON."""

import json
from pathlib import Path
from typing import Union, Dict, Any, Optional

from .models import MemoryEntry, MemoryStore


PathLike = Union[str, Path]


class MemoryPersistence:
    """Save and load MemoryStore instances as JSON."""

    @staticmethod
    def _to_path(path: PathLike) -> Path:
        return path if isinstance(path, Path) else Path(path)

    def save(
        self,
        store: MemoryStore,
        path: PathLike,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Serialize a MemoryStore to a JSON file.

        All MemoryEntry fields (including compression_history / turn_number)
        are preserved via pydantic model_dump.
        """
        p = self._to_path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "version": 1,
            "encoding_name": store.encoding_name,
            "entries": [entry.model_dump() for entry in store.entries],
        }
        if extra:
            payload["extra"] = extra

        # Overwrite atomically
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return p

    def load(self, path: PathLike) -> MemoryStore:
        """Load a MemoryStore from JSON on disk.

        Raises FileNotFoundError if the file does not exist.
        """
        p = self._to_path(path)
        if not p.exists():
            raise FileNotFoundError(f"Memory store file not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        encoding_name = payload.get("encoding_name", "cl100k_base")
        store = MemoryStore(encoding_name=encoding_name)

        for raw in payload.get("entries", []):
            store.add_entry(MemoryEntry(**raw))
        return store

    def load_payload(self, path: PathLike) -> Dict[str, Any]:
        """Load the raw payload dict (includes extra/report if saved)."""
        p = self._to_path(path)
        if not p.exists():
            raise FileNotFoundError(f"Memory store file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
