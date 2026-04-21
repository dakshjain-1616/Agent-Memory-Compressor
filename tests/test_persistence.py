"""Tests for MemoryPersistence and MemoryCompressor save/load."""

import json
from pathlib import Path

import pytest

from agent_memory_compressor.models import MemoryEntry, MemoryStore
from agent_memory_compressor.orchestrator import MemoryCompressor
from agent_memory_compressor.persistence import MemoryPersistence


def _make_store() -> MemoryStore:
    store = MemoryStore()
    store.add_entry(MemoryEntry(content="hello world", role="user", turn_number=1))
    entry = MemoryEntry(
        content="compressed stuff",
        role="compressed",
        turn_number=2,
        metadata={"source": "summarize"},
        importance_score=0.75,
        compression_history=[{"strategy": "SUMMARIZE", "original_tokens": 42}],
    )
    store.add_entry(entry)
    store.add_entry(MemoryEntry(content="latest reply", role="assistant", turn_number=3))
    return store


def test_save_load_roundtrip_preserves_all_fields(tmp_path: Path):
    store = _make_store()
    target = tmp_path / "store.json"
    MemoryPersistence().save(store, target)

    loaded = MemoryPersistence().load(target)

    assert len(loaded.entries) == len(store.entries)
    for orig, new in zip(store.entries, loaded.entries):
        assert orig.id == new.id
        assert orig.content == new.content
        assert orig.role == new.role
        assert orig.turn_number == new.turn_number
        assert orig.metadata == new.metadata
        assert orig.importance_score == new.importance_score
        assert orig.compression_history == new.compression_history


def test_overwrites_existing_file(tmp_path: Path):
    target = tmp_path / "store.json"
    persistence = MemoryPersistence()
    persistence.save(_make_store(), target)
    first_size = target.stat().st_size

    small_store = MemoryStore()
    small_store.add_entry(MemoryEntry(content="x", role="user", turn_number=1))
    persistence.save(small_store, target)
    second_size = target.stat().st_size
    assert second_size < first_size

    reloaded = persistence.load(target)
    assert len(reloaded.entries) == 1
    assert reloaded.entries[0].content == "x"


def test_load_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        MemoryPersistence().load(missing)


def test_path_str_and_path_are_equivalent(tmp_path: Path):
    store = _make_store()
    target = tmp_path / "store.json"
    persistence = MemoryPersistence()
    persistence.save(store, str(target))

    loaded_str = persistence.load(str(target))
    loaded_path = persistence.load(target)
    assert len(loaded_str.entries) == len(loaded_path.entries) == len(store.entries)


def test_empty_store_roundtrip(tmp_path: Path):
    target = tmp_path / "empty.json"
    MemoryPersistence().save(MemoryStore(), target)
    loaded = MemoryPersistence().load(target)
    assert len(loaded.entries) == 0
    assert loaded.token_total() == 0


def test_compressor_save_load_with_report(tmp_path: Path):
    store = _make_store()
    # Add bulk content so compression triggers
    for i in range(10):
        store.add_entry(MemoryEntry(
            content="filler content " * 30,
            role="user",
            turn_number=10 + i,
        ))

    compressor = MemoryCompressor(token_budget=50, protected_recent=2)
    compressor.compress(store)
    assert compressor.last_report is not None

    target = tmp_path / "compressor.json"
    compressor.save(store, target)
    assert target.exists()

    payload = json.loads(target.read_text())
    assert "extra" in payload
    assert "last_report" in payload["extra"]

    new_compressor = MemoryCompressor(token_budget=50)
    restored = new_compressor.load(target)
    assert len(restored.entries) == len(store.entries)
    assert new_compressor.last_report is not None
    assert new_compressor.last_report.initial_tokens == compressor.last_report.initial_tokens


def test_save_creates_parent_directory(tmp_path: Path):
    nested = tmp_path / "nested" / "dir" / "store.json"
    MemoryPersistence().save(_make_store(), nested)
    assert nested.exists()
