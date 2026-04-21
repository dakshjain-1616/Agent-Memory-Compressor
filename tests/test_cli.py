"""Tests for the memory-cli Click CLI."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_memory_compressor.cli import cli
from agent_memory_compressor.models import MemoryEntry, MemoryStore
from agent_memory_compressor.persistence import MemoryPersistence


def _write_big_store(path: Path) -> MemoryStore:
    store = MemoryStore()
    store.add_entry(MemoryEntry(content="system boot", role="system", turn_number=0))
    for i in range(1, 25):
        role = "user" if i % 2 == 1 else "assistant"
        content = ("user question " if role == "user" else "detailed assistant answer ") * 15
        store.add_entry(MemoryEntry(
            content=content + f"#{i}",
            role=role,
            turn_number=i,
            importance_score=(i % 5) / 5.0,
        ))
    MemoryPersistence().save(store, path)
    return store


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "inspect" in result.output
    assert "compress" in result.output
    assert "demo" in result.output


def test_inspect_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", "--help"])
    assert result.exit_code == 0


def test_compress_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["compress", "--help"])
    assert result.exit_code == 0
    assert "--budget" in result.output
    assert "--output" in result.output


def test_demo_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "--help"])
    assert result.exit_code == 0


def test_inspect_shows_counts(tmp_path: Path):
    store_path = tmp_path / "store.json"
    store = _write_big_store(store_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", str(store_path)])
    assert result.exit_code == 0, result.output
    assert "total_entries" in result.output
    assert str(len(store.entries)) in result.output
    assert "total_tokens" in result.output


def test_compress_produces_output_file(tmp_path: Path):
    store_path = tmp_path / "store.json"
    output_path = tmp_path / "compressed.json"
    _write_big_store(store_path)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "compress", str(store_path),
        "--budget", "80",
        "--output", str(output_path),
    ])
    assert result.exit_code == 0, result.output
    assert output_path.exists()
    payload = json.loads(result.output.strip().splitlines()[-1])
    for key in ("before", "after", "saved", "reduction_pct"):
        assert key in payload


def test_compress_reduces_tokens(tmp_path: Path):
    store_path = tmp_path / "store.json"
    output_path = tmp_path / "compressed.json"
    _write_big_store(store_path)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "compress", str(store_path),
        "--budget", "60",
        "--output", str(output_path),
    ])
    assert result.exit_code == 0, result.output
    bar = json.loads(result.output.strip().splitlines()[-1])
    assert bar["before"] > bar["after"]
    assert bar["saved"] >= 0
    assert bar["reduction_pct"] >= 0


def test_demo_runs():
    runner = CliRunner()
    result = runner.invoke(cli, ["demo"])
    # The demo prints output and exits 0
    assert result.exit_code == 0, result.output
    assert "DEMO COMPLETE" in result.output or "Compression" in result.output
