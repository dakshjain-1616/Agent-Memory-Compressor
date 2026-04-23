# Agent Memory Compressor

> Autonomously built using [NEO](https://heyneo.com) — Your Autonomous AI Engineering Agent
>
> [![VS Code Extension](https://img.shields.io/badge/VS%20Code-NEO%20Extension-007ACC?logo=visual-studio-code&logoColor=white)](https://marketplace.visualstudio.com/items?itemName=NeoResearchInc.heyneo) [![Cursor Extension](https://img.shields.io/badge/Cursor-NEO%20Extension-000000?logo=cursor&logoColor=white)](https://marketplace.cursorapi.com/items/?itemName=NeoResearchInc.heyneo)

![Infographic](assets/infographic.svg)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

`agent-memory-compressor` is a Python library that implements an intelligent memory
compression pipeline for long-running LLM agents. It combines importance-based
scoring, LLM-driven summarization, a forgetting curve trigger, and a
token-budgeted context builder so agents can run indefinitely without exhausting
their context windows — while preserving the facts and decisions that matter.

## Problem: Context Window Exhaustion

A 10-turn agent session can easily accumulate 20,000+ tokens of raw history,
leaving almost no room for the current task. Naive truncation drops older turns
wholesale — including the decisions and discovered facts the agent needs to
avoid repeating work. Developers need a principled way to *compress* history
rather than *discard* it.

This library addresses the problem along three axes:

- **What to keep.** A multi-signal importance scorer ranks every memory entry.
- **How to shrink.** Three pluggable compression strategies
  (summarize, extract facts, archive) replace low-value entries with compact
  equivalents using any OpenAI-compatible LLM.
- **When to act.** A forgetting curve fires compression automatically when
  either a turn interval or a token threshold is crossed.

## Compression Strategy

Compression is driven by an **ImportanceScorer** that combines three signals:

| Signal           | Purpose                                                                 |
| ---------------- | ----------------------------------------------------------------------- |
| Recency          | Exponential decay — newer entries score higher.                         |
| Type weight      | Decisions and system notes outrank routine turns and tool noise.        |
| Keyword boost    | Entries matching goal-related keywords are promoted.                    |

Given a scored store, the **CompressionEngine** exposes three strategies:

- `summarize(entry)` — asks the LLM for a short summary that preserves all
  decisions and facts.
- `extract_facts(entry)` — asks the LLM for a bullet list of facts and
  decisions, stored as high-importance compressed entries.
- `archive(entry)` — replaces the entry with a minimal reference; the original
  content is retained in the entry's `compression_history` for audit.

The **MemoryCompressor** orchestrates the pipeline: score, pick the
lowest-scoring non-protected entries, apply the least-destructive strategy
first, and iterate until the store is under `token_budget`. Every successful
replacement is verified to actually reduce the token count, so compression
can never make the context larger.

## Forgetting Curve

The **ForgettingCurve** decides *when* to compress. It combines two triggers:

- **Turn-based** — fires once the number of turns since the last compression
  reaches `compression_interval_turns` (default 10).
- **Token-based** — fires once `MemoryStore.token_total()` exceeds
  `compression_threshold_tokens` (default 6000), with hysteresis to prevent
  thrashing.

`should_compress(store)` returns `True` as soon as either condition is met.
`get_compression_priority(store)` returns entries sorted by importance, so the
orchestrator always attacks the least-valuable history first.

## Installation

```bash
pip install -e .
# optional, for live LLM calls
pip install openai
```

The package depends on `pydantic`, `tiktoken` (for `cl100k_base` token counts),
`click`, and `rich`.

## Usage Example

```python
from agent_memory_compressor import MemoryEntry, MemoryStore, MemoryCompressor
from agent_memory_compressor.triggers import ForgettingCurve
from agent_memory_compressor.context import ContextBuilder, ContextConfig
from agent_memory_compressor.strategies import LLMClient, CompressionEngine

store = MemoryStore()
for turn, (role, content) in enumerate(conversation, start=1):
    store.add_entry(MemoryEntry(content=content, role=role, turn_number=turn))

llm = LLMClient(api_key="sk-...", model="gpt-4o-mini")
compressor = MemoryCompressor(
    token_budget=4000,
    protected_recent=3,
    engine=CompressionEngine(llm_client=llm),
)

curve = ForgettingCurve(compression_interval_turns=10,
                       compression_threshold_tokens=6000)

if curve.should_compress(store):
    report = compressor.compress(store)
    curve.mark_compressed(store)
    print(f"Saved {report.tokens_saved} tokens "
          f"({report.compression_ratio:.0%} reduction)")

context = ContextBuilder(ContextConfig(max_tokens=4000)).build_context(
    store, system_message="You are a helpful assistant."
)
```

Without an API key, `LLMClient` falls back to a deterministic short stub so
pipelines remain runnable in tests and offline demos. A full end-to-end demo
lives at [`demos/long_run_demo.py`](demos/long_run_demo.py).

## API Reference

| Class                 | Module         | Responsibility                                                             |
| --------------------- | -------------- | -------------------------------------------------------------------------- |
| `MemoryEntry`         | `models`       | Pydantic model for a single turn / note.                                   |
| `MemoryStore`         | `models`       | Ordered collection of entries with `tiktoken` token counting.              |
| `ImportanceScorer`    | `scoring`      | Scores entries via recency, type weight, keyword boost.                    |
| `CompressionEngine`   | `strategies`   | `summarize`, `extract_facts`, `archive` strategies.                        |
| `LLMClient`           | `strategies`   | OpenAI-compatible client with offline fallback and mock-response support.  |
| `MemoryCompressor`    | `orchestrator` | Iterative budget-driven compression loop, emits `CompressionReport`.       |
| `ForgettingCurve`     | `triggers`     | Turn- and token-based compression triggers.                                |
| `ContextBuilder`      | `context`      | Assembles a `max_tokens`-bounded context, protected recent turns first.    |
| `SessionAdapter`      | `adapters`     | Bridges to `agent-session-manager` sessions.                               |
| `MemoryPersistence`   | `persistence`  | JSON save/load of stores and reports.                                      |

A `memory-cli` entrypoint (`click`-based) is installed for quick inspection,
compression, and demo runs.

## Integration with the Session Manager

The `adapters` module wires the compressor directly into the
[Stateful Agent Session Manager](https://github.com/dakshjain-1616/agent-session-manager):

```python
from agent_memory_compressor.adapters import compress_session

compressed_messages, report = compress_session(
    session,              # anything exposing get_messages() / get_metadata()
    token_budget=4000,
    protected_recent=3,
)
```

`SessionAdapter.session_to_store` projects session messages into a
`MemoryStore`, `compressor.compress(...)` runs the pipeline, and
`store_to_session` projects the compressed entries back into the session's
message format — preserving original roles and retaining the compression
history on each compacted entry.

## License

MIT.
