"""Click-based CLI: `memory-cli` for inspecting and compressing memory stores."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .models import MemoryStore
from .orchestrator import MemoryCompressor
from .persistence import MemoryPersistence


@click.group()
def cli():
    """Agent Memory Compressor CLI.

    Utilities for inspecting and compressing persisted memory stores.
    """


@cli.command("inspect")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def inspect_cmd(path: Path):
    """Print a Rich table summary of a persisted memory store."""
    store = MemoryPersistence().load(path)
    console = Console()

    total_entries = len(store)
    total_tokens = store.token_total()
    turns = [e.turn_number for e in store.entries]
    oldest = min(turns) if turns else 0
    newest = max(turns) if turns else 0
    role_dist = Counter(e.role for e in store.entries)

    table = Table(title=f"Memory Store: {path}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("total_entries", str(total_entries))
    table.add_row("total_tokens", str(total_tokens))
    table.add_row("oldest_turn", str(oldest))
    table.add_row("newest_turn", str(newest))
    for role, count in sorted(role_dist.items()):
        table.add_row(f"role:{role}", str(count))
    console.print(table)

    # Top-5 most important
    scored = [e for e in store.entries if e.importance_score is not None]
    scored.sort(key=lambda e: e.importance_score or 0, reverse=True)
    top_table = Table(title="Top 5 Most Important Entries")
    top_table.add_column("Rank", style="cyan")
    top_table.add_column("Score", style="green")
    top_table.add_column("Turn", style="yellow")
    top_table.add_column("Role")
    top_table.add_column("Preview", overflow="fold")
    for i, e in enumerate(scored[:5], start=1):
        preview = (e.content[:60] + "...") if len(e.content) > 60 else e.content
        top_table.add_row(
            str(i),
            f"{e.importance_score:.3f}" if e.importance_score is not None else "-",
            str(e.turn_number),
            e.role,
            preview,
        )
    console.print(top_table)


@cli.command("compress")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--budget", type=int, required=True, help="Token budget for compression.")
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="Output path for the compressed store.")
def compress_cmd(path: Path, budget: int, output: Path):
    """Compress a persisted store to the given token budget and save output."""
    persistence = MemoryPersistence()
    store = persistence.load(path)
    before = store.token_total()

    compressor = MemoryCompressor(token_budget=budget)
    compressor.compress(store)
    after = store.token_total()

    compressor.save(store, output)

    saved = before - after
    reduction_pct = (saved / before * 100.0) if before > 0 else 0.0
    bar = {
        "before": before,
        "after": after,
        "saved": saved,
        "reduction_pct": round(reduction_pct, 2),
    }
    click.echo(json.dumps(bar))


@cli.command("demo")
def demo_cmd():
    """Run the long-run demo programmatically."""
    # Import with a direct file load so we don't depend on sys.path hacks
    import importlib.util
    demo_path = Path(__file__).resolve().parent.parent / "demos" / "long_run_demo.py"
    spec = importlib.util.spec_from_file_location("long_run_demo", demo_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Could not load demo at {demo_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def main():
    cli()


if __name__ == "__main__":
    main()
