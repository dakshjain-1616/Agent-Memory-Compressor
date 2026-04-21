"""MemoryCompressor orchestrator: iterative compression loop based on token budget."""

import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field

from .models import MemoryEntry, MemoryStore
from .scoring import ImportanceScorer, ScoringConfig
from .strategies import CompressionEngine, CompressionStrategy


@dataclass
class CompressionReport:
    """Report of a compression operation."""
    initial_tokens: int
    final_tokens: int
    tokens_saved: int
    entries_compressed: int
    entries_archived: int
    entries_summarized: int
    entries_extracted: int
    compression_ratio: float
    iterations: int
    duration_seconds: float
    protected_entries: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "initial_tokens": self.initial_tokens,
            "final_tokens": self.final_tokens,
            "tokens_saved": self.tokens_saved,
            "entries_compressed": self.entries_compressed,
            "entries_archived": self.entries_archived,
            "entries_summarized": self.entries_summarized,
            "entries_extracted": self.entries_extracted,
            "compression_ratio": self.compression_ratio,
            "iterations": self.iterations,
            "duration_seconds": self.duration_seconds,
            "protected_entries": self.protected_entries,
        }


class MemoryCompressor:
    """Orchestrates memory compression based on token budget constraints."""
    
    def __init__(
        self,
        token_budget: int = 4000,
        protected_recent: int = 3,
        scorer: Optional[ImportanceScorer] = None,
        engine: Optional[CompressionEngine] = None,
        strategy_priority: Optional[List[CompressionStrategy]] = None,
        min_entries_to_compress: int = 1,
        max_iterations: int = 100
    ):
        """Initialize the memory compressor.
        
        Args:
            token_budget: Maximum tokens allowed in the store.
            protected_recent: Number of most recent entries to protect from compression.
            scorer: ImportanceScorer instance. Uses default if not provided.
            engine: CompressionEngine instance. Uses default if not provided.
            strategy_priority: Order of strategies to try (default: SUMMARIZE, EXTRACT_FACTS, ARCHIVE).
            min_entries_to_compress: Minimum entries to compress per iteration.
            max_iterations: Maximum compression iterations to prevent infinite loops.
        """
        self.token_budget = token_budget
        self.protected_recent = protected_recent
        self.scorer = scorer or ImportanceScorer()
        self.engine = engine or CompressionEngine()
        self.strategy_priority = strategy_priority or [
            CompressionStrategy.SUMMARIZE,
            CompressionStrategy.EXTRACT_FACTS,
            CompressionStrategy.ARCHIVE
        ]
        self.min_entries_to_compress = min_entries_to_compress
        self.max_iterations = max_iterations
        self.last_report: Optional["CompressionReport"] = None
    
    def _get_compressible_entries(self, store: MemoryStore) -> List[MemoryEntry]:
        """Get entries that can be compressed (excluding protected recent entries).
        
        Args:
            store: The memory store.
            
        Returns:
            List of compressible entries.
        """
        if len(store.entries) <= self.protected_recent:
            return []
        
        # Protect the N most recent entries
        return store.entries[:-self.protected_recent] if self.protected_recent > 0 else list(store.entries)
    
    def _select_entries_to_compress(
        self,
        store: MemoryStore,
        target_reduction: int
    ) -> List[MemoryEntry]:
        """Select entries to compress based on importance scores.
        
        Args:
            store: The memory store.
            target_reduction: Target token reduction.
            
        Returns:
            List of entries to compress (least important first).
        """
        compressible = self._get_compressible_entries(store)
        if not compressible:
            return []
        
        # Score compressible entries
        self.scorer.update_entry_scores(store)
        
        # Sort by importance (lowest first - least important get compressed first)
        scored = [(entry, entry.importance_score or 0) for entry in compressible]
        scored.sort(key=lambda x: x[1])
        
        # Select entries until we meet the target reduction
        selected = []
        current_reduction = 0
        
        for entry, score in scored:
            if current_reduction >= target_reduction:
                break
            selected.append(entry)
            # Estimate tokens saved (rough approximation)
            current_reduction += len(entry.content.split()) // 2
        
        # Ensure minimum entries
        while len(selected) < self.min_entries_to_compress and scored:
            entry = scored[len(selected)][0] if len(selected) < len(scored) else None
            if entry and entry not in selected:
                selected.append(entry)
            else:
                break
        
        return selected
    
    def _choose_strategy(self, entry: MemoryEntry, iteration: int) -> CompressionStrategy:
        """Choose compression strategy based on entry and iteration.
        
        Args:
            entry: Entry to compress.
            iteration: Current compression iteration.
            
        Returns:
            Compression strategy to use.
        """
        # Use more aggressive strategies in later iterations
        strategy_index = min(iteration, len(self.strategy_priority) - 1)
        return self.strategy_priority[strategy_index]
    
    def compress(
        self,
        store: MemoryStore,
        target_budget: Optional[int] = None
    ) -> CompressionReport:
        """Run iterative compression until token budget is met.
        
        Args:
            store: The memory store to compress.
            target_budget: Override the default token budget.
            
        Returns:
            CompressionReport with compression statistics.
        """
        budget = target_budget or self.token_budget
        start_time = time.time()
        
        initial_tokens = store.token_total()
        initial_count = len(store)
        
        # Guard clause: if already within budget, return no-op report
        if initial_tokens <= budget:
            self.last_report = CompressionReport(
                initial_tokens=initial_tokens,
                final_tokens=initial_tokens,
                tokens_saved=0,
                entries_compressed=0,
                entries_archived=0,
                entries_summarized=0,
                entries_extracted=0,
                compression_ratio=0.0,
                iterations=0,
                duration_seconds=0.0,
                protected_entries=min(self.protected_recent, initial_count)
            )
            return self.last_report

        # Track statistics
        entries_archived = 0
        entries_summarized = 0
        entries_extracted = 0
        iterations = 0
        
        # Iterative compression loop
        while store.token_total() > budget and iterations < self.max_iterations:
            iterations += 1
            
            # Calculate target reduction
            current_tokens = store.token_total()
            target_reduction = current_tokens - budget
            
            # Select entries to compress
            to_compress = self._select_entries_to_store(store, target_reduction)
            
            if not to_compress:
                # No more compressible entries
                break
            
            # Compress selected entries
            for entry in to_compress:
                strategy = self._choose_strategy(entry, iterations - 1)
                result = self.engine.compress(entry, strategy)
                
                if result.success and result.compressed_entry:
                    # Check if compression actually saves tokens
                    original_tokens = len(entry.content.split())
                    compressed_tokens = len(result.compressed_entry.content.split())

                    # Only replace if we actually save tokens
                    if compressed_tokens >= original_tokens:
                        continue
                    for i, e in enumerate(store.entries):
                        if e.id == entry.id:
                            store.entries[i] = result.compressed_entry
                            break
                    
                    # Update statistics
                    if strategy == CompressionStrategy.ARCHIVE:
                        entries_archived += 1
                    elif strategy == CompressionStrategy.SUMMARIZE:
                        entries_summarized += 1
                    elif strategy == CompressionStrategy.EXTRACT_FACTS:
                        entries_extracted += 1
            
            # Check if we're making progress
            if store.token_total() >= current_tokens:
                # No progress made, try more aggressive strategy
                if iterations >= len(self.strategy_priority):
                    break
        
        final_tokens = store.token_total()
        duration = time.time() - start_time
        
        # Calculate compression ratio
        compression_ratio = (initial_tokens - final_tokens) / initial_tokens if initial_tokens > 0 else 0
        
        report = CompressionReport(
            initial_tokens=initial_tokens,
            final_tokens=final_tokens,
            tokens_saved=initial_tokens - final_tokens,
            entries_compressed=entries_archived + entries_summarized + entries_extracted,
            entries_archived=entries_archived,
            entries_summarized=entries_summarized,
            entries_extracted=entries_extracted,
            compression_ratio=compression_ratio,
            iterations=iterations,
            duration_seconds=duration,
            protected_entries=min(self.protected_recent, initial_count)
        )
        self.last_report = report
        return report

    def _select_entries_to_store(self, store: MemoryStore, target_reduction: int) -> List[MemoryEntry]:
        """Alias for _select_entries_to_compress for backward compatibility."""
        return self._select_entries_to_compress(store, target_reduction)
    
    def should_compress(self, store: MemoryStore, budget: Optional[int] = None) -> bool:
        """Check if compression is needed.
        
        Args:
            store: The memory store to check.
            budget: Token budget to check against.
            
        Returns:
            True if store exceeds budget and has compressible entries.
        """
        target = budget or self.token_budget
        if store.token_total() <= target:
            return False
        
        # Check if there are compressible entries
        return len(self._get_compressible_entries(store)) > 0
    
    def get_stats(self, store: MemoryStore) -> Dict[str, Any]:
        """Get current compression statistics.
        
        Args:
            store: The memory store.
            
        Returns:
            Dictionary with statistics.
        """
        total_tokens = store.token_total()
        total_entries = len(store)
        compressible = len(self._get_compressible_entries(store))
        
        return {
            "total_tokens": total_tokens,
            "total_entries": total_entries,
            "token_budget": self.token_budget,
            "over_budget": max(0, total_tokens - self.token_budget),
            "compressible_entries": compressible,
            "protected_entries": total_entries - compressible,
            "needs_compression": self.should_compress(store),
        }


    def save(self, store: MemoryStore, path) -> "Path":
        """Persist the memory store plus the latest compression report."""
        from .persistence import MemoryPersistence
        extra = {"last_report": self.last_report.to_dict()} if self.last_report else None
        return MemoryPersistence().save(store, path, extra=extra)

    def load(self, path) -> MemoryStore:
        """Load a memory store from disk. Restores last_report if present."""
        from .persistence import MemoryPersistence
        persistence = MemoryPersistence()
        payload = persistence.load_payload(path)
        store = persistence.load(path)
        extra = payload.get("extra") or {}
        report_data = extra.get("last_report")
        if report_data:
            try:
                self.last_report = CompressionReport(**report_data)
            except TypeError:
                self.last_report = None
        return store


def create_compressor(
    token_budget: int = 4000,
    protected_recent: int = 3,
    **kwargs
) -> MemoryCompressor:
    """Factory function to create a MemoryCompressor.
    
    Args:
        token_budget: Maximum tokens allowed.
        protected_recent: Number of recent entries to protect.
        **kwargs: Additional arguments for MemoryCompressor.
        
    Returns:
        Configured MemoryCompressor.
    """
    return MemoryCompressor(
        token_budget=token_budget,
        protected_recent=protected_recent,
        **kwargs
    )
