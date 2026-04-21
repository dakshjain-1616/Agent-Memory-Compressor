"""Importance scoring with exponential decay, type-based weights, and keyword boosting."""

import time
import math
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from .models import MemoryEntry, MemoryStore


@dataclass
class ScoringConfig:
    """Configuration for importance scoring."""
    # Recency decay parameters
    decay_half_life: float = 3600.0  # seconds (1 hour default)
    
    # Type-based weights
    type_weights: Dict[str, float] = None
    
    # Keyword boosting
    boost_keywords: Dict[str, float] = None
    case_sensitive: bool = False
    
    def __post_init__(self):
        if self.type_weights is None:
            self.type_weights = {
                "system": 2.0,
                "user": 1.5,
                "assistant": 1.0,
                "tool": 0.8,
                "compressed": 0.5,
            }
        if self.boost_keywords is None:
            self.boost_keywords = {}


class ImportanceScorer:
    """Calculate importance scores for memory entries."""
    
    def __init__(self, config: Optional[ScoringConfig] = None):
        """Initialize the importance scorer.
        
        Args:
            config: Scoring configuration. Uses defaults if not provided.
        """
        self.config = config or ScoringConfig()
        self.current_time: Optional[float] = None
    
    def set_reference_time(self, timestamp: float) -> None:
        """Set the reference time for recency calculations.
        
        Args:
            timestamp: Reference timestamp (usually current time).
        """
        self.current_time = timestamp
    
    def _get_current_time(self) -> float:
        """Get current time for recency calculation."""
        return self.current_time if self.current_time is not None else time.time()
    
    def calculate_recency_score(self, entry: MemoryEntry) -> float:
        """Calculate recency score using exponential decay.
        
        Score = exp(-lambda * age) where lambda = ln(2) / half_life
        
        Args:
            entry: The memory entry to score.
            
        Returns:
            Recency score between 0 and 1.
        """
        age_seconds = self._get_current_time() - entry.timestamp
        lambda_val = math.log(2) / self.config.decay_half_life
        return math.exp(-lambda_val * age_seconds)
    
    def calculate_type_weight(self, entry: MemoryEntry) -> float:
        """Get type-based weight for an entry.
        
        Args:
            entry: The memory entry to score.
            
        Returns:
            Weight multiplier for the entry type.
        """
        return self.config.type_weights.get(entry.role, 1.0)
    
    def calculate_keyword_boost(self, entry: MemoryEntry) -> float:
        """Calculate keyword-based boost for an entry.
        
        Args:
            entry: The memory entry to score.
            
        Returns:
            Boost multiplier (1.0 + sum of matched keyword boosts).
        """
        if not self.config.boost_keywords:
            return 1.0
        
        content = entry.content
        if not self.config.case_sensitive:
            content = content.lower()
        
        total_boost = 0.0
        for keyword, boost in self.config.boost_keywords.items():
            search_keyword = keyword if self.config.case_sensitive else keyword.lower()
            if search_keyword in content:
                total_boost += boost
        
        return 1.0 + total_boost
    
    def score_entry(self, entry: MemoryEntry) -> float:
        """Calculate overall importance score for an entry.
        
        Score = recency * type_weight * keyword_boost
        
        Args:
            entry: The memory entry to score.
            
        Returns:
            Importance score (higher = more important).
        """
        recency = self.calculate_recency_score(entry)
        type_weight = self.calculate_type_weight(entry)
        keyword_boost = self.calculate_keyword_boost(entry)
        
        return recency * type_weight * keyword_boost
    
    def score_store(self, store: MemoryStore) -> List[tuple[MemoryEntry, float]]:
        """Score all entries in a memory store.
        
        Args:
            store: The memory store to score.
            
        Returns:
            List of (entry, score) tuples sorted by score (highest first).
        """
        scored = [(entry, self.score_entry(entry)) for entry in store.entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def update_entry_scores(self, store: MemoryStore) -> None:
        """Update importance_score field for all entries in store.
        
        Args:
            store: The memory store to update.
        """
        for entry in store.entries:
            entry.importance_score = self.score_entry(entry)
    
    def get_least_important(self, store: MemoryStore, n: int = 1) -> List[MemoryEntry]:
        """Get the N least important entries from the store.
        
        Args:
            store: The memory store to search.
            n: Number of entries to return.
            
        Returns:
            List of N least important entries.
        """
        scored = self.score_store(store)
        # Return lowest scored entries
        return [entry for entry, _ in scored[-n:]] if n > 0 else []
    
    def get_most_important(self, store: MemoryStore, n: int = 1) -> List[MemoryEntry]:
        """Get the N most important entries from the store.
        
        Args:
            store: The memory store to search.
            n: Number of entries to return.
            
        Returns:
            List of N most important entries.
        """
        scored = self.score_store(store)
        return [entry for entry, _ in scored[:n]] if n > 0 else []


def create_scorer(
    decay_half_life: float = 3600.0,
    type_weights: Optional[Dict[str, float]] = None,
    boost_keywords: Optional[Dict[str, float]] = None,
    case_sensitive: bool = False
) -> ImportanceScorer:
    """Factory function to create an ImportanceScorer with custom config.
    
    Args:
        decay_half_life: Half-life in seconds for recency decay.
        type_weights: Custom type weights dictionary.
        boost_keywords: Keywords to boost and their boost values.
        case_sensitive: Whether keyword matching is case sensitive.
        
    Returns:
        Configured ImportanceScorer instance.
    """
    config = ScoringConfig(
        decay_half_life=decay_half_life,
        type_weights=type_weights,
        boost_keywords=boost_keywords,
        case_sensitive=case_sensitive
    )
    return ImportanceScorer(config)
