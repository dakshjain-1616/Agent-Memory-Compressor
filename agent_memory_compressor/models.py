"""Memory models with Pydantic and tiktoken integration."""

import uuid
import time
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single memory entry in the agent's conversation history."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    role: Literal["system", "user", "assistant", "tool", "compressed"]
    timestamp: float = Field(default_factory=time.time)
    turn_number: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    importance_score: Optional[float] = None
    compression_history: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryStore:
    """In-memory storage for conversation history with token counting."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.entries: List[MemoryEntry] = []
        self.encoding_name = encoding_name
        self._encoding = None
        self._total_tokens: int = 0
        
    def _get_encoding(self):
        """Lazy load tiktoken encoding."""
        if self._encoding is None:
            try:
                import tiktoken
                self._encoding = tiktoken.get_encoding(self.encoding_name)
            except ImportError:
                raise ImportError(
                    "tiktoken is required for token counting. "
                    "Install with: pip install tiktoken"
                )
        return self._encoding
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string using tiktoken."""
        try:
            encoding = self._get_encoding()
            return len(encoding.encode(text))
        except Exception:
            return int(len(text.split()) * 1.3)
    
    def entry_tokens(self, entry: MemoryEntry) -> int:
        """Count tokens for a single memory entry."""
        return self.count_tokens(entry.content)
    
    def token_total(self) -> int:
        """Calculate total tokens across all entries."""
        return sum(self.entry_tokens(entry) for entry in self.entries)
    
    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a memory entry to the store."""
        self.entries.append(entry)
        self._total_tokens = None
    
    def add_entries(self, entries: List[MemoryEntry]) -> None:
        """Add multiple memory entries to the store."""
        self.entries.extend(entries)
        self._total_tokens = None
    
    def remove_entry(self, entry_id: str) -> bool:
        """Remove a memory entry by ID."""
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                self.entries.pop(i)
                self._total_tokens = None
                return True
        return False
    
    def update_entry(self, entry_id: str, **updates) -> bool:
        """Update a memory entry by ID."""
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                entry_data = entry.model_dump()
                entry_data.update(updates)
                self.entries[i] = MemoryEntry(**entry_data)
                self._total_tokens = None
                return True
        return False
    
    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory entry by ID."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None
    
    def get_recent(self, n: int) -> List[MemoryEntry]:
        """Get the N most recent entries."""
        return self.entries[-n:] if n > 0 else []
    
    def clear(self) -> None:
        """Clear all entries from the store."""
        self.entries.clear()
        self._total_tokens = 0
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self):
        return iter(self.entries)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert store to dictionary representation."""
        return {
            "encoding_name": self.encoding_name,
            "total_entries": len(self.entries),
            "total_tokens": self.token_total(),
            "entries": [entry.model_dump() for entry in self.entries],
        }
