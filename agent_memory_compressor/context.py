"""ContextBuilder: assemble final prompt context respecting max_tokens and recent N turns protection."""

from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass

from .models import MemoryEntry, MemoryStore


@dataclass
class ContextConfig:
    """Configuration for context building."""
    max_tokens: int = 4000
    protected_recent: int = 3
    system_message_reserve: int = 500
    format_template: str = "{role}: {content}"
    separator: str = "\n\n"


class ContextBuilder:
    """Builds prompt context from memory store with token budget constraints."""
    
    def __init__(self, config: Optional[ContextConfig] = None):
        """Initialize context builder.
        
        Args:
            config: Configuration for context building.
        """
        self.config = config or ContextConfig()
    
    def _format_entry(self, entry: MemoryEntry) -> str:
        """Format a single entry for context.
        
        Args:
            entry: Memory entry to format.
            
        Returns:
            Formatted string.
        """
        return self.config.format_template.format(
            role=entry.role,
            content=entry.content
        )
    
    def _count_context_tokens(self, text: str) -> int:
        """Count tokens in context text.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Approximate token count.
        """
        # Rough approximation: 1 token ≈ 0.75 words
        return int(len(text.split()) / 0.75)
    
    def build_context(
        self,
        store: MemoryStore,
        system_message: Optional[str] = None
    ) -> str:
        """Build context string respecting max_tokens and protected recent turns.
        
        Args:
            store: Memory store to build context from.
            system_message: Optional system message to include at start.
            
        Returns:
            Formatted context string.
        """
        # Reserve either the configured system_message_reserve OR the actual
        # system message size, whichever is larger. Subtracting both
        # double-counts the reservation.
        if system_message:
            system_cost = max(
                self.config.system_message_reserve,
                self._count_context_tokens(system_message),
            )
        else:
            system_cost = 0
        available_tokens = self.config.max_tokens - system_cost
        
        # Always include protected recent entries
        protected_entries = store.get_recent(self.config.protected_recent)
        protected_context = self.config.separator.join(
            self._format_entry(e) for e in protected_entries
        )
        protected_tokens = self._count_context_tokens(protected_context)
        
        # Remaining tokens for older entries
        remaining_tokens = available_tokens - protected_tokens
        
        # Get older entries (excluding protected)
        if len(store.entries) > self.config.protected_recent:
            older_entries = store.entries[:-self.config.protected_recent]
        else:
            older_entries = []
        
        # Build context from older entries, most recent first
        selected_older = []
        current_tokens = 0
        
        for entry in reversed(older_entries):
            entry_text = self._format_entry(entry)
            entry_tokens = self._count_context_tokens(entry_text)
            
            if current_tokens + entry_tokens <= remaining_tokens:
                selected_older.insert(0, entry)  # Keep chronological order
                current_tokens += entry_tokens
            else:
                break
        
        # Assemble final context
        parts = []
        
        if system_message:
            parts.append(f"system: {system_message}")
        
        if selected_older:
            parts.append(self.config.separator.join(
                self._format_entry(e) for e in selected_older
            ))
        
        if protected_entries:
            parts.append(protected_context)
        
        return self.config.separator.join(parts)
    
    def build_messages(
        self,
        store: MemoryStore,
        system_message: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build context as list of message dictionaries.
        
        Args:
            store: Memory store to build context from.
            system_message: Optional system message.
            
        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        if system_message:
            system_cost = max(
                self.config.system_message_reserve,
                self._count_context_tokens(system_message),
            )
        else:
            system_cost = 0
        available_tokens = self.config.max_tokens - system_cost
        
        # Always include protected recent entries
        protected_entries = store.get_recent(self.config.protected_recent)
        protected_tokens = sum(
            self._count_context_tokens(e.content) for e in protected_entries
        )
        
        remaining_tokens = available_tokens - protected_tokens
        
        # Get older entries
        if len(store.entries) > self.config.protected_recent:
            older_entries = store.entries[:-self.config.protected_recent]
        else:
            older_entries = []
        
        # Select older entries that fit
        selected_older = []
        current_tokens = 0
        
        for entry in reversed(older_entries):
            entry_tokens = self._count_context_tokens(entry.content)
            if current_tokens + entry_tokens <= remaining_tokens:
                selected_older.insert(0, entry)
                current_tokens += entry_tokens
            else:
                break
        
        # Add selected older entries
        for entry in selected_older:
            messages.append({"role": entry.role, "content": entry.content})
        
        # Add protected entries
        for entry in protected_entries:
            messages.append({"role": entry.role, "content": entry.content})
        
        return messages
    
    def get_context_stats(
        self,
        store: MemoryStore,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get statistics about the context.
        
        Args:
            store: Memory store.
            system_message: Optional system message.
            
        Returns:
            Dictionary with context statistics.
        """
        context = self.build_context(store, system_message)
        messages = self.build_messages(store, system_message)
        
        total_tokens = self._count_context_tokens(context)
        
        protected_entries = store.get_recent(self.config.protected_recent)
        older_included = len(messages) - len(protected_entries) - (1 if system_message else 0)
        
        return {
            "total_tokens": total_tokens,
            "max_tokens": self.config.max_tokens,
            "entries_included": len(messages) - (1 if system_message else 0),
            "protected_entries": len(protected_entries),
            "older_entries_included": older_included,
            "entries_excluded": len(store.entries) - len(messages) + (1 if system_message else 0),
            "has_system_message": system_message is not None,
        }
    
    def verify_recent_protected(
        self,
        store: MemoryStore,
        context: str
    ) -> bool:
        """Verify that recent N turns are present in context.
        
        Args:
            store: Memory store.
            context: Built context string.
            
        Returns:
            True if all protected recent entries are in context.
        """
        protected = store.get_recent(self.config.protected_recent)
        
        for entry in protected:
            entry_text = self._format_entry(entry)
            if entry_text not in context:
                return False
        
        return True


def create_context_builder(
    max_tokens: int = 4000,
    protected_recent: int = 3,
    **kwargs
) -> ContextBuilder:
    """Factory function to create a ContextBuilder.
    
    Args:
        max_tokens: Maximum tokens for context.
        protected_recent: Number of recent entries to protect.
        **kwargs: Additional configuration options.
        
    Returns:
        Configured ContextBuilder.
    """
    config = ContextConfig(
        max_tokens=max_tokens,
        protected_recent=protected_recent,
        **kwargs
    )
    return ContextBuilder(config)
