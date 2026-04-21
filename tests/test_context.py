"""Tests for context module."""

import sys
sys.path.insert(0, '/app/agent_memory_compressor_0556')

import pytest
from agent_memory_compressor.models import MemoryEntry, MemoryStore
from agent_memory_compressor.context import ContextBuilder, ContextConfig


class TestContextBuilder:
    """Test ContextBuilder class."""
    
    def test_create_builder(self):
        """Test creating a context builder."""
        config = ContextConfig(max_tokens=1000, protected_recent=3)
        builder = ContextBuilder(config)
        assert builder.config.max_tokens == 1000
        assert builder.config.protected_recent == 3
    
    def test_build_context_includes_recent(self):
        """Test that context includes recent entries."""
        store = MemoryStore()
        for i in range(10):
            store.add_entry(MemoryEntry(
                content=f"Message {i}",
                role="user" if i % 2 == 0 else "assistant",
                turn_number=i+1
            ))
        
        config = ContextConfig(max_tokens=500, protected_recent=3)
        builder = ContextBuilder(config)
        context = builder.build_context(store)
        
        # Should include recent messages
        assert "Message 9" in context or "Message 8" in context
    
    def test_build_context_respects_budget(self):
        """Test that context respects token budget."""
        store = MemoryStore()
        for i in range(20):
            store.add_entry(MemoryEntry(
                content=f"This is a long message with many words to test token counting {i}",
                role="user",
                turn_number=i+1
            ))
        
        config = ContextConfig(max_tokens=100, protected_recent=3)
        builder = ContextBuilder(config)
        context = builder.build_context(store)
        
        # Context should be limited
        tokens = builder._count_context_tokens(context)
        assert tokens <= 100
    
    def test_protected_recent_in_context(self):
        """Test that protected recent entries are always in context."""
        store = MemoryStore()
        for i in range(10):
            store.add_entry(MemoryEntry(
                content=f"Message {i}",
                role="user",
                turn_number=i+1
            ))
        
        config = ContextConfig(max_tokens=200, protected_recent=3)
        builder = ContextBuilder(config)
        context = builder.build_context(store)
        
        # Last 3 messages should be present
        assert "Message 9" in context
        assert "Message 8" in context
        assert "Message 7" in context
    
    def test_build_messages(self):
        """Test building message list."""
        store = MemoryStore()
        for i in range(5):
            store.add_entry(MemoryEntry(
                content=f"Msg {i}",
                role="user" if i % 2 == 0 else "assistant",
                turn_number=i+1
            ))
        
        config = ContextConfig(max_tokens=500, protected_recent=2)
        builder = ContextBuilder(config)
        messages = builder.build_messages(store, system_message="System prompt")
        
        assert len(messages) > 0
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"
    
    def test_get_context_stats(self):
        """Test getting context stats."""
        store = MemoryStore()
        for i in range(5):
            store.add_entry(MemoryEntry(content=f"Msg {i}", role="user", turn_number=i+1))
        
        config = ContextConfig(max_tokens=500, protected_recent=2)
        builder = ContextBuilder(config)
        stats = builder.get_context_stats(store)
        
        assert "entries_included" in stats
        assert "entries_included" in stats
        assert "total_tokens" in stats
        assert "max_tokens" in stats
    
    def test_verify_recent_protected(self):
        """Test verifying recent entries are protected."""
        store = MemoryStore()
        for i in range(5):
            store.add_entry(MemoryEntry(content=f"Msg {i}", role="user", turn_number=i+1))
        
        config = ContextConfig(max_tokens=500, protected_recent=3)
        builder = ContextBuilder(config)
        
        # Should pass with all entries
        context = builder.build_context(store)
        assert builder.verify_recent_protected(store, context) == True
        
        # Build context string
        # Should fail with truncated context (only old entries)
        old_only_context = builder.config.separator.join(builder._format_entry(e) for e in store.entries[:2])
        assert builder.verify_recent_protected(store, old_only_context) == False
    
    def test_empty_store(self):
        """Test with empty store."""
        store = MemoryStore()
        config = ContextConfig(max_tokens=500, protected_recent=3)
        builder = ContextBuilder(config)
        
        context = builder.build_context(store)
        assert context == ""
        
        messages = builder.build_messages(store)
        assert len(messages) == 0



    def test_context_protected_turns(self):
        """Test that last 3 turns appear in context when max_tokens is very low."""
        store = MemoryStore()
        for i in range(10):
            store.add_entry(MemoryEntry(
                content=f"Message {i}",
                role="user",
                turn_number=i+1
            ))
        
        # Set max_tokens very low (50) with protected_recent=3
        config = ContextConfig(max_tokens=50, protected_recent=3)
        builder = ContextBuilder(config)
        context = builder.build_context(store)
        
        # Last 3 turns (8, 9, 10) should still appear in context
        assert "Message 9" in context, "Turn 9 (Message 9) should be in context"
        assert "Message 8" in context, "Turn 8 (Message 8) should be in context"
        assert "Message 7" in context, "Turn 7 (Message 7) should be in context"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
