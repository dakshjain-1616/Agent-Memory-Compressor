"""Tests for models module."""

import sys
sys.path.insert(0, '/app/agent_memory_compressor_0556')

import pytest
from agent_memory_compressor.models import MemoryEntry, MemoryStore


class TestMemoryEntry:
    """Test MemoryEntry class."""
    
    def test_create_entry(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            content="Test content",
            role="user",
            turn_number=1
        )
        assert entry.content == "Test content"
        assert entry.role == "user"
        assert entry.turn_number == 1
        assert entry.id is not None
        assert entry.timestamp > 0
    
    def test_entry_roles(self):
        """Test all valid entry roles."""
        for role in ["system", "user", "assistant", "tool", "compressed"]:
            entry = MemoryEntry(content="Test", role=role, turn_number=1)
            assert entry.role == role
    
    def test_entry_metadata(self):
        """Test entry metadata."""
        entry = MemoryEntry(
            content="Test",
            role="user",
            turn_number=1,
            metadata={"key": "value"}
        )
        assert entry.metadata["key"] == "value"


class TestMemoryStore:
    """Test MemoryStore class."""
    
    def test_create_store(self):
        """Test creating a memory store."""
        store = MemoryStore()
        assert len(store) == 0
        assert store.token_total() == 0
    
    def test_add_entry(self):
        """Test adding an entry."""
        store = MemoryStore()
        entry = MemoryEntry(content="Hello", role="user", turn_number=1)
        store.add_entry(entry)
        assert len(store) == 1
        assert store.token_total() > 0
    
    def test_add_entries(self):
        """Test adding multiple entries."""
        store = MemoryStore()
        entries = [
            MemoryEntry(content="Hello", role="user", turn_number=1),
            MemoryEntry(content="Hi", role="assistant", turn_number=2),
        ]
        store.add_entries(entries)
        assert len(store) == 2
    
    def test_remove_entry(self):
        """Test removing an entry."""
        store = MemoryStore()
        entry = MemoryEntry(content="Hello", role="user", turn_number=1)
        store.add_entry(entry)
        assert store.remove_entry(entry.id) == True
        assert len(store) == 0
        assert store.remove_entry("nonexistent") == False
    
    def test_get_recent(self):
        """Test getting recent entries."""
        store = MemoryStore()
        for i in range(5):
            store.add_entry(MemoryEntry(content=f"Msg {i}", role="user", turn_number=i+1))
        
        recent = store.get_recent(3)
        assert len(recent) == 3
        assert recent[0].turn_number == 3
        assert recent[2].turn_number == 5
    
    def test_token_counting(self):
        """Test token counting."""
        store = MemoryStore()
        entry = MemoryEntry(content="Hello world", role="user", turn_number=1)
        store.add_entry(entry)
        tokens = store.token_total()
        assert tokens > 0
        
        # Add another entry
        entry2 = MemoryEntry(content="Another message", role="assistant", turn_number=2)
        store.add_entry(entry2)
        assert store.token_total() > tokens
    
    def test_update_entry(self):
        """Test updating an entry."""
        store = MemoryStore()
        entry = MemoryEntry(content="Original", role="user", turn_number=1)
        store.add_entry(entry)
        
        assert store.update_entry(entry.id, content="Updated") == True
        updated = store.get_entry(entry.id)
        assert updated.content == "Updated"
        assert store.update_entry("nonexistent", content="Test") == False
    
    def test_clear(self):
        """Test clearing store."""
        store = MemoryStore()
        store.add_entry(MemoryEntry(content="Test", role="user", turn_number=1))
        store.clear()
        assert len(store) == 0
        assert store.token_total() == 0
    
    def test_to_dict(self):
        """Test store serialization."""
        store = MemoryStore()
        store.add_entry(MemoryEntry(content="Test", role="user", turn_number=1))
        data = store.to_dict()
        assert "total_entries" in data
        assert "total_tokens" in data
        assert data["total_entries"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
