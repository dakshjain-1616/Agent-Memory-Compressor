"""Tests for orchestrator module."""

import sys
sys.path.insert(0, '/app/agent_memory_compressor_0556')

import pytest
from agent_memory_compressor.models import MemoryEntry, MemoryStore
from agent_memory_compressor.orchestrator import MemoryCompressor
from agent_memory_compressor.strategies import CompressionStrategy, LLMClient, CompressionResult


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self):
        self.responses = {}
    
    def set_response(self, prompt_pattern, response):
        self.responses[prompt_pattern] = response
    
    def complete(self, prompt, **kwargs):
        for pattern, response in self.responses.items():
            if pattern in prompt:
                return response
        return "Mock summary"
    
    def compress(self, entry, strategy):
        """Mock compress method that returns a simple compressed entry."""
        compressed_content = "[Compressed]"
        compressed_entry = MemoryEntry(
            content=compressed_content,
            role="compressed",
            turn_number=entry.turn_number,
            timestamp=entry.timestamp,
            metadata={
                **entry.metadata,
                "original_role": entry.role,
                "compression_strategy": strategy.value,
            },
            compression_history=[
                *entry.compression_history,
                {
                    "strategy": strategy.value,
                    "original_content": entry.content,
                    "timestamp": __import__('time').time(),
                    "metadata": {}
                }
            ]
        )
        
        return CompressionResult(
            original_entry=entry,
            compressed_entry=compressed_entry,
            strategy=strategy,
            tokens_saved=len(entry.content.split()) - len(compressed_content.split()),
            success=True
        )


class TestMemoryCompressor:
    """Test MemoryCompressor class."""
    
    def test_create_compressor(self):
        """Test creating a compressor."""
        llm = MockLLMClient()
        compressor = MemoryCompressor(token_budget=50, protected_recent=3, engine=llm)
        
        assert compressor.token_budget == 50
        assert compressor.protected_recent == 3
    
    def test_compress_reduces_tokens(self):
        """Test that compression reduces tokens."""
        llm = MockLLMClient()
        llm.set_response("summarize", "Summary of conversation")
        
        compressor = MemoryCompressor(token_budget=50, protected_recent=3, engine=llm)
        
        store = MemoryStore()
        # Add entries that exceed budget
        for i in range(10):
            store.add_entry(MemoryEntry(
                content=f"This is message number {i} with some content",
                role="user" if i % 2 == 0 else "assistant",
                turn_number=i+1
            ))
        
        initial_tokens = store.token_total()
        assert initial_tokens > 50  # Should exceed budget
        
        report = compressor.compress(store)
        
        # Should have compressed something
        assert report.entries_compressed > 0
        assert report.tokens_saved > 0
        assert store.token_total() <= initial_tokens
    
    def test_protected_recent_not_compressed(self):
        """Test that recent entries are protected."""
        llm = MockLLMClient()
        llm.set_response("summarize", "Summary")
        
        compressor = MemoryCompressor(token_budget=50, protected_recent=3, engine=llm)
        
        store = MemoryStore()
        for i in range(10):
            store.add_entry(MemoryEntry(
                content=f"Message {i}",
                role="user",
                turn_number=i+1
            ))
        
        report = compressor.compress(store)
        
        # Check that last 3 entries are still there and not compressed
        recent = store.get_recent(3)
        for entry in recent:
            assert entry.role == "user"  # Original role preserved
            assert "Message" in entry.content  # Original content
    
    def test_get_compressible_entries(self):
        """Test getting compressible entries."""
        llm = MockLLMClient()
        compressor = MemoryCompressor(token_budget=50, protected_recent=3, engine=llm)
        
        store = MemoryStore()
        for i in range(5):
            store.add_entry(MemoryEntry(content=f"Msg {i}", role="user", turn_number=i+1))
        
        compressible = compressor._get_compressible_entries(store)
        # Should exclude last 3 entries (protected_recent=3), leaving 2 compressible
        assert len(compressible) == 2
        assert compressible[0].turn_number == 1
        assert compressible[-1].turn_number == 2
    
    def test_should_compress(self):
        """Test should_compress check."""
        llm = MockLLMClient()
        compressor = MemoryCompressor(token_budget=50, protected_recent=3, engine=llm)
        
        store = MemoryStore()
        # Add small amount of content
        store.add_entry(MemoryEntry(content="Hi", role="user", turn_number=1))
        
        # Should not need compression
        assert compressor.should_compress(store) == False
        
        # Add more content to exceed budget
        for i in range(20):
            store.add_entry(MemoryEntry(
                content=f"This is a longer message with many tokens {i}",
                role="user",
                turn_number=i+2
            ))
        
        # Should need compression now
        assert compressor.should_compress(store) == True
    
    def test_compression_report(self):
        """Test compression report."""
        llm = MockLLMClient()
        llm.set_response("summarize", "Summary")
        
        compressor = MemoryCompressor(token_budget=50, protected_recent=3, engine=llm)
        
        store = MemoryStore()
        for i in range(10):
            store.add_entry(MemoryEntry(content=f"Message {i}", role="user", turn_number=i+1))
        
        report = compressor.compress(store)
        
        assert report.entries_compressed >= 0
        assert report.tokens_saved >= 0
        assert report.compression_ratio >= 0.0
        assert report.compression_ratio <= 1.0
    
    def test_get_stats(self):
        """Test getting compressor stats."""
        llm = MockLLMClient()
        compressor = MemoryCompressor(token_budget=50, protected_recent=3, engine=llm)
        
        store = MemoryStore()
        for i in range(5):
            store.add_entry(MemoryEntry(content=f"Msg {i}", role="user", turn_number=i+1))
        
        stats = compressor.get_stats(store)
        
        assert "total_entries" in stats
        assert "total_tokens" in stats
        assert "token_budget" in stats
        assert "protected_entries" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
