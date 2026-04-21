"""Tests for strategies module."""

import sys
sys.path.insert(0, '/app/agent_memory_compressor_0556')

import pytest
from agent_memory_compressor.models import MemoryEntry
from agent_memory_compressor.strategies import (
    CompressionStrategy, CompressionEngine, CompressionResult
)


class MockLLMClient:
    """Mock LLM client for testing compression strategies."""
    
    def __init__(self):
        self.responses = {}
    
    def set_response(self, prompt_pattern, response):
        self.responses[prompt_pattern] = response
    
    def complete(self, prompt, **kwargs):
        for pattern, response in self.responses.items():
            if pattern in prompt.lower():
                return response
        if "summarize" in prompt.lower():
            return "Summary of the conversation content."
        elif "extract" in prompt.lower() or "fact" in prompt.lower():
            return "- Key fact 1\n- Key fact 2"
        elif "archive" in prompt.lower():
            return "ARCHIVED"
        return "Mock response"


class TestCompressionStrategy:
    def test_strategy_values(self):
        assert CompressionStrategy.SUMMARIZE.value == "summarize"
        assert CompressionStrategy.EXTRACT_FACTS.value == "extract_facts"
        assert CompressionStrategy.ARCHIVE.value == "archive"


class TestCompressionEngine:
    def test_summarize_strategy(self):
        llm = MockLLMClient()
        llm.set_response("summarize", "This is a summary.")
        engine = CompressionEngine(llm_client=llm)
        
        entry = MemoryEntry(
            content="This is a very long conversation with many details.",
            role="assistant",
            turn_number=5
        )
        
        result = engine.compress(entry, CompressionStrategy.SUMMARIZE)
        
        assert result.success is True
        assert result.compressed_entry is not None
        assert result.compressed_entry.role == "compressed"
        assert result.tokens_saved > 0
    
    def test_extract_facts_strategy(self):
        llm = MockLLMClient()
        llm.set_response("extract", "- Fact 1\n- Fact 2")
        engine = CompressionEngine(llm_client=llm)
        
        entry = MemoryEntry(
            content="The user mentioned they like Python programming.",
            role="user",
            turn_number=3
        )
        
        result = engine.compress(entry, CompressionStrategy.EXTRACT_FACTS)
        
        assert result.success is True
        assert result.compressed_entry is not None
        assert result.compressed_entry.role == "compressed"
    
    def test_archive_strategy(self):
        llm = MockLLMClient()
        engine = CompressionEngine(llm_client=llm)
        
        entry = MemoryEntry(
            content="Old conversation content",
            role="assistant",
            turn_number=1
        )
        
        result = engine.compress(entry, CompressionStrategy.ARCHIVE)
        
        assert result.success is True
        assert result.compressed_entry is not None
    
    def test_compression_preserves_turn_number(self):
        llm = MockLLMClient()
        engine = CompressionEngine(llm_client=llm)
        
        entry = MemoryEntry(content="Test", role="user", turn_number=42)
        result = engine.compress(entry, CompressionStrategy.SUMMARIZE)
        
        assert result.compressed_entry.turn_number == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
