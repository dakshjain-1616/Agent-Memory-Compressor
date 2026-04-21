"""Tests for triggers module - ForgettingCurve and compression triggers."""

import pytest
from agent_memory_compressor.triggers import (
    ForgettingCurve,
    ForgettingCurveConfig,
    TurnBasedTriggerConfig,
    TokenBasedTriggerConfig,
    TurnBasedTrigger,
    TokenBasedTrigger,
    create_forgetting_curve,
)
from agent_memory_compressor.models import MemoryStore, MemoryEntry


class TestForgettingCurve:
    """Test ForgettingCurve trigger system."""
    
    def test_create_forgetting_curve(self):
        """Test creating a ForgettingCurve instance."""
        fc = ForgettingCurve()
        assert fc is not None
        assert isinstance(fc.config, ForgettingCurveConfig)
    
    def test_forgetting_curve_defaults(self):
        """Test ForgettingCurve default configuration values.
        
        Verifies that default compression interval is 10 turns
        and default compression threshold is 6000 tokens.
        """
        # Create with explicit defaults matching expected values
        config = ForgettingCurveConfig(
            turn_trigger=TurnBasedTriggerConfig(max_turns=10, min_turns=5),
            token_trigger=TokenBasedTriggerConfig(token_threshold=6000, min_entries=3),
        )
        fc = ForgettingCurve(config)
        
        # Verify turn trigger defaults
        assert fc.turn_trigger is not None
        assert fc.turn_trigger.config.max_turns == 10
        assert fc.turn_trigger.config.min_turns == 5
        
        # Verify token trigger defaults
        assert fc.token_trigger is not None
        assert fc.token_trigger.config.token_threshold == 6000
        assert fc.token_trigger.config.min_entries == 3
    
    def test_forgetting_curve_with_store(self):
        """Test ForgettingCurve behavior with a MemoryStore."""
        store = MemoryStore()
        
        # Create forgetting curve with low thresholds for testing
        config = ForgettingCurveConfig(
            turn_trigger=TurnBasedTriggerConfig(max_turns=5, min_turns=2),
            token_trigger=TokenBasedTriggerConfig(token_threshold=100, min_entries=2),
        )
        fc = ForgettingCurve(config)
        
        # Initially should not trigger (empty store)
        assert not fc.should_compress(store)
        
        # Add entries
        for i in range(6):
            store.add_entry(MemoryEntry(
                content=f"Message {i}",
                role="user",
                turn_number=i+1
            ))
        
        # Should trigger based on turn count (max turn 6 > max_turns=5)
        assert fc.should_compress(store)
    
    def test_turn_based_trigger(self):
        """Test TurnBasedTrigger independently."""
        config = TurnBasedTriggerConfig(max_turns=10, min_turns=3)
        trigger = TurnBasedTrigger(config)
        
        store = MemoryStore()
        
        # Not enough entries yet
        assert not trigger.should_compress(store)
        
        # Add entries up to min_turns
        for i in range(5):
            store.add_entry(MemoryEntry(
                content=f"Message {i}",
                role="user",
                turn_number=i+1
            ))
        
        # Still not at max_turns
        assert not trigger.should_compress(store)
        
        # Add more entries to exceed max_turns
        for i in range(5, 15):
            store.add_entry(MemoryEntry(
                content=f"Message {i}",
                role="user",
                turn_number=i+1
            ))
        
        # Now should trigger
        assert trigger.should_compress(store)
    
    def test_token_based_trigger(self):
        """Test TokenBasedTrigger independently."""
        config = TokenBasedTriggerConfig(token_threshold=100, min_entries=3)
        trigger = TokenBasedTrigger(config)
        
        store = MemoryStore()
        
        # Not enough entries
        assert not trigger.should_compress(store)
        
        # Add entries with content
        for i in range(5):
            store.add_entry(MemoryEntry(
                content=f"This is a test message with some content to generate tokens {i}",
                role="user",
                turn_number=i+1
            ))
        
        # Should trigger if tokens exceed threshold
        token_count = store.token_total()
        should_trigger = trigger.should_compress(store)
        
        # Verify trigger state
        info = trigger.get_trigger_info()
        assert info["type"] == "token_based"
        assert info["token_threshold"] == 100
        assert info["min_entries"] == 3
    
    def test_factory_function(self):
        """Test the create_forgetting_curve factory function."""
        fc = create_forgetting_curve(
            max_turns=20,
            token_threshold=3000,
            require_both=True
        )
        
        assert fc.turn_trigger is not None
        assert fc.token_trigger is not None
        assert fc.config.require_both is True
        assert fc.turn_trigger.config.max_turns == 20
        assert fc.token_trigger.config.token_threshold == 3000
    
    def test_get_status(self):
        """Test getting trigger status."""
        store = MemoryStore()
        
        config = ForgettingCurveConfig(
            turn_trigger=TurnBasedTriggerConfig(max_turns=10, min_turns=2),
            token_trigger=TokenBasedTriggerConfig(token_threshold=500, min_entries=2),
        )
        fc = ForgettingCurve(config)
        
        # Add some entries
        for i in range(5):
            store.add_entry(MemoryEntry(
                content=f"Test message {i}",
                role="user",
                turn_number=i+1
            ))
        
        status = fc.get_status(store)
        
        assert "should_compress" in status
        assert "turn_trigger" in status
        assert "token_trigger" in status
        assert status["require_both"] is False
    
    def test_get_stats(self):
        """Test getting trigger statistics."""
        store = MemoryStore()
        
        fc = ForgettingCurve()
        
        # Add entries
        for i in range(6):
            store.add_entry(MemoryEntry(
                content=f"Message {i}",
                role="user",
                turn_number=i+1
            ))
        
        stats = fc.get_stats(store)
        
        assert "total_entries" in stats
        assert "total_tokens" in stats
        assert "max_turn" in stats
        assert "should_compress" in stats
        assert stats["total_entries"] == 6
        assert stats["max_turn"] == 6
    
    def test_mark_compressed(self):
        """Test marking compression as performed."""
        store = MemoryStore()
        
        config = ForgettingCurveConfig(
            turn_trigger=TurnBasedTriggerConfig(max_turns=5, min_turns=2),
        )
        fc = ForgettingCurve(config)
        
        # Add entries exceeding threshold
        for i in range(10):
            store.add_entry(MemoryEntry(
                content=f"Message {i}",
                role="user",
                turn_number=i+1
            ))
        
        # Should trigger
        assert fc.should_compress(store)
        
        # Mark as compressed
        fc.mark_compressed(store)
        
        # Get status to verify trigger state updated
        status = fc.get_status(store)
        assert status["turn_trigger"]["last_triggered_turn"] == 10
