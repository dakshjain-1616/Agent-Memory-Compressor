"""ForgettingCurve: turn-based and token-based triggers for compression."""

import time
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .models import MemoryStore


class CompressionTrigger(ABC):
    """Abstract base class for compression triggers."""
    
    @abstractmethod
    def should_compress(self, store: MemoryStore) -> bool:
        """Check if compression should be triggered.
        
        Args:
            store: The memory store to check.
            
        Returns:
            True if compression should be triggered.
        """
        pass
    
    @abstractmethod
    def get_trigger_info(self) -> Dict[str, Any]:
        """Get information about the trigger state.
        
        Returns:
            Dictionary with trigger information.
        """
        pass


@dataclass
class TurnBasedTriggerConfig:
    """Configuration for turn-based compression trigger."""
    max_turns: int = 50  # Trigger after this many turns
    min_turns: int = 10  # Minimum turns before triggering
    
    def __post_init__(self):
        if self.min_turns >= self.max_turns:
            raise ValueError("min_turns must be less than max_turns")


class TurnBasedTrigger(CompressionTrigger):
    """Trigger compression based on number of turns."""
    
    def __init__(self, config: Optional[TurnBasedTriggerConfig] = None):
        """Initialize turn-based trigger.
        
        Args:
            config: Configuration for the trigger.
        """
        self.config = config or TurnBasedTriggerConfig()
        self._last_triggered_turn: Optional[int] = None
    
    def should_compress(self, store: MemoryStore) -> bool:
        """Check if compression should be triggered based on turn count.
        
        Args:
            store: The memory store.
            
        Returns:
            True if turn count exceeds threshold.
        """
        if len(store.entries) < self.config.min_turns:
            return False
        
        # Get max turn number
        max_turn = max(
            (entry.turn_number for entry in store.entries),
            default=0
        )
        
        # Check if we've exceeded max_turns since last trigger
        if self._last_triggered_turn is not None:
            turns_since_trigger = max_turn - self._last_triggered_turn
            return turns_since_trigger >= self.config.max_turns
        
        return max_turn >= self.config.max_turns
    
    def mark_triggered(self, store: MemoryStore) -> None:
        """Mark that compression was triggered.
        
        Args:
            store: The memory store.
        """
        self._last_triggered_turn = max(
            (entry.turn_number for entry in store.entries),
            default=0
        )
    
    def get_trigger_info(self) -> Dict[str, Any]:
        """Get trigger information."""
        return {
            "type": "turn_based",
            "max_turns": self.config.max_turns,
            "min_turns": self.config.min_turns,
            "last_triggered_turn": self._last_triggered_turn,
        }


@dataclass
class TokenBasedTriggerConfig:
    """Configuration for token-based compression trigger."""
    token_threshold: int = 4000  # Trigger when exceeding this
    min_entries: int = 5  # Minimum entries before triggering
    hysteresis: float = 0.8  # Must drop below threshold * hysteresis to re-trigger


class TokenBasedTrigger(CompressionTrigger):
    """Trigger compression based on token count."""
    
    def __init__(self, config: Optional[TokenBasedTriggerConfig] = None):
        """Initialize token-based trigger.
        
        Args:
            config: Configuration for the trigger.
        """
        self.config = config or TokenBasedTriggerConfig()
        self._last_triggered_tokens: int = 0
        self._was_triggered: bool = False
    
    def should_compress(self, store: MemoryStore) -> bool:
        """Check if compression should be triggered based on token count.
        
        Args:
            store: The memory store.
            
        Returns:
            True if token count exceeds threshold.
        """
        if len(store.entries) < self.config.min_entries:
            return False
        
        current_tokens = store.token_total()
        
        # If previously triggered, check hysteresis
        if self._was_triggered:
            retrigger_threshold = self.config.token_threshold * self.config.hysteresis
            if current_tokens < retrigger_threshold:
                # Reset trigger state when tokens drop enough
                self._was_triggered = False
                return False
            # Still above threshold, don't re-trigger
            return False
        
        # Check if we've exceeded threshold
        if current_tokens > self.config.token_threshold:
            return True
        
        return False
    
    def mark_triggered(self, store: MemoryStore) -> None:
        """Mark that compression was triggered.
        
        Args:
            store: The memory store.
        """
        self._was_triggered = True
        self._last_triggered_tokens = store.token_total()
    
    def get_trigger_info(self) -> Dict[str, Any]:
        """Get trigger information."""
        return {
            "type": "token_based",
            "token_threshold": self.config.token_threshold,
            "min_entries": self.config.min_entries,
            "hysteresis": self.config.hysteresis,
            "was_triggered": self._was_triggered,
            "last_triggered_tokens": self._last_triggered_tokens,
        }


@dataclass
class ForgettingCurveConfig:
    """Configuration for forgetting curve-based compression."""
    # Turn-based settings
    turn_trigger: Optional[TurnBasedTriggerConfig] = None
    
    # Token-based settings
    token_trigger: Optional[TokenBasedTriggerConfig] = None
    
    # Combined logic
    require_both: bool = False  # If True, both triggers must fire


class ForgettingCurve:
    """Manages compression triggers based on forgetting curve principles.
    
    Combines turn-based and token-based triggers to determine when
    memory compression should occur.
    """
    
    def __init__(
        self,
        config: Optional[ForgettingCurveConfig] = None,
        compression_interval_turns: int = 10,
        compression_threshold_tokens: int = 6000,
    ):
        """Initialize forgetting curve manager.

        Args:
            config: Optional full configuration for triggers. If provided,
                it overrides the simple interval/threshold defaults.
            compression_interval_turns: Default turn interval between compressions.
            compression_threshold_tokens: Default token threshold to trigger compression.
        """
        # Expose the simple spec-level defaults directly on the instance
        self.compression_interval_turns = compression_interval_turns
        self.compression_threshold_tokens = compression_threshold_tokens

        if config is None:
            # Build a default config from the simple defaults
            config = ForgettingCurveConfig(
                turn_trigger=TurnBasedTriggerConfig(
                    max_turns=compression_interval_turns,
                    min_turns=max(1, compression_interval_turns // 2),
                ),
                token_trigger=TokenBasedTriggerConfig(
                    token_threshold=compression_threshold_tokens,
                ),
            )
        self.config = config

        # Initialize triggers
        self.turn_trigger: Optional[TurnBasedTrigger] = None
        self.token_trigger: Optional[TokenBasedTrigger] = None

        if self.config.turn_trigger:
            self.turn_trigger = TurnBasedTrigger(self.config.turn_trigger)

        if self.config.token_trigger:
            self.token_trigger = TokenBasedTrigger(self.config.token_trigger)
    
    def should_compress(self, store: MemoryStore) -> bool:
        """Check if compression should be triggered.
        
        Args:
            store: The memory store.
            
        Returns:
            True if compression should be triggered.
        """
        turn_should = (
            self.turn_trigger.should_compress(store)
            if self.turn_trigger else False
        )
        
        token_should = (
            self.token_trigger.should_compress(store)
            if self.token_trigger else False
        )
        
        if self.config.require_both:
            # Both triggers must fire
            if self.turn_trigger and self.token_trigger:
                return turn_should and token_should
            elif self.turn_trigger:
                return turn_should
            elif self.token_trigger:
                return token_should
            return False
        else:
            # Either trigger can fire
            return turn_should or token_should
    
    def mark_compressed(self, store: MemoryStore) -> None:
        """Mark that compression was performed.
        
        Args:
            store: The memory store.
        """
        if self.turn_trigger:
            self.turn_trigger.mark_triggered(store)
        if self.token_trigger:
            self.token_trigger.mark_triggered(store)
    
    def get_status(self, store: MemoryStore) -> Dict[str, Any]:
        """Get current status of all triggers.
        
        Args:
            store: The memory store.
            
        Returns:
            Dictionary with trigger status.
        """
        status = {
            "should_compress": self.should_compress(store),
            "require_both": self.config.require_both,
        }
        
        if self.turn_trigger:
            status["turn_trigger"] = {
                **self.turn_trigger.get_trigger_info(),
                "should_trigger": self.turn_trigger.should_compress(store),
            }
        
        if self.token_trigger:
            status["token_trigger"] = {
                **self.token_trigger.get_trigger_info(),
                "current_tokens": store.token_total(),
                "should_trigger": self.token_trigger.should_compress(store),
            }
        
        return status
    
    def get_stats(self, store: MemoryStore) -> Dict[str, Any]:
        """Get statistics about the store and triggers.
        
        Args:
            store: The memory store.
            
        Returns:
            Dictionary with statistics.
        """
        max_turn = max(
            (entry.turn_number for entry in store.entries),
            default=0
        )
        
        return {
            "total_entries": len(store),
            "total_tokens": store.token_total(),
            "max_turn": max_turn,
            "should_compress": self.should_compress(store),
        }


def create_forgetting_curve(
    max_turns: int = 50,
    token_threshold: int = 4000,
    require_both: bool = False,
    **kwargs
) -> ForgettingCurve:
    """Factory function to create a ForgettingCurve.
    
    Args:
        max_turns: Maximum turns before triggering (None to disable).
        token_threshold: Token threshold for triggering (None to disable).
        require_both: Whether both triggers must fire.
        **kwargs: Additional configuration options.
        
    Returns:
        Configured ForgettingCurve instance.
    """
    config = ForgettingCurveConfig(
        turn_trigger=TurnBasedTriggerConfig(max_turns=max_turns) if max_turns else None,
        token_trigger=TokenBasedTriggerConfig(token_threshold=token_threshold) if token_threshold else None,
        require_both=require_both,
    )
    return ForgettingCurve(config)
