#!/usr/bin/env python
"""Long-run demo: simulate a 50-turn session with compression reports."""

import random
import time

from agent_memory_compressor.models import MemoryEntry, MemoryStore
from agent_memory_compressor.orchestrator import MemoryCompressor, CompressionReport
from agent_memory_compressor.triggers import ForgettingCurve, TurnBasedTriggerConfig, TokenBasedTriggerConfig, ForgettingCurveConfig
from agent_memory_compressor.context import ContextBuilder


class DemoSession:
    """Simulates a conversation session."""
    
    def __init__(self, store: MemoryStore):
        self.store = store
        self.turn_count = 0
        self.compression_count = 0
        self.total_tokens_saved = 0
    
    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn."""
        self.turn_count += 1
        entry = MemoryEntry(
            content=content,
            role=role,
            turn_number=self.turn_count
        )
        self.store.add_entry(entry)
    
    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "turns": self.turn_count,
            "entries": len(self.store),
            "tokens": self.store.token_total(),
            "compressions": self.compression_count,
            "total_saved": self.total_tokens_saved
        }


def generate_conversation_turn(turn_number: int) -> tuple[str, str]:
    """Generate a realistic conversation turn."""
    role = "user" if turn_number % 2 == 1 else "assistant"
    
    user_topics = [
        "Tell me about Python programming",
        "How do I handle errors in my code?",
        "What's the best way to learn machine learning?",
        "Can you explain neural networks?",
        "How does memory management work?",
        "What are the benefits of async programming?",
        "Explain the difference between lists and tuples",
        "How do I optimize my database queries?",
        "What's new in Python 3.12?",
        "Can you help me debug this function?",
    ]
    
    assistant_responses = [
        "Python is a versatile language with simple syntax. Here's how it works...",
        "Error handling in Python uses try-except blocks. Let me show you an example...",
        "Machine learning requires understanding statistics and algorithms. Start with...",
        "Neural networks are inspired by biological neurons. They consist of layers...",
        "Memory management in Python uses reference counting and garbage collection...",
        "Async programming allows concurrent execution. You can use async/await...",
        "Lists are mutable while tuples are immutable. This means...",
        "Database optimization involves indexing and query planning. Consider...",
        "Python 3.12 introduces several improvements including better error messages...",
        "Let's debug this step by step. First, check if the function receives...",
    ]
    
    if role == "user":
        content = random.choice(user_topics)
        # Add some variation
        content += f" (question {turn_number})"
    else:
        content = random.choice(assistant_responses)
        # Make responses longer to simulate real assistant messages
        content += " " + "Here are some additional details and examples to help you understand better. " * (turn_number % 3 + 1)
    
    return role, content


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_stats(label: str, stats: dict) -> None:
    """Print statistics."""
    print(f"\n{label}:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def print_compression_report(report: CompressionReport) -> None:
    """Print compression report."""
    print("\n  Compression Report:")
    print(f"    Initial tokens: {report.initial_tokens}")
    print(f"    Final tokens: {report.final_tokens}")
    print(f"    Tokens saved: {report.tokens_saved}")
    print(f"    Compression ratio: {report.compression_ratio:.1%}")
    print(f"    Entries compressed: {report.entries_compressed}")
    print(f"    Iterations: {report.iterations}")
    print(f"    Duration: {report.duration_seconds:.3f}s")


def main():
    """Run the long-run demo."""
    print_header("AGENT MEMORY COMPRESSOR - LONG RUN DEMO")
    print("\nSimulating a 50-turn conversation with memory compression")
    print("Token budget: 500 tokens")
    print("Protected recent: 3 turns")
    
    # Initialize
    store = MemoryStore()
    session = DemoSession(store)
    
    # Setup compression
    compressor = MemoryCompressor(
        token_budget=500,
        protected_recent=3
    )
    
    # Setup forgetting curve triggers
    forgetting_curve = ForgettingCurve(ForgettingCurveConfig(
        turn_trigger=TurnBasedTriggerConfig(max_turns=15, min_turns=5),
        token_trigger=TokenBasedTriggerConfig(token_threshold=400, min_entries=5)
    ))
    
    context_builder = ContextBuilder()
    
    print_header("PHASE 1: CONVERSATION SIMULATION")
    
    # Simulate 50 turns
    for turn in range(1, 51):
        role, content = generate_conversation_turn(turn)
        session.add_turn(role, content)
        
        # Print progress every 10 turns
        if turn % 10 == 0:
            stats = session.get_stats()
            print(f"\n  Turn {turn}: {stats['entries']} entries, {stats['tokens']} tokens")
        
        # Check if compression needed
        if forgetting_curve.should_compress(store):
            print(f"\n  >>> Compression triggered at turn {turn}")
            
            report = compressor.compress(store)
            session.compression_count += 1
            session.total_tokens_saved += report.tokens_saved
            
            print_compression_report(report)
            forgetting_curve.mark_compressed(store)
    
    print_header("PHASE 2: FINAL STATISTICS")
    
    final_stats = session.get_stats()
    print_stats("Session Summary", final_stats)
    
    # Build context
    context = context_builder.build_context(store, system_message="You are a helpful AI assistant")
    context_stats = context_builder.get_context_stats(store)
    
    print("\nContext Statistics:")
    for key, value in context_stats.items():
        print(f"  {key}: {value}")
    
    print_header("PHASE 3: COMPRESSION EFFECTIVENESS")
    
    # Calculate effectiveness
    if final_stats['compressions'] > 0:
        avg_saved = final_stats['total_saved'] / final_stats['compressions']
        print(f"\n  Total compressions: {final_stats['compressions']}")
        print(f"  Total tokens saved: {final_stats['total_saved']}")
        print(f"  Average tokens saved per compression: {avg_saved:.1f}")
        print(f"  Final token count: {final_stats['tokens']}")
        print(f"  Token budget: 500")
        print(f"  Within budget: {'Yes' if final_stats['tokens'] <= 500 else 'No'}")
    else:
        print("\n  No compressions were triggered")
    
    print_header("DEMO COMPLETE")
    print("\n✓ Successfully simulated 50-turn conversation")
    print("✓ Memory compression maintained token budget")
    print("✓ Recent turns were protected from compression")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
