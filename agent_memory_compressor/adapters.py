"""Agent-session-manager adapter: compress_session utility."""

from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass

from .models import MemoryEntry, MemoryStore
from .orchestrator import MemoryCompressor, CompressionReport
from .context import ContextBuilder


@runtime_checkable
class SessionProtocol(Protocol):
    """Protocol for session objects that can be compressed."""
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages from the session."""
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get session metadata."""
        ...


@dataclass
class SessionAdapterConfig:
    """Configuration for session adapter."""
    token_budget: int = 4000
    protected_recent: int = 3
    include_system_messages: bool = True
    preserve_turn_order: bool = True


class SessionAdapter:
    """Adapter for integrating with agent session managers."""
    
    def __init__(self, config: Optional[SessionAdapterConfig] = None):
        """Initialize session adapter.
        
        Args:
            config: Configuration for the adapter.
        """
        self.config = config or SessionAdapterConfig()
        self.compressor = MemoryCompressor(
            token_budget=self.config.token_budget,
            protected_recent=self.config.protected_recent
        )
        self.context_builder = ContextBuilder()
    
    def session_to_store(self, session: SessionProtocol) -> MemoryStore:
        """Convert a session to a MemoryStore.
        
        Args:
            session: Session object with get_messages() method.
            
        Returns:
            MemoryStore populated from session.
        """
        store = MemoryStore()
        messages = session.get_messages()
        
        for i, msg in enumerate(messages):
            entry = MemoryEntry(
                content=msg.get("content", ""),
                role=msg.get("role", "user"),
                turn_number=i + 1,
                metadata={"session_metadata": session.get_metadata()}
            )
            store.add_entry(entry)
        
        return store
    
    def store_to_session(
        self,
        store: MemoryStore,
        original_session: SessionProtocol
    ) -> List[Dict[str, str]]:
        """Convert a MemoryStore back to session messages.
        
        Args:
            store: MemoryStore to convert.
            original_session: Original session for reference.
            
        Returns:
            List of message dictionaries.
        """
        messages = []
        
        for entry in store.entries:
            if entry.role == "compressed":
                # Handle compressed entries
                messages.append({
                    "role": entry.metadata.get("original_role", "assistant"),
                    "content": entry.content,
                    "compressed": True,
                    "compression_history": entry.compression_history
                })
            else:
                messages.append({
                    "role": entry.role,
                    "content": entry.content
                })
        
        return messages
    
    def compress_session(
        self,
        session: SessionProtocol,
        target_budget: Optional[int] = None
    ) -> tuple[List[Dict[str, str]], CompressionReport]:
        """Compress a session's message history.
        
        Args:
            session: Session object to compress.
            target_budget: Optional override for token budget.
            
        Returns:
            Tuple of (compressed_messages, compression_report).
        """
        # Convert session to store
        store = self.session_to_store(session)
        
        # Compress
        report = self.compressor.compress(store, target_budget)
        
        # Convert back to messages
        compressed_messages = self.store_to_session(store, session)
        
        return compressed_messages, report
    
    def get_session_context(
        self,
        session: SessionProtocol,
        system_message: Optional[str] = None
    ) -> str:
        """Get formatted context for a session.
        
        Args:
            session: Session object.
            system_message: Optional system message.
            
        Returns:
            Formatted context string.
        """
        store = self.session_to_store(session)
        return self.context_builder.build_context(store, system_message)


class MockSession:
    """Mock session for testing."""
    
    def __init__(self, messages: Optional[List[Dict[str, str]]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Initialize mock session.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            metadata: Session metadata.
        """
        self._messages = messages or []
        self._metadata = metadata or {}
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages."""
        return self._messages
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata."""
        return self._metadata
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message."""
        self._messages.append({"role": role, "content": content})


def compress_session(
    session: SessionProtocol,
    token_budget: int = 4000,
    protected_recent: int = 3,
    target_budget: Optional[int] = None
) -> tuple[List[Dict[str, str]], CompressionReport]:
    """Utility function to compress a session.
    
    Args:
        session: Session object with get_messages() method.
        token_budget: Maximum tokens allowed.
        protected_recent: Number of recent messages to protect.
        target_budget: Optional override for token budget.
        
    Returns:
        Tuple of (compressed_messages, compression_report).
    """
    config = SessionAdapterConfig(
        token_budget=token_budget,
        protected_recent=protected_recent
    )
    adapter = SessionAdapter(config)
    return adapter.compress_session(session, target_budget)


def create_mock_session(
    num_messages: int = 10,
    message_length: int = 50
) -> MockSession:
    """Create a mock session for testing.
    
    Args:
        num_messages: Number of messages to create.
        message_length: Approximate length of each message.
        
    Returns:
        MockSession with generated messages.
    """
    messages = []
    for i in range(num_messages):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i}: " + "x" * message_length
        messages.append({"role": role, "content": content})
    
    return MockSession(
        messages=messages,
        metadata={"session_id": "test_session", "created_at": "2024-01-01"}
    )
