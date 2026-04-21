"""Compression strategies: summarize, extract_facts, and archive using OpenAI-compatible LLM calls."""

import json
from typing import List, Dict, Any, Optional, Callable, Literal
from dataclasses import dataclass
from enum import Enum

from .models import MemoryEntry, MemoryStore


class CompressionStrategy(Enum):
    """Available compression strategies."""
    SUMMARIZE = "summarize"
    EXTRACT_FACTS = "extract_facts"
    ARCHIVE = "archive"


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_entry: MemoryEntry
    compressed_entry: Optional[MemoryEntry]
    strategy: CompressionStrategy
    tokens_saved: int
    success: bool
    error: Optional[str] = None


class LLMClient:
    """OpenAI-compatible LLM client for compression operations."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        mock_responses: Optional[Dict[str, str]] = None
    ):
        """Initialize LLM client.
        
        Args:
            api_key: API key for the LLM service.
            base_url: Base URL for the API (defaults to OpenAI).
            model: Model name to use.
            mock_responses: For testing - map of prompts to mock responses.
        """
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model
        self.mock_responses = mock_responses or {}
        self._client = None
    
    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
        return self._client
    
    def complete(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> str:
        """Get completion from LLM.
        
        Args:
            prompt: The user prompt.
            system_message: Optional system message.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            
        Returns:
            The LLM response text.
        """
        # Check for mock response first (for testing)
        if prompt in self.mock_responses:
            return self.mock_responses[prompt]
        
        # If no API key, return a fallback response
        if not self.api_key:
            return self._fallback_response(prompt, system_message)
        
        try:
            client = self._get_client()
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback on error
            return self._fallback_response(prompt, system_message)
    
    def _fallback_response(self, prompt: str, system_message: Optional[str]) -> str:
        """Generate a simple fallback response when LLM is unavailable."""
        if "summarize" in (system_message or "").lower():
            return "[Summary of conversation]"
        elif "fact" in (system_message or "").lower():
            return '["Key fact extracted"]'
        elif "archive" in (system_message or "").lower():
            return "[Archived content]"
        return "[Compressed content]"


class CompressionEngine:
    """Engine for compressing memory entries using various strategies."""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_summary_tokens: int = 100,
        max_facts: int = 5
    ):
        """Initialize compression engine.
        
        Args:
            llm_client: LLM client for compression operations.
            max_summary_tokens: Maximum tokens for summaries.
            max_facts: Maximum facts to extract.
        """
        self.llm = llm_client or LLMClient()
        self.max_summary_tokens = max_summary_tokens
        self.max_facts = max_facts
    
    def _create_compressed_entry(
        self,
        original: MemoryEntry,
        new_content: str,
        strategy: CompressionStrategy,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryEntry:
        """Create a compressed entry preserving original metadata."""
        compression_record = {
            "strategy": strategy.value,
            "original_content": original.content,
            "timestamp": __import__('time').time(),
            "metadata": metadata or {}
        }
        
        return MemoryEntry(
            content=new_content,
            role="compressed",
            turn_number=original.turn_number,
            timestamp=original.timestamp,
            metadata={
                **original.metadata,
                "original_role": original.role,
                "compression_strategy": strategy.value,
            },
            compression_history=[
                *original.compression_history,
                compression_record
            ]
        )
    
    def summarize(self, entry: MemoryEntry) -> CompressionResult:
        """Summarize a memory entry.
        
        Args:
            entry: Entry to summarize.
            
        Returns:
            CompressionResult with the summarized entry.
        """
        system_msg = "You are a helpful assistant that summarizes text concisely."
        prompt = f"Summarize the following in 1-2 sentences:\n\n{entry.content}"
        
        try:
            summary = self.llm.complete(
                prompt=prompt,
                system_message=system_msg,
                max_tokens=self.max_summary_tokens
            )
            
            compressed = self._create_compressed_entry(
                entry, summary, CompressionStrategy.SUMMARIZE
            )
            
            # Calculate tokens saved
            original_tokens = len(entry.content.split())
            compressed_tokens = len(summary.split())
            tokens_saved = max(0, original_tokens - compressed_tokens)
            
            return CompressionResult(
                original_entry=entry,
                compressed_entry=compressed,
                strategy=CompressionStrategy.SUMMARIZE,
                tokens_saved=tokens_saved,
                success=True
            )
        except Exception as e:
            return CompressionResult(
                original_entry=entry,
                compressed_entry=None,
                strategy=CompressionStrategy.SUMMARIZE,
                tokens_saved=0,
                success=False,
                error=str(e)
            )
    
    def extract_facts(self, entry: MemoryEntry) -> CompressionResult:
        """Extract key facts from a memory entry.
        
        Args:
            entry: Entry to extract facts from.
            
        Returns:
            CompressionResult with facts as bullet points.
        """
        system_msg = "You are a helpful assistant that extracts key facts."
        prompt = (
            f"Extract up to {self.max_facts} key facts from the following text. "
            f"Return as a JSON array of strings:\n\n{entry.content}"
        )
        
        try:
            response = self.llm.complete(
                prompt=prompt,
                system_message=system_msg,
                max_tokens=self.max_summary_tokens * 2
            )
            
            # Try to parse JSON, fallback to bullet points
            try:
                facts = json.loads(response)
                if isinstance(facts, list):
                    facts_text = "\n".join(f"• {fact}" for fact in facts)
                else:
                    facts_text = f"• {response}"
            except json.JSONDecodeError:
                # Use response as-is, formatted as bullet points
                facts_text = "\n".join(f"• {line.strip()}" for line in response.split("\n") if line.strip())
            
            compressed = self._create_compressed_entry(
                entry, facts_text, CompressionStrategy.EXTRACT_FACTS,
                metadata={"num_facts": facts_text.count("•")}
            )
            
            original_tokens = len(entry.content.split())
            compressed_tokens = len(facts_text.split())
            tokens_saved = max(0, original_tokens - compressed_tokens)
            
            return CompressionResult(
                original_entry=entry,
                compressed_entry=compressed,
                strategy=CompressionStrategy.EXTRACT_FACTS,
                tokens_saved=tokens_saved,
                success=True
            )
        except Exception as e:
            return CompressionResult(
                original_entry=entry,
                compressed_entry=None,
                strategy=CompressionStrategy.EXTRACT_FACTS,
                tokens_saved=0,
                success=False,
                error=str(e)
            )
    
    def archive(self, entry: MemoryEntry) -> CompressionResult:
        """Archive a memory entry (minimal retention).
        
        Args:
            entry: Entry to archive.
            
        Returns:
            CompressionResult with archived reference.
        """
        # Create a minimal archive reference
        archive_content = f"[Archived {entry.role} message from turn {entry.turn_number}]"
        
        compressed = self._create_compressed_entry(
            entry, archive_content, CompressionStrategy.ARCHIVE,
            metadata={"archived": True}
        )
        
        original_tokens = len(entry.content.split())
        compressed_tokens = len(archive_content.split())
        tokens_saved = max(0, original_tokens - compressed_tokens)
        
        return CompressionResult(
            original_entry=entry,
            compressed_entry=compressed,
            strategy=CompressionStrategy.ARCHIVE,
            tokens_saved=tokens_saved,
            success=True
        )
    
    def compress(
        self,
        entry: MemoryEntry,
        strategy: CompressionStrategy = CompressionStrategy.SUMMARIZE
    ) -> CompressionResult:
        """Compress an entry using the specified strategy.
        
        Args:
            entry: Entry to compress.
            strategy: Compression strategy to use.
            
        Returns:
            CompressionResult.
        """
        if strategy == CompressionStrategy.SUMMARIZE:
            return self.summarize(entry)
        elif strategy == CompressionStrategy.EXTRACT_FACTS:
            return self.extract_facts(entry)
        elif strategy == CompressionStrategy.ARCHIVE:
            return self.archive(entry)
        else:
            return CompressionResult(
                original_entry=entry,
                compressed_entry=None,
                strategy=strategy,
                tokens_saved=0,
                success=False,
                error=f"Unknown strategy: {strategy}"
            )
    
    def compress_batch(
        self,
        entries: List[MemoryEntry],
        strategy: CompressionStrategy = CompressionStrategy.SUMMARIZE
    ) -> List[CompressionResult]:
        """Compress multiple entries.
        
        Args:
            entries: Entries to compress.
            strategy: Compression strategy to use.
            
        Returns:
            List of CompressionResults.
        """
        return [self.compress(entry, strategy) for entry in entries]


def create_engine(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    mock_responses: Optional[Dict[str, str]] = None,
    max_summary_tokens: int = 100,
    max_facts: int = 5
) -> CompressionEngine:
    """Factory function to create a CompressionEngine.
    
    Args:
        api_key: API key for LLM service.
        base_url: Base URL for API.
        model: Model name.
        mock_responses: Mock responses for testing.
        max_summary_tokens: Max tokens for summaries.
        max_facts: Max facts to extract.
        
    Returns:
        Configured CompressionEngine.
    """
    llm = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        mock_responses=mock_responses
    )
    return CompressionEngine(
        llm_client=llm,
        max_summary_tokens=max_summary_tokens,
        max_facts=max_facts
    )
