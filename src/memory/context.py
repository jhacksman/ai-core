"""
Context Manager - Intelligent context pruning, retrieval, and relevance scoring.

This module provides intelligent context management for the Venice.ai scaffolding
system with dynamic context pruning and relevance scoring capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from .vector_store import VectorStore, VectorDocument, SearchResult
from .long_term_memory import LongTermMemory, Memory, MemoryType, MemoryImportance
from ..venice.client import VeniceClient

logger = logging.getLogger(__name__)


class ContextRelevance(Enum):
    """Context relevance levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"


@dataclass
class ContextItem:
    """Represents a context item with relevance scoring."""
    item_id: str
    content: str
    context_type: str
    relevance_score: float
    importance: MemoryImportance
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_memory_id: Optional[str] = None


@dataclass
class ContextWindow:
    """Represents a context window with size constraints."""
    window_id: str
    items: List[ContextItem]
    max_tokens: int
    current_tokens: int
    priority_threshold: float
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """
    Intelligent context management with pruning and relevance scoring.
    
    Manages context windows for the Venice.ai scaffolding system with
    intelligent pruning based on relevance, recency, and importance.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        long_term_memory: LongTermMemory,
        vector_store: VectorStore,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the context manager.
        
        Args:
            venice_client: Venice.ai client for analysis
            long_term_memory: Long-term memory system
            vector_store: Vector store for semantic search
            config: Configuration options
        """
        self.venice_client = venice_client
        self.long_term_memory = long_term_memory
        self.vector_store = vector_store
        self.config = config or {}
        
        self.context_windows: Dict[str, ContextWindow] = {}
        self.active_contexts: Dict[str, List[ContextItem]] = {}
        
        self.max_context_tokens = self.config.get("max_context_tokens", 32000)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.3)
        self.decay_factor = self.config.get("decay_factor", 0.95)
        
        self.stats = {
            "contexts_created": 0,
            "items_pruned": 0,
            "average_relevance_score": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize the context manager."""
        logger.info("Initializing Context Manager")
        
        try:
            logger.info("Context Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Context Manager: {e}")
            raise
    
    async def create_context_window(
        self,
        window_id: str,
        max_tokens: int,
        priority_threshold: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextWindow:
        """
        Create a new context window.
        
        Args:
            window_id: Unique window identifier
            max_tokens: Maximum tokens for this window
            priority_threshold: Minimum priority for inclusion
            metadata: Additional metadata
            
        Returns:
            Created context window
        """
        try:
            context_window = ContextWindow(
                window_id=window_id,
                items=[],
                max_tokens=max_tokens,
                current_tokens=0,
                priority_threshold=priority_threshold,
                metadata=metadata or {}
            )
            
            self.context_windows[window_id] = context_window
            self.active_contexts[window_id] = []
            
            self.stats["contexts_created"] += 1
            
            logger.debug(f"Created context window: {window_id}")
            return context_window
            
        except Exception as e:
            logger.error(f"Failed to create context window: {e}")
            raise
    
    async def add_context_item(
        self,
        window_id: str,
        content: str,
        context_type: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_memory_id: Optional[str] = None
    ) -> str:
        """
        Add an item to a context window.
        
        Args:
            window_id: Target window ID
            content: Context content
            context_type: Type of context
            importance: Importance level
            tags: Associated tags
            metadata: Additional metadata
            source_memory_id: Source memory ID if applicable
            
        Returns:
            Context item ID
        """
        try:
            if window_id not in self.context_windows:
                raise ValueError(f"Context window {window_id} not found")
            
            item_id = self._generate_item_id()
            
            relevance_score = await self._calculate_relevance_score(
                content, context_type, importance
            )
            
            context_item = ContextItem(
                item_id=item_id,
                content=content,
                context_type=context_type,
                relevance_score=relevance_score,
                importance=importance,
                tags=tags or [],
                metadata=metadata or {},
                source_memory_id=source_memory_id
            )
            
            window = self.context_windows[window_id]
            
            estimated_tokens = await self._estimate_tokens(content)
            
            if window.current_tokens + estimated_tokens > window.max_tokens:
                await self._prune_context_window(window_id, estimated_tokens)
            
            window.items.append(context_item)
            window.current_tokens += estimated_tokens
            self.active_contexts[window_id].append(context_item)
            
            logger.debug(f"Added context item {item_id} to window {window_id}")
            return item_id
            
        except Exception as e:
            logger.error(f"Failed to add context item: {e}")
            raise
    
    async def retrieve_relevant_context(
        self,
        query: str,
        window_id: Optional[str] = None,
        max_items: int = 10,
        relevance_threshold: Optional[float] = None
    ) -> List[ContextItem]:
        """
        Retrieve relevant context items based on query.
        
        Args:
            query: Search query
            window_id: Specific window to search (optional)
            max_items: Maximum items to return
            relevance_threshold: Minimum relevance score
            
        Returns:
            List of relevant context items
        """
        try:
            threshold = relevance_threshold or self.relevance_threshold
            relevant_items = []
            
            search_windows = [window_id] if window_id else list(self.context_windows.keys())
            
            for wid in search_windows:
                if wid in self.active_contexts:
                    for item in self.active_contexts[wid]:
                        semantic_score = await self._calculate_semantic_similarity(
                            query, item.content
                        )
                        
                        combined_score = (
                            semantic_score * 0.6 + 
                            item.relevance_score * 0.4
                        )
                        
                        if combined_score >= threshold:
                            item.relevance_score = combined_score
                            item.last_accessed = datetime.now()
                            item.access_count += 1
                            relevant_items.append(item)
            
            relevant_items.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return relevant_items[:max_items]
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant context: {e}")
            return []
    
    async def prune_context(
        self,
        window_id: str,
        target_reduction: Optional[int] = None
    ) -> int:
        """
        Prune context items from a window.
        
        Args:
            window_id: Window to prune
            target_reduction: Target token reduction (optional)
            
        Returns:
            Number of items pruned
        """
        try:
            if window_id not in self.context_windows:
                return 0
            
            return await self._prune_context_window(window_id, target_reduction)
            
        except Exception as e:
            logger.error(f"Failed to prune context: {e}")
            return 0
    
    async def get_context_summary(self, window_id: str) -> Dict[str, Any]:
        """Get summary of context window."""
        if window_id not in self.context_windows:
            return {}
        
        window = self.context_windows[window_id]
        items = self.active_contexts.get(window_id, [])
        
        return {
            "window_id": window_id,
            "total_items": len(items),
            "current_tokens": window.current_tokens,
            "max_tokens": window.max_tokens,
            "utilization": window.current_tokens / window.max_tokens,
            "average_relevance": sum(item.relevance_score for item in items) / len(items) if items else 0.0,
            "context_types": list(set(item.context_type for item in items))
        }
    
    async def cleanup(self) -> None:
        """Cleanup context manager resources."""
        logger.info("Cleaning up Context Manager")
        
        try:
            for window_id in list(self.context_windows.keys()):
                await self._archive_context_window(window_id)
            
            logger.info("Context Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Context Manager cleanup: {e}")
    
    async def _prune_context_window(
        self,
        window_id: str,
        target_reduction: Optional[int] = None
    ) -> int:
        """Prune items from a context window."""
        window = self.context_windows[window_id]
        items = self.active_contexts[window_id]
        
        if not items:
            return 0
        
        target_tokens = target_reduction or (window.max_tokens * 0.2)
        
        items_with_scores = []
        for item in items:
            pruning_score = await self._calculate_pruning_score(item)
            items_with_scores.append((item, pruning_score))
        
        items_with_scores.sort(key=lambda x: x[1])
        
        pruned_count = 0
        tokens_freed = 0
        
        for item, score in items_with_scores:
            if tokens_freed >= target_tokens:
                break
            
            item_tokens = await self._estimate_tokens(item.content)
            
            if score < window.priority_threshold:
                items.remove(item)
                window.items.remove(item)
                window.current_tokens -= item_tokens
                tokens_freed += item_tokens
                pruned_count += 1
                
                await self._archive_context_item(item)
        
        self.stats["items_pruned"] += pruned_count
        
        logger.debug(f"Pruned {pruned_count} items from window {window_id}")
        return pruned_count
    
    async def _calculate_relevance_score(
        self,
        content: str,
        context_type: str,
        importance: MemoryImportance
    ) -> float:
        """Calculate relevance score for context item."""
        base_score = 0.5
        
        importance_boost = {
            MemoryImportance.CRITICAL: 0.4,
            MemoryImportance.HIGH: 0.3,
            MemoryImportance.MEDIUM: 0.2,
            MemoryImportance.LOW: 0.1
        }
        
        type_boost = {
            "current_task": 0.3,
            "recent_memory": 0.2,
            "tool_usage": 0.15,
            "research": 0.1,
            "general": 0.0
        }
        
        relevance = (
            base_score + 
            importance_boost.get(importance, 0.0) + 
            type_boost.get(context_type, 0.0)
        )
        
        return min(1.0, relevance)
    
    async def _calculate_pruning_score(self, item: ContextItem) -> float:
        """Calculate pruning score (lower = more likely to be pruned)."""
        recency_factor = self._calculate_recency_factor(item.last_accessed)
        access_factor = min(1.0, item.access_count / 10.0)
        
        pruning_score = (
            item.relevance_score * 0.4 +
            recency_factor * 0.3 +
            access_factor * 0.3
        )
        
        return pruning_score
    
    async def _calculate_semantic_similarity(
        self,
        query: str,
        content: str
    ) -> float:
        """Calculate semantic similarity between query and content."""
        try:
            search_results = await self.vector_store.search(
                query=query,
                n_results=1
            )
            
            if search_results and search_results[0].document.content == content:
                return search_results[0].similarity_score
            
            return 0.3
            
        except Exception:
            return 0.3
    
    def _calculate_recency_factor(self, last_accessed: datetime) -> float:
        """Calculate recency factor based on last access time."""
        time_diff = datetime.now() - last_accessed
        hours_ago = time_diff.total_seconds() / 3600
        
        if hours_ago < 1:
            return 1.0
        elif hours_ago < 24:
            return 0.8
        elif hours_ago < 168:  # 1 week
            return 0.5
        else:
            return 0.2
    
    async def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        return len(content.split()) * 1.3
    
    async def _archive_context_item(self, item: ContextItem) -> None:
        """Archive a context item to long-term memory."""
        try:
            await self.long_term_memory.store_memory(
                content=item.content,
                memory_type=MemoryType.CONVERSATION,
                importance=item.importance,
                tags=item.tags + ["archived_context"],
                metadata={
                    "original_item_id": item.item_id,
                    "context_type": item.context_type,
                    "relevance_score": item.relevance_score,
                    "access_count": item.access_count
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to archive context item: {e}")
    
    async def _archive_context_window(self, window_id: str) -> None:
        """Archive a context window."""
        try:
            if window_id in self.active_contexts:
                for item in self.active_contexts[window_id]:
                    await self._archive_context_item(item)
                
                del self.active_contexts[window_id]
                del self.context_windows[window_id]
            
        except Exception as e:
            logger.error(f"Failed to archive context window: {e}")
    
    def _generate_item_id(self) -> str:
        """Generate unique context item ID."""
        timestamp = int(datetime.now().timestamp())
        return f"context_{timestamp}_{len(self.active_contexts)}"
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get context management statistics."""
        total_items = sum(len(items) for items in self.active_contexts.values())
        
        return {
            "active_windows": len(self.context_windows),
            "total_active_items": total_items,
            "performance_stats": self.stats.copy()
        }
