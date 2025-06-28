"""
Long-term Memory - Persistent memory retrieval and experience storage.

This module provides long-term memory capabilities for the Venice.ai scaffolding
system with persistent storage and intelligent retrieval mechanisms.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import pickle
from pathlib import Path

from .vector_store import VectorStore, VectorDocument, SearchResult
from ..venice.client import VeniceClient

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories stored in long-term memory."""
    EXPERIENCE = "experience"
    OBSERVATION = "observation"
    RESEARCH = "research"
    REFLECTION = "reflection"


class MemoryImportance(Enum):
    """Importance levels for memories."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class MemoryEntry:
    """Represents a memory entry in the long-term memory system."""
    content: str
    memory_type: MemoryType
    importance: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory entry to dictionary."""
        return {
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create memory entry from dictionary."""
        return cls(
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            importance=data["importance"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class Memory:
    """Represents a memory in the long-term memory system."""
    memory_id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_memories: List[str] = field(default_factory=list)
    decay_factor: float = 1.0


@dataclass
class MemoryRetrievalResult:
    """Result of memory retrieval operation."""
    memory: Memory
    relevance_score: float
    retrieval_reason: str
    context_match: float = 0.0


class LongTermMemory:
    """
    Long-term memory system with persistent storage and intelligent retrieval.
    
    Provides sophisticated memory management for the Venice.ai scaffolding system
    including memory consolidation, decay, and pattern recognition.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize the long-term memory system.
        
        Args:
            config: Configuration options
            vector_store: Vector store for semantic search
        """
        self.config = config or {}
        self.vector_store = vector_store
        self.storage_path = Path(self.config.get("storage_path", "./memory_storage"))
        
        self.memories: Dict[str, Memory] = {}
        
        self.retention_days = self.config.get("retention_days", 90)
        self.importance_threshold = self.config.get("importance_threshold", 0.3)
        self.auto_cleanup = self.config.get("auto_cleanup", True)
        self.backup_interval_hours = self.config.get("backup_interval_hours", 24)
        
        self.max_memories = self.config.get("max_memories", 100000)
        self.decay_rate = self.config.get("decay_rate", 0.01)
        self.retrieval_limit = self.config.get("retrieval_limit", 20)
        
        self.stats = {
            "memories_stored": 0,
            "memories_retrieved": 0,
            "average_retrieval_time": 0.0
        }
        
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> None:
        """Initialize the long-term memory system."""
        logger.info("Initializing Long-term Memory")
        
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            await self._load_memories()
            
            logger.info(f"Long-term Memory initialized with {len(self.memories)} memories")
            
        except Exception as e:
            logger.error(f"Failed to initialize Long-term Memory: {e}")
            raise
    
    async def store_memory(self, memory_entry: MemoryEntry) -> Optional[str]:
        """
        Store a new memory in long-term memory.
        
        Args:
            memory_entry: Memory entry to store
            
        Returns:
            Memory ID if stored, None if below threshold
        """
        try:
            if memory_entry.importance < self.importance_threshold:
                return None
            
            memory_id = self._generate_memory_id()
            
            if self.vector_store:
                await self.vector_store.add_document(
                    document_id=memory_id,
                    content=memory_entry.content,
                    metadata={
                        "memory_type": memory_entry.memory_type.value,
                        "importance": memory_entry.importance,
                        "timestamp": memory_entry.timestamp.isoformat(),
                        **memory_entry.metadata
                    }
                )
            
            self.stats["memories_stored"] += 1
            
            logger.debug(f"Stored memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def retrieve_memories(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        min_importance: Optional[float] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            memory_type: Filter by memory type
            min_importance: Minimum importance threshold
            time_range: Time range filter
            
        Returns:
            List of memory search results
        """
        start_time = datetime.now()
        
        try:
            if not self.vector_store:
                return []
            
            filters = {}
            if memory_type:
                filters["memory_type"] = memory_type.value
            if min_importance:
                filters["importance"] = {"$gte": min_importance}
            if time_range:
                filters["timestamp"] = {
                    "$gte": time_range[0].isoformat(),
                    "$lte": time_range[1].isoformat()
                }
            
            search_results = await self.vector_store.search(
                query=query,
                limit=limit,
                filters=filters if filters else None
            )
            
            retrieval_time = (datetime.now() - start_time).total_seconds()
            self.stats["memories_retrieved"] += len(search_results)
            self._update_average_retrieval_time(retrieval_time)
            
            logger.debug(f"Retrieved {len(search_results)} memories in {retrieval_time:.3f}s")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories for query '{query}': {e}")
            return []
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        memory = self.memories.get(memory_id)
        if memory:
            memory.last_accessed = datetime.now()
            memory.access_count += 1
        return memory
    
    async def cleanup(self) -> None:
        """Cleanup long-term memory resources."""
        logger.info("Cleaning up Long-term Memory")
        
        try:
            self._shutdown_event.set()
            
            await self._save_memories()
            
            logger.info("Long-term Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Long-term Memory cleanup: {e}")
    
    async def _load_memories(self) -> None:
        """Load memories from persistent storage."""
        try:
            memories_file = self.storage_path / "memories.pkl"
            
            if memories_file.exists():
                with open(memories_file, 'rb') as f:
                    self.memories = pickle.load(f)
                logger.debug(f"Loaded {len(self.memories)} memories from storage")
            
        except Exception as e:
            logger.error(f"Failed to load memories from storage: {e}")
    
    async def _save_memories(self) -> None:
        """Save memories to persistent storage."""
        try:
            memories_file = self.storage_path / "memories.pkl"
            
            with open(memories_file, 'wb') as f:
                pickle.dump(self.memories, f)
            
            logger.debug("Saved memories to persistent storage")
            
        except Exception as e:
            logger.error(f"Failed to save memories to storage: {e}")
    
    def _build_search_filters(
        self,
        memory_types: Optional[List[MemoryType]],
        importance_threshold: Optional[MemoryImportance]
    ) -> Optional[Dict[str, Any]]:
        """Build search filters for vector store query."""
        filters = {}
        
        if memory_types:
            filters["memory_type"] = {"$in": [mt.value for mt in memory_types]}
        
        if importance_threshold:
            importance_values = {
                MemoryImportance.LOW: 1,
                MemoryImportance.MEDIUM: 2,
                MemoryImportance.HIGH: 3,
                MemoryImportance.CRITICAL: 4
            }
            threshold_value = importance_values[importance_threshold]
            filters["importance_value"] = {"$gte": threshold_value}
        
        return filters if filters else None
    
    def _calculate_relevance_score(
        self,
        memory: Memory,
        query: str,
        similarity_score: float
    ) -> float:
        """Calculate relevance score for a memory."""
        relevance = similarity_score
        
        importance_boost = {
            MemoryImportance.CRITICAL: 0.3,
            MemoryImportance.HIGH: 0.2,
            MemoryImportance.MEDIUM: 0.1,
            MemoryImportance.LOW: 0.0
        }
        relevance += importance_boost.get(memory.importance, 0.0)
        
        relevance *= memory.decay_factor
        
        return min(1.0, relevance)
    
    def _generate_memory_id(self) -> str:
        """Generate unique memory ID."""
        timestamp = int(datetime.now().timestamp())
        return f"memory_{timestamp}_{len(self.memories)}"
    
    def _update_average_retrieval_time(self, retrieval_time: float) -> None:
        """Update the average retrieval time metric."""
        current_avg = self.stats["average_retrieval_time"]
        retrieval_count = self.stats["memories_retrieved"]
        
        if retrieval_count == 1:
            self.stats["average_retrieval_time"] = retrieval_time
        else:
            new_avg = ((current_avg * (retrieval_count - 1)) + retrieval_time) / retrieval_count
            self.stats["average_retrieval_time"] = new_avg
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_memories": len(self.memories),
            "performance_stats": self.stats.copy()
        }
