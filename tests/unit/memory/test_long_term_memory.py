"""
Unit tests for Long Term Memory system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List

from src.memory.long_term_memory import LongTermMemory, MemoryEntry, MemoryType
from src.memory.vector_store import VectorStore


class TestLongTermMemory:
    """Test cases for LongTermMemory."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        vector_store = Mock(spec=VectorStore)
        vector_store.add_document = AsyncMock()
        vector_store.search = AsyncMock()
        vector_store.delete_document = AsyncMock()
        vector_store.get_collection_stats = AsyncMock(return_value={"count": 0})
        return vector_store
    
    @pytest.fixture
    def long_term_memory(self, mock_vector_store):
        """Create LongTermMemory instance for testing."""
        config = {
            "retention_days": 90,
            "importance_threshold": 0.3,
            "auto_cleanup": True,
            "backup_interval_hours": 24
        }
        return LongTermMemory(config=config, vector_store=mock_vector_store)
    
    @pytest.mark.asyncio
    async def test_memory_initialization(self, long_term_memory):
        """Test memory system initializes correctly."""
        await long_term_memory.initialize()
        
        assert long_term_memory.retention_days == 90
        assert long_term_memory.importance_threshold == 0.3
        assert long_term_memory.auto_cleanup is True
        assert long_term_memory.backup_interval_hours == 24
    
    @pytest.mark.asyncio
    async def test_store_memory_success(self, long_term_memory, mock_vector_store):
        """Test successful memory storage."""
        await long_term_memory.initialize()
        
        memory_entry = MemoryEntry(
            content="Test memory content",
            memory_type=MemoryType.EXPERIENCE,
            importance=0.8,
            metadata={"source": "test"},
            timestamp=datetime.now()
        )
        
        mock_vector_store.add_document.return_value = "memory-123"
        
        memory_id = await long_term_memory.store_memory(memory_entry)
        
        assert memory_id == "memory-123"
        mock_vector_store.add_document.assert_called_once()
        
        call_args = mock_vector_store.add_document.call_args
        assert call_args[1]["content"] == "Test memory content"
        assert call_args[1]["metadata"]["memory_type"] == "experience"
        assert call_args[1]["metadata"]["importance"] == 0.8
    
    @pytest.mark.asyncio
    async def test_store_memory_below_threshold(self, long_term_memory, mock_vector_store):
        """Test memory storage below importance threshold."""
        await long_term_memory.initialize()
        
        memory_entry = MemoryEntry(
            content="Low importance memory",
            memory_type=MemoryType.OBSERVATION,
            importance=0.1,  # Below threshold of 0.3
            metadata={"source": "test"},
            timestamp=datetime.now()
        )
        
        memory_id = await long_term_memory.store_memory(memory_entry)
        
        assert memory_id is None
        mock_vector_store.add_document.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_retrieve_memories_success(self, long_term_memory, mock_vector_store):
        """Test successful memory retrieval."""
        await long_term_memory.initialize()
        
        mock_vector_store.search.return_value = [
            {
                "id": "memory-1",
                "content": "Retrieved memory 1",
                "metadata": {
                    "memory_type": "experience",
                    "importance": 0.8,
                    "timestamp": datetime.now().isoformat()
                },
                "score": 0.9
            },
            {
                "id": "memory-2", 
                "content": "Retrieved memory 2",
                "metadata": {
                    "memory_type": "observation",
                    "importance": 0.6,
                    "timestamp": datetime.now().isoformat()
                },
                "score": 0.7
            }
        ]
        
        memories = await long_term_memory.retrieve_memories(
            query="test query",
            limit=5,
            memory_type=MemoryType.EXPERIENCE
        )
        
        assert len(memories) == 2
        assert memories[0]["content"] == "Retrieved memory 1"
        assert memories[1]["content"] == "Retrieved memory 2"
        mock_vector_store.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_memories_with_filters(self, long_term_memory, mock_vector_store):
        """Test memory retrieval with filters."""
        await long_term_memory.initialize()
        
        mock_vector_store.search.return_value = []
        
        await long_term_memory.retrieve_memories(
            query="test query",
            limit=10,
            memory_type=MemoryType.RESEARCH,
            min_importance=0.5,
            time_range=(datetime.now() - timedelta(days=7), datetime.now())
        )
        
        call_args = mock_vector_store.search.call_args
        assert call_args[1]["query"] == "test query"
        assert call_args[1]["limit"] == 10
        assert "memory_type" in call_args[1]["filters"]
        assert "importance" in call_args[1]["filters"]
        assert "timestamp" in call_args[1]["filters"]
    
    @pytest.mark.asyncio
    async def test_update_memory_importance(self, long_term_memory, mock_vector_store):
        """Test updating memory importance."""
        await long_term_memory.initialize()
        
        mock_vector_store.update_metadata = AsyncMock()
        
        await long_term_memory.update_memory_importance("memory-123", 0.9)
        
        mock_vector_store.update_metadata.assert_called_once_with(
            "memory-123",
            {"importance": 0.9}
        )
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, long_term_memory, mock_vector_store):
        """Test memory deletion."""
        await long_term_memory.initialize()
        
        result = await long_term_memory.delete_memory("memory-123")
        
        assert result is True
        mock_vector_store.delete_document.assert_called_once_with("memory-123")
    
    @pytest.mark.asyncio
    async def test_cleanup_old_memories(self, long_term_memory, mock_vector_store):
        """Test cleanup of old memories."""
        await long_term_memory.initialize()
        
        old_timestamp = datetime.now() - timedelta(days=100)
        mock_vector_store.search.return_value = [
            {
                "id": "old-memory-1",
                "metadata": {"timestamp": old_timestamp.isoformat()}
            },
            {
                "id": "old-memory-2", 
                "metadata": {"timestamp": old_timestamp.isoformat()}
            }
        ]
        
        deleted_count = await long_term_memory.cleanup_old_memories()
        
        assert deleted_count == 2
        assert mock_vector_store.delete_document.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_memory_stats(self, long_term_memory, mock_vector_store):
        """Test getting memory statistics."""
        await long_term_memory.initialize()
        
        mock_vector_store.get_collection_stats.return_value = {
            "count": 150,
            "size_mb": 25.5
        }
        
        mock_vector_store.search.side_effect = [
            [{"id": f"exp-{i}"} for i in range(50)],  # experiences
            [{"id": f"obs-{i}"} for i in range(75)],  # observations  
            [{"id": f"res-{i}"} for i in range(25)]   # research
        ]
        
        stats = await long_term_memory.get_memory_stats()
        
        assert stats["total_memories"] == 150
        assert stats["storage_size_mb"] == 25.5
        assert stats["memory_types"]["experience"] == 50
        assert stats["memory_types"]["observation"] == 75
        assert stats["memory_types"]["research"] == 25
    
    @pytest.mark.asyncio
    async def test_backup_memories(self, long_term_memory, mock_vector_store):
        """Test memory backup functionality."""
        await long_term_memory.initialize()
        
        with patch('src.memory.long_term_memory.LongTermMemory._create_backup_file') as mock_backup:
            mock_backup.return_value = "/path/to/backup.json"
            
            backup_path = await long_term_memory.backup_memories()
            
            assert backup_path == "/path/to/backup.json"
            mock_backup.assert_called_once()


class TestMemoryEntry:
    """Test cases for MemoryEntry."""
    
    def test_memory_entry_creation(self):
        """Test MemoryEntry creation."""
        timestamp = datetime.now()
        entry = MemoryEntry(
            content="Test content",
            memory_type=MemoryType.EXPERIENCE,
            importance=0.8,
            metadata={"key": "value"},
            timestamp=timestamp
        )
        
        assert entry.content == "Test content"
        assert entry.memory_type == MemoryType.EXPERIENCE
        assert entry.importance == 0.8
        assert entry.metadata == {"key": "value"}
        assert entry.timestamp == timestamp
    
    def test_memory_entry_to_dict(self):
        """Test MemoryEntry serialization."""
        timestamp = datetime.now()
        entry = MemoryEntry(
            content="Test content",
            memory_type=MemoryType.RESEARCH,
            importance=0.6,
            metadata={"source": "test"},
            timestamp=timestamp
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict["content"] == "Test content"
        assert entry_dict["memory_type"] == "research"
        assert entry_dict["importance"] == 0.6
        assert entry_dict["metadata"] == {"source": "test"}
        assert entry_dict["timestamp"] == timestamp.isoformat()
    
    def test_memory_entry_from_dict(self):
        """Test MemoryEntry deserialization."""
        timestamp = datetime.now()
        entry_dict = {
            "content": "Test content",
            "memory_type": "observation",
            "importance": 0.4,
            "metadata": {"tag": "test"},
            "timestamp": timestamp.isoformat()
        }
        
        entry = MemoryEntry.from_dict(entry_dict)
        
        assert entry.content == "Test content"
        assert entry.memory_type == MemoryType.OBSERVATION
        assert entry.importance == 0.4
        assert entry.metadata == {"tag": "test"}
        assert entry.timestamp == timestamp


class TestMemoryType:
    """Test cases for MemoryType enum."""
    
    def test_memory_type_values(self):
        """Test MemoryType enum values."""
        assert MemoryType.EXPERIENCE.value == "experience"
        assert MemoryType.OBSERVATION.value == "observation"
        assert MemoryType.RESEARCH.value == "research"
        assert MemoryType.REFLECTION.value == "reflection"
    
    def test_memory_type_from_string(self):
        """Test creating MemoryType from string."""
        assert MemoryType("experience") == MemoryType.EXPERIENCE
        assert MemoryType("observation") == MemoryType.OBSERVATION
        assert MemoryType("research") == MemoryType.RESEARCH
        assert MemoryType("reflection") == MemoryType.REFLECTION
