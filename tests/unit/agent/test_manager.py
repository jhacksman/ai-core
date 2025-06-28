"""
Unit tests for Agent Manager.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.agent.manager import AgentManager, Task, TaskStatus, TaskPriority
from src.agent.research_coordinator import ResearchCoordinator
from src.agent.tool_creator import ToolCreator
from src.agent.executor import Executor


class TestAgentManager:
    """Test cases for AgentManager."""
    
    @pytest.fixture
    def mock_research_coordinator(self):
        """Mock research coordinator for testing."""
        coordinator = Mock(spec=ResearchCoordinator)
        coordinator.conduct_research = AsyncMock()
        coordinator.analyze_findings = AsyncMock()
        return coordinator
    
    @pytest.fixture
    def mock_tool_creator(self):
        """Mock tool creator for testing."""
        creator = Mock(spec=ToolCreator)
        creator.create_tool = AsyncMock()
        creator.validate_tool = AsyncMock()
        return creator
    
    @pytest.fixture
    def mock_executor(self):
        """Mock executor for testing."""
        executor = Mock(spec=Executor)
        executor.execute_task = AsyncMock()
        executor.get_status = Mock()
        return executor
    
    @pytest.fixture
    def agent_manager(self, mock_research_coordinator, mock_tool_creator, mock_executor):
        """Create AgentManager instance for testing."""
        config = {
            "max_concurrent_tasks": 5,
            "task_timeout_minutes": 30,
            "retry_attempts": 3,
            "coordination_interval_seconds": 10
        }
        return AgentManager(
            config=config,
            research_coordinator=mock_research_coordinator,
            tool_creator=mock_tool_creator,
            executor=mock_executor
        )
    
    @pytest.mark.asyncio
    async def test_agent_manager_initialization(self, agent_manager):
        """Test agent manager initializes correctly."""
        await agent_manager.initialize()
        
        assert agent_manager.max_concurrent_tasks == 5
        assert agent_manager.task_timeout_minutes == 30
        assert agent_manager.retry_attempts == 3
        assert agent_manager.coordination_interval_seconds == 10
        assert len(agent_manager.active_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_submit_task_success(self, agent_manager):
        """Test successful task submission."""
        await agent_manager.initialize()
        
        task_id = await agent_manager.submit_task(
            description="Test task",
            task_type="research",
            priority=TaskPriority.HIGH,
            metadata={"key": "value"}
        )
        
        assert task_id is not None
        assert task_id in agent_manager.active_tasks
        
        task = agent_manager.active_tasks[task_id]
        assert task.description == "Test task"
        assert task.task_type == "research"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_submit_task_max_capacity(self, agent_manager):
        """Test task submission at maximum capacity."""
        agent_manager.max_concurrent_tasks = 2
        await agent_manager.initialize()
        
        task_id_1 = await agent_manager.submit_task("Task 1", "research")
        task_id_2 = await agent_manager.submit_task("Task 2", "analysis")
        
        task_id_3 = await agent_manager.submit_task("Task 3", "tool_creation")
        
        assert task_id_1 is not None
        assert task_id_2 is not None
        assert task_id_3 is None  # Should be rejected
    
    @pytest.mark.asyncio
    async def test_execute_research_task(self, agent_manager, mock_research_coordinator):
        """Test executing research task."""
        await agent_manager.initialize()
        
        mock_research_coordinator.conduct_research.return_value = {
            "findings": ["Finding 1", "Finding 2"],
            "sources": ["Source 1", "Source 2"],
            "confidence": 0.8
        }
        
        task_id = await agent_manager.submit_task(
            description="Research AI trends",
            task_type="research",
            metadata={"domain": "AI"}
        )
        
        await agent_manager._execute_task(task_id)
        
        task = agent_manager.active_tasks[task_id]
        assert task.status == TaskStatus.COMPLETED
        assert "findings" in task.result
        mock_research_coordinator.conduct_research.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_tool_creation_task(self, agent_manager, mock_tool_creator):
        """Test executing tool creation task."""
        await agent_manager.initialize()
        
        mock_tool_creator.create_tool.return_value = {
            "tool_id": "new-tool-123",
            "tool_name": "test_tool",
            "status": "created"
        }
        
        task_id = await agent_manager.submit_task(
            description="Create data analysis tool",
            task_type="tool_creation",
            metadata={"purpose": "data analysis"}
        )
        
        await agent_manager._execute_task(task_id)
        
        task = agent_manager.active_tasks[task_id]
        assert task.status == TaskStatus.COMPLETED
        assert task.result["tool_id"] == "new-tool-123"
        mock_tool_creator.create_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_analysis_task(self, agent_manager, mock_executor):
        """Test executing analysis task."""
        await agent_manager.initialize()
        
        mock_executor.execute_task.return_value = {
            "analysis_result": "Comprehensive analysis completed",
            "metrics": {"accuracy": 0.95, "confidence": 0.88}
        }
        
        task_id = await agent_manager.submit_task(
            description="Analyze system performance",
            task_type="analysis",
            metadata={"data_source": "metrics"}
        )
        
        await agent_manager._execute_task(task_id)
        
        task = agent_manager.active_tasks[task_id]
        assert task.status == TaskStatus.COMPLETED
        assert "analysis_result" in task.result
        mock_executor.execute_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_failure_and_retry(self, agent_manager, mock_research_coordinator):
        """Test task failure and retry mechanism."""
        await agent_manager.initialize()
        
        mock_research_coordinator.conduct_research.side_effect = [
            Exception("Temporary failure"),
            {"findings": ["Success after retry"], "confidence": 0.7}
        ]
        
        task_id = await agent_manager.submit_task(
            description="Research with retry",
            task_type="research"
        )
        
        await agent_manager._execute_task(task_id)
        
        task = agent_manager.active_tasks[task_id]
        assert task.status == TaskStatus.COMPLETED
        assert task.retry_count == 1
        assert mock_research_coordinator.conduct_research.call_count == 2
    
    @pytest.mark.asyncio
    async def test_task_timeout(self, agent_manager):
        """Test task timeout handling."""
        agent_manager.task_timeout_minutes = 0.01  # Very short timeout
        await agent_manager.initialize()
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = asyncio.TimeoutError()
            
            task_id = await agent_manager.submit_task(
                description="Long running task",
                task_type="research"
            )
            
            await agent_manager._execute_task(task_id)
            
            task = agent_manager.active_tasks[task_id]
            assert task.status == TaskStatus.FAILED
            assert "timeout" in task.error.lower()
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, agent_manager):
        """Test getting task status."""
        await agent_manager.initialize()
        
        task_id = await agent_manager.submit_task(
            description="Status test task",
            task_type="research"
        )
        
        status = await agent_manager.get_task_status(task_id)
        
        assert status["task_id"] == task_id
        assert status["status"] == "pending"
        assert status["description"] == "Status test task"
        assert "created_at" in status
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, agent_manager):
        """Test task cancellation."""
        await agent_manager.initialize()
        
        task_id = await agent_manager.submit_task(
            description="Task to cancel",
            task_type="research"
        )
        
        success = await agent_manager.cancel_task(task_id)
        
        assert success is True
        task = agent_manager.active_tasks[task_id]
        assert task.status == TaskStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_list_active_tasks(self, agent_manager):
        """Test listing active tasks."""
        await agent_manager.initialize()
        
        task_ids = []
        for i in range(3):
            task_id = await agent_manager.submit_task(
                description=f"Task {i}",
                task_type="research"
            )
            task_ids.append(task_id)
        
        active_tasks = await agent_manager.list_active_tasks()
        
        assert len(active_tasks) == 3
        assert all(task["task_id"] in task_ids for task in active_tasks)
    
    @pytest.mark.asyncio
    async def test_coordination_loop(self, agent_manager):
        """Test agent coordination loop."""
        await agent_manager.initialize()
        
        with patch('asyncio.sleep') as mock_sleep:
            coordination_task = asyncio.create_task(
                agent_manager._coordination_loop()
            )
            
            await asyncio.sleep(0.1)
            
            coordination_task.cancel()
            
            try:
                await coordination_task
            except asyncio.CancelledError:
                pass
            
            mock_sleep.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_manager_stats(self, agent_manager):
        """Test getting manager statistics."""
        await agent_manager.initialize()
        
        await agent_manager.submit_task("Task 1", "research")
        await agent_manager.submit_task("Task 2", "analysis")
        
        stats = agent_manager.get_stats()
        
        assert stats["active_tasks"] == 2
        assert stats["max_concurrent_tasks"] == 5
        assert "task_types" in stats
        assert "priority_distribution" in stats


class TestTask:
    """Test cases for Task class."""
    
    def test_task_creation(self):
        """Test Task creation."""
        task = Task(
            task_id="test-123",
            description="Test task",
            task_type="research",
            priority=TaskPriority.HIGH,
            metadata={"key": "value"}
        )
        
        assert task.task_id == "test-123"
        assert task.description == "Test task"
        assert task.task_type == "research"
        assert task.priority == TaskPriority.HIGH
        assert task.status == TaskStatus.PENDING
        assert task.metadata == {"key": "value"}
        assert task.retry_count == 0
    
    def test_task_to_dict(self):
        """Test Task serialization."""
        task = Task(
            task_id="test-123",
            description="Test task",
            task_type="analysis",
            priority=TaskPriority.MEDIUM
        )
        
        task_dict = task.to_dict()
        
        assert task_dict["task_id"] == "test-123"
        assert task_dict["description"] == "Test task"
        assert task_dict["task_type"] == "analysis"
        assert task_dict["priority"] == "medium"
        assert task_dict["status"] == "pending"


class TestTaskEnums:
    """Test cases for task-related enums."""
    
    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
    
    def test_task_priority_values(self):
        """Test TaskPriority enum values."""
        assert TaskPriority.LOW.value == "low"
        assert TaskPriority.MEDIUM.value == "medium"
        assert TaskPriority.HIGH.value == "high"
        assert TaskPriority.CRITICAL.value == "critical"
