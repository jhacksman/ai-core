"""
Agent Manager - Central orchestrator for Venice.ai scaffolding operations.

This module coordinates research, tool creation, and solution execution
for the PDX Hackerspace AI Agent system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from ..venice.client import VeniceClient
from ..mcp.server_manager import MCPServerManager
from ..mcp.registry import MCPRegistry
from ..mcp.factory import ServerFactory
from ..mcp.capability_analyzer import CapabilityAnalyzer
from .research_coordinator import ResearchCoordinator
from .tool_creator import ToolCreator
from .executor import TaskExecutor

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of agent tasks."""
    PENDING = "pending"
    RESEARCHING = "researching"
    ANALYZING = "analyzing"
    CREATING_TOOLS = "creating_tools"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(Enum):
    """Priority levels for agent tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class AgentTask:
    """Represents a task for the agent system."""
    task_id: str
    description: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    research_context: Dict[str, Any] = field(default_factory=dict)
    tool_requirements: List[str] = field(default_factory=list)
    execution_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


Task = AgentTask


@dataclass
class AgentCapabilities:
    """Current capabilities of the agent system."""
    available_tools: List[str] = field(default_factory=list)
    active_servers: List[str] = field(default_factory=list)
    research_sources: List[str] = field(default_factory=list)
    execution_capacity: int = 5
    memory_usage: float = 0.0


class AgentManager:
    """
    Central orchestrator for Venice.ai scaffolding operations.
    
    Coordinates research, tool creation, and solution execution for
    the PDX Hackerspace AI Agent system.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        server_manager: MCPServerManager,
        registry: MCPRegistry,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent manager.
        
        Args:
            venice_client: Venice.ai client for LLM operations
            server_manager: MCP server manager
            registry: MCP server registry
            config: Configuration options
        """
        self.venice_client = venice_client
        self.server_manager = server_manager
        self.registry = registry
        self.config = config or {}
        
        self.server_factory = ServerFactory(
            venice_client=venice_client,
            server_manager=server_manager,
            registry=registry
        )
        
        self.capability_analyzer = CapabilityAnalyzer(
            venice_client=venice_client,
            existing_capabilities=[]
        )
        
        self.research_coordinator = ResearchCoordinator(
            venice_client=venice_client,
            config=self.config.get("research", {})
        )
        
        self.tool_creator = ToolCreator(
            server_factory=self.server_factory,
            capability_analyzer=self.capability_analyzer,
            research_coordinator=self.research_coordinator
        )
        
        self.task_executor = TaskExecutor(
            venice_client=venice_client,
            server_manager=server_manager,
            registry=registry
        )
        
        self.active_tasks: Dict[str, AgentTask] = {}
        self.task_queue: List[str] = []
        self.capabilities = AgentCapabilities()
        
        self.performance_metrics = {
            "tasks_completed": 0,
            "tools_created": 0,
            "research_sessions": 0,
            "average_completion_time": 0.0,
            "success_rate": 0.0
        }
        
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> None:
        """Start the agent manager and background processes."""
        logger.info("Starting Agent Manager")
        
        try:
            await self._initialize_components()
            
            await self._start_background_processes()
            
            await self._update_capabilities()
            
            logger.info("Agent Manager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Agent Manager: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the agent manager and cleanup resources."""
        logger.info("Stopping Agent Manager")
        
        try:
            self._shutdown_event.set()
            
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            await self._cleanup_components()
            
            logger.info("Agent Manager stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping Agent Manager: {e}")
    
    async def solve_problem(
        self,
        problem_description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentTask:
        """
        Main entry point for problem-solving using the Venice.ai scaffolding approach.
        
        Args:
            problem_description: Description of the problem to solve
            priority: Task priority level
            context: Additional context for the problem
            
        Returns:
            Agent task representing the problem-solving process
        """
        logger.info(f"Starting problem-solving for: {problem_description[:100]}...")
        
        try:
            task = AgentTask(
                task_id=self._generate_task_id(),
                description=problem_description,
                priority=priority,
                metadata=context or {}
            )
            
            self.active_tasks[task.task_id] = task
            self.task_queue.append(task.task_id)
            
            await self._execute_problem_solving_pipeline(task)
            
            return task
            
        except Exception as e:
            logger.error(f"Failed to solve problem: {e}")
            raise
    
    async def _execute_problem_solving_pipeline(self, task: AgentTask) -> None:
        """
        Execute the complete problem-solving pipeline.
        
        This implements the research → analyze → design → create → deploy pattern.
        """
        try:
            task.status = TaskStatus.RESEARCHING
            task.updated_at = datetime.now()
            
            research_context = await self.research_coordinator.comprehensive_research(
                problem=task.description,
                depth="deep",
                sources=["web", "memory", "documentation"]
            )
            
            task.research_context = research_context
            self.performance_metrics["research_sessions"] += 1
            
            task.status = TaskStatus.ANALYZING
            task.updated_at = datetime.now()
            
            requirements = await self.capability_analyzer.analyze_problem_requirements(
                problem_description=task.description,
                research_context=research_context
            )
            
            capability_gaps = await self.capability_analyzer.identify_gaps(
                problem_description=task.description,
                research_context=research_context,
                existing_capabilities=self.capabilities.available_tools
            )
            
            task.tool_requirements = [gap.capability_name for gap in capability_gaps]
            
            if capability_gaps:
                task.status = TaskStatus.CREATING_TOOLS
                task.updated_at = datetime.now()
                
                created_tools = await self.tool_creator.create_tools_for_gaps(
                    gaps=capability_gaps,
                    research_context=research_context
                )
                
                await self._update_capabilities()
                self.performance_metrics["tools_created"] += len(created_tools)
            
            task.status = TaskStatus.EXECUTING
            task.updated_at = datetime.now()
            
            execution_result = await self.task_executor.execute_solution(
                task=task,
                available_tools=self.capabilities.available_tools
            )
            
            task.execution_results = execution_result
            
            task.status = TaskStatus.COMPLETED
            task.updated_at = datetime.now()
            
            self.performance_metrics["tasks_completed"] += 1
            completion_time = (task.updated_at - task.created_at).total_seconds()
            self._update_average_completion_time(completion_time)
            
            logger.info(f"Successfully completed task: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute problem-solving pipeline for task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.updated_at = datetime.now()
            task.execution_results = {"error": str(e)}
    
    async def get_task_status(self, task_id: str) -> Optional[AgentTask]:
        """Get the status of a specific task."""
        return self.active_tasks.get(task_id)
    
    async def list_active_tasks(self) -> List[AgentTask]:
        """List all active tasks."""
        return list(self.active_tasks.values())
    
    async def get_capabilities(self) -> AgentCapabilities:
        """Get current agent capabilities."""
        await self._update_capabilities()
        return self.capabilities
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        total_tasks = len(self.active_tasks)
        completed_tasks = sum(
            1 for task in self.active_tasks.values() 
            if task.status == TaskStatus.COMPLETED
        )
        
        if total_tasks > 0:
            self.performance_metrics["success_rate"] = completed_tasks / total_tasks
        
        return self.performance_metrics.copy()
    
    async def _initialize_components(self) -> None:
        """Initialize all agent components."""
        logger.debug("Initializing agent components")
        
        await self.research_coordinator.initialize()
        
        await self.tool_creator.initialize()
        
        await self.task_executor.initialize()
        
        logger.debug("Agent components initialized")
    
    async def _start_background_processes(self) -> None:
        """Start background monitoring and maintenance processes."""
        logger.debug("Starting background processes")
        
        task_processor = asyncio.create_task(self._process_task_queue())
        self._background_tasks.append(task_processor)
        
        capability_monitor = asyncio.create_task(self._monitor_capabilities())
        self._background_tasks.append(capability_monitor)
        
        performance_tracker = asyncio.create_task(self._track_performance())
        self._background_tasks.append(performance_tracker)
        
        logger.debug("Background processes started")
    
    async def _process_task_queue(self) -> None:
        """Background process to handle task queue."""
        while not self._shutdown_event.is_set():
            try:
                if self.task_queue:
                    prioritized_tasks = self._prioritize_tasks()
                    
                    for task_id in prioritized_tasks[:self.capabilities.execution_capacity]:
                        if task_id in self.active_tasks:
                            task = self.active_tasks[task_id]
                            if task.status == TaskStatus.PENDING:
                                asyncio.create_task(self._execute_problem_solving_pipeline(task))
                                self.task_queue.remove(task_id)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in task queue processor: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_capabilities(self) -> None:
        """Background process to monitor system capabilities."""
        while not self._shutdown_event.is_set():
            try:
                await self._update_capabilities()
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in capability monitor: {e}")
                await asyncio.sleep(60)
    
    async def _track_performance(self) -> None:
        """Background process to track performance metrics."""
        while not self._shutdown_event.is_set():
            try:
                await self.get_performance_metrics()
                
                if self.performance_metrics["tasks_completed"] > 0:
                    logger.info(
                        f"Performance: {self.performance_metrics['tasks_completed']} tasks completed, "
                        f"{self.performance_metrics['tools_created']} tools created, "
                        f"{self.performance_metrics['success_rate']:.2f} success rate"
                    )
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance tracker: {e}")
                await asyncio.sleep(600)
    
    async def _update_capabilities(self) -> None:
        """Update current system capabilities."""
        try:
            tools = await self.registry.list_tools()
            self.capabilities.available_tools = [tool.name for tool in tools]
            
            active_servers = await self.server_manager.list_active_servers()
            self.capabilities.active_servers = [server.name for server in active_servers]
            
            self.capabilities.research_sources = await self.research_coordinator.get_available_sources()
            
            self.capabilities.memory_usage = 0.0  # TODO: Implement actual memory tracking
            
        except Exception as e:
            logger.error(f"Failed to update capabilities: {e}")
    
    async def _cleanup_components(self) -> None:
        """Cleanup all agent components."""
        logger.debug("Cleaning up agent components")
        
        try:
            await self.task_executor.cleanup()
            await self.tool_creator.cleanup()
            await self.research_coordinator.cleanup()
            
        except Exception as e:
            logger.error(f"Error during component cleanup: {e}")
    
    def _generate_task_id(self) -> str:
        """Generate a unique task ID."""
        timestamp = int(datetime.now().timestamp())
        return f"task_{timestamp}_{len(self.active_tasks)}"
    
    def _prioritize_tasks(self) -> List[str]:
        """Prioritize tasks in the queue based on priority and creation time."""
        priority_order = {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        
        def task_priority_score(task_id: str) -> Tuple[int, float]:
            task = self.active_tasks.get(task_id)
            if not task:
                return (0, 0.0)
            
            priority_score = priority_order.get(task.priority, 1)
            time_score = task.created_at.timestamp()  # Earlier tasks get higher priority
            
            return (priority_score, -time_score)  # Negative time for reverse order
        
        return sorted(self.task_queue, key=task_priority_score, reverse=True)
    
    def _update_average_completion_time(self, completion_time: float) -> None:
        """Update the average completion time metric."""
        current_avg = self.performance_metrics["average_completion_time"]
        completed_count = self.performance_metrics["tasks_completed"]
        
        if completed_count == 1:
            self.performance_metrics["average_completion_time"] = completion_time
        else:
            new_avg = ((current_avg * (completed_count - 1)) + completion_time) / completed_count
            self.performance_metrics["average_completion_time"] = new_avg
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "background_tasks": len(self._background_tasks),
            "capabilities": {
                "available_tools": len(self.capabilities.available_tools),
                "active_servers": len(self.capabilities.active_servers),
                "research_sources": len(self.capabilities.research_sources)
            },
            "performance": self.performance_metrics
        }
