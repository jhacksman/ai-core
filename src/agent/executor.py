"""
Task Executor - Background task execution for the Venice.ai scaffolding system.

This module handles background task execution with intelligent retry logic
and result storage for the PDX Hackerspace AI Agent system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import traceback

from ..venice.client import VeniceClient
from ..mcp.server_manager import MCPServerManager
from ..mcp.registry import MCPRegistry
from ..mcp.client import MCPClient

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class RetryStrategy(Enum):
    """Retry strategies for failed tasks."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    CUSTOM = "custom"


@dataclass
class ExecutionTask:
    """Represents a task for execution."""
    task_id: str
    name: str
    description: str
    tool_calls: List[Dict[str, Any]]
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    timeout: float = 300.0  # 5 minutes default
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of task execution."""
    task_id: str
    status: ExecutionStatus
    results: Dict[str, Any]
    execution_time: float
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskExecutor:
    """
    Background task executor for the Venice.ai scaffolding system.
    
    Handles execution of solutions using available tools with intelligent
    retry logic and result storage.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        server_manager: MCPServerManager,
        registry: MCPRegistry,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the task executor.
        
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
        
        self.max_concurrent_tasks = self.config.get("max_concurrent_tasks", 5)
        self.default_timeout = self.config.get("default_timeout", 300)
        self.max_retries = self.config.get("max_retries", 3)
        
        self.active_tasks: Dict[str, ExecutionTask] = {}
        self.completed_tasks: Dict[str, ExecutionTask] = {}
        self.task_queue: List[str] = []
        
        self.mcp_client: Optional[MCPClient] = None
        
        self.execution_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "retried_tasks": 0,
            "average_execution_time": 0.0,
            "tool_usage_stats": {},
            "error_patterns": {}
        }
        
        self._execution_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self._shutdown_event = asyncio.Event()
        self._background_tasks: List[asyncio.Task] = []
    
    async def initialize(self) -> None:
        """Initialize the task executor."""
        logger.info("Initializing Task Executor")
        
        try:
            self.mcp_client = MCPClient()
            
            processor_task = asyncio.create_task(self._process_task_queue())
            self._background_tasks.append(processor_task)
            
            logger.info("Task Executor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Task Executor: {e}")
            raise
    
    async def execute_solution(
        self,
        task: Any,  # AgentTask from manager.py
        available_tools: List[str]
    ) -> Dict[str, Any]:
        """
        Execute a solution for an agent task.
        
        Args:
            task: Agent task to execute
            available_tools: List of available tool names
            
        Returns:
            Execution results
        """
        logger.info(f"Executing solution for task: {task.task_id}")
        
        try:
            execution_plan = await self._generate_execution_plan(
                problem=task.description,
                research_context=task.research_context,
                available_tools=available_tools
            )
            
            exec_task = ExecutionTask(
                task_id=f"exec_{task.task_id}",
                name=f"Execute solution for {task.task_id}",
                description=task.description,
                tool_calls=execution_plan["tool_calls"],
                metadata={
                    "agent_task_id": task.task_id,
                    "execution_plan": execution_plan
                }
            )
            
            result = await self.execute_task(exec_task)
            
            return {
                "execution_result": result,
                "execution_plan": execution_plan,
                "tool_results": result.tool_results,
                "success": result.status == ExecutionStatus.COMPLETED
            }
            
        except Exception as e:
            logger.error(f"Failed to execute solution for task {task.task_id}: {e}")
            return {
                "execution_result": None,
                "error": str(e),
                "success": False
            }
    
    async def execute_task(self, task: ExecutionTask) -> ExecutionResult:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            
        Returns:
            Execution result
        """
        logger.info(f"Executing task: {task.name}")
        
        self.active_tasks[task.task_id] = task
        
        start_time = datetime.now()
        task.started_at = start_time
        task.status = ExecutionStatus.RUNNING
        
        try:
            result = await asyncio.wait_for(
                self._execute_task_internal(task),
                timeout=task.timeout
            )
            
            task.status = ExecutionStatus.COMPLETED
            task.completed_at = datetime.now()
            task.results = result
            
            execution_time = (task.completed_at - start_time).total_seconds()
            
            execution_result = ExecutionResult(
                task_id=task.task_id,
                status=task.status,
                results=result,
                execution_time=execution_time,
                tool_results=result.get("tool_results", []),
                metadata=task.metadata
            )
            
            self._update_execution_metrics(execution_result, True)
            
            self.completed_tasks[task.task_id] = task
            del self.active_tasks[task.task_id]
            
            logger.info(f"Task {task.name} completed successfully in {execution_time:.2f}s")
            return execution_result
            
        except asyncio.TimeoutError:
            logger.error(f"Task {task.name} timed out after {task.timeout}s")
            return await self._handle_task_failure(task, "Task timed out", start_time)
            
        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
            return await self._handle_task_failure(task, str(e), start_time)
    
    async def execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute a single tool call.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            timeout: Execution timeout
            
        Returns:
            Tool execution result
        """
        logger.debug(f"Executing tool call: {tool_name}")
        
        try:
            if not self.mcp_client:
                raise Exception("MCP client not initialized")
            
            tools = await self.registry.find_tools_by_name(tool_name)
            if not tools:
                raise Exception(f"Tool {tool_name} not found in registry")
            
            tool_info = tools[0]
            
            start_time = datetime.now()
            
            result = await self.mcp_client.call_tool(
                name=tool_name,
                arguments=arguments
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.execution_metrics["tool_usage_stats"][tool_name] = (
                self.execution_metrics["tool_usage_stats"].get(tool_name, 0) + 1
            )
            
            return {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "execution_time": execution_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Tool call {tool_name} failed: {e}")
            
            error_key = f"{tool_name}:{type(e).__name__}"
            self.execution_metrics["error_patterns"][error_key] = (
                self.execution_metrics["error_patterns"].get(error_key, 0) + 1
            )
            
            return {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": None,
                "error": str(e),
                "success": False
            }
    
    async def schedule_task(self, task: ExecutionTask) -> str:
        """
        Schedule a task for background execution.
        
        Args:
            task: Task to schedule
            
        Returns:
            Task ID
        """
        logger.info(f"Scheduling task: {task.name}")
        
        self.task_queue.append(task.task_id)
        self.active_tasks[task.task_id] = task
        
        return task.task_id
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running or queued task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled
        """
        logger.info(f"Cancelling task: {task_id}")
        
        try:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = ExecutionStatus.CANCELLED
                
                if task_id in self.task_queue:
                    self.task_queue.remove(task_id)
                
                self.completed_tasks[task_id] = task
                del self.active_tasks[task_id]
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[ExecutionTask]:
        """Get status of a task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None
    
    async def list_active_tasks(self) -> List[ExecutionTask]:
        """List all active tasks."""
        return list(self.active_tasks.values())
    
    async def cleanup(self) -> None:
        """Cleanup executor resources."""
        logger.info("Cleaning up Task Executor")
        
        try:
            self._shutdown_event.set()
            
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            for task_id in list(self.active_tasks.keys()):
                await self.cancel_task(task_id)
            
            logger.info("Task Executor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Task Executor cleanup: {e}")
    
    async def _execute_task_internal(self, task: ExecutionTask) -> Dict[str, Any]:
        """Internal task execution logic."""
        results = {
            "task_id": task.task_id,
            "tool_results": [],
            "summary": "",
            "success": True
        }
        
        try:
            for i, tool_call in enumerate(task.tool_calls):
                logger.debug(f"Executing tool call {i+1}/{len(task.tool_calls)}: {tool_call.get('tool_name', 'unknown')}")
                
                tool_result = await self.execute_tool_call(
                    tool_name=tool_call["tool_name"],
                    arguments=tool_call.get("arguments", {}),
                    timeout=tool_call.get("timeout")
                )
                
                results["tool_results"].append(tool_result)
                
                if not tool_result.get("success", False):
                    results["success"] = False
                    task.errors.append(f"Tool call {tool_call['tool_name']} failed: {tool_result.get('error', 'Unknown error')}")
            
            if results["tool_results"]:
                summary = await self._generate_execution_summary(task, results["tool_results"])
                results["summary"] = summary
            
            return results
            
        except Exception as e:
            logger.error(f"Internal execution failed for task {task.task_id}: {e}")
            task.errors.append(f"Internal execution error: {e}")
            results["success"] = False
            results["error"] = str(e)
            return results
    
    async def _generate_execution_plan(
        self,
        problem: str,
        research_context: Dict[str, Any],
        available_tools: List[str]
    ) -> Dict[str, Any]:
        """Generate execution plan for a problem."""
        try:
            plan_prompt = f"""
            Generate an execution plan to solve this problem:
            
            Problem: {problem}
            
            Available Tools: {available_tools}
            
            Research Context: {json.dumps(research_context, indent=2)[:1000]}...
            
            Create a step-by-step execution plan using the available tools.
            Return as JSON with this structure:
            {{
                "plan_description": "Brief description of the plan",
                "tool_calls": [
                    {{
                        "step": 1,
                        "tool_name": "tool_name",
                        "arguments": {{"arg1": "value1"}},
                        "description": "What this step does"
                    }}
                ],
                "expected_outcome": "What the plan should achieve"
            }}
            """
            
            response = await self.venice_client.chat_completion(
                messages=[{"role": "user", "content": plan_prompt}],
                model="qwen-qwq-32b",
                temperature=0.3
            )
            
            try:
                plan = json.loads(response.content)
            except json.JSONDecodeError:
                plan = self._create_fallback_execution_plan(problem, available_tools)
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to generate execution plan: {e}")
            return self._create_fallback_execution_plan(problem, available_tools)
    
    async def _generate_execution_summary(
        self,
        task: ExecutionTask,
        tool_results: List[Dict[str, Any]]
    ) -> str:
        """Generate summary of execution results."""
        try:
            results_summary = []
            for result in tool_results:
                summary_item = f"Tool: {result['tool_name']}\n"
                summary_item += f"Success: {result['success']}\n"
                if result['success']:
                    summary_item += f"Result: {str(result['result'])[:200]}...\n"
                else:
                    summary_item += f"Error: {result.get('error', 'Unknown error')}\n"
                results_summary.append(summary_item)
            
            summary_prompt = f"""
            Summarize the execution results for this task:
            
            Task: {task.description}
            
            Tool Execution Results:
            {chr(10).join(results_summary)}
            
            Provide a concise summary of what was accomplished and any issues encountered.
            """
            
            response = await self.venice_client.chat_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                model="llama-4",
                temperature=0.2
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate execution summary: {e}")
            return f"Executed {len(tool_results)} tool calls for task: {task.description}"
    
    async def _handle_task_failure(
        self,
        task: ExecutionTask,
        error_message: str,
        start_time: datetime
    ) -> ExecutionResult:
        """Handle task failure with retry logic."""
        task.errors.append(error_message)
        
        if task.retry_count < task.max_retries and task.retry_strategy != RetryStrategy.NONE:
            task.retry_count += 1
            task.status = ExecutionStatus.RETRYING
            
            delay = self._calculate_retry_delay(task)
            
            logger.info(f"Retrying task {task.name} in {delay}s (attempt {task.retry_count}/{task.max_retries})")
            
            await asyncio.sleep(delay)
            return await self.execute_task(task)
        
        else:
            task.status = ExecutionStatus.FAILED
            task.completed_at = datetime.now()
            
            execution_time = (task.completed_at - start_time).total_seconds()
            
            result = ExecutionResult(
                task_id=task.task_id,
                status=task.status,
                results={"error": error_message},
                execution_time=execution_time,
                errors=task.errors,
                metadata=task.metadata
            )
            
            self._update_execution_metrics(result, False)
            
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            return result
    
    async def _process_task_queue(self) -> None:
        """Background process to handle task queue."""
        while not self._shutdown_event.is_set():
            try:
                if self.task_queue:
                    async with self._execution_semaphore:
                        if self.task_queue:
                            task_id = self.task_queue.pop(0)
                            if task_id in self.active_tasks:
                                task = self.active_tasks[task_id]
                                if task.status == ExecutionStatus.PENDING:
                                    asyncio.create_task(self.execute_task(task))
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in task queue processor: {e}")
                await asyncio.sleep(5)
    
    def _calculate_retry_delay(self, task: ExecutionTask) -> float:
        """Calculate delay before retry based on strategy."""
        if task.retry_strategy == RetryStrategy.LINEAR:
            return task.retry_count * 5.0  # 5, 10, 15 seconds
        elif task.retry_strategy == RetryStrategy.EXPONENTIAL:
            return min(2 ** task.retry_count, 60.0)  # 2, 4, 8, 16... max 60 seconds
        else:
            return 5.0  # Default 5 seconds
    
    def _create_fallback_execution_plan(
        self,
        problem: str,
        available_tools: List[str]
    ) -> Dict[str, Any]:
        """Create fallback execution plan when AI generation fails."""
        return {
            "plan_description": f"Basic execution plan for: {problem}",
            "tool_calls": [
                {
                    "step": 1,
                    "tool_name": available_tools[0] if available_tools else "echo",
                    "arguments": {"input": problem},
                    "description": "Execute basic tool call"
                }
            ],
            "expected_outcome": "Basic problem processing"
        }
    
    def _update_execution_metrics(self, result: ExecutionResult, success: bool) -> None:
        """Update execution metrics."""
        self.execution_metrics["total_tasks"] += 1
        
        if success:
            self.execution_metrics["successful_tasks"] += 1
        else:
            self.execution_metrics["failed_tasks"] += 1
        
        total_tasks = self.execution_metrics["total_tasks"]
        current_avg = self.execution_metrics["average_execution_time"]
        
        if total_tasks == 1:
            self.execution_metrics["average_execution_time"] = result.execution_time
        else:
            new_avg = ((current_avg * (total_tasks - 1)) + result.execution_time) / total_tasks
            self.execution_metrics["average_execution_time"] = new_avg
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "metrics": self.execution_metrics.copy(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "queued_tasks": len(self.task_queue)
        }


Executor = TaskExecutor
