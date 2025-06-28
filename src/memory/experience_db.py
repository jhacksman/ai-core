"""
Experience Database - Tracks tool usage effectiveness and learning patterns.

This module tracks tool usage effectiveness, solution outcomes, and learning
patterns for future problem-solving in the Venice.ai scaffolding system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics

from .vector_store import VectorStore, VectorDocument, SearchResult
from .long_term_memory import LongTermMemory, Memory, MemoryType, MemoryImportance
from ..venice.client import VeniceClient

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Types of solution outcomes."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


class LearningPattern(Enum):
    """Types of learning patterns identified."""
    TOOL_EFFECTIVENESS = "tool_effectiveness"
    PROBLEM_COMPLEXITY = "problem_complexity"
    SOLUTION_APPROACH = "solution_approach"
    DEPENDENCY_PATTERN = "dependency_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    ERROR_PATTERN = "error_pattern"


@dataclass
class ToolUsageRecord:
    """Records tool usage and effectiveness."""
    record_id: str
    tool_name: str
    problem_context: str
    usage_timestamp: datetime
    execution_time: float
    outcome: OutcomeType
    success_metrics: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    user_feedback: Optional[str] = None
    effectiveness_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionOutcome:
    """Records complete solution outcomes."""
    outcome_id: str
    problem_description: str
    solution_approach: str
    tools_used: List[str]
    total_execution_time: float
    outcome_type: OutcomeType
    success_rate: float
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningInsight:
    """Represents a learning insight derived from experience."""
    insight_id: str
    pattern_type: LearningPattern
    description: str
    confidence_score: float
    supporting_evidence: List[str] = field(default_factory=list)
    actionable_recommendations: List[str] = field(default_factory=list)
    impact_assessment: str = ""
    discovered_at: datetime = field(default_factory=datetime.now)
    validation_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolEffectivenessProfile:
    """Profile of tool effectiveness across different contexts."""
    tool_name: str
    total_usage_count: int
    success_rate: float
    average_execution_time: float
    context_effectiveness: Dict[str, float] = field(default_factory=dict)
    common_failure_modes: List[str] = field(default_factory=list)
    optimal_use_cases: List[str] = field(default_factory=list)
    performance_trends: Dict[str, List[float]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class ExperienceDatabase:
    """
    Tracks tool usage effectiveness and learning patterns.
    
    Maintains comprehensive records of tool usage, solution outcomes, and
    learning patterns to improve future problem-solving effectiveness.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        vector_store: VectorStore,
        long_term_memory: LongTermMemory,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the experience database.
        
        Args:
            venice_client: Venice.ai client for analysis
            vector_store: Vector store for semantic search
            long_term_memory: Long-term memory system
            config: Configuration options
        """
        self.venice_client = venice_client
        self.vector_store = vector_store
        self.long_term_memory = long_term_memory
        self.config = config or {}
        
        self.tool_usage_records: Dict[str, ToolUsageRecord] = {}
        self.solution_outcomes: Dict[str, SolutionOutcome] = {}
        self.learning_insights: Dict[str, LearningInsight] = {}
        self.tool_effectiveness_profiles: Dict[str, ToolEffectivenessProfile] = {}
        
        self.min_usage_for_profile = self.config.get("min_usage_for_profile", 5)
        self.learning_confidence_threshold = self.config.get("learning_confidence_threshold", 0.7)
        self.pattern_detection_window = self.config.get("pattern_detection_window", 30)  # days
        
        self.stats = {
            "total_tool_usages": 0,
            "total_solutions": 0,
            "insights_discovered": 0,
            "average_success_rate": 0.0,
            "learning_accuracy": 0.0
        }
        
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> None:
        """Initialize the experience database."""
        logger.info("Initializing Experience Database")
        
        try:
            await self._load_experience_data()
            
            await self._start_background_processes()
            
            logger.info(f"Experience Database initialized with {len(self.tool_usage_records)} usage records")
            
        except Exception as e:
            logger.error(f"Failed to initialize Experience Database: {e}")
            raise
    
    async def record_tool_usage(
        self,
        tool_name: str,
        problem_context: str,
        execution_time: float,
        outcome: OutcomeType,
        success_metrics: Optional[Dict[str, Any]] = None,
        error_details: Optional[str] = None,
        resource_usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a tool usage instance.
        
        Args:
            tool_name: Name of the tool used
            problem_context: Context in which tool was used
            execution_time: Time taken for execution
            outcome: Outcome of the tool usage
            success_metrics: Metrics indicating success
            error_details: Details of any errors
            resource_usage: Resource usage information
            metadata: Additional metadata
            
        Returns:
            Record ID
        """
        try:
            record_id = self._generate_record_id()
            
            effectiveness_score = self._calculate_effectiveness_score(
                outcome, execution_time, success_metrics or {}
            )
            
            record = ToolUsageRecord(
                record_id=record_id,
                tool_name=tool_name,
                problem_context=problem_context,
                usage_timestamp=datetime.now(),
                execution_time=execution_time,
                outcome=outcome,
                success_metrics=success_metrics or {},
                error_details=error_details,
                resource_usage=resource_usage or {},
                effectiveness_score=effectiveness_score,
                metadata=metadata or {}
            )
            
            self.tool_usage_records[record_id] = record
            
            vector_doc = VectorDocument(
                id=record_id,
                content=f"Tool: {tool_name}\nContext: {problem_context}\nOutcome: {outcome.value}",
                metadata={
                    "type": "tool_usage_record",
                    "tool_name": tool_name,
                    "outcome": outcome.value,
                    "execution_time": execution_time,
                    "effectiveness_score": effectiveness_score,
                    "timestamp": record.usage_timestamp.isoformat(),
                    **(metadata or {})
                }
            )
            
            await self.vector_store.add_document(vector_doc)
            
            await self._update_tool_effectiveness_profile(tool_name, record)
            
            await self.long_term_memory.store_memory(
                content=f"Tool Usage: {tool_name} for {problem_context} - {outcome.value}",
                memory_type=MemoryType.TOOL_USAGE,
                importance=self._outcome_to_importance(outcome),
                tags=[tool_name, outcome.value],
                metadata={
                    "record_id": record_id,
                    "effectiveness_score": effectiveness_score
                }
            )
            
            self.stats["total_tool_usages"] += 1
            
            logger.debug(f"Recorded tool usage: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to record tool usage: {e}")
            raise
    
    async def record_solution_outcome(
        self,
        problem_description: str,
        solution_approach: str,
        tools_used: List[str],
        total_execution_time: float,
        outcome_type: OutcomeType,
        success_rate: float,
        quality_metrics: Optional[Dict[str, Any]] = None,
        lessons_learned: Optional[List[str]] = None,
        improvement_suggestions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a complete solution outcome.
        
        Args:
            problem_description: Description of the problem solved
            solution_approach: Approach used to solve the problem
            tools_used: List of tools used in the solution
            total_execution_time: Total time taken for the solution
            outcome_type: Type of outcome achieved
            success_rate: Success rate of the solution
            quality_metrics: Quality metrics for the solution
            lessons_learned: Lessons learned from the solution
            improvement_suggestions: Suggestions for improvement
            metadata: Additional metadata
            
        Returns:
            Outcome ID
        """
        try:
            outcome_id = self._generate_outcome_id()
            
            outcome = SolutionOutcome(
                outcome_id=outcome_id,
                problem_description=problem_description,
                solution_approach=solution_approach,
                tools_used=tools_used,
                total_execution_time=total_execution_time,
                outcome_type=outcome_type,
                success_rate=success_rate,
                quality_metrics=quality_metrics or {},
                lessons_learned=lessons_learned or [],
                improvement_suggestions=improvement_suggestions or [],
                metadata=metadata or {}
            )
            
            self.solution_outcomes[outcome_id] = outcome
            
            vector_doc = VectorDocument(
                id=outcome_id,
                content=f"Problem: {problem_description}\nSolution: {solution_approach}\nOutcome: {outcome_type.value}",
                metadata={
                    "type": "solution_outcome",
                    "tools_used": tools_used,
                    "outcome_type": outcome_type.value,
                    "success_rate": success_rate,
                    "execution_time": total_execution_time,
                    "created_at": outcome.created_at.isoformat(),
                    **(metadata or {})
                }
            )
            
            await self.vector_store.add_document(vector_doc)
            
            await self.long_term_memory.store_memory(
                content=f"Solution Outcome: {problem_description} - {outcome_type.value}",
                memory_type=MemoryType.EXPERIENCE,
                importance=self._outcome_to_importance(outcome_type),
                tags=["solution", outcome_type.value] + tools_used,
                metadata={
                    "outcome_id": outcome_id,
                    "success_rate": success_rate,
                    "tools_used": tools_used
                }
            )
            
            self.stats["total_solutions"] += 1
            self._update_average_success_rate(success_rate)
            
            logger.debug(f"Recorded solution outcome: {outcome_id}")
            return outcome_id
            
        except Exception as e:
            logger.error(f"Failed to record solution outcome: {e}")
            raise
    
    async def discover_learning_insights(self) -> List[LearningInsight]:
        """
        Discover learning insights from experience data.
        
        Returns:
            List of discovered learning insights
        """
        try:
            insights = []
            
            tool_insights = await self._analyze_tool_effectiveness_patterns()
            insights.extend(tool_insights)
            
            complexity_insights = await self._analyze_problem_complexity_patterns()
            insights.extend(complexity_insights)
            
            approach_insights = await self._analyze_solution_approach_patterns()
            insights.extend(approach_insights)
            
            for insight in insights:
                if insight.confidence_score >= self.learning_confidence_threshold:
                    self.learning_insights[insight.insight_id] = insight
                    self.stats["insights_discovered"] += 1
            
            logger.info(f"Discovered {len(insights)} learning insights")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to discover learning insights: {e}")
            return []
    
    async def get_tool_effectiveness_profile(self, tool_name: str) -> Optional[ToolEffectivenessProfile]:
        """Get effectiveness profile for a specific tool."""
        return self.tool_effectiveness_profiles.get(tool_name)
    
    async def get_similar_experiences(
        self,
        problem_context: str,
        limit: int = 10
    ) -> List[Tuple[ToolUsageRecord, float]]:
        """
        Get similar experiences based on problem context.
        
        Args:
            problem_context: Context to find similar experiences for
            limit: Maximum number of results
            
        Returns:
            List of (record, similarity_score) tuples
        """
        try:
            search_results = await self.vector_store.search(
                query=problem_context,
                n_results=limit,
                where={"type": "tool_usage_record"}
            )
            
            similar_experiences = []
            
            for result in search_results:
                record_id = result.document.id
                if record_id in self.tool_usage_records:
                    record = self.tool_usage_records[record_id]
                    similar_experiences.append((record, result.similarity_score))
            
            logger.debug(f"Found {len(similar_experiences)} similar experiences")
            return similar_experiences
            
        except Exception as e:
            logger.error(f"Failed to get similar experiences: {e}")
            return []
    
    async def cleanup(self) -> None:
        """Cleanup experience database resources."""
        logger.info("Cleaning up Experience Database")
        
        try:
            self._shutdown_event.set()
            
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            await self._save_experience_data()
            
            logger.info("Experience Database cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Experience Database cleanup: {e}")
    
    async def _update_tool_effectiveness_profile(
        self,
        tool_name: str,
        record: ToolUsageRecord
    ) -> None:
        """Update tool effectiveness profile with new usage record."""
        try:
            if tool_name not in self.tool_effectiveness_profiles:
                profile = ToolEffectivenessProfile(
                    tool_name=tool_name,
                    total_usage_count=0,
                    success_rate=0.0,
                    average_execution_time=0.0
                )
                self.tool_effectiveness_profiles[tool_name] = profile
            else:
                profile = self.tool_effectiveness_profiles[tool_name]
            
            profile.total_usage_count += 1
            
            is_success = record.outcome in [OutcomeType.SUCCESS, OutcomeType.PARTIAL_SUCCESS]
            current_successes = profile.success_rate * (profile.total_usage_count - 1)
            if is_success:
                current_successes += 1
            profile.success_rate = current_successes / profile.total_usage_count
            
            total_time = profile.average_execution_time * (profile.total_usage_count - 1)
            profile.average_execution_time = (total_time + record.execution_time) / profile.total_usage_count
            
            context_key = self._extract_context_key(record.problem_context)
            if context_key not in profile.context_effectiveness:
                profile.context_effectiveness[context_key] = record.effectiveness_score
            else:
                current_avg = profile.context_effectiveness[context_key]
                profile.context_effectiveness[context_key] = (current_avg + record.effectiveness_score) / 2
            
            if not is_success and record.error_details:
                error_pattern = self._extract_error_pattern(record.error_details)
                if error_pattern not in profile.common_failure_modes:
                    profile.common_failure_modes.append(error_pattern)
            
            profile.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update tool effectiveness profile for {tool_name}: {e}")
    
    async def _analyze_tool_effectiveness_patterns(self) -> List[LearningInsight]:
        """Analyze tool effectiveness patterns."""
        insights = []
        
        try:
            for tool_name, profile in self.tool_effectiveness_profiles.items():
                if profile.total_usage_count >= self.min_usage_for_profile:
                    if profile.success_rate > 0.8:
                        insight = LearningInsight(
                            insight_id=self._generate_insight_id(),
                            pattern_type=LearningPattern.TOOL_EFFECTIVENESS,
                            description=f"Tool {tool_name} shows high effectiveness with {profile.success_rate:.2f} success rate",
                            confidence_score=min(profile.success_rate, 0.95),
                            supporting_evidence=[f"Success rate: {profile.success_rate:.2f}", f"Usage count: {profile.total_usage_count}"],
                            actionable_recommendations=[f"Prioritize {tool_name} for similar contexts"]
                        )
                        insights.append(insight)
                    
                    elif profile.success_rate < 0.4:
                        insight = LearningInsight(
                            insight_id=self._generate_insight_id(),
                            pattern_type=LearningPattern.TOOL_EFFECTIVENESS,
                            description=f"Tool {tool_name} shows low effectiveness with {profile.success_rate:.2f} success rate",
                            confidence_score=1.0 - profile.success_rate,
                            supporting_evidence=[f"Success rate: {profile.success_rate:.2f}", f"Common failures: {profile.common_failure_modes}"],
                            actionable_recommendations=[f"Investigate alternatives to {tool_name}", "Improve tool implementation"]
                        )
                        insights.append(insight)
            
        except Exception as e:
            logger.error(f"Failed to analyze tool effectiveness patterns: {e}")
        
        return insights
    
    async def _analyze_problem_complexity_patterns(self) -> List[LearningInsight]:
        """Analyze problem complexity patterns."""
        insights = []
        
        try:
            execution_times = []
            success_rates = []
            
            for outcome in self.solution_outcomes.values():
                execution_times.append(outcome.total_execution_time)
                success_rates.append(outcome.success_rate)
            
            if len(execution_times) >= 10:
                avg_time = statistics.mean(execution_times)
                
                long_tasks = [sr for et, sr in zip(execution_times, success_rates) if et > avg_time]
                short_tasks = [sr for et, sr in zip(execution_times, success_rates) if et <= avg_time]
                
                if long_tasks and short_tasks:
                    long_avg = statistics.mean(long_tasks)
                    short_avg = statistics.mean(short_tasks)
                    
                    if abs(long_avg - short_avg) > 0.2:
                        insight = LearningInsight(
                            insight_id=self._generate_insight_id(),
                            pattern_type=LearningPattern.PROBLEM_COMPLEXITY,
                            description=f"Complex problems (longer execution) have different success patterns",
                            confidence_score=0.8,
                            supporting_evidence=[f"Long tasks avg success: {long_avg:.2f}", f"Short tasks avg success: {short_avg:.2f}"],
                            actionable_recommendations=["Adjust approach based on estimated complexity"]
                        )
                        insights.append(insight)
            
        except Exception as e:
            logger.error(f"Failed to analyze problem complexity patterns: {e}")
        
        return insights
    
    async def _analyze_solution_approach_patterns(self) -> List[LearningInsight]:
        """Analyze solution approach patterns."""
        insights = []
        
        try:
            approach_outcomes = {}
            
            for outcome in self.solution_outcomes.values():
                approach = outcome.solution_approach
                if approach not in approach_outcomes:
                    approach_outcomes[approach] = []
                approach_outcomes[approach].append(outcome)
            
            for approach, outcomes in approach_outcomes.items():
                if len(outcomes) >= 3:
                    success_rates = [o.success_rate for o in outcomes]
                    avg_success = statistics.mean(success_rates)
                    
                    if avg_success > 0.8:
                        insight = LearningInsight(
                            insight_id=self._generate_insight_id(),
                            pattern_type=LearningPattern.SOLUTION_APPROACH,
                            description=f"Solution approach '{approach}' shows high effectiveness",
                            confidence_score=min(avg_success, 0.9),
                            supporting_evidence=[f"Average success rate: {avg_success:.2f}", f"Sample size: {len(outcomes)}"],
                            actionable_recommendations=[f"Prefer '{approach}' approach for similar problems"]
                        )
                        insights.append(insight)
            
        except Exception as e:
            logger.error(f"Failed to analyze solution approach patterns: {e}")
        
        return insights
    
    async def _start_background_processes(self) -> None:
        """Start background learning processes."""
        logger.debug("Starting experience database background processes")
        
        insight_task = asyncio.create_task(self._periodic_insight_discovery())
        self._background_tasks.append(insight_task)
        
        profile_task = asyncio.create_task(self._periodic_profile_maintenance())
        self._background_tasks.append(profile_task)
    
    async def _periodic_insight_discovery(self) -> None:
        """Background process for periodic insight discovery."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(21600)
                await self.discover_learning_insights()
                
            except Exception as e:
                logger.error(f"Error in periodic insight discovery: {e}")
                await asyncio.sleep(3600)
    
    async def _periodic_profile_maintenance(self) -> None:
        """Background process for profile maintenance."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(43200)
                
                for tool_name, profile in list(self.tool_effectiveness_profiles.items()):
                    if profile.total_usage_count < 2 and (datetime.now() - profile.last_updated).days > 30:
                        del self.tool_effectiveness_profiles[tool_name]
                
            except Exception as e:
                logger.error(f"Error in periodic profile maintenance: {e}")
                await asyncio.sleep(3600)
    
    async def _load_experience_data(self) -> None:
        """Load experience data from storage."""
        pass
    
    async def _save_experience_data(self) -> None:
        """Save experience data to storage."""
        pass
    
    def _calculate_effectiveness_score(
        self,
        outcome: OutcomeType,
        execution_time: float,
        success_metrics: Dict[str, Any]
    ) -> float:
        """Calculate effectiveness score for a tool usage."""
        base_score = {
            OutcomeType.SUCCESS: 1.0,
            OutcomeType.PARTIAL_SUCCESS: 0.7,
            OutcomeType.FAILURE: 0.2,
            OutcomeType.TIMEOUT: 0.1,
            OutcomeType.ERROR: 0.0
        }.get(outcome, 0.0)
        
        time_factor = max(0.5, 1.0 - (execution_time / 300.0))  # Normalize to 5 minutes
        
        metrics_factor = 1.0
        if success_metrics:
            quality_score = success_metrics.get("quality_score", 1.0)
            metrics_factor = min(1.0, max(0.5, quality_score))
        
        return base_score * time_factor * metrics_factor
    
    def _outcome_to_importance(self, outcome: OutcomeType) -> MemoryImportance:
        """Convert outcome type to memory importance."""
        outcome_mapping = {
            OutcomeType.SUCCESS: MemoryImportance.HIGH,
            OutcomeType.PARTIAL_SUCCESS: MemoryImportance.MEDIUM,
            OutcomeType.FAILURE: MemoryImportance.MEDIUM,
            OutcomeType.TIMEOUT: MemoryImportance.LOW,
            OutcomeType.ERROR: MemoryImportance.MEDIUM
        }
        return outcome_mapping.get(outcome, MemoryImportance.MEDIUM)
    
    def _extract_context_key(self, problem_context: str) -> str:
        """Extract a key from problem context for categorization."""
        words = problem_context.lower().split()[:3]
        return "_".join(words)
    
    def _extract_error_pattern(self, error_details: str) -> str:
        """Extract error pattern from error details."""
        if "timeout" in error_details.lower():
            return "timeout"
        elif "connection" in error_details.lower():
            return "connection_error"
        elif "permission" in error_details.lower():
            return "permission_error"
        else:
            return "unknown_error"
    
    def _update_average_success_rate(self, success_rate: float) -> None:
        """Update the average success rate metric."""
        current_avg = self.stats["average_success_rate"]
        solution_count = self.stats["total_solutions"]
        
        if solution_count == 1:
            self.stats["average_success_rate"] = success_rate
        else:
            new_avg = ((current_avg * (solution_count - 1)) + success_rate) / solution_count
            self.stats["average_success_rate"] = new_avg
    
    def _generate_record_id(self) -> str:
        """Generate unique record ID."""
        timestamp = int(datetime.now().timestamp())
        return f"record_{timestamp}_{len(self.tool_usage_records)}"
    
    def _generate_outcome_id(self) -> str:
        """Generate unique outcome ID."""
        timestamp = int(datetime.now().timestamp())
        return f"outcome_{timestamp}_{len(self.solution_outcomes)}"
    
    def _generate_insight_id(self) -> str:
        """Generate unique insight ID."""
        timestamp = int(datetime.now().timestamp())
        return f"insight_{timestamp}_{len(self.learning_insights)}"
    
    def get_experience_stats(self) -> Dict[str, Any]:
        """Get experience database statistics."""
        return {
            "tool_usage_records": len(self.tool_usage_records),
            "solution_outcomes": len(self.solution_outcomes),
            "learning_insights": len(self.learning_insights),
            "tool_profiles": len(self.tool_effectiveness_profiles),
            "performance_stats": self.stats.copy()
        }
