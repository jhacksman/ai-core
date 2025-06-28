"""
Research Memory - Scaffolding component for storing and indexing research findings.

This module stores and indexes research findings, problem-solution mappings,
and tool creation decisions for the Venice.ai scaffolding system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from .vector_store import VectorStore, VectorDocument, SearchResult
from .long_term_memory import LongTermMemory, Memory, MemoryType, MemoryImportance
from ..venice.client import VeniceClient

logger = logging.getLogger(__name__)


class ResearchCategory(Enum):
    """Categories of research findings."""
    PROBLEM_ANALYSIS = "problem_analysis"
    SOLUTION_MAPPING = "solution_mapping"
    TOOL_CREATION = "tool_creation"
    API_RESEARCH = "api_research"
    TECHNOLOGY_STACK = "technology_stack"
    BEST_PRACTICES = "best_practices"
    IMPLEMENTATION_PATTERNS = "implementation_patterns"
    FAILURE_ANALYSIS = "failure_analysis"


class ResearchConfidence(Enum):
    """Confidence levels for research findings."""
    VERIFIED = "verified"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EXPERIMENTAL = "experimental"


@dataclass
class ResearchFinding:
    """Represents a research finding in the scaffolding system."""
    finding_id: str
    title: str
    content: str
    category: ResearchCategory
    confidence: ResearchConfidence
    sources: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    problem_context: Optional[str] = None
    solution_mapping: Optional[Dict[str, Any]] = None
    tool_decisions: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    effectiveness_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemSolutionMapping:
    """Maps problems to solutions with effectiveness tracking."""
    mapping_id: str
    problem_description: str
    problem_hash: str
    solution_approach: str
    tools_created: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    usage_count: int = 0
    average_completion_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCreationDecision:
    """Records decisions made during tool creation."""
    decision_id: str
    problem_context: str
    tool_name: str
    creation_strategy: str
    implementation_approach: str
    dependencies_chosen: List[str] = field(default_factory=list)
    alternatives_considered: List[str] = field(default_factory=list)
    decision_rationale: str = ""
    outcome_success: Optional[bool] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearchMemory:
    """
    Scaffolding component for storing and indexing research findings.
    
    Manages research findings, problem-solution mappings, and tool creation
    decisions to improve the Venice.ai scaffolding system's effectiveness.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        vector_store: VectorStore,
        long_term_memory: LongTermMemory,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the research memory system.
        
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
        
        self.research_findings: Dict[str, ResearchFinding] = {}
        self.problem_solution_mappings: Dict[str, ProblemSolutionMapping] = {}
        self.tool_creation_decisions: Dict[str, ToolCreationDecision] = {}
        
        self.stats = {
            "findings_stored": 0,
            "mappings_created": 0,
            "decisions_recorded": 0,
            "successful_retrievals": 0,
            "average_effectiveness": 0.0
        }
        
        self.max_findings = self.config.get("max_findings", 50000)
        self.effectiveness_threshold = self.config.get("effectiveness_threshold", 0.7)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
    
    async def initialize(self) -> None:
        """Initialize the research memory system."""
        logger.info("Initializing Research Memory")
        
        try:
            await self._load_research_data()
            
            logger.info(f"Research Memory initialized with {len(self.research_findings)} findings")
            
        except Exception as e:
            logger.error(f"Failed to initialize Research Memory: {e}")
            raise
    
    async def store_research_finding(
        self,
        title: str,
        content: str,
        category: ResearchCategory,
        confidence: ResearchConfidence = ResearchConfidence.MEDIUM,
        sources: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        problem_context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a research finding.
        
        Args:
            title: Finding title
            content: Finding content
            category: Research category
            confidence: Confidence level
            sources: Information sources
            tags: Associated tags
            problem_context: Related problem context
            metadata: Additional metadata
            
        Returns:
            Finding ID
        """
        try:
            finding_id = self._generate_finding_id()
            
            finding = ResearchFinding(
                finding_id=finding_id,
                title=title,
                content=content,
                category=category,
                confidence=confidence,
                sources=sources or [],
                tags=tags or [],
                problem_context=problem_context,
                metadata=metadata or {}
            )
            
            self.research_findings[finding_id] = finding
            
            vector_doc = VectorDocument(
                id=finding_id,
                content=f"{title}\n\n{content}",
                metadata={
                    "type": "research_finding",
                    "category": category.value,
                    "confidence": confidence.value,
                    "tags": tags or [],
                    "sources": sources or [],
                    "created_at": finding.created_at.isoformat(),
                    **(metadata or {})
                }
            )
            
            await self.vector_store.add_document(vector_doc)
            
            await self.long_term_memory.store_memory(
                content=f"Research Finding: {title}\n{content}",
                memory_type=MemoryType.RESEARCH,
                importance=self._confidence_to_importance(confidence),
                tags=tags or [],
                metadata={
                    "finding_id": finding_id,
                    "category": category.value,
                    "sources": sources or []
                }
            )
            
            self.stats["findings_stored"] += 1
            
            logger.debug(f"Stored research finding: {finding_id}")
            return finding_id
            
        except Exception as e:
            logger.error(f"Failed to store research finding: {e}")
            raise
    
    async def create_problem_solution_mapping(
        self,
        problem_description: str,
        solution_approach: str,
        tools_created: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a problem-solution mapping.
        
        Args:
            problem_description: Description of the problem
            solution_approach: Approach used to solve the problem
            tools_created: List of tools created for this solution
            metadata: Additional metadata
            
        Returns:
            Mapping ID
        """
        try:
            mapping_id = self._generate_mapping_id()
            problem_hash = str(hash(problem_description.lower().strip()))
            
            mapping = ProblemSolutionMapping(
                mapping_id=mapping_id,
                problem_description=problem_description,
                problem_hash=problem_hash,
                solution_approach=solution_approach,
                tools_created=tools_created or [],
                metadata=metadata or {}
            )
            
            self.problem_solution_mappings[mapping_id] = mapping
            
            vector_doc = VectorDocument(
                id=mapping_id,
                content=f"Problem: {problem_description}\nSolution: {solution_approach}",
                metadata={
                    "type": "problem_solution_mapping",
                    "problem_hash": problem_hash,
                    "tools_created": tools_created or [],
                    "created_at": mapping.created_at.isoformat(),
                    **(metadata or {})
                }
            )
            
            await self.vector_store.add_document(vector_doc)
            
            self.stats["mappings_created"] += 1
            
            logger.debug(f"Created problem-solution mapping: {mapping_id}")
            return mapping_id
            
        except Exception as e:
            logger.error(f"Failed to create problem-solution mapping: {e}")
            raise
    
    async def record_tool_creation_decision(
        self,
        problem_context: str,
        tool_name: str,
        creation_strategy: str,
        implementation_approach: str,
        dependencies_chosen: Optional[List[str]] = None,
        alternatives_considered: Optional[List[str]] = None,
        decision_rationale: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a tool creation decision.
        
        Args:
            problem_context: Context of the problem being solved
            tool_name: Name of the tool created
            creation_strategy: Strategy used for creation
            implementation_approach: Implementation approach chosen
            dependencies_chosen: Dependencies selected
            alternatives_considered: Alternative approaches considered
            decision_rationale: Rationale for the decision
            metadata: Additional metadata
            
        Returns:
            Decision ID
        """
        try:
            decision_id = self._generate_decision_id()
            
            decision = ToolCreationDecision(
                decision_id=decision_id,
                problem_context=problem_context,
                tool_name=tool_name,
                creation_strategy=creation_strategy,
                implementation_approach=implementation_approach,
                dependencies_chosen=dependencies_chosen or [],
                alternatives_considered=alternatives_considered or [],
                decision_rationale=decision_rationale,
                metadata=metadata or {}
            )
            
            self.tool_creation_decisions[decision_id] = decision
            
            vector_doc = VectorDocument(
                id=decision_id,
                content=f"Tool Creation: {tool_name}\nContext: {problem_context}\nStrategy: {creation_strategy}\nRationale: {decision_rationale}",
                metadata={
                    "type": "tool_creation_decision",
                    "tool_name": tool_name,
                    "creation_strategy": creation_strategy,
                    "implementation_approach": implementation_approach,
                    "created_at": decision.created_at.isoformat(),
                    **(metadata or {})
                }
            )
            
            await self.vector_store.add_document(vector_doc)
            
            self.stats["decisions_recorded"] += 1
            
            logger.debug(f"Recorded tool creation decision: {decision_id}")
            return decision_id
            
        except Exception as e:
            logger.error(f"Failed to record tool creation decision: {e}")
            raise
    
    async def find_similar_problems(
        self,
        problem_description: str,
        limit: int = 5
    ) -> List[Tuple[ProblemSolutionMapping, float]]:
        """
        Find similar problems and their solutions.
        
        Args:
            problem_description: Problem to find similarities for
            limit: Maximum number of results
            
        Returns:
            List of (mapping, similarity_score) tuples
        """
        try:
            search_results = await self.vector_store.search(
                query=problem_description,
                n_results=limit * 2,
                where={"type": "problem_solution_mapping"}
            )
            
            similar_problems = []
            
            for result in search_results:
                mapping_id = result.document.id
                if mapping_id in self.problem_solution_mappings:
                    mapping = self.problem_solution_mappings[mapping_id]
                    
                    mapping.last_used = datetime.now()
                    mapping.usage_count += 1
                    
                    similar_problems.append((mapping, result.similarity_score))
            
            similar_problems.sort(key=lambda x: x[1], reverse=True)
            similar_problems = similar_problems[:limit]
            
            self.stats["successful_retrievals"] += len(similar_problems)
            
            logger.debug(f"Found {len(similar_problems)} similar problems")
            return similar_problems
            
        except Exception as e:
            logger.error(f"Failed to find similar problems: {e}")
            return []
    
    async def get_relevant_research(
        self,
        query: str,
        categories: Optional[List[ResearchCategory]] = None,
        confidence_threshold: Optional[ResearchConfidence] = None,
        limit: int = 10
    ) -> List[Tuple[ResearchFinding, float]]:
        """
        Get relevant research findings for a query.
        
        Args:
            query: Search query
            categories: Filter by categories
            confidence_threshold: Minimum confidence level
            limit: Maximum number of results
            
        Returns:
            List of (finding, relevance_score) tuples
        """
        try:
            where_filters = {"type": "research_finding"}
            
            if categories:
                where_filters["category"] = {"$in": [cat.value for cat in categories]}
            
            search_results = await self.vector_store.search(
                query=query,
                n_results=limit * 2,
                where=where_filters
            )
            
            relevant_research = []
            
            for result in search_results:
                finding_id = result.document.id
                if finding_id in self.research_findings:
                    finding = self.research_findings[finding_id]
                    
                    if confidence_threshold and not self._meets_confidence_threshold(
                        finding.confidence, confidence_threshold
                    ):
                        continue
                    
                    finding.usage_count += 1
                    
                    relevant_research.append((finding, result.similarity_score))
            
            relevant_research.sort(key=lambda x: x[1], reverse=True)
            relevant_research = relevant_research[:limit]
            
            logger.debug(f"Found {len(relevant_research)} relevant research findings")
            return relevant_research
            
        except Exception as e:
            logger.error(f"Failed to get relevant research: {e}")
            return []
    
    async def get_tool_creation_insights(
        self,
        problem_context: str,
        limit: int = 5
    ) -> List[Tuple[ToolCreationDecision, float]]:
        """
        Get insights from previous tool creation decisions.
        
        Args:
            problem_context: Context to find insights for
            limit: Maximum number of results
            
        Returns:
            List of (decision, relevance_score) tuples
        """
        try:
            search_results = await self.vector_store.search(
                query=problem_context,
                n_results=limit * 2,
                where={"type": "tool_creation_decision"}
            )
            
            insights = []
            
            for result in search_results:
                decision_id = result.document.id
                if decision_id in self.tool_creation_decisions:
                    decision = self.tool_creation_decisions[decision_id]
                    insights.append((decision, result.similarity_score))
            
            insights.sort(key=lambda x: x[1], reverse=True)
            insights = insights[:limit]
            
            logger.debug(f"Found {len(insights)} tool creation insights")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get tool creation insights: {e}")
            return []
    
    async def update_solution_effectiveness(
        self,
        mapping_id: str,
        success: bool,
        completion_time: Optional[float] = None
    ) -> bool:
        """
        Update the effectiveness of a solution mapping.
        
        Args:
            mapping_id: ID of the mapping to update
            success: Whether the solution was successful
            completion_time: Time taken to complete the solution
            
        Returns:
            True if update was successful
        """
        try:
            if mapping_id not in self.problem_solution_mappings:
                return False
            
            mapping = self.problem_solution_mappings[mapping_id]
            
            total_uses = mapping.usage_count
            current_successes = mapping.success_rate * total_uses
            
            if success:
                current_successes += 1
            
            mapping.success_rate = current_successes / (total_uses + 1) if total_uses > 0 else (1.0 if success else 0.0)
            
            if completion_time is not None:
                if mapping.average_completion_time == 0.0:
                    mapping.average_completion_time = completion_time
                else:
                    mapping.average_completion_time = (
                        (mapping.average_completion_time * total_uses) + completion_time
                    ) / (total_uses + 1)
            
            mapping.usage_count += 1
            mapping.last_used = datetime.now()
            
            logger.debug(f"Updated solution effectiveness for mapping: {mapping_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update solution effectiveness: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup research memory resources."""
        logger.debug("Cleaning up Research Memory")
        
        try:
            await self._save_research_data()
            
        except Exception as e:
            logger.error(f"Error during Research Memory cleanup: {e}")
    
    async def _load_research_data(self) -> None:
        """Load research data from storage."""
        pass
    
    async def _save_research_data(self) -> None:
        """Save research data to storage."""
        pass
    
    def _confidence_to_importance(self, confidence: ResearchConfidence) -> MemoryImportance:
        """Convert research confidence to memory importance."""
        confidence_mapping = {
            ResearchConfidence.VERIFIED: MemoryImportance.CRITICAL,
            ResearchConfidence.HIGH: MemoryImportance.HIGH,
            ResearchConfidence.MEDIUM: MemoryImportance.MEDIUM,
            ResearchConfidence.LOW: MemoryImportance.LOW,
            ResearchConfidence.EXPERIMENTAL: MemoryImportance.LOW
        }
        return confidence_mapping.get(confidence, MemoryImportance.MEDIUM)
    
    def _meets_confidence_threshold(
        self,
        confidence: ResearchConfidence,
        threshold: ResearchConfidence
    ) -> bool:
        """Check if confidence meets threshold."""
        confidence_levels = {
            ResearchConfidence.EXPERIMENTAL: 1,
            ResearchConfidence.LOW: 2,
            ResearchConfidence.MEDIUM: 3,
            ResearchConfidence.HIGH: 4,
            ResearchConfidence.VERIFIED: 5
        }
        
        return confidence_levels.get(confidence, 0) >= confidence_levels.get(threshold, 0)
    
    def _generate_finding_id(self) -> str:
        """Generate unique finding ID."""
        timestamp = int(datetime.now().timestamp())
        return f"finding_{timestamp}_{len(self.research_findings)}"
    
    def _generate_mapping_id(self) -> str:
        """Generate unique mapping ID."""
        timestamp = int(datetime.now().timestamp())
        return f"mapping_{timestamp}_{len(self.problem_solution_mappings)}"
    
    def _generate_decision_id(self) -> str:
        """Generate unique decision ID."""
        timestamp = int(datetime.now().timestamp())
        return f"decision_{timestamp}_{len(self.tool_creation_decisions)}"
    
    def get_research_stats(self) -> Dict[str, Any]:
        """Get research memory statistics."""
        return {
            "research_findings": len(self.research_findings),
            "problem_solution_mappings": len(self.problem_solution_mappings),
            "tool_creation_decisions": len(self.tool_creation_decisions),
            "performance_stats": self.stats.copy()
        }
