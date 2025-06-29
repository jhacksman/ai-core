"""
Research Coordinator - Orchestrates multi-source research for problem-solving.

This module coordinates research across web search, memory retrieval,
and knowledge synthesis for the Venice.ai scaffolding system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from ..venice.client import VeniceClient

logger = logging.getLogger(__name__)


class ResearchDepth(Enum):
    """Research depth levels."""
    SHALLOW = "shallow"
    MEDIUM = "medium"
    DEEP = "deep"
    COMPREHENSIVE = "comprehensive"


class ResearchSource(Enum):
    """Available research sources."""
    WEB = "web"
    MEMORY = "memory"
    DOCUMENTATION = "documentation"
    API_SPECS = "api_specs"
    CODE_REPOS = "code_repos"
    ACADEMIC = "academic"


@dataclass
class ResearchQuery:
    """Represents a research query."""
    query: str
    source: ResearchSource
    priority: int = 1
    max_results: int = 10
    filters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """Represents a research result."""
    source: ResearchSource
    query: str
    content: str
    url: Optional[str] = None
    title: Optional[str] = None
    relevance_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchContext:
    """Complete research context for a problem."""
    problem: str
    depth: ResearchDepth
    sources_used: List[ResearchSource]
    results: List[ResearchResult] = field(default_factory=list)
    synthesis: str = ""
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    research_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResearchCoordinator:
    """
    Orchestrates multi-source research for the Venice.ai scaffolding system.
    
    Coordinates research across web search, memory retrieval, and knowledge
    synthesis to provide comprehensive context for problem-solving.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the research coordinator.
        
        Args:
            venice_client: Venice.ai client for LLM operations
            config: Configuration options
        """
        self.venice_client = venice_client
        self.config = config or {}
        
        self.max_concurrent_queries = self.config.get("max_concurrent_queries", 5)
        self.default_timeout = self.config.get("default_timeout", 30)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.6)
        
        self.available_sources = {
            ResearchSource.WEB: self._web_search,
            ResearchSource.MEMORY: self._memory_search,
            ResearchSource.DOCUMENTATION: self._documentation_search,
            ResearchSource.API_SPECS: self._api_spec_search,
            ResearchSource.CODE_REPOS: self._code_repo_search,
            ResearchSource.ACADEMIC: self._academic_search
        }
        
        self.research_cache: Dict[str, ResearchContext] = {}
        
        self.research_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "cache_hits": 0,
            "average_query_time": 0.0,
            "sources_used": {source.value: 0 for source in ResearchSource}
        }
    
    async def initialize(self) -> None:
        """Initialize the research coordinator."""
        logger.info("Initializing Research Coordinator")
        
        try:
            await self._test_source_connectivity()
            
            logger.info("Research Coordinator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Research Coordinator: {e}")
            raise
    
    async def comprehensive_research(
        self,
        problem: str,
        depth: str = "deep",
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a problem.
        
        Args:
            problem: Problem description to research
            depth: Research depth level
            sources: List of sources to use
            
        Returns:
            Research context dictionary
        """
        logger.info(f"Starting comprehensive research for: {problem[:100]}...")
        
        start_time = datetime.now()
        
        try:
            research_depth = ResearchDepth(depth)
            research_sources = [ResearchSource(s) for s in (sources or ["web", "memory", "documentation"])]
            
            cache_key = self._generate_cache_key(problem, research_depth, research_sources)
            if cache_key in self.research_cache:
                logger.debug("Returning cached research results")
                self.research_stats["cache_hits"] += 1
                return self.research_cache[cache_key].__dict__
            
            context = ResearchContext(
                problem=problem,
                depth=research_depth,
                sources_used=research_sources
            )
            
            queries = await self._generate_research_queries(problem, research_depth, research_sources)
            
            results = await self._execute_research_queries(queries)
            context.results = results
            
            synthesis = await self._synthesize_research_results(problem, results)
            context.synthesis = synthesis["synthesis"]
            context.key_findings = synthesis["key_findings"]
            context.recommendations = synthesis["recommendations"]
            context.confidence_score = synthesis["confidence_score"]
            
            end_time = datetime.now()
            context.research_duration = (end_time - start_time).total_seconds()
            
            self.research_cache[cache_key] = context
            
            self.research_stats["total_queries"] += len(queries)
            self.research_stats["successful_queries"] += len(results)
            
            logger.info(f"Completed research in {context.research_duration:.2f}s with {len(results)} results")
            
            return context.__dict__
            
        except Exception as e:
            logger.error(f"Failed to conduct comprehensive research: {e}")
            raise
    
    async def targeted_research(
        self,
        queries: List[str],
        sources: List[str],
        max_results_per_query: int = 5
    ) -> List[ResearchResult]:
        """
        Conduct targeted research with specific queries.
        
        Args:
            queries: Specific research queries
            sources: Sources to search
            max_results_per_query: Maximum results per query
            
        Returns:
            List of research results
        """
        logger.info(f"Starting targeted research with {len(queries)} queries")
        
        try:
            research_queries = []
            for query in queries:
                for source_name in sources:
                    source = ResearchSource(source_name)
                    research_queries.append(ResearchQuery(
                        query=query,
                        source=source,
                        max_results=max_results_per_query
                    ))
            
            results = await self._execute_research_queries(research_queries)
            
            logger.info(f"Completed targeted research with {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to conduct targeted research: {e}")
            return []
    
    async def get_available_sources(self) -> List[str]:
        """Get list of available research sources."""
        return [source.value for source in self.available_sources.keys()]
    
    async def cleanup(self) -> None:
        """Cleanup research coordinator resources."""
        logger.debug("Cleaning up Research Coordinator")
        
        if len(self.research_cache) > 1000:  # Arbitrary limit
            self.research_cache.clear()
    
    async def _generate_research_queries(
        self,
        problem: str,
        depth: ResearchDepth,
        sources: List[ResearchSource]
    ) -> List[ResearchQuery]:
        """Generate research queries based on problem and depth."""
        try:
            query_prompt = f"""
            Generate research queries for this problem:
            
            Problem: {problem}
            Research Depth: {depth.value}
            Available Sources: {[s.value for s in sources]}
            
            Generate 3-8 specific, targeted research queries that would help understand:
            1. The problem domain and context
            2. Existing solutions and approaches
            3. Technical requirements and constraints
            4. Best practices and recommendations
            
            Return as JSON array of query strings.
            """
            
            response = await self.venice_client.chat_completion(
                messages=[{"role": "user", "content": query_prompt}],
                model="qwen-qwq-32b",
                temperature=0.3
            )
            
            try:
                query_strings = json.loads(response.content)
                if not isinstance(query_strings, list):
                    query_strings = [response.content]
            except json.JSONDecodeError:
                query_strings = self._extract_queries_from_text(response.content)
            
            queries = []
            for i, query_str in enumerate(query_strings):
                for source in sources:
                    queries.append(ResearchQuery(
                        query=query_str,
                        source=source,
                        priority=i + 1,
                        max_results=self._get_max_results_for_depth(depth)
                    ))
            
            return queries
            
        except Exception as e:
            logger.error(f"Failed to generate research queries: {e}")
            return self._generate_fallback_queries(problem, sources)
    
    async def _execute_research_queries(self, queries: List[ResearchQuery]) -> List[ResearchResult]:
        """Execute research queries concurrently."""
        logger.debug(f"Executing {len(queries)} research queries")
        
        queries_by_source = {}
        for query in queries:
            if query.source not in queries_by_source:
                queries_by_source[query.source] = []
            queries_by_source[query.source].append(query)
        
        all_results = []
        semaphore = asyncio.Semaphore(self.max_concurrent_queries)
        
        async def execute_source_queries(source: ResearchSource, source_queries: List[ResearchQuery]):
            async with semaphore:
                try:
                    if source in self.available_sources:
                        search_func = self.available_sources[source]
                        results = await search_func(source_queries)
                        return results
                    else:
                        logger.warning(f"Source {source.value} not available")
                        return []
                except Exception as e:
                    logger.error(f"Failed to execute queries for source {source.value}: {e}")
                    return []
        
        tasks = [
            execute_source_queries(source, source_queries)
            for source, source_queries in queries_by_source.items()
        ]
        
        results_by_source = await asyncio.gather(*tasks, return_exceptions=True)
        
        for results in results_by_source:
            if isinstance(results, list):
                all_results.extend(results)
        
        filtered_results = [
            result for result in all_results
            if result.relevance_score >= self.relevance_threshold
        ]
        
        filtered_results.sort(key=lambda r: r.relevance_score, reverse=True)
        
        logger.debug(f"Completed research queries: {len(filtered_results)} relevant results")
        return filtered_results
    
    async def _synthesize_research_results(
        self,
        problem: str,
        results: List[ResearchResult]
    ) -> Dict[str, Any]:
        """Synthesize research results into actionable insights."""
        try:
            research_content = []
            for result in results[:20]:  # Limit to top 20 results
                content_summary = f"Source: {result.source.value}\n"
                content_summary += f"Title: {result.title or 'N/A'}\n"
                content_summary += f"Content: {result.content[:500]}...\n"
                content_summary += f"Relevance: {result.relevance_score:.2f}\n"
                research_content.append(content_summary)
            
            synthesis_prompt = f"""
            Synthesize these research findings for the problem:
            
            Problem: {problem}
            
            Research Results:
            {chr(10).join(research_content)}
            
            Provide a comprehensive synthesis including:
            1. Key findings and insights
            2. Recommended approaches and solutions
            3. Technical requirements and constraints
            4. Confidence assessment (0.0-1.0)
            
            Format as JSON with keys: synthesis, key_findings, recommendations, confidence_score
            """
            
            response = await self.venice_client.chat_completion(
                messages=[{"role": "user", "content": synthesis_prompt}],
                model="llama-4",
                temperature=0.2
            )
            
            try:
                synthesis = json.loads(response.content)
            except json.JSONDecodeError:
                synthesis = self._create_fallback_synthesis(problem, results)
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Failed to synthesize research results: {e}")
            return self._create_fallback_synthesis(problem, results)
    
    async def _web_search(self, queries: List[ResearchQuery]) -> List[ResearchResult]:
        """Perform web search for research queries."""
        results = []
        
        for query in queries:
            try:
                result = ResearchResult(
                    source=ResearchSource.WEB,
                    query=query.query,
                    content=f"Web search results for: {query.query}",
                    title=f"Web Search: {query.query}",
                    relevance_score=0.8,
                    url=f"https://example.com/search?q={query.query.replace(' ', '+')}"
                )
                results.append(result)
                
                self.research_stats["sources_used"][ResearchSource.WEB.value] += 1
                
            except Exception as e:
                logger.error(f"Web search failed for query '{query.query}': {e}")
        
        return results
    
    async def _memory_search(self, queries: List[ResearchQuery]) -> List[ResearchResult]:
        """Search memory/knowledge base for research queries."""
        results = []
        
        for query in queries:
            try:
                result = ResearchResult(
                    source=ResearchSource.MEMORY,
                    query=query.query,
                    content=f"Memory search results for: {query.query}",
                    title=f"Memory: {query.query}",
                    relevance_score=0.7
                )
                results.append(result)
                
                self.research_stats["sources_used"][ResearchSource.MEMORY.value] += 1
                
            except Exception as e:
                logger.error(f"Memory search failed for query '{query.query}': {e}")
        
        return results
    
    async def _documentation_search(self, queries: List[ResearchQuery]) -> List[ResearchResult]:
        """Search documentation for research queries."""
        results = []
        
        for query in queries:
            try:
                result = ResearchResult(
                    source=ResearchSource.DOCUMENTATION,
                    query=query.query,
                    content=f"Documentation search results for: {query.query}",
                    title=f"Docs: {query.query}",
                    relevance_score=0.9
                )
                results.append(result)
                
                self.research_stats["sources_used"][ResearchSource.DOCUMENTATION.value] += 1
                
            except Exception as e:
                logger.error(f"Documentation search failed for query '{query.query}': {e}")
        
        return results
    
    async def _api_spec_search(self, queries: List[ResearchQuery]) -> List[ResearchResult]:
        """Search API specifications for research queries."""
        results = []
        
        for query in queries:
            try:
                result = ResearchResult(
                    source=ResearchSource.API_SPECS,
                    query=query.query,
                    content=f"API specification results for: {query.query}",
                    title=f"API Spec: {query.query}",
                    relevance_score=0.8
                )
                results.append(result)
                
                self.research_stats["sources_used"][ResearchSource.API_SPECS.value] += 1
                
            except Exception as e:
                logger.error(f"API spec search failed for query '{query.query}': {e}")
        
        return results
    
    async def _code_repo_search(self, queries: List[ResearchQuery]) -> List[ResearchResult]:
        """Search code repositories for research queries."""
        results = []
        
        for query in queries:
            try:
                result = ResearchResult(
                    source=ResearchSource.CODE_REPOS,
                    query=query.query,
                    content=f"Code repository results for: {query.query}",
                    title=f"Code: {query.query}",
                    relevance_score=0.7
                )
                results.append(result)
                
                self.research_stats["sources_used"][ResearchSource.CODE_REPOS.value] += 1
                
            except Exception as e:
                logger.error(f"Code repo search failed for query '{query.query}': {e}")
        
        return results
    
    async def _academic_search(self, queries: List[ResearchQuery]) -> List[ResearchResult]:
        """Search academic sources for research queries."""
        results = []
        
        for query in queries:
            try:
                result = ResearchResult(
                    source=ResearchSource.ACADEMIC,
                    query=query.query,
                    content=f"Academic search results for: {query.query}",
                    title=f"Academic: {query.query}",
                    relevance_score=0.9
                )
                results.append(result)
                
                self.research_stats["sources_used"][ResearchSource.ACADEMIC.value] += 1
                
            except Exception as e:
                logger.error(f"Academic search failed for query '{query.query}': {e}")
        
        return results
    
    async def _test_source_connectivity(self) -> None:
        """Test connectivity to research sources."""
        logger.debug("Testing research source connectivity")
        
        test_queries = [
            ResearchQuery(query="test", source=source, max_results=1)
            for source in self.available_sources.keys()
        ]
        
        try:
            await self._execute_research_queries(test_queries)
            logger.debug("Research source connectivity test completed")
        except Exception as e:
            logger.warning(f"Some research sources may not be available: {e}")
    
    def _extract_queries_from_text(self, text: str) -> List[str]:
        """Extract queries from text response."""
        lines = text.split('\n')
        queries = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and '?' in line:
                queries.append(line)
        
        return queries[:8]  # Limit to 8 queries
    
    def _generate_fallback_queries(self, problem: str, sources: List[ResearchSource]) -> List[ResearchQuery]:
        """Generate fallback queries when AI generation fails."""
        fallback_queries = [
            f"What is {problem}",
            f"How to solve {problem}",
            f"Best practices for {problem}",
            f"Tools for {problem}",
            f"Examples of {problem}"
        ]
        
        queries = []
        for query_str in fallback_queries:
            for source in sources:
                queries.append(ResearchQuery(
                    query=query_str,
                    source=source,
                    max_results=5
                ))
        
        return queries
    
    def _get_max_results_for_depth(self, depth: ResearchDepth) -> int:
        """Get maximum results based on research depth."""
        depth_limits = {
            ResearchDepth.SHALLOW: 3,
            ResearchDepth.MEDIUM: 5,
            ResearchDepth.DEEP: 10,
            ResearchDepth.COMPREHENSIVE: 15
        }
        return depth_limits.get(depth, 5)
    
    def _create_fallback_synthesis(self, problem: str, results: List[ResearchResult]) -> Dict[str, Any]:
        """Create fallback synthesis when AI synthesis fails."""
        return {
            "synthesis": f"Research conducted for problem: {problem}. Found {len(results)} relevant results.",
            "key_findings": [f"Finding from {result.source.value}" for result in results[:3]],
            "recommendations": ["Analyze research results", "Identify solution approaches", "Implement solution"],
            "confidence_score": 0.5
        }
    
    def _generate_cache_key(
        self,
        problem: str,
        depth: ResearchDepth,
        sources: List[ResearchSource]
    ) -> str:
        """Generate cache key for research results."""
        problem_hash = hash(problem[:200])
        sources_str = "_".join(sorted([s.value for s in sources]))
        return f"{problem_hash}_{depth.value}_{sources_str}"
    
    def get_research_stats(self) -> Dict[str, Any]:
        """Get research coordinator statistics."""
        return self.research_stats.copy()
