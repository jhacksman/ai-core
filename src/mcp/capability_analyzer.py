"""
Capability Analyzer for identifying tool gaps and suggesting implementations.

This module analyzes problems and existing capabilities to identify what new
tools need to be created for the Venice.ai scaffolding system.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

from ..venice.client import VeniceClient

logger = logging.getLogger(__name__)


class GapPriority(Enum):
    """Priority levels for capability gaps."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapCategory(Enum):
    """Categories of capability gaps."""
    DATA_ACCESS = "data_access"
    API_INTEGRATION = "api_integration"
    WEB_AUTOMATION = "web_automation"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    MONITORING = "monitoring"
    ANALYSIS = "analysis"
    WORKFLOW = "workflow"
    UNKNOWN = "unknown"


@dataclass
class ToolGap:
    """Represents a gap in available tool capabilities."""
    capability_name: str
    description: str
    priority: GapPriority
    category: GapCategory
    required_inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    similar_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImplementationSuggestion:
    """Suggested implementation approach for a tool gap."""
    strategy: str
    input_schema: Dict[str, Any]
    dependencies: List[str]
    integrations: List[str]
    tags: List[str]
    complexity_estimate: float
    confidence_score: float
    implementation_notes: str
    example_usage: Dict[str, Any]


class CapabilityAnalyzer:
    """
    Analyzer for identifying capability gaps and suggesting tool implementations.
    
    This is a core component of the Venice.ai scaffolding system that determines
    what new tools need to be created based on problem analysis.
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        existing_capabilities: Optional[List[str]] = None
    ):
        """
        Initialize the capability analyzer.
        
        Args:
            venice_client: Venice.ai client for LLM analysis
            existing_capabilities: List of existing tool capabilities
        """
        self.venice_client = venice_client
        self.existing_capabilities = set(existing_capabilities or [])
        self.capability_patterns = self._initialize_capability_patterns()
        self.implementation_strategies = self._initialize_implementation_strategies()
        
        self.analysis_cache: Dict[str, List[ToolGap]] = {}
        self.suggestion_cache: Dict[str, ImplementationSuggestion] = {}
    
    async def identify_gaps(
        self,
        problem_description: str,
        research_context: Dict[str, Any],
        existing_capabilities: Optional[List[str]] = None
    ) -> List[ToolGap]:
        """
        Identify capability gaps for solving a specific problem.
        
        Args:
            problem_description: Description of the problem to solve
            research_context: Context from research phase
            existing_capabilities: Current available capabilities
            
        Returns:
            List of identified tool gaps
        """
        logger.info(f"Analyzing capability gaps for problem: {problem_description[:100]}...")
        
        cache_key = self._generate_cache_key(problem_description, research_context)
        if cache_key in self.analysis_cache:
            logger.debug("Returning cached gap analysis")
            return self.analysis_cache[cache_key]
        
        try:
            if existing_capabilities:
                self.existing_capabilities.update(existing_capabilities)
            
            required_capabilities = await self._extract_required_capabilities(
                problem_description, research_context
            )
            
            missing_capabilities = self._find_missing_capabilities(required_capabilities)
            
            tool_gaps = []
            for capability in missing_capabilities:
                gap = await self._analyze_capability_gap(
                    capability, problem_description, research_context
                )
                if gap:
                    tool_gaps.append(gap)
            
            prioritized_gaps = self._prioritize_gaps(tool_gaps, problem_description)
            
            self.analysis_cache[cache_key] = prioritized_gaps
            
            logger.info(f"Identified {len(prioritized_gaps)} capability gaps")
            return prioritized_gaps
            
        except Exception as e:
            logger.error(f"Failed to identify capability gaps: {e}")
            return []
    
    async def suggest_implementation(
        self,
        gap: ToolGap,
        research_context: Dict[str, Any]
    ) -> ImplementationSuggestion:
        """
        Suggest implementation approach for a capability gap.
        
        Args:
            gap: Tool gap to implement
            research_context: Research context for implementation decisions
            
        Returns:
            Implementation suggestion
        """
        logger.info(f"Suggesting implementation for gap: {gap.capability_name}")
        
        cache_key = f"{gap.capability_name}_{hash(str(research_context))}"
        if cache_key in self.suggestion_cache:
            return self.suggestion_cache[cache_key]
        
        try:
            strategy = self._select_implementation_strategy(gap, research_context)
            
            input_schema = await self._generate_input_schema(gap, strategy)
            
            dependencies = self._determine_dependencies(gap, strategy)
            
            integrations = self._identify_integrations(gap, research_context)
            
            tags = self._generate_tags(gap, strategy)
            
            complexity = self._estimate_complexity(gap, strategy)
            
            confidence = self._calculate_confidence(gap, strategy, research_context)
            
            notes = await self._generate_implementation_notes(gap, strategy, research_context)
            
            example = self._create_example_usage(gap, input_schema)
            
            suggestion = ImplementationSuggestion(
                strategy=strategy,
                input_schema=input_schema,
                dependencies=dependencies,
                integrations=integrations,
                tags=tags,
                complexity_estimate=complexity,
                confidence_score=confidence,
                implementation_notes=notes,
                example_usage=example
            )
            
            self.suggestion_cache[cache_key] = suggestion
            
            logger.debug(f"Generated implementation suggestion with {confidence:.2f} confidence")
            return suggestion
            
        except Exception as e:
            logger.error(f"Failed to suggest implementation for {gap.capability_name}: {e}")
            return ImplementationSuggestion(
                strategy="basic_implementation",
                input_schema={"type": "object", "properties": {}},
                dependencies=["mcp"],
                integrations=[],
                tags=["basic"],
                complexity_estimate=5.0,
                confidence_score=0.5,
                implementation_notes="Basic implementation fallback",
                example_usage={}
            )
    
    async def analyze_problem_requirements(
        self,
        problem_description: str,
        research_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze problem requirements for tool creation.
        
        Args:
            problem_description: Description of the problem
            research_context: Research findings
            
        Returns:
            Analysis of problem requirements
        """
        logger.info("Analyzing problem requirements")
        
        try:
            analysis_prompt = f"""
            Analyze this problem for tool creation requirements:
            
            Problem: {problem_description}
            
            Research Context: {json.dumps(research_context, indent=2)}
            
            Provide a structured analysis including:
            1. Domain: What field/area does this problem belong to?
            2. Required Actions: What specific actions need to be performed?
            3. Data Requirements: What data sources or types are needed?
            4. Integration Points: What systems need to be integrated?
            5. Complexity Assessment: Rate complexity 1-10 and explain
            6. Success Criteria: How to measure if the problem is solved?
            
            Format as JSON with these keys: domain, actions, data_requirements, 
            integrations, complexity, success_criteria
            """
            
            response = await self.venice_client.chat_completion(
                messages=[{"role": "user", "content": analysis_prompt}],
                model="qwen-qwq-32b",
                temperature=0.3
            )
            
            try:
                requirements = json.loads(response.content)
            except json.JSONDecodeError:
                requirements = self._parse_requirements_fallback(response.content)
            
            return requirements
            
        except Exception as e:
            logger.error(f"Failed to analyze problem requirements: {e}")
            return {
                "domain": "general",
                "actions": ["analyze", "process"],
                "data_requirements": ["text_input"],
                "integrations": [],
                "complexity": 5,
                "success_criteria": ["task_completion"]
            }
    
    async def _extract_required_capabilities(
        self,
        problem_description: str,
        research_context: Dict[str, Any]
    ) -> List[str]:
        """Extract required capabilities from problem description."""
        try:
            extraction_prompt = f"""
            Extract the specific capabilities needed to solve this problem:
            
            Problem: {problem_description}
            Research Context: {json.dumps(research_context, indent=2)}
            
            List the specific tool capabilities needed, such as:
            - API integration capabilities
            - Data processing capabilities  
            - Web scraping capabilities
            - Database access capabilities
            - Communication capabilities
            - Analysis capabilities
            
            Return as a JSON list of capability names.
            """
            
            response = await self.venice_client.chat_completion(
                messages=[{"role": "user", "content": extraction_prompt}],
                model="llama-4",
                temperature=0.2
            )
            
            try:
                capabilities = json.loads(response.content)
                if isinstance(capabilities, list):
                    return capabilities
            except json.JSONDecodeError:
                pass
            
            return self._extract_capabilities_fallback(response.content)
            
        except Exception as e:
            logger.error(f"Failed to extract required capabilities: {e}")
            return self._extract_capabilities_pattern_matching(problem_description)
    
    def _find_missing_capabilities(self, required_capabilities: List[str]) -> List[str]:
        """Find capabilities that are missing from existing capabilities."""
        missing = []
        
        for capability in required_capabilities:
            capability_lower = capability.lower()
            found = False
            
            for existing in self.existing_capabilities:
                if (capability_lower in existing.lower() or 
                    existing.lower() in capability_lower):
                    found = True
                    break
            
            if not found:
                missing.append(capability)
        
        return missing
    
    async def _analyze_capability_gap(
        self,
        capability: str,
        problem_description: str,
        research_context: Dict[str, Any]
    ) -> Optional[ToolGap]:
        """Analyze a specific capability gap."""
        try:
            category = self._categorize_capability(capability)
            
            priority = self._determine_priority(capability, problem_description, research_context)
            
            complexity = self._estimate_capability_complexity(capability, research_context)
            
            similar_tools = self._find_similar_tools(capability)
            
            inputs, outputs = self._extract_io_requirements(capability, research_context)
            
            dependencies = self._determine_capability_dependencies(capability, category)
            
            gap = ToolGap(
                capability_name=capability,
                description=f"Tool for {capability.lower()}",
                priority=priority,
                category=category,
                required_inputs=inputs,
                expected_outputs=outputs,
                complexity_score=complexity,
                dependencies=dependencies,
                similar_tools=similar_tools,
                metadata={
                    "problem_context": problem_description[:200],
                    "research_summary": research_context.get("summary", ""),
                    "identified_at": datetime.now().isoformat()
                }
            )
            
            return gap
            
        except Exception as e:
            logger.error(f"Failed to analyze capability gap for {capability}: {e}")
            return None
    
    def _prioritize_gaps(self, gaps: List[ToolGap], problem_description: str) -> List[ToolGap]:
        """Prioritize capability gaps based on problem context."""
        priority_order = {
            GapPriority.CRITICAL: 4,
            GapPriority.HIGH: 3,
            GapPriority.MEDIUM: 2,
            GapPriority.LOW: 1
        }
        
        def gap_score(gap: ToolGap) -> float:
            priority_score = priority_order.get(gap.priority, 1)
            complexity_penalty = gap.complexity_score / 10.0  # Prefer simpler tools
            return priority_score - complexity_penalty
        
        return sorted(gaps, key=gap_score, reverse=True)
    
    def _categorize_capability(self, capability: str) -> GapCategory:
        """Categorize a capability based on its description."""
        capability_lower = capability.lower()
        
        if any(term in capability_lower for term in ["api", "rest", "http", "endpoint"]):
            return GapCategory.API_INTEGRATION
        elif any(term in capability_lower for term in ["web", "scrape", "browser", "selenium"]):
            return GapCategory.WEB_AUTOMATION
        elif any(term in capability_lower for term in ["database", "sql", "query", "data"]):
            return GapCategory.DATA_ACCESS
        elif any(term in capability_lower for term in ["process", "transform", "parse", "analyze"]):
            return GapCategory.DATA_PROCESSING
        elif any(term in capability_lower for term in ["slack", "discord", "email", "notify"]):
            return GapCategory.COMMUNICATION
        elif any(term in capability_lower for term in ["monitor", "watch", "alert", "check"]):
            return GapCategory.MONITORING
        elif any(term in capability_lower for term in ["analyze", "report", "metrics", "stats"]):
            return GapCategory.ANALYSIS
        elif any(term in capability_lower for term in ["workflow", "orchestrate", "coordinate"]):
            return GapCategory.WORKFLOW
        else:
            return GapCategory.UNKNOWN
    
    def _determine_priority(
        self,
        capability: str,
        problem_description: str,
        research_context: Dict[str, Any]
    ) -> GapPriority:
        """Determine priority of a capability gap."""
        capability_lower = capability.lower()
        problem_lower = problem_description.lower()
        
        if capability_lower in problem_lower:
            return GapPriority.CRITICAL
        
        domain = research_context.get("domain", "").lower()
        if domain and any(term in capability_lower for term in domain.split()):
            return GapPriority.HIGH
        
        research_text = str(research_context).lower()
        if capability_lower in research_text:
            return GapPriority.MEDIUM
        
        return GapPriority.LOW
    
    def _estimate_capability_complexity(
        self,
        capability: str,
        research_context: Dict[str, Any]
    ) -> float:
        """Estimate complexity of implementing a capability (1-10 scale)."""
        capability_lower = capability.lower()
        
        complexity_map = {
            "api": 3.0,
            "web": 6.0,
            "database": 4.0,
            "process": 2.0,
            "analyze": 5.0,
            "monitor": 4.0,
            "workflow": 7.0
        }
        
        base_complexity = 3.0  # Default
        for term, complexity in complexity_map.items():
            if term in capability_lower:
                base_complexity = complexity
                break
        
        if research_context.get("complexity", 0) > 7:
            base_complexity += 2.0
        elif research_context.get("complexity", 0) > 5:
            base_complexity += 1.0
        
        return min(base_complexity, 10.0)
    
    def _find_similar_tools(self, capability: str) -> List[str]:
        """Find similar existing tools."""
        similar = []
        capability_lower = capability.lower()
        
        for existing in self.existing_capabilities:
            existing_lower = existing.lower()
            
            if (any(word in existing_lower for word in capability_lower.split()) or
                any(word in capability_lower for word in existing_lower.split())):
                similar.append(existing)
        
        return similar[:5]  # Limit to top 5
    
    def _extract_io_requirements(
        self,
        capability: str,
        research_context: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Extract input and output requirements for a capability."""
        capability_lower = capability.lower()
        
        if "api" in capability_lower:
            inputs = ["url", "method", "headers", "data"]
            outputs = ["response_data", "status_code"]
        elif "web" in capability_lower:
            inputs = ["url", "selector", "wait_time"]
            outputs = ["scraped_content", "links", "metadata"]
        elif "database" in capability_lower:
            inputs = ["query", "parameters", "connection_string"]
            outputs = ["query_results", "affected_rows"]
        elif "process" in capability_lower:
            inputs = ["input_data", "processing_options"]
            outputs = ["processed_data", "processing_stats"]
        else:
            inputs = ["input_data"]
            outputs = ["result"]
        
        data_requirements = research_context.get("data_requirements", [])
        if data_requirements:
            inputs.extend(data_requirements)
        
        return inputs, outputs
    
    def _determine_capability_dependencies(
        self,
        capability: str,
        category: GapCategory
    ) -> List[str]:
        """Determine dependencies for a capability."""
        base_deps = ["mcp"]
        capability_lower = capability.lower()
        
        if category == GapCategory.API_INTEGRATION:
            base_deps.extend(["aiohttp", "pydantic"])
        elif category == GapCategory.WEB_AUTOMATION:
            base_deps.extend(["aiohttp", "beautifulsoup4", "selenium"])
        elif category == GapCategory.DATA_ACCESS:
            base_deps.extend(["sqlalchemy", "asyncpg"])
        elif category == GapCategory.DATA_PROCESSING:
            base_deps.extend(["pandas", "numpy"])
        elif category == GapCategory.COMMUNICATION:
            base_deps.extend(["aiohttp", "slack-sdk"])
        elif category == GapCategory.MONITORING:
            base_deps.extend(["psutil", "aiofiles"])
        
        if "oauth" in capability_lower:
            base_deps.append("authlib")
        if "json" in capability_lower:
            base_deps.append("jsonschema")
        if "xml" in capability_lower:
            base_deps.append("lxml")
        
        return list(set(base_deps))  # Remove duplicates
    
    def _extract_capabilities_fallback(self, text: str) -> List[str]:
        """Fallback method to extract capabilities from text."""
        capabilities = []
        
        patterns = [
            r"(\w+)\s+capability",
            r"need\s+to\s+(\w+)",
            r"requires?\s+(\w+)",
            r"(\w+)\s+integration",
            r"(\w+)\s+tool"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            capabilities.extend(matches)
        
        cleaned = []
        for cap in capabilities:
            if len(cap) > 2 and cap.lower() not in ["the", "and", "for", "with"]:
                cleaned.append(cap.lower())
        
        return list(set(cleaned))
    
    def _extract_capabilities_pattern_matching(self, problem_description: str) -> List[str]:
        """Extract capabilities using pattern matching."""
        capabilities = []
        text_lower = problem_description.lower()
        
        capability_patterns = {
            "api_integration": ["api", "rest", "http", "endpoint", "service"],
            "web_scraping": ["scrape", "web", "browser", "crawl", "extract"],
            "data_processing": ["process", "transform", "parse", "analyze", "filter"],
            "database_access": ["database", "sql", "query", "store", "retrieve"],
            "file_operations": ["file", "read", "write", "upload", "download"],
            "communication": ["notify", "send", "message", "email", "slack"],
            "monitoring": ["monitor", "watch", "check", "alert", "status"],
            "automation": ["automate", "schedule", "trigger", "workflow"]
        }
        
        for capability, keywords in capability_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities
    
    def _select_implementation_strategy(
        self,
        gap: ToolGap,
        research_context: Dict[str, Any]
    ) -> str:
        """Select implementation strategy for a gap."""
        category = gap.category
        
        strategy_map = {
            GapCategory.API_INTEGRATION: "api_call",
            GapCategory.WEB_AUTOMATION: "web_scraping",
            GapCategory.DATA_ACCESS: "database_query",
            GapCategory.DATA_PROCESSING: "data_transformation",
            GapCategory.COMMUNICATION: "message_sending",
            GapCategory.MONITORING: "system_monitoring",
            GapCategory.ANALYSIS: "data_analysis",
            GapCategory.WORKFLOW: "workflow_orchestration"
        }
        
        return strategy_map.get(category, "basic_implementation")
    
    async def _generate_input_schema(self, gap: ToolGap, strategy: str) -> Dict[str, Any]:
        """Generate input schema for a tool gap."""
        base_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for input_name in gap.required_inputs:
            base_schema["properties"][input_name] = {
                "type": "string",
                "description": f"Input parameter: {input_name}"
            }
            base_schema["required"].append(input_name)
        
        if strategy == "api_call":
            base_schema["properties"].update({
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                "headers": {"type": "object", "description": "HTTP headers"}
            })
        elif strategy == "web_scraping":
            base_schema["properties"].update({
                "selector": {"type": "string", "description": "CSS selector"},
                "wait_time": {"type": "number", "description": "Wait time in seconds"}
            })
        
        return base_schema
    
    def _determine_dependencies(self, gap: ToolGap, strategy: str) -> List[str]:
        """Determine dependencies for implementation."""
        return gap.dependencies
    
    def _identify_integrations(
        self,
        gap: ToolGap,
        research_context: Dict[str, Any]
    ) -> List[str]:
        """Identify integration points for a gap."""
        integrations = []
        
        if gap.category in [GapCategory.ANALYSIS, GapCategory.DATA_PROCESSING]:
            integrations.append("venice_ai")
        
        if "memory" in gap.description.lower() or "store" in gap.description.lower():
            integrations.append("memory")
        
        research_text = str(research_context).lower()
        if "database" in research_text:
            integrations.append("database")
        if "api" in research_text:
            integrations.append("api_gateway")
        
        return integrations
    
    def _generate_tags(self, gap: ToolGap, strategy: str) -> List[str]:
        """Generate tags for a tool gap."""
        tags = [gap.category.value, strategy]
        
        tags.append(gap.priority.value)
        
        capability_lower = gap.capability_name.lower()
        if "web" in capability_lower:
            tags.append("web")
        if "api" in capability_lower:
            tags.append("api")
        if "data" in capability_lower:
            tags.append("data")
        
        return list(set(tags))
    
    def _estimate_complexity(self, gap: ToolGap, strategy: str) -> float:
        """Estimate implementation complexity."""
        return gap.complexity_score
    
    def _calculate_confidence(
        self,
        gap: ToolGap,
        strategy: str,
        research_context: Dict[str, Any]
    ) -> float:
        """Calculate confidence in implementation suggestion."""
        base_confidence = 0.7
        
        if gap.similar_tools:
            base_confidence += 0.1  # Similar tools exist
        
        if gap.category != GapCategory.UNKNOWN:
            base_confidence += 0.1  # Clear category
        
        if len(gap.required_inputs) > 0:
            base_confidence += 0.1  # Clear requirements
        
        if research_context.get("complexity", 5) < 5:
            base_confidence += 0.1  # Lower complexity
        
        return min(base_confidence, 1.0)
    
    async def _generate_implementation_notes(
        self,
        gap: ToolGap,
        strategy: str,
        research_context: Dict[str, Any]
    ) -> str:
        """Generate implementation notes for a gap."""
        notes = f"Implementation strategy: {strategy}\n"
        notes += f"Category: {gap.category.value}\n"
        notes += f"Priority: {gap.priority.value}\n"
        notes += f"Complexity: {gap.complexity_score}/10\n"
        
        if gap.similar_tools:
            notes += f"Similar tools: {', '.join(gap.similar_tools)}\n"
        
        if gap.dependencies:
            notes += f"Dependencies: {', '.join(gap.dependencies)}\n"
        
        if strategy == "api_call":
            notes += "Consider authentication, rate limiting, and error handling.\n"
        elif strategy == "web_scraping":
            notes += "Handle dynamic content, respect robots.txt, implement delays.\n"
        elif strategy == "database_query":
            notes += "Use parameterized queries, handle connections properly.\n"
        
        return notes
    
    def _create_example_usage(self, gap: ToolGap, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create example usage for a tool gap."""
        example = {
            "tool_name": gap.capability_name.lower().replace(" ", "_"),
            "description": gap.description,
            "example_input": {}
        }
        
        for prop_name, prop_schema in input_schema.get("properties", {}).items():
            if prop_schema.get("type") == "string":
                example["example_input"][prop_name] = f"example_{prop_name}"
            elif prop_schema.get("type") == "number":
                example["example_input"][prop_name] = 42
            elif prop_schema.get("type") == "boolean":
                example["example_input"][prop_name] = True
            elif prop_schema.get("type") == "object":
                example["example_input"][prop_name] = {}
        
        return example
    
    def _parse_requirements_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback parser for requirements analysis."""
        return {
            "domain": "general",
            "actions": ["process", "analyze"],
            "data_requirements": ["input_data"],
            "integrations": [],
            "complexity": 5,
            "success_criteria": ["task_completion"]
        }
    
    def _generate_cache_key(self, problem: str, context: Dict[str, Any]) -> str:
        """Generate cache key for analysis results."""
        problem_hash = hash(problem[:200])  # Use first 200 chars
        context_hash = hash(str(sorted(context.items())))
        return f"{problem_hash}_{context_hash}"
    
    def _initialize_capability_patterns(self) -> Dict[str, List[str]]:
        """Initialize capability pattern matching."""
        return {
            "api_integration": [
                "api call", "rest api", "http request", "web service",
                "endpoint", "api integration", "service call"
            ],
            "web_scraping": [
                "web scraping", "scrape website", "extract data", "crawl web",
                "browser automation", "web data extraction"
            ],
            "data_processing": [
                "process data", "transform data", "parse data", "analyze data",
                "data manipulation", "data transformation"
            ],
            "database_operations": [
                "database query", "sql query", "data storage", "database access",
                "data retrieval", "database operations"
            ],
            "file_operations": [
                "file handling", "read file", "write file", "file processing",
                "file upload", "file download"
            ],
            "communication": [
                "send message", "notification", "email", "slack message",
                "communication", "messaging"
            ],
            "monitoring": [
                "monitor system", "health check", "status monitoring",
                "alert system", "system monitoring"
            ]
        }
    
    def _initialize_implementation_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize implementation strategy templates."""
        return {
            "api_call": {
                "dependencies": ["aiohttp", "pydantic"],
                "complexity_base": 3.0,
                "template": "api_server"
            },
            "web_scraping": {
                "dependencies": ["aiohttp", "beautifulsoup4", "selenium"],
                "complexity_base": 6.0,
                "template": "web_scraping_server"
            },
            "database_query": {
                "dependencies": ["sqlalchemy", "asyncpg"],
                "complexity_base": 4.0,
                "template": "database_server"
            },
            "data_transformation": {
                "dependencies": ["pandas", "numpy"],
                "complexity_base": 3.0,
                "template": "basic_server"
            },
            "basic_implementation": {
                "dependencies": ["mcp"],
                "complexity_base": 2.0,
                "template": "basic_server"
            }
        }
    
    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "existing_capabilities_count": len(self.existing_capabilities),
            "cached_analyses": len(self.analysis_cache),
            "cached_suggestions": len(self.suggestion_cache),
            "capability_patterns": len(self.capability_patterns),
            "implementation_strategies": len(self.implementation_strategies)
        }
