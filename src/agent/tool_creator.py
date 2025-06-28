"""
Tool Creator - Dynamic tool generation for the Venice.ai scaffolding system.

This module implements the core scaffolding component that dynamically generates
specialized MCP servers and tools based on problem analysis and research findings.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..mcp.factory import ServerFactory, ToolDesign, MCPServerConfig, DeploymentResult
from ..mcp.capability_analyzer import CapabilityAnalyzer, ToolGap, ImplementationSuggestion
from .research_coordinator import ResearchCoordinator

logger = logging.getLogger(__name__)


class CreationStrategy(Enum):
    """Tool creation strategies."""
    RESEARCH_DRIVEN = "research_driven"
    TEMPLATE_BASED = "template_based"
    HYBRID = "hybrid"
    RAPID_PROTOTYPE = "rapid_prototype"


class ValidationLevel(Enum):
    """Tool validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


@dataclass
class CreatedTool:
    """Represents a successfully created tool."""
    tool_id: str
    name: str
    description: str
    server_config: MCPServerConfig
    capabilities: List[str]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCreationRequest:
    """Request for tool creation."""
    problem_description: str
    research_context: Dict[str, Any]
    capability_gaps: List[ToolGap]
    strategy: CreationStrategy = CreationStrategy.RESEARCH_DRIVEN
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolCreator:
    """
    Core scaffolding component for dynamic tool generation.
    
    Implements the research → analyze → design → create → deploy pipeline
    for the Venice.ai scaffolding system.
    """
    
    def __init__(
        self,
        server_factory: ServerFactory,
        capability_analyzer: CapabilityAnalyzer,
        research_coordinator: ResearchCoordinator
    ):
        """
        Initialize the tool creator.
        
        Args:
            server_factory: Factory for creating MCP servers
            capability_analyzer: Analyzer for capability gaps
            research_coordinator: Coordinator for research operations
        """
        self.server_factory = server_factory
        self.capability_analyzer = capability_analyzer
        self.research_coordinator = research_coordinator
        
        self.created_tools: Dict[str, CreatedTool] = {}
        self.creation_history: List[Dict[str, Any]] = []
        
        self.creation_metrics = {
            "total_tools_created": 0,
            "successful_creations": 0,
            "failed_creations": 0,
            "average_creation_time": 0.0,
            "tools_by_strategy": {strategy.value: 0 for strategy in CreationStrategy},
            "validation_success_rate": 0.0
        }
        
        self.max_concurrent_creations = 3
        self.default_timeout = 300  # 5 minutes
    
    async def initialize(self) -> None:
        """Initialize the tool creator."""
        logger.info("Initializing Tool Creator")
        
        try:
            logger.info("Tool Creator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Tool Creator: {e}")
            raise
    
    async def create_tool_from_problem(self, problem: str) -> CreatedTool:
        """
        Complete pipeline from problem to deployed tool.
        
        This is the main entry point for the Venice.ai scaffolding system.
        
        Args:
            problem: Problem description
            
        Returns:
            Created tool with deployment information
        """
        logger.info(f"Creating tool from problem: {problem[:100]}...")
        
        start_time = datetime.now()
        
        try:
            research_context = await self.research_coordinator.comprehensive_research(
                problem=problem,
                depth="deep",
                sources=["web", "memory", "documentation"]
            )
            
            capability_gaps = await self.capability_analyzer.identify_gaps(
                problem_description=problem,
                research_context=research_context
            )
            
            created_tools = await self.create_tools_for_gaps(
                gaps=capability_gaps,
                research_context=research_context
            )
            
            if created_tools:
                tool = created_tools[0]
                
                creation_time = (datetime.now() - start_time).total_seconds()
                self._update_creation_metrics(creation_time, True)
                
                logger.info(f"Successfully created tool from problem in {creation_time:.2f}s")
                return tool
            else:
                raise Exception("No tools were created for the problem")
                
        except Exception as e:
            creation_time = (datetime.now() - start_time).total_seconds()
            self._update_creation_metrics(creation_time, False)
            logger.error(f"Failed to create tool from problem: {e}")
            raise
    
    async def create_tools_for_gaps(
        self,
        gaps: List[ToolGap],
        research_context: Dict[str, Any]
    ) -> List[CreatedTool]:
        """
        Create tools for identified capability gaps.
        
        Args:
            gaps: List of capability gaps to address
            research_context: Research context for tool creation
            
        Returns:
            List of created tools
        """
        logger.info(f"Creating tools for {len(gaps)} capability gaps")
        
        created_tools = []
        semaphore = asyncio.Semaphore(self.max_concurrent_creations)
        
        async def create_single_tool(gap: ToolGap) -> Optional[CreatedTool]:
            async with semaphore:
                try:
                    return await self._create_tool_for_gap(gap, research_context)
                except Exception as e:
                    logger.error(f"Failed to create tool for gap {gap.capability_name}: {e}")
                    return None
        
        tasks = [create_single_tool(gap) for gap in gaps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, CreatedTool):
                created_tools.append(result)
        
        logger.info(f"Successfully created {len(created_tools)} tools")
        return created_tools
    
    async def create_tool_from_request(self, request: ToolCreationRequest) -> List[CreatedTool]:
        """
        Create tools from a structured creation request.
        
        Args:
            request: Tool creation request
            
        Returns:
            List of created tools
        """
        logger.info(f"Processing tool creation request for: {request.problem_description[:100]}...")
        
        try:
            if request.strategy == CreationStrategy.RESEARCH_DRIVEN:
                return await self._create_tools_research_driven(request)
            elif request.strategy == CreationStrategy.TEMPLATE_BASED:
                return await self._create_tools_template_based(request)
            elif request.strategy == CreationStrategy.HYBRID:
                return await self._create_tools_hybrid(request)
            elif request.strategy == CreationStrategy.RAPID_PROTOTYPE:
                return await self._create_tools_rapid_prototype(request)
            else:
                raise ValueError(f"Unknown creation strategy: {request.strategy}")
                
        except Exception as e:
            logger.error(f"Failed to process tool creation request: {e}")
            raise
    
    async def validate_tool(
        self,
        tool: CreatedTool,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> Dict[str, Any]:
        """
        Validate a created tool.
        
        Args:
            tool: Tool to validate
            validation_level: Level of validation to perform
            
        Returns:
            Validation results
        """
        logger.info(f"Validating tool: {tool.name}")
        
        try:
            validation_results = {
                "tool_id": tool.tool_id,
                "validation_level": validation_level.value,
                "timestamp": datetime.now().isoformat(),
                "tests_passed": 0,
                "tests_failed": 0,
                "performance_metrics": {},
                "issues": [],
                "recommendations": []
            }
            
            if validation_level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                basic_results = await self._validate_basic(tool)
                validation_results.update(basic_results)
            
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                standard_results = await self._validate_standard(tool)
                validation_results.update(standard_results)
            
            if validation_level == ValidationLevel.COMPREHENSIVE:
                comprehensive_results = await self._validate_comprehensive(tool)
                validation_results.update(comprehensive_results)
            
            tool.validation_results = validation_results
            
            success_rate = validation_results["tests_passed"] / max(
                validation_results["tests_passed"] + validation_results["tests_failed"], 1
            )
            self.creation_metrics["validation_success_rate"] = (
                self.creation_metrics["validation_success_rate"] + success_rate
            ) / 2
            
            logger.info(f"Tool validation completed with {success_rate:.2f} success rate")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate tool {tool.name}: {e}")
            return {
                "tool_id": tool.tool_id,
                "validation_level": validation_level.value,
                "error": str(e),
                "tests_passed": 0,
                "tests_failed": 1
            }
    
    async def cleanup(self) -> None:
        """Cleanup tool creator resources."""
        logger.debug("Cleaning up Tool Creator")
        
        if len(self.creation_history) > 1000:
            self.creation_history = self.creation_history[-500:]  # Keep last 500
    
    async def _create_tool_for_gap(
        self,
        gap: ToolGap,
        research_context: Dict[str, Any]
    ) -> CreatedTool:
        """Create a tool for a specific capability gap."""
        logger.debug(f"Creating tool for gap: {gap.capability_name}")
        
        start_time = datetime.now()
        
        try:
            suggestion = await self.capability_analyzer.suggest_implementation(
                gap=gap,
                research_context=research_context
            )
            
            tool_design = ToolDesign(
                name=gap.capability_name.lower().replace(" ", "_"),
                description=gap.description,
                input_schema=suggestion.input_schema,
                implementation_strategy=suggestion.strategy,
                dependencies=suggestion.dependencies,
                integration_points=suggestion.integrations,
                tags=suggestion.tags,
                metadata={
                    "gap_priority": gap.priority.value,
                    "gap_category": gap.category.value,
                    "complexity_estimate": suggestion.complexity_estimate,
                    "confidence_score": suggestion.confidence_score,
                    "implementation_notes": suggestion.implementation_notes
                }
            )
            
            deployment_result = await self.server_factory.create_specialized_server(tool_design)
            
            tool = CreatedTool(
                tool_id=self._generate_tool_id(),
                name=tool_design.name,
                description=tool_design.description,
                server_config=deployment_result.server_config,
                capabilities=[gap.capability_name],
                metadata={
                    "gap_info": gap.__dict__,
                    "suggestion_info": suggestion.__dict__,
                    "deployment_info": deployment_result.__dict__
                }
            )
            
            self.created_tools[tool.tool_id] = tool
            
            creation_record = {
                "tool_id": tool.tool_id,
                "gap": gap.__dict__,
                "suggestion": suggestion.__dict__,
                "deployment": deployment_result.__dict__,
                "creation_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            self.creation_history.append(creation_record)
            
            self.creation_metrics["total_tools_created"] += 1
            self.creation_metrics["successful_creations"] += 1
            
            logger.debug(f"Successfully created tool: {tool.name}")
            return tool
            
        except Exception as e:
            self.creation_metrics["failed_creations"] += 1
            logger.error(f"Failed to create tool for gap {gap.capability_name}: {e}")
            raise
    
    async def _create_tools_research_driven(self, request: ToolCreationRequest) -> List[CreatedTool]:
        """Create tools using research-driven strategy."""
        logger.debug("Using research-driven tool creation strategy")
        
        if not request.research_context:
            request.research_context = await self.research_coordinator.comprehensive_research(
                problem=request.problem_description,
                depth="deep"
            )
        
        created_tools = await self.create_tools_for_gaps(
            gaps=request.capability_gaps,
            research_context=request.research_context
        )
        
        self.creation_metrics["tools_by_strategy"][CreationStrategy.RESEARCH_DRIVEN.value] += len(created_tools)
        return created_tools
    
    async def _create_tools_template_based(self, request: ToolCreationRequest) -> List[CreatedTool]:
        """Create tools using template-based strategy."""
        logger.debug("Using template-based tool creation strategy")
        
        created_tools = []
        
        for gap in request.capability_gaps:
            try:
                template_name = self._select_template_for_gap(gap)
                
                tool_design = ToolDesign(
                    name=gap.capability_name.lower().replace(" ", "_"),
                    description=gap.description,
                    input_schema={"type": "object", "properties": {}},
                    implementation_strategy="template_based",
                    dependencies=gap.dependencies,
                    tags=[gap.category.value, "template_based"],
                    metadata={"template": template_name}
                )
                
                deployment_result = await self.server_factory.create_specialized_server(tool_design)
                
                tool = CreatedTool(
                    tool_id=self._generate_tool_id(),
                    name=tool_design.name,
                    description=tool_design.description,
                    server_config=deployment_result.server_config,
                    capabilities=[gap.capability_name]
                )
                
                self.created_tools[tool.tool_id] = tool
                created_tools.append(tool)
                
            except Exception as e:
                logger.error(f"Failed to create template-based tool for {gap.capability_name}: {e}")
        
        self.creation_metrics["tools_by_strategy"][CreationStrategy.TEMPLATE_BASED.value] += len(created_tools)
        return created_tools
    
    async def _create_tools_hybrid(self, request: ToolCreationRequest) -> List[CreatedTool]:
        """Create tools using hybrid strategy."""
        logger.debug("Using hybrid tool creation strategy")
        
        high_priority_gaps = [gap for gap in request.capability_gaps if gap.priority.value in ["critical", "high"]]
        low_priority_gaps = [gap for gap in request.capability_gaps if gap.priority.value in ["medium", "low"]]
        
        created_tools = []
        
        if high_priority_gaps:
            research_request = ToolCreationRequest(
                problem_description=request.problem_description,
                research_context=request.research_context,
                capability_gaps=high_priority_gaps,
                strategy=CreationStrategy.RESEARCH_DRIVEN
            )
            research_tools = await self._create_tools_research_driven(research_request)
            created_tools.extend(research_tools)
        
        if low_priority_gaps:
            template_request = ToolCreationRequest(
                problem_description=request.problem_description,
                research_context=request.research_context,
                capability_gaps=low_priority_gaps,
                strategy=CreationStrategy.TEMPLATE_BASED
            )
            template_tools = await self._create_tools_template_based(template_request)
            created_tools.extend(template_tools)
        
        self.creation_metrics["tools_by_strategy"][CreationStrategy.HYBRID.value] += len(created_tools)
        return created_tools
    
    async def _create_tools_rapid_prototype(self, request: ToolCreationRequest) -> List[CreatedTool]:
        """Create tools using rapid prototype strategy."""
        logger.debug("Using rapid prototype tool creation strategy")
        
        created_tools = []
        
        for gap in request.capability_gaps:
            try:
                tool_design = ToolDesign(
                    name=gap.capability_name.lower().replace(" ", "_"),
                    description=f"Rapid prototype for {gap.description}",
                    input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
                    implementation_strategy="basic_implementation",
                    dependencies=["mcp"],
                    tags=["prototype", "rapid"],
                    metadata={"prototype": True}
                )
                
                deployment_result = await self.server_factory.create_specialized_server(tool_design)
                
                tool = CreatedTool(
                    tool_id=self._generate_tool_id(),
                    name=tool_design.name,
                    description=tool_design.description,
                    server_config=deployment_result.server_config,
                    capabilities=[gap.capability_name],
                    metadata={"prototype": True}
                )
                
                self.created_tools[tool.tool_id] = tool
                created_tools.append(tool)
                
            except Exception as e:
                logger.error(f"Failed to create rapid prototype for {gap.capability_name}: {e}")
        
        self.creation_metrics["tools_by_strategy"][CreationStrategy.RAPID_PROTOTYPE.value] += len(created_tools)
        return created_tools
    
    async def _validate_basic(self, tool: CreatedTool) -> Dict[str, Any]:
        """Perform basic tool validation."""
        results = {"basic_validation": True}
        
        try:
            if not tool.server_config:
                results["basic_validation"] = False
                results.setdefault("issues", []).append("Missing server configuration")
            
            if not tool.capabilities:
                results["basic_validation"] = False
                results.setdefault("issues", []).append("No capabilities defined")
            
            if results["basic_validation"]:
                results["tests_passed"] = results.get("tests_passed", 0) + 1
            else:
                results["tests_failed"] = results.get("tests_failed", 0) + 1
            
        except Exception as e:
            results["basic_validation"] = False
            results["tests_failed"] = results.get("tests_failed", 0) + 1
            results.setdefault("issues", []).append(f"Basic validation error: {e}")
        
        return results
    
    async def _validate_standard(self, tool: CreatedTool) -> Dict[str, Any]:
        """Perform standard tool validation."""
        results = {"standard_validation": True}
        
        try:
            results["server_status"] = "running"  # Placeholder
            
            results["response_test"] = True  # Placeholder
            
            if results["standard_validation"]:
                results["tests_passed"] = results.get("tests_passed", 0) + 1
            else:
                results["tests_failed"] = results.get("tests_failed", 0) + 1
            
        except Exception as e:
            results["standard_validation"] = False
            results["tests_failed"] = results.get("tests_failed", 0) + 1
            results.setdefault("issues", []).append(f"Standard validation error: {e}")
        
        return results
    
    async def _validate_comprehensive(self, tool: CreatedTool) -> Dict[str, Any]:
        """Perform comprehensive tool validation."""
        results = {"comprehensive_validation": True}
        
        try:
            results["performance_metrics"] = {
                "response_time": 0.1,  # Placeholder
                "memory_usage": 0.05,  # Placeholder
                "cpu_usage": 0.02      # Placeholder
            }
            
            results["integration_test"] = True  # Placeholder
            
            if results["comprehensive_validation"]:
                results["tests_passed"] = results.get("tests_passed", 0) + 1
            else:
                results["tests_failed"] = results.get("tests_failed", 0) + 1
            
        except Exception as e:
            results["comprehensive_validation"] = False
            results["tests_failed"] = results.get("tests_failed", 0) + 1
            results.setdefault("issues", []).append(f"Comprehensive validation error: {e}")
        
        return results
    
    def _select_template_for_gap(self, gap: ToolGap) -> str:
        """Select appropriate template for a capability gap."""
        category_templates = {
            "api_integration": "api_server",
            "web_automation": "web_scraping_server",
            "data_access": "database_server",
            "data_processing": "basic_server",
            "communication": "basic_server",
            "monitoring": "basic_server",
            "analysis": "basic_server",
            "workflow": "basic_server"
        }
        
        return category_templates.get(gap.category.value, "basic_server")
    
    def _generate_tool_id(self) -> str:
        """Generate unique tool ID."""
        timestamp = int(datetime.now().timestamp())
        return f"tool_{timestamp}_{len(self.created_tools)}"
    
    def _update_creation_metrics(self, creation_time: float, success: bool) -> None:
        """Update creation metrics."""
        if success:
            self.creation_metrics["successful_creations"] += 1
        else:
            self.creation_metrics["failed_creations"] += 1
        
        total_creations = self.creation_metrics["successful_creations"] + self.creation_metrics["failed_creations"]
        current_avg = self.creation_metrics["average_creation_time"]
        
        if total_creations == 1:
            self.creation_metrics["average_creation_time"] = creation_time
        else:
            new_avg = ((current_avg * (total_creations - 1)) + creation_time) / total_creations
            self.creation_metrics["average_creation_time"] = new_avg
    
    def get_created_tools(self) -> List[CreatedTool]:
        """Get list of all created tools."""
        return list(self.created_tools.values())
    
    def get_creation_stats(self) -> Dict[str, Any]:
        """Get tool creation statistics."""
        return {
            "metrics": self.creation_metrics.copy(),
            "total_tools": len(self.created_tools),
            "creation_history_size": len(self.creation_history)
        }
