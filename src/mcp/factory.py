"""
MCP Server Factory for dynamic server creation and deployment.

This module provides capabilities for on-the-fly MCP server generation
using templates and automated deployment based on problem analysis.
"""

import asyncio
import logging
import json
import tempfile
import subprocess
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import yaml
import shutil

from .template_engine import TemplateEngine, ServerTemplate
from .capability_analyzer import CapabilityAnalyzer, ToolGap
from .server_manager import MCPServerManager, ServerConfig, TransportType
from .registry import MCPRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolDesign:
    """Design specification for a new MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    implementation_strategy: str
    dependencies: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class MCPServerConfig:
    """Configuration for a dynamically created MCP server."""
    name: str
    code: str
    dependencies: List[str]
    transport: TransportType = TransportType.STDIO
    lifecycle: str = "on-demand"  # "persistent" or "on-demand"
    working_dir: Optional[str] = None
    env_vars: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentResult:
    """Result of MCP server deployment."""
    success: bool
    server_name: str
    server_path: Optional[str] = None
    process_id: Optional[int] = None
    error_message: Optional[str] = None
    deployment_time: Optional[datetime] = None


class ServerFactory:
    """
    Factory for creating and deploying MCP servers dynamically.
    
    Provides the core infrastructure for the Venice.ai scaffolding system
    to create specialized tools based on problem analysis and research.
    """
    
    def __init__(
        self,
        template_engine: TemplateEngine,
        capability_analyzer: CapabilityAnalyzer,
        server_manager: MCPServerManager,
        registry: MCPRegistry,
        workspace_dir: str = "/tmp/mcp_servers"
    ):
        """
        Initialize the server factory.
        
        Args:
            template_engine: Engine for generating server code
            capability_analyzer: Analyzer for identifying tool gaps
            server_manager: Manager for server lifecycle
            registry: Registry for server discovery
            workspace_dir: Directory for generated servers
        """
        self.template_engine = template_engine
        self.capability_analyzer = capability_analyzer
        self.server_manager = server_manager
        self.registry = registry
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.created_servers: Dict[str, MCPServerConfig] = {}
        self.deployment_history: List[DeploymentResult] = []
    
    async def analyze_and_create_tools(
        self,
        problem_description: str,
        research_context: Dict[str, Any],
        existing_capabilities: Optional[List[str]] = None
    ) -> List[DeploymentResult]:
        """
        Analyze a problem and create necessary tools to solve it.
        
        This is the main entry point for the Venice.ai scaffolding system
        to dynamically create tools based on problem analysis.
        
        Args:
            problem_description: Description of the problem to solve
            research_context: Context from research phase
            existing_capabilities: List of existing tool capabilities
            
        Returns:
            List of deployment results for created servers
        """
        logger.info(f"Analyzing problem and creating tools: {problem_description}")
        
        try:
            tool_gaps = await self.capability_analyzer.identify_gaps(
                problem_description=problem_description,
                research_context=research_context,
                existing_capabilities=existing_capabilities or []
            )
            
            if not tool_gaps:
                logger.info("No tool gaps identified, existing capabilities are sufficient")
                return []
            
            tool_designs = []
            for gap in tool_gaps:
                design = await self._design_tool_for_gap(gap, research_context)
                if design:
                    tool_designs.append(design)
            
            deployment_results = []
            for design in tool_designs:
                result = await self.create_specialized_server(design)
                deployment_results.append(result)
            
            logger.info(f"Created {len(deployment_results)} specialized servers")
            return deployment_results
            
        except Exception as e:
            logger.error(f"Failed to analyze and create tools: {e}")
            return [DeploymentResult(
                success=False,
                server_name="analysis_failed",
                error_message=str(e),
                deployment_time=datetime.now()
            )]
    
    async def create_specialized_server(self, tool_design: ToolDesign) -> DeploymentResult:
        """
        Generate a new MCP server with specialized tools.
        
        Args:
            tool_design: Design specification for the tool
            
        Returns:
            Deployment result
        """
        server_name = f"{tool_design.name}-server"
        logger.info(f"Creating specialized server: {server_name}")
        
        try:
            server_code = await self.template_engine.generate_server(
                name=server_name,
                tools=[tool_design],
                integrations=tool_design.integration_points
            )
            
            server_config = MCPServerConfig(
                name=server_name,
                code=server_code,
                dependencies=tool_design.dependencies,
                transport=TransportType.STDIO,  # Default to stdio transport
                lifecycle="on-demand",  # Can be "persistent" or "on-demand"
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "tool_design": tool_design.metadata,
                    "tags": tool_design.tags
                }
            )
            
            deployment_result = await self.deploy_server(server_config)
            
            if deployment_result.success:
                await self._register_server_with_registry(server_config, tool_design)
                self.created_servers[server_name] = server_config
            
            return deployment_result
            
        except Exception as e:
            logger.error(f"Failed to create specialized server {server_name}: {e}")
            return DeploymentResult(
                success=False,
                server_name=server_name,
                error_message=str(e),
                deployment_time=datetime.now()
            )
    
    async def deploy_server(self, server_config: MCPServerConfig) -> DeploymentResult:
        """
        Deploy an MCP server to the filesystem and start it.
        
        Args:
            server_config: Server configuration
            
        Returns:
            Deployment result
        """
        server_name = server_config.name
        deployment_time = datetime.now()
        
        try:
            server_dir = self.workspace_dir / server_name
            server_dir.mkdir(parents=True, exist_ok=True)
            
            server_file = server_dir / "server.py"
            with open(server_file, 'w') as f:
                f.write(server_config.code)
            
            if server_config.dependencies:
                requirements_file = server_dir / "requirements.txt"
                with open(requirements_file, 'w') as f:
                    for dep in server_config.dependencies:
                        f.write(f"{dep}\n")
            
            manager_config = ServerConfig(
                name=server_name,
                command=["python", str(server_file)],
                transport=server_config.transport,
                enabled=True,
                working_dir=str(server_dir),
                env=server_config.env_vars,
                metadata=server_config.metadata
            )
            
            await self.server_manager.create_dynamic_server(
                name=server_name,
                command=manager_config.command,
                transport=server_config.transport,
                env=server_config.env_vars,
                auto_start=(server_config.lifecycle == "persistent")
            )
            
            result = DeploymentResult(
                success=True,
                server_name=server_name,
                server_path=str(server_file),
                deployment_time=deployment_time
            )
            
            self.deployment_history.append(result)
            logger.info(f"Successfully deployed server {server_name} at {server_file}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to deploy server {server_name}: {e}")
            result = DeploymentResult(
                success=False,
                server_name=server_name,
                error_message=str(e),
                deployment_time=deployment_time
            )
            self.deployment_history.append(result)
            return result
    
    async def create_from_api_spec(
        self,
        api_spec: Dict[str, Any],
        server_name: Optional[str] = None
    ) -> DeploymentResult:
        """
        Create an MCP server from an OpenAPI specification.
        
        This method integrates with mcp-openapi-schema-explorer patterns
        to automatically generate servers from API documentation.
        
        Args:
            api_spec: OpenAPI specification
            server_name: Optional server name (auto-generated if None)
            
        Returns:
            Deployment result
        """
        if not server_name:
            api_title = api_spec.get("info", {}).get("title", "api")
            server_name = f"{api_title.lower().replace(' ', '-')}-server"
        
        logger.info(f"Creating server from API spec: {server_name}")
        
        try:
            tools = await self._extract_tools_from_api_spec(api_spec)
            
            server_code = await self.template_engine.generate_api_server(
                name=server_name,
                api_spec=api_spec,
                tools=tools
            )
            
            dependencies = ["aiohttp", "pydantic", "mcp"]
            if "security" in api_spec:
                dependencies.append("authlib")
            
            server_config = MCPServerConfig(
                name=server_name,
                code=server_code,
                dependencies=dependencies,
                transport=TransportType.STDIO,
                lifecycle="persistent",  # API servers are typically persistent
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "source": "api_spec",
                    "api_info": api_spec.get("info", {}),
                    "endpoints": len(api_spec.get("paths", {}))
                }
            )
            
            return await self.deploy_server(server_config)
            
        except Exception as e:
            logger.error(f"Failed to create server from API spec: {e}")
            return DeploymentResult(
                success=False,
                server_name=server_name,
                error_message=str(e),
                deployment_time=datetime.now()
            )
    
    async def create_from_template(
        self,
        template_name: str,
        parameters: Dict[str, Any],
        server_name: Optional[str] = None
    ) -> DeploymentResult:
        """
        Create an MCP server from a predefined template.
        
        Args:
            template_name: Name of the template to use
            parameters: Template parameters
            server_name: Optional server name
            
        Returns:
            Deployment result
        """
        if not server_name:
            server_name = f"{template_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"Creating server from template {template_name}: {server_name}")
        
        try:
            server_code = await self.template_engine.generate_from_template(
                template_name=template_name,
                parameters=parameters,
                server_name=server_name
            )
            
            template = await self.template_engine.get_template(template_name)
            
            server_config = MCPServerConfig(
                name=server_name,
                code=server_code,
                dependencies=template.dependencies if template else ["mcp"],
                transport=TransportType.STDIO,
                lifecycle="on-demand",
                metadata={
                    "created_at": datetime.now().isoformat(),
                    "source": "template",
                    "template_name": template_name,
                    "parameters": parameters
                }
            )
            
            return await self.deploy_server(server_config)
            
        except Exception as e:
            logger.error(f"Failed to create server from template {template_name}: {e}")
            return DeploymentResult(
                success=False,
                server_name=server_name,
                error_message=str(e),
                deployment_time=datetime.now()
            )
    
    async def remove_server(self, server_name: str) -> bool:
        """
        Remove a dynamically created server.
        
        Args:
            server_name: Name of the server to remove
            
        Returns:
            True if removal successful
        """
        logger.info(f"Removing server: {server_name}")
        
        try:
            await self.server_manager.remove_server(server_name)
            
            await self.registry.unregister_server(server_name)
            
            server_dir = self.workspace_dir / server_name
            if server_dir.exists():
                shutil.rmtree(server_dir)
            
            if server_name in self.created_servers:
                del self.created_servers[server_name]
            
            logger.info(f"Successfully removed server {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove server {server_name}: {e}")
            return False
    
    async def list_created_servers(self) -> List[Dict[str, Any]]:
        """
        List all servers created by this factory.
        
        Returns:
            List of server information
        """
        servers = []
        
        for server_name, config in self.created_servers.items():
            status = self.server_manager.get_server_status(server_name)
            
            servers.append({
                "name": server_name,
                "lifecycle": config.lifecycle,
                "transport": config.transport.value,
                "dependencies": config.dependencies,
                "metadata": config.metadata,
                "status": status
            })
        
        return servers
    
    async def get_deployment_history(self) -> List[Dict[str, Any]]:
        """
        Get deployment history for analysis and debugging.
        
        Returns:
            List of deployment results
        """
        return [
            {
                "success": result.success,
                "server_name": result.server_name,
                "server_path": result.server_path,
                "error_message": result.error_message,
                "deployment_time": result.deployment_time.isoformat() if result.deployment_time else None
            }
            for result in self.deployment_history
        ]
    
    async def _design_tool_for_gap(
        self,
        gap: ToolGap,
        research_context: Dict[str, Any]
    ) -> Optional[ToolDesign]:
        """
        Design a tool to fill a specific capability gap.
        
        Args:
            gap: Identified tool gap
            research_context: Research context for design decisions
            
        Returns:
            Tool design or None if design fails
        """
        try:
            implementation = await self.capability_analyzer.suggest_implementation(
                gap, research_context
            )
            
            design = ToolDesign(
                name=gap.capability_name.lower().replace(" ", "_"),
                description=gap.description,
                input_schema=implementation.get("input_schema", {}),
                implementation_strategy=implementation.get("strategy", ""),
                dependencies=implementation.get("dependencies", []),
                integration_points=implementation.get("integrations", []),
                metadata={
                    "gap_priority": gap.priority,
                    "gap_category": gap.category,
                    "research_context": research_context.get("summary", "")
                },
                tags=implementation.get("tags", [])
            )
            
            return design
            
        except Exception as e:
            logger.error(f"Failed to design tool for gap {gap.capability_name}: {e}")
            return None
    
    async def _extract_tools_from_api_spec(
        self,
        api_spec: Dict[str, Any]
    ) -> List[ToolDesign]:
        """
        Extract tool designs from an OpenAPI specification.
        
        Args:
            api_spec: OpenAPI specification
            
        Returns:
            List of tool designs
        """
        tools = []
        paths = api_spec.get("paths", {})
        
        for path, methods in paths.items():
            for method, spec in methods.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    tool_name = f"{method}_{path.replace('/', '_').replace('{', '').replace('}', '').strip('_')}"
                    
                    input_schema = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                    
                    if "parameters" in spec:
                        for param in spec["parameters"]:
                            if param.get("in") == "path":
                                input_schema["properties"][param["name"]] = {
                                    "type": param.get("schema", {}).get("type", "string"),
                                    "description": param.get("description", "")
                                }
                                if param.get("required", False):
                                    input_schema["required"].append(param["name"])
                    
                    if "requestBody" in spec:
                        content = spec["requestBody"].get("content", {})
                        if "application/json" in content:
                            schema = content["application/json"].get("schema", {})
                            input_schema["properties"]["body"] = schema
                            input_schema["required"].append("body")
                    
                    tool = ToolDesign(
                        name=tool_name,
                        description=spec.get("summary", f"{method.upper()} {path}"),
                        input_schema=input_schema,
                        implementation_strategy="api_call",
                        dependencies=["aiohttp"],
                        metadata={
                            "api_path": path,
                            "api_method": method.upper(),
                            "api_spec": spec
                        },
                        tags=["api", method.lower()]
                    )
                    
                    tools.append(tool)
        
        return tools
    
    async def _register_server_with_registry(
        self,
        server_config: MCPServerConfig,
        tool_design: ToolDesign
    ):
        """
        Register a newly created server with the MCP registry.
        
        Args:
            server_config: Server configuration
            tool_design: Tool design used to create the server
        """
        try:
            await self.registry.register_server(
                server_name=server_config.name,
                capabilities=None,  # Would need to query the actual server
                tools=None,  # Would need to query the actual server
                metadata={
                    "factory_created": True,
                    "tool_design": tool_design.name,
                    "lifecycle": server_config.lifecycle,
                    "created_at": datetime.now().isoformat()
                }
            )
            
            logger.debug(f"Registered server {server_config.name} with registry")
            
        except Exception as e:
            logger.error(f"Failed to register server {server_config.name} with registry: {e}")
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive factory statistics.
        
        Returns:
            Factory statistics
        """
        successful_deployments = sum(1 for r in self.deployment_history if r.success)
        failed_deployments = len(self.deployment_history) - successful_deployments
        
        return {
            "total_servers_created": len(self.created_servers),
            "successful_deployments": successful_deployments,
            "failed_deployments": failed_deployments,
            "workspace_dir": str(self.workspace_dir),
            "deployment_history_count": len(self.deployment_history),
            "server_lifecycles": {
                "persistent": sum(1 for c in self.created_servers.values() if c.lifecycle == "persistent"),
                "on_demand": sum(1 for c in self.created_servers.values() if c.lifecycle == "on-demand")
            }
        }
