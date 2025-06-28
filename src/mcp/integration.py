"""
MCP Integration Module - Connects meta-server capabilities with the core framework.

This module provides the integration layer between the meta-server, existing MCP servers,
and the Venice.ai scaffolding system for seamless dynamic server creation and management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

from .meta_server import MetaMCPServer
from .server_manager import MCPServerManager
from .factory import ServerFactory
from .template_engine import TemplateEngine
from .capability_analyzer import CapabilityAnalyzer
from .registry import MCPRegistry
from .client import MCPClientManager

from ..venice.client import VeniceClient
from ..memory.long_term_memory import LongTermMemory
from ..agent.manager import AgentManager

logger = logging.getLogger(__name__)

@dataclass
class IntegratedServerInstance:
    """Integrated server instance data structure."""
    server_id: str
    server_type: str  # 'foundational' or 'generated'
    name: str
    status: str  # 'running', 'stopped', 'error'
    tools: List[str]
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]

class MCPIntegrationManager:
    """
    MCP Integration Manager for coordinating all MCP server capabilities.
    
    This class provides the central coordination point for:
    - Foundational MCP servers (Slack, Discord, Infrastructure, Automation)
    - Meta-server capabilities for dynamic server creation
    - Server lifecycle management and monitoring
    - Integration with Venice.ai scaffolding system
    """
    
    def __init__(
        self,
        venice_client: VeniceClient,
        long_term_memory: LongTermMemory,
        agent_manager: AgentManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MCP Integration Manager.
        
        Args:
            venice_client: Venice.ai client for AI processing
            long_term_memory: Long-term memory system
            agent_manager: Agent manager for coordination
            config: Configuration options
        """
        self.venice_client = venice_client
        self.long_term_memory = long_term_memory
        self.agent_manager = agent_manager
        self.config = config or {}
        
        self.server_manager = MCPServerManager(config.get("server_manager", {}))
        self.registry = MCPRegistry(config.get("registry", {}))
        self.client_manager = MCPClientManager(config.get("client_manager", {}))
        
        self.server_factory = ServerFactory(
            venice_client=venice_client,
            config=config.get("server_factory", {})
        )
        self.template_engine = TemplateEngine(config.get("template_engine", {}))
        self.capability_analyzer = CapabilityAnalyzer(
            venice_client=venice_client,
            long_term_memory=long_term_memory,
            config=config.get("capability_analyzer", {})
        )
        
        self.meta_server = MetaMCPServer(
            venice_client=venice_client,
            long_term_memory=long_term_memory,
            server_factory=self.server_factory,
            template_engine=self.template_engine,
            capability_analyzer=self.capability_analyzer,
            config=config.get("meta_server", {})
        )
        
        self.server_instances: Dict[str, IntegratedServerInstance] = {}
        
        self.foundational_servers = {
            "slack": {
                "module": "src.mcp_servers.slack_server",
                "class": "SlackMCPServer",
                "description": "Slack communication and AI processing"
            },
            "discord": {
                "module": "src.mcp_servers.discord_server", 
                "class": "DiscordMCPServer",
                "description": "Discord communication and management"
            },
            "infrastructure": {
                "module": "src.mcp_servers.infrastructure_server",
                "class": "InfrastructureMCPServer", 
                "description": "System monitoring and management"
            },
            "automation": {
                "module": "src.mcp_servers.automation_server",
                "class": "AutomationMCPServer",
                "description": "Browser automation and web search"
            }
        }
        
        self.stats = {
            "foundational_servers_active": 0,
            "generated_servers_active": 0,
            "total_tools_available": 0,
            "integration_requests": 0
        }
    
    async def initialize(self) -> None:
        """Initialize the MCP Integration Manager."""
        logger.info("Initializing MCP Integration Manager")
        
        try:
            await self.server_manager.initialize()
            await self.registry.initialize()
            await self.client_manager.initialize()
            
            await self.server_factory.initialize()
            await self.template_engine.initialize()
            await self.capability_analyzer.initialize()
            await self.meta_server.initialize()
            
            await self._register_foundational_servers()
            
            logger.info("MCP Integration Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Integration Manager: {e}")
            raise
    
    async def start_foundational_servers(
        self,
        servers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Start foundational MCP servers.
        
        Args:
            servers: List of server names to start (all if None)
            
        Returns:
            Startup results for each server
        """
        try:
            servers_to_start = servers or list(self.foundational_servers.keys())
            results = {}
            
            for server_name in servers_to_start:
                if server_name not in self.foundational_servers:
                    results[server_name] = {"error": "Unknown foundational server"}
                    continue
                
                try:
                    server_config = self.foundational_servers[server_name]
                    
                    start_result = await self.server_manager.start_server(
                        server_id=server_name,
                        server_config=server_config
                    )
                    
                    if start_result.get("success"):
                        instance = IntegratedServerInstance(
                            server_id=server_name,
                            server_type="foundational",
                            name=server_name,
                            status="running",
                            tools=await self._get_server_tools(server_name),
                            created_at=datetime.now(),
                            last_activity=datetime.now(),
                            metadata=server_config
                        )
                        
                        self.server_instances[server_name] = instance
                        await self.registry.register_server(server_name, instance)
                        
                        self.stats["foundational_servers_active"] += 1
                        self.stats["total_tools_available"] += len(instance.tools)
                    
                    results[server_name] = start_result
                    
                except Exception as e:
                    logger.error(f"Failed to start server {server_name}: {e}")
                    results[server_name] = {"error": str(e)}
            
            return {
                "started_servers": len([r for r in results.values() if r.get("success")]),
                "failed_servers": len([r for r in results.values() if "error" in r]),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to start foundational servers: {e}")
            return {"error": str(e)}
    
    async def create_dynamic_server(
        self,
        purpose: str,
        api_url: Optional[str] = None,
        research_depth: str = "medium"
    ) -> Dict[str, Any]:
        """
        Create a dynamic MCP server based on purpose and optional API.
        
        Args:
            purpose: Description of what the server should do
            api_url: Optional API to integrate with
            research_depth: Depth of research for server creation
            
        Returns:
            Dynamic server creation results
        """
        try:
            self.stats["integration_requests"] += 1
            
            if api_url:
                creation_result = await self.meta_server.generate_mcp_server(
                    purpose=purpose,
                    api_url=api_url
                )
            else:
                creation_result = await self.meta_server.research_and_create_server(
                    problem_description=purpose,
                    research_depth=research_depth
                )
            
            if "error" in creation_result:
                return creation_result
            
            if "server_creation" in creation_result:
                server_info = creation_result["server_creation"]
            else:
                server_info = creation_result
            
            server_id = server_info.get("server_id")
            server_name = server_info.get("server_name")
            
            if not server_id or not server_name:
                return {"error": "Invalid server creation result"}
            
            instance = IntegratedServerInstance(
                server_id=server_id,
                server_type="generated",
                name=server_name,
                status="created",
                tools=server_info.get("tools", []),
                created_at=datetime.now(),
                last_activity=datetime.now(),
                metadata={
                    "purpose": purpose,
                    "api_url": api_url,
                    "file_path": server_info.get("file_path"),
                    "research_depth": research_depth
                }
            )
            
            self.server_instances[server_id] = instance
            await self.registry.register_server(server_id, instance)
            
            self.stats["generated_servers_active"] += 1
            self.stats["total_tools_available"] += len(instance.tools)
            
            return {
                "server_id": server_id,
                "server_name": server_name,
                "status": "created",
                "tools_count": len(instance.tools),
                "creation_details": creation_result
            }
            
        except Exception as e:
            logger.error(f"Failed to create dynamic server: {e}")
            return {"error": str(e)}
    
    async def analyze_capability_gaps(
        self,
        problem_description: str
    ) -> Dict[str, Any]:
        """
        Analyze capability gaps for a given problem.
        
        Args:
            problem_description: Description of the problem to analyze
            
        Returns:
            Capability gap analysis results
        """
        try:
            current_capabilities = await self._get_current_capabilities()
            
            gap_analysis = await self.capability_analyzer.analyze_capability_gaps(
                problem_description=problem_description,
                current_capabilities=current_capabilities
            )
            
            return gap_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze capability gaps: {e}")
            return {"error": str(e)}
    
    async def get_server_status(self) -> Dict[str, Any]:
        """Get status of all MCP servers."""
        try:
            foundational_status = {}
            generated_status = {}
            
            for server_id, instance in self.server_instances.items():
                server_info = {
                    "name": instance.name,
                    "status": instance.status,
                    "tools_count": len(instance.tools),
                    "created_at": instance.created_at.isoformat(),
                    "last_activity": instance.last_activity.isoformat()
                }
                
                if instance.server_type == "foundational":
                    foundational_status[server_id] = server_info
                else:
                    generated_status[server_id] = server_info
            
            return {
                "foundational_servers": foundational_status,
                "generated_servers": generated_status,
                "statistics": self.stats.copy(),
                "registry_status": await self.registry.get_status()
            }
            
        except Exception as e:
            logger.error(f"Failed to get server status: {e}")
            return {"error": str(e)}
    
    async def execute_tool_across_servers(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        preferred_server: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool across available MCP servers.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            preferred_server: Preferred server to use (optional)
            
        Returns:
            Tool execution results
        """
        try:
            available_servers = []
            
            for server_id, instance in self.server_instances.items():
                if tool_name in instance.tools:
                    available_servers.append(server_id)
            
            if not available_servers:
                return {"error": f"Tool '{tool_name}' not found in any active server"}
            
            target_server = preferred_server if preferred_server in available_servers else available_servers[0]
            
            result = await self.client_manager.call_tool(
                server_id=target_server,
                tool_name=tool_name,
                arguments=arguments
            )
            
            if target_server in self.server_instances:
                self.server_instances[target_server].last_activity = datetime.now()
            
            return {
                "server_used": target_server,
                "tool_name": tool_name,
                "result": result,
                "available_servers": available_servers
            }
            
        except Exception as e:
            logger.error(f"Failed to execute tool across servers: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown all MCP servers and cleanup resources."""
        logger.info("Shutting down MCP Integration Manager")
        
        try:
            for server_id in list(self.server_instances.keys()):
                try:
                    await self.server_manager.stop_server(server_id)
                    del self.server_instances[server_id]
                except Exception as e:
                    logger.error(f"Error stopping server {server_id}: {e}")
            
            await self.meta_server.cleanup()
            await self.client_manager.cleanup()
            await self.server_manager.cleanup()
            
            logger.info("MCP Integration Manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during MCP Integration Manager shutdown: {e}")
    
    async def _register_foundational_servers(self) -> None:
        """Register foundational servers in the registry."""
        try:
            for server_name, server_config in self.foundational_servers.items():
                await self.registry.register_server_type(
                    server_type=server_name,
                    config=server_config
                )
                
        except Exception as e:
            logger.error(f"Failed to register foundational servers: {e}")
    
    async def _get_server_tools(self, server_name: str) -> List[str]:
        """Get list of tools for a server."""
        try:
            
            tool_mappings = {
                "slack": ["send_message", "get_channel_history", "analyze_sentiment"],
                "discord": ["send_message", "get_guild_info", "manage_roles"],
                "infrastructure": ["get_system_metrics", "check_service_status", "analyze_system_health"],
                "automation": ["create_browser_session", "perform_web_search", "extract_page_content"]
            }
            
            return tool_mappings.get(server_name, [])
            
        except Exception as e:
            logger.error(f"Failed to get server tools for {server_name}: {e}")
            return []
    
    async def _get_current_capabilities(self) -> Dict[str, List[str]]:
        """Get current capabilities from all active servers."""
        try:
            capabilities = {}
            
            for server_id, instance in self.server_instances.items():
                if instance.status == "running":
                    capabilities[server_id] = instance.tools
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Failed to get current capabilities: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration manager statistics."""
        return {
            "server_instances": len(self.server_instances),
            "performance_stats": self.stats.copy(),
            "component_stats": {
                "server_manager": self.server_manager.get_stats(),
                "registry": self.registry.get_stats(),
                "meta_server": self.meta_server.get_stats(),
                "capability_analyzer": self.capability_analyzer.get_stats()
            }
        }


integration_manager = None

async def get_integration_manager(
    venice_client: VeniceClient,
    long_term_memory: LongTermMemory,
    agent_manager: AgentManager,
    config: Optional[Dict[str, Any]] = None
) -> MCPIntegrationManager:
    """Get or create the global integration manager instance."""
    global integration_manager
    
    if integration_manager is None:
        integration_manager = MCPIntegrationManager(
            venice_client=venice_client,
            long_term_memory=long_term_memory,
            agent_manager=agent_manager,
            config=config
        )
        await integration_manager.initialize()
    
    return integration_manager
