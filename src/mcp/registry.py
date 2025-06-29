"""
MCP Server Registry for discovery and capability tracking.

This module provides a comprehensive registry system for MCP servers,
including capability discovery, tool registration, and dynamic server management.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
from pathlib import Path
import yaml

from mcp.types import Tool, Resource, Prompt, ServerCapabilities

logger = logging.getLogger(__name__)


class CapabilityType(Enum):
    """Types of MCP server capabilities."""
    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"
    LOGGING = "logging"
    EXPERIMENTAL = "experimental"


@dataclass
class ToolInfo:
    """Information about an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str
    last_updated: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceInfo:
    """Information about an MCP resource."""
    uri: str
    name: str
    description: str
    mime_type: Optional[str]
    server_name: str
    last_updated: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerInfo:
    """Information about a registered MCP server."""
    name: str
    capabilities: ServerCapabilities
    status: str
    last_heartbeat: datetime
    tools: List[ToolInfo] = field(default_factory=list)
    resources: List[ResourceInfo] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPRegistry:
    """
    Registry for MCP servers and their capabilities.
    
    Provides discovery, registration, and management of MCP servers
    with support for dynamic capability updates and usage analytics.
    """
    
    def __init__(self, registry_file: Optional[str] = None):
        """
        Initialize MCP registry.
        
        Args:
            registry_file: Optional file to persist registry data
        """
        self.registry_file = registry_file
        self.servers: Dict[str, ServerInfo] = {}
        self.tools: Dict[str, ToolInfo] = {}  # tool_name -> ToolInfo
        self.resources: Dict[str, ResourceInfo] = {}  # uri -> ResourceInfo
        self.capability_index: Dict[str, Set[str]] = {}  # capability -> server_names
        self.tag_index: Dict[str, Set[str]] = {}  # tag -> tool_names
        self.update_callbacks: List[Callable[[str, str], None]] = []  # (event_type, server_name)
        self._lock = asyncio.Lock()
    
    async def register_server(
        self,
        server_name: str,
        capabilities: ServerCapabilities,
        tools: Optional[List[Tool]] = None,
        resources: Optional[List[Resource]] = None,
        prompts: Optional[List[Prompt]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register an MCP server with its capabilities.
        
        Args:
            server_name: Unique name for the server
            capabilities: Server capabilities
            tools: List of available tools
            resources: List of available resources
            prompts: List of available prompts
            metadata: Additional server metadata
            
        Returns:
            True if registration successful
        """
        async with self._lock:
            try:
                logger.info(f"Registering MCP server: {server_name}")
                
                server_info = ServerInfo(
                    name=server_name,
                    capabilities=capabilities,
                    status="active",
                    last_heartbeat=datetime.now(),
                    metadata=metadata or {}
                )
                
                if tools:
                    for tool in tools:
                        tool_info = ToolInfo(
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=tool.inputSchema.dict() if tool.inputSchema else {},
                            server_name=server_name,
                            tags=set(tool.inputSchema.get("tags", [])) if tool.inputSchema else set()
                        )
                        server_info.tools.append(tool_info)
                        self.tools[tool.name] = tool_info
                        
                        for tag in tool_info.tags:
                            if tag not in self.tag_index:
                                self.tag_index[tag] = set()
                            self.tag_index[tag].add(tool.name)
                
                if resources:
                    for resource in resources:
                        resource_info = ResourceInfo(
                            uri=resource.uri,
                            name=resource.name or resource.uri,
                            description=resource.description or "",
                            mime_type=resource.mimeType,
                            server_name=server_name
                        )
                        server_info.resources.append(resource_info)
                        self.resources[resource.uri] = resource_info
                
                if prompts:
                    server_info.prompts = [prompt.name for prompt in prompts]
                
                self._update_capability_index(server_name, capabilities)
                
                self.servers[server_name] = server_info
                
                await self._persist_registry()
                
                for callback in self.update_callbacks:
                    try:
                        callback("server_registered", server_name)
                    except Exception as e:
                        logger.error(f"Error in registry callback: {e}")
                
                logger.info(f"Successfully registered server {server_name} with {len(server_info.tools)} tools")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register server {server_name}: {e}")
                return False
    
    async def unregister_server(self, server_name: str) -> bool:
        """
        Unregister an MCP server.
        
        Args:
            server_name: Name of the server to unregister
            
        Returns:
            True if unregistration successful
        """
        async with self._lock:
            if server_name not in self.servers:
                logger.warning(f"Server {server_name} not found in registry")
                return False
            
            try:
                server_info = self.servers[server_name]
                
                for tool_info in server_info.tools:
                    if tool_info.name in self.tools:
                        del self.tools[tool_info.name]
                    
                    for tag in tool_info.tags:
                        if tag in self.tag_index:
                            self.tag_index[tag].discard(tool_info.name)
                            if not self.tag_index[tag]:
                                del self.tag_index[tag]
                
                for resource_info in server_info.resources:
                    if resource_info.uri in self.resources:
                        del self.resources[resource_info.uri]
                
                for capability_set in self.capability_index.values():
                    capability_set.discard(server_name)
                
                self.capability_index = {
                    cap: servers for cap, servers in self.capability_index.items()
                    if servers
                }
                
                del self.servers[server_name]
                
                await self._persist_registry()
                
                for callback in self.update_callbacks:
                    try:
                        callback("server_unregistered", server_name)
                    except Exception as e:
                        logger.error(f"Error in registry callback: {e}")
                
                logger.info(f"Successfully unregistered server {server_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister server {server_name}: {e}")
                return False
    
    async def update_server_heartbeat(self, server_name: str) -> bool:
        """
        Update server heartbeat timestamp.
        
        Args:
            server_name: Name of the server
            
        Returns:
            True if update successful
        """
        async with self._lock:
            if server_name in self.servers:
                self.servers[server_name].last_heartbeat = datetime.now()
                return True
            return False
    
    async def find_tools_by_capability(self, capability: str) -> List[ToolInfo]:
        """
        Find tools that provide a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of matching tools
        """
        matching_tools = []
        
        for tool_info in self.tools.values():
            if (capability.lower() in tool_info.description.lower() or
                capability.lower() in tool_info.name.lower() or
                capability in tool_info.tags):
                matching_tools.append(tool_info)
        
        matching_tools.sort(
            key=lambda t: (t.success_rate, t.usage_count),
            reverse=True
        )
        
        return matching_tools
    
    async def find_tools_by_tags(self, tags: List[str]) -> List[ToolInfo]:
        """
        Find tools that match any of the given tags.
        
        Args:
            tags: List of tags to search for
            
        Returns:
            List of matching tools
        """
        matching_tool_names = set()
        
        for tag in tags:
            if tag in self.tag_index:
                matching_tool_names.update(self.tag_index[tag])
        
        return [self.tools[name] for name in matching_tool_names if name in self.tools]
    
    async def find_resources_by_type(self, mime_type: str) -> List[ResourceInfo]:
        """
        Find resources by MIME type.
        
        Args:
            mime_type: MIME type to search for
            
        Returns:
            List of matching resources
        """
        return [
            resource for resource in self.resources.values()
            if resource.mime_type == mime_type
        ]
    
    async def get_server_capabilities(self, server_name: str) -> Optional[ServerCapabilities]:
        """
        Get capabilities for a specific server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server capabilities or None if not found
        """
        if server_name in self.servers:
            return self.servers[server_name].capabilities
        return None
    
    async def get_servers_by_capability(self, capability_type: CapabilityType) -> List[str]:
        """
        Get servers that support a specific capability type.
        
        Args:
            capability_type: Type of capability
            
        Returns:
            List of server names
        """
        capability_key = capability_type.value
        return list(self.capability_index.get(capability_key, set()))
    
    async def record_tool_usage(
        self,
        tool_name: str,
        success: bool,
        execution_time: float
    ):
        """
        Record tool usage statistics.
        
        Args:
            tool_name: Name of the tool
            success: Whether the execution was successful
            execution_time: Execution time in seconds
        """
        async with self._lock:
            if tool_name in self.tools:
                tool_info = self.tools[tool_name]
                tool_info.usage_count += 1
                
                alpha = 0.1  # Learning rate
                if tool_info.usage_count == 1:
                    tool_info.success_rate = 1.0 if success else 0.0
                else:
                    current_success = 1.0 if success else 0.0
                    tool_info.success_rate = (
                        alpha * current_success + 
                        (1 - alpha) * tool_info.success_rate
                    )
                
                if tool_info.usage_count == 1:
                    tool_info.average_execution_time = execution_time
                else:
                    tool_info.average_execution_time = (
                        alpha * execution_time + 
                        (1 - alpha) * tool_info.average_execution_time
                    )
                
                tool_info.last_updated = datetime.now()
    
    async def get_tool_recommendations(
        self,
        query: str,
        limit: int = 10
    ) -> List[ToolInfo]:
        """
        Get tool recommendations based on a query.
        
        Args:
            query: Search query
            limit: Maximum number of recommendations
            
        Returns:
            List of recommended tools
        """
        query_lower = query.lower()
        scored_tools = []
        
        for tool_info in self.tools.values():
            score = 0.0
            
            if query_lower in tool_info.name.lower():
                score += 10.0
            
            if query_lower in tool_info.description.lower():
                score += 5.0
            
            for tag in tool_info.tags:
                if query_lower in tag.lower():
                    score += 3.0
            
            score *= (1 + tool_info.success_rate)
            score *= (1 + min(tool_info.usage_count / 100, 1.0))  # Cap usage boost
            
            if score > 0:
                scored_tools.append((score, tool_info))
        
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        return [tool_info for _, tool_info in scored_tools[:limit]]
    
    def add_update_callback(self, callback: Callable[[str, str], None]):
        """
        Add a callback for registry updates.
        
        Args:
            callback: Function to call on updates (event_type, server_name)
        """
        self.update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable[[str, str], None]):
        """
        Remove an update callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    async def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        active_servers = sum(
            1 for server in self.servers.values()
            if server.status == "active"
        )
        
        total_tools = len(self.tools)
        total_resources = len(self.resources)
        
        if self.tools:
            avg_success_rate = sum(
                tool.success_rate for tool in self.tools.values()
            ) / len(self.tools)
        else:
            avg_success_rate = 0.0
        
        most_used_tools = sorted(
            self.tools.values(),
            key=lambda t: t.usage_count,
            reverse=True
        )[:5]
        
        return {
            "total_servers": len(self.servers),
            "active_servers": active_servers,
            "total_tools": total_tools,
            "total_resources": total_resources,
            "total_tags": len(self.tag_index),
            "average_success_rate": avg_success_rate,
            "most_used_tools": [
                {
                    "name": tool.name,
                    "usage_count": tool.usage_count,
                    "success_rate": tool.success_rate,
                    "server": tool.server_name
                }
                for tool in most_used_tools
            ],
            "capability_distribution": {
                cap: len(servers) for cap, servers in self.capability_index.items()
            }
        }
    
    async def export_registry(self) -> Dict[str, Any]:
        """
        Export the entire registry to a dictionary.
        
        Returns:
            Registry data as dictionary
        """
        return {
            "servers": {
                name: {
                    "name": server.name,
                    "capabilities": server.capabilities.dict(),
                    "status": server.status,
                    "last_heartbeat": server.last_heartbeat.isoformat(),
                    "tools": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                            "usage_count": tool.usage_count,
                            "success_rate": tool.success_rate,
                            "tags": list(tool.tags)
                        }
                        for tool in server.tools
                    ],
                    "resources": [
                        {
                            "uri": resource.uri,
                            "name": resource.name,
                            "description": resource.description,
                            "mime_type": resource.mime_type,
                            "access_count": resource.access_count
                        }
                        for resource in server.resources
                    ],
                    "prompts": server.prompts,
                    "metadata": server.metadata
                }
                for name, server in self.servers.items()
            },
            "export_timestamp": datetime.now().isoformat(),
            "registry_version": "1.0"
        }
    
    async def import_registry(self, data: Dict[str, Any]) -> bool:
        """
        Import registry data from a dictionary.
        
        Args:
            data: Registry data to import
            
        Returns:
            True if import successful
        """
        try:
            async with self._lock:
                self.servers.clear()
                self.tools.clear()
                self.resources.clear()
                self.capability_index.clear()
                self.tag_index.clear()
                
                servers_data = data.get("servers", {})
                for server_name, server_data in servers_data.items():
                    capabilities = ServerCapabilities(**server_data["capabilities"])
                    
                    tools = []
                    for tool_data in server_data.get("tools", []):
                        tool_info = ToolInfo(
                            name=tool_data["name"],
                            description=tool_data["description"],
                            input_schema=tool_data["input_schema"],
                            server_name=server_name,
                            usage_count=tool_data.get("usage_count", 0),
                            success_rate=tool_data.get("success_rate", 0.0),
                            tags=set(tool_data.get("tags", []))
                        )
                        tools.append(tool_info)
                    
                    resources = []
                    for resource_data in server_data.get("resources", []):
                        resource_info = ResourceInfo(
                            uri=resource_data["uri"],
                            name=resource_data["name"],
                            description=resource_data["description"],
                            mime_type=resource_data.get("mime_type"),
                            server_name=server_name,
                            access_count=resource_data.get("access_count", 0)
                        )
                        resources.append(resource_info)
                    
                    server_info = ServerInfo(
                        name=server_name,
                        capabilities=capabilities,
                        status=server_data.get("status", "inactive"),
                        last_heartbeat=datetime.fromisoformat(server_data["last_heartbeat"]),
                        tools=tools,
                        resources=resources,
                        prompts=server_data.get("prompts", []),
                        metadata=server_data.get("metadata", {})
                    )
                    
                    self.servers[server_name] = server_info
                    
                    for tool_info in tools:
                        self.tools[tool_info.name] = tool_info
                        for tag in tool_info.tags:
                            if tag not in self.tag_index:
                                self.tag_index[tag] = set()
                            self.tag_index[tag].add(tool_info.name)
                    
                    for resource_info in resources:
                        self.resources[resource_info.uri] = resource_info
                    
                    self._update_capability_index(server_name, capabilities)
                
                logger.info(f"Successfully imported registry with {len(self.servers)} servers")
                return True
                
        except Exception as e:
            logger.error(f"Failed to import registry: {e}")
            return False
    
    def _update_capability_index(self, server_name: str, capabilities: ServerCapabilities):
        """Update the capability index for a server."""
        if capabilities.tools:
            if "tool" not in self.capability_index:
                self.capability_index["tool"] = set()
            self.capability_index["tool"].add(server_name)
        
        if capabilities.resources:
            if "resource" not in self.capability_index:
                self.capability_index["resource"] = set()
            self.capability_index["resource"].add(server_name)
        
        if capabilities.prompts:
            if "prompt" not in self.capability_index:
                self.capability_index["prompt"] = set()
            self.capability_index["prompt"].add(server_name)
        
        if capabilities.logging:
            if "logging" not in self.capability_index:
                self.capability_index["logging"] = set()
            self.capability_index["logging"].add(server_name)
        
        if capabilities.experimental:
            if "experimental" not in self.capability_index:
                self.capability_index["experimental"] = set()
            self.capability_index["experimental"].add(server_name)
    
    async def _persist_registry(self):
        """Persist registry data to file if configured."""
        if not self.registry_file:
            return
        
        try:
            registry_data = await self.export_registry()
            registry_path = Path(self.registry_file)
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.debug(f"Registry persisted to {self.registry_file}")
            
        except Exception as e:
            logger.error(f"Failed to persist registry: {e}")
    
    async def load_registry(self) -> bool:
        """
        Load registry data from file.
        
        Returns:
            True if load successful
        """
        if not self.registry_file:
            return True
        
        try:
            registry_path = Path(self.registry_file)
            if not registry_path.exists():
                logger.info("No existing registry file found, starting with empty registry")
                return True
            
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            return await self.import_registry(registry_data)
            
        except Exception as e:
            logger.error(f"Failed to load registry from {self.registry_file}: {e}")
            return False
