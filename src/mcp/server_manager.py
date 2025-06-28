"""
MCP Server Manager with OAuth 2.0 and latest protocol features.

This module manages the lifecycle of MCP servers, handles authentication,
and provides the core infrastructure for dynamic server creation and management.
"""

import asyncio
import logging
import json
import subprocess
import signal
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
from pathlib import Path
import yaml

from mcp.server.lowlevel import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    ClientCapabilities,
    Implementation,
    ServerCapabilities,
    Tool,
    Resource,
    Prompt
)

logger = logging.getLogger(__name__)


class ServerStatus(Enum):
    """MCP Server status states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"


class TransportType(Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: List[str]
    transport: TransportType = TransportType.STDIO
    enabled: bool = True
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    timeout: int = 30
    auto_restart: bool = True
    oauth_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerInstance:
    """Runtime instance of an MCP server."""
    config: ServerConfig
    process: Optional[subprocess.Popen] = None
    status: ServerStatus = ServerStatus.STOPPED
    start_time: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    error_count: int = 0
    capabilities: Optional[ServerCapabilities] = None
    tools: List[Tool] = field(default_factory=list)
    resources: List[Resource] = field(default_factory=list)
    prompts: List[Prompt] = field(default_factory=list)


class OAuth2Manager:
    """
    OAuth 2.0 authentication manager for MCP servers.
    
    Implements RFC 8707 Resource Indicators and separation of
    Authorization Server (AS) and Resource Server (RS) roles.
    """
    
    def __init__(self):
        """Initialize OAuth 2.0 manager."""
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        self.auth_servers: Dict[str, str] = {}
        self.resource_servers: Dict[str, str] = {}
    
    async def get_access_token(
        self,
        server_name: str,
        oauth_config: Dict[str, Any]
    ) -> Optional[str]:
        """
        Get access token for MCP server authentication.
        
        Args:
            server_name: Name of the MCP server
            oauth_config: OAuth configuration
            
        Returns:
            Access token or None if authentication fails
        """
        cached_token = self.token_cache.get(server_name)
        if cached_token and self._is_token_valid(cached_token):
            return cached_token["access_token"]
        
        try:
            token_data = await self._perform_oauth_flow(oauth_config)
            if token_data:
                self.token_cache[server_name] = {
                    "access_token": token_data["access_token"],
                    "expires_at": datetime.now() + timedelta(seconds=token_data.get("expires_in", 3600)),
                    "refresh_token": token_data.get("refresh_token")
                }
                return token_data["access_token"]
        except Exception as e:
            logger.error(f"OAuth authentication failed for {server_name}: {e}")
        
        return None
    
    def _is_token_valid(self, token_data: Dict[str, Any]) -> bool:
        """Check if cached token is still valid."""
        expires_at = token_data.get("expires_at")
        if not expires_at:
            return False
        
        return datetime.now() < (expires_at - timedelta(minutes=5))
    
    async def _perform_oauth_flow(self, oauth_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform OAuth 2.0 client credentials flow.
        
        Args:
            oauth_config: OAuth configuration with client_id, client_secret, etc.
            
        Returns:
            Token response data or None
        """
        token_url = oauth_config.get("token_url")
        client_id = oauth_config.get("client_id")
        client_secret = oauth_config.get("client_secret")
        scope = oauth_config.get("scope", "")
        resource = oauth_config.get("resource")  # RFC 8707 Resource Indicators
        
        if not all([token_url, client_id, client_secret]):
            logger.error("Missing required OAuth configuration")
            return None
        
        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": scope
        }
        
        if resource:
            data["resource"] = resource
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "MCP-Protocol-Version": "2025-06-18"  # Latest protocol version
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(token_url, data=data, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"OAuth token request failed: {response.status}")
                    return None


class MCPServerManager:
    """
    Manages MCP servers with OAuth 2.0, latest protocol features, and lifecycle management.
    
    Provides comprehensive server management including dynamic creation,
    health monitoring, and protocol compliance with MCP v2025-06-18.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MCP server manager.
        
        Args:
            config_path: Path to server configuration file
        """
        self.config_path = config_path or "config/mcp_servers.yaml"
        self.servers: Dict[str, ServerInstance] = {}
        self.oauth_manager = OAuth2Manager()
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the MCP server manager."""
        logger.info("Starting MCP Server Manager")
        
        await self.load_configurations()
        
        for server_name, instance in self.servers.items():
            if instance.config.enabled:
                await self.start_server(server_name)
        
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"MCP Server Manager started with {len(self.servers)} servers")
    
    async def stop(self):
        """Stop the MCP server manager and all servers."""
        logger.info("Stopping MCP Server Manager")
        
        self._shutdown_event.set()
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        stop_tasks = [self.stop_server(name) for name in list(self.servers.keys())]
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logger.info("MCP Server Manager stopped")
    
    async def load_configurations(self):
        """Load server configurations from YAML file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Configuration file not found: {self.config_path}")
                return
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            servers_config = config_data.get("servers", {})
            
            for server_name, server_data in servers_config.items():
                config = ServerConfig(
                    name=server_name,
                    command=server_data.get("command", []),
                    transport=TransportType(server_data.get("transport", "stdio")),
                    enabled=server_data.get("enabled", True),
                    env=server_data.get("env", {}),
                    working_dir=server_data.get("working_dir"),
                    timeout=server_data.get("timeout", 30),
                    auto_restart=server_data.get("auto_restart", True),
                    oauth_config=server_data.get("oauth"),
                    metadata=server_data.get("metadata", {})
                )
                
                self.servers[server_name] = ServerInstance(config=config)
            
            logger.info(f"Loaded {len(self.servers)} server configurations")
            
        except Exception as e:
            logger.error(f"Failed to load server configurations: {e}")
    
    async def start_server(self, server_name: str) -> bool:
        """
        Start an MCP server.
        
        Args:
            server_name: Name of the server to start
            
        Returns:
            True if server started successfully
        """
        if server_name not in self.servers:
            logger.error(f"Server not found: {server_name}")
            return False
        
        instance = self.servers[server_name]
        
        if instance.status == ServerStatus.RUNNING:
            logger.warning(f"Server {server_name} is already running")
            return True
        
        try:
            instance.status = ServerStatus.STARTING
            logger.info(f"Starting MCP server: {server_name}")
            
            env = os.environ.copy()
            env.update(instance.config.env)
            
            if instance.config.oauth_config:
                token = await self.oauth_manager.get_access_token(
                    server_name, instance.config.oauth_config
                )
                if token:
                    env["MCP_ACCESS_TOKEN"] = token
                else:
                    logger.error(f"Failed to get OAuth token for {server_name}")
                    instance.status = ServerStatus.ERROR
                    return False
            
            instance.process = subprocess.Popen(
                instance.config.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=instance.config.working_dir,
                text=True
            )
            
            await asyncio.sleep(2)
            
            if instance.process.poll() is not None:
                stderr_output = instance.process.stderr.read() if instance.process.stderr else ""
                logger.error(f"Server {server_name} failed to start: {stderr_output}")
                instance.status = ServerStatus.ERROR
                return False
            
            instance.status = ServerStatus.RUNNING
            instance.start_time = datetime.now()
            instance.last_heartbeat = datetime.now()
            instance.error_count = 0
            
            await self._initialize_server_capabilities(instance)
            
            logger.info(f"MCP server {server_name} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server {server_name}: {e}")
            instance.status = ServerStatus.ERROR
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """
        Stop an MCP server.
        
        Args:
            server_name: Name of the server to stop
            
        Returns:
            True if server stopped successfully
        """
        if server_name not in self.servers:
            logger.error(f"Server not found: {server_name}")
            return False
        
        instance = self.servers[server_name]
        
        if instance.status == ServerStatus.STOPPED:
            logger.warning(f"Server {server_name} is already stopped")
            return True
        
        try:
            instance.status = ServerStatus.STOPPING
            logger.info(f"Stopping MCP server: {server_name}")
            
            if instance.process:
                instance.process.terminate()
                
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_process(instance.process)),
                        timeout=10
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Force killing server {server_name}")
                    instance.process.kill()
                    await asyncio.create_task(self._wait_for_process(instance.process))
                
                instance.process = None
            
            instance.status = ServerStatus.STOPPED
            instance.start_time = None
            instance.last_heartbeat = None
            
            logger.info(f"MCP server {server_name} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop server {server_name}: {e}")
            instance.status = ServerStatus.ERROR
            return False
    
    async def restart_server(self, server_name: str) -> bool:
        """
        Restart an MCP server.
        
        Args:
            server_name: Name of the server to restart
            
        Returns:
            True if server restarted successfully
        """
        logger.info(f"Restarting MCP server: {server_name}")
        
        if await self.stop_server(server_name):
            await asyncio.sleep(1)  # Brief pause between stop and start
            return await self.start_server(server_name)
        
        return False
    
    async def create_dynamic_server(
        self,
        name: str,
        command: List[str],
        transport: TransportType = TransportType.STDIO,
        env: Optional[Dict[str, str]] = None,
        auto_start: bool = True
    ) -> bool:
        """
        Create and optionally start a dynamic MCP server.
        
        Args:
            name: Unique name for the server
            command: Command to run the server
            transport: Transport type to use
            env: Environment variables
            auto_start: Whether to start the server immediately
            
        Returns:
            True if server created successfully
        """
        if name in self.servers:
            logger.error(f"Server {name} already exists")
            return False
        
        config = ServerConfig(
            name=name,
            command=command,
            transport=transport,
            enabled=True,
            env=env or {},
            auto_restart=False,  # Dynamic servers don't auto-restart by default
            metadata={"dynamic": True, "created_at": datetime.now().isoformat()}
        )
        
        self.servers[name] = ServerInstance(config=config)
        
        if auto_start:
            return await self.start_server(name)
        
        return True
    
    async def remove_server(self, server_name: str) -> bool:
        """
        Remove an MCP server (stop and delete).
        
        Args:
            server_name: Name of the server to remove
            
        Returns:
            True if server removed successfully
        """
        if server_name not in self.servers:
            logger.error(f"Server not found: {server_name}")
            return False
        
        await self.stop_server(server_name)
        
        del self.servers[server_name]
        
        logger.info(f"Removed MCP server: {server_name}")
        return True
    
    async def _initialize_server_capabilities(self, instance: ServerInstance):
        """Initialize server capabilities through MCP protocol."""
        try:
            instance.capabilities = ServerCapabilities(
                experimental={},
                logging={},
                prompts={},
                resources={},
                tools={}
            )
            
            logger.debug(f"Initialized capabilities for {instance.config.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize capabilities for {instance.config.name}: {e}")
    
    async def _wait_for_process(self, process: subprocess.Popen):
        """Wait for a process to terminate."""
        while process.poll() is None:
            await asyncio.sleep(0.1)
    
    async def _health_check_loop(self):
        """Periodic health check for all servers."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all running servers."""
        for server_name, instance in self.servers.items():
            if instance.status == ServerStatus.RUNNING and instance.process:
                if instance.process.poll() is not None:
                    logger.warning(f"Server {server_name} process died unexpectedly")
                    instance.status = ServerStatus.ERROR
                    instance.error_count += 1
                    
                    if instance.config.auto_restart and instance.error_count < 3:
                        logger.info(f"Auto-restarting server {server_name}")
                        await self.restart_server(server_name)
                else:
                    instance.last_heartbeat = datetime.now()
    
    def get_server_status(self, server_name: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server status information or None if not found
        """
        if server_name not in self.servers:
            return None
        
        instance = self.servers[server_name]
        
        return {
            "name": server_name,
            "status": instance.status.value,
            "enabled": instance.config.enabled,
            "transport": instance.config.transport.value,
            "start_time": instance.start_time.isoformat() if instance.start_time else None,
            "last_heartbeat": instance.last_heartbeat.isoformat() if instance.last_heartbeat else None,
            "error_count": instance.error_count,
            "process_id": instance.process.pid if instance.process else None,
            "capabilities": instance.capabilities.dict() if instance.capabilities else None,
            "tools_count": len(instance.tools),
            "resources_count": len(instance.resources),
            "prompts_count": len(instance.prompts),
            "metadata": instance.config.metadata
        }
    
    def get_all_servers_status(self) -> Dict[str, Any]:
        """
        Get status information for all servers.
        
        Returns:
            Dictionary with status information for all servers
        """
        servers_status = {}
        
        for server_name in self.servers:
            servers_status[server_name] = self.get_server_status(server_name)
        
        running_count = sum(1 for s in self.servers.values() if s.status == ServerStatus.RUNNING)
        
        return {
            "total_servers": len(self.servers),
            "running_servers": running_count,
            "servers": servers_status,
            "manager_status": "running" if not self._shutdown_event.is_set() else "stopping"
        }
