"""
MCP Client for JSON-RPC communication with MCP servers.

This module provides a client for communicating with MCP servers using
the JSON-RPC protocol with support for the latest MCP features.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import uuid

from .transport import MCPTransport, MCPMessage, TransportFactory
from .registry import MCPRegistry, ToolInfo

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """Represents an MCP request."""
    id: str
    method: str
    params: Dict[str, Any]
    timestamp: datetime
    timeout: float = 30.0


@dataclass
class MCPResponse:
    """Represents an MCP response."""
    id: str
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class MCPClientError(Exception):
    """Exception raised for MCP client errors."""
    def __init__(self, message: str, error_code: Optional[int] = None, error_data: Optional[Any] = None):
        self.message = message
        self.error_code = error_code
        self.error_data = error_data
        super().__init__(self.message)


class MCPClient:
    """
    Client for communicating with MCP servers via JSON-RPC.
    
    Provides high-level methods for tool execution, resource access,
    and server management with support for the latest MCP protocol features.
    """
    
    def __init__(
        self,
        server_name: str,
        transport: MCPTransport,
        registry: Optional[MCPRegistry] = None,
        timeout: float = 30.0
    ):
        """
        Initialize MCP client.
        
        Args:
            server_name: Name of the MCP server
            transport: Transport layer for communication
            registry: Optional registry for capability tracking
            timeout: Default timeout for requests
        """
        self.server_name = server_name
        self.transport = transport
        self.registry = registry
        self.timeout = timeout
        self.connected = False
        self.pending_requests: Dict[str, MCPRequest] = {}
        self.response_futures: Dict[str, asyncio.Future] = {}
        self.notification_handlers: Dict[str, Callable] = {}
        self._request_id_counter = 0
        
        self.transport.set_message_handler(self._handle_message)
    
    async def connect(self) -> bool:
        """
        Connect to the MCP server.
        
        Returns:
            True if connection successful
        """
        try:
            logger.info(f"Connecting to MCP server: {self.server_name}")
            
            if await self.transport.start():
                self.connected = True
                
                await self._initialize_connection()
                
                logger.info(f"Successfully connected to {self.server_name}")
                return True
            else:
                logger.error(f"Failed to start transport for {self.server_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to {self.server_name}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server."""
        logger.info(f"Disconnecting from MCP server: {self.server_name}")
        
        self.connected = False
        
        for request_id, future in self.response_futures.items():
            if not future.done():
                future.cancel()
        
        self.response_futures.clear()
        self.pending_requests.clear()
        
        await self.transport.stop()
        
        logger.info(f"Disconnected from {self.server_name}")
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Any:
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            timeout: Request timeout (uses default if None)
            
        Returns:
            Tool execution result
            
        Raises:
            MCPClientError: If tool call fails
        """
        if not self.connected:
            raise MCPClientError("Client not connected to server")
        
        start_time = datetime.now()
        
        try:
            logger.debug(f"Calling tool {tool_name} with arguments: {arguments}")
            
            response = await self._send_request(
                method="tools/call",
                params={
                    "name": tool_name,
                    "arguments": arguments
                },
                timeout=timeout or self.timeout
            )
            
            if response.error:
                raise MCPClientError(
                    f"Tool call failed: {response.error.get('message', 'Unknown error')}",
                    error_code=response.error.get('code'),
                    error_data=response.error.get('data')
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            if self.registry:
                await self.registry.record_tool_usage(
                    tool_name=tool_name,
                    success=True,
                    execution_time=execution_time
                )
            
            logger.debug(f"Tool {tool_name} executed successfully in {execution_time:.2f}s")
            return response.result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            if self.registry:
                await self.registry.record_tool_usage(
                    tool_name=tool_name,
                    success=False,
                    execution_time=execution_time
                )
            
            if isinstance(e, MCPClientError):
                raise
            else:
                raise MCPClientError(f"Tool call error: {e}")
    
    async def list_tools(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        List available tools from the MCP server.
        
        Args:
            timeout: Request timeout
            
        Returns:
            List of tool definitions
        """
        if not self.connected:
            raise MCPClientError("Client not connected to server")
        
        try:
            response = await self._send_request(
                method="tools/list",
                params={},
                timeout=timeout or self.timeout
            )
            
            if response.error:
                raise MCPClientError(
                    f"Failed to list tools: {response.error.get('message', 'Unknown error')}",
                    error_code=response.error.get('code')
                )
            
            tools = response.result.get("tools", [])
            logger.debug(f"Listed {len(tools)} tools from {self.server_name}")
            return tools
            
        except Exception as e:
            if isinstance(e, MCPClientError):
                raise
            else:
                raise MCPClientError(f"List tools error: {e}")
    
    async def get_resource(
        self,
        uri: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get a resource from the MCP server.
        
        Args:
            uri: Resource URI
            timeout: Request timeout
            
        Returns:
            Resource data
        """
        if not self.connected:
            raise MCPClientError("Client not connected to server")
        
        try:
            response = await self._send_request(
                method="resources/read",
                params={"uri": uri},
                timeout=timeout or self.timeout
            )
            
            if response.error:
                raise MCPClientError(
                    f"Failed to get resource: {response.error.get('message', 'Unknown error')}",
                    error_code=response.error.get('code')
                )
            
            return response.result
            
        except Exception as e:
            if isinstance(e, MCPClientError):
                raise
            else:
                raise MCPClientError(f"Get resource error: {e}")
    
    async def list_resources(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        List available resources from the MCP server.
        
        Args:
            timeout: Request timeout
            
        Returns:
            List of resource definitions
        """
        if not self.connected:
            raise MCPClientError("Client not connected to server")
        
        try:
            response = await self._send_request(
                method="resources/list",
                params={},
                timeout=timeout or self.timeout
            )
            
            if response.error:
                raise MCPClientError(
                    f"Failed to list resources: {response.error.get('message', 'Unknown error')}",
                    error_code=response.error.get('code')
                )
            
            resources = response.result.get("resources", [])
            logger.debug(f"Listed {len(resources)} resources from {self.server_name}")
            return resources
            
        except Exception as e:
            if isinstance(e, MCPClientError):
                raise
            else:
                raise MCPClientError(f"List resources error: {e}")
    
    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get a prompt from the MCP server.
        
        Args:
            name: Prompt name
            arguments: Prompt arguments
            timeout: Request timeout
            
        Returns:
            Prompt data
        """
        if not self.connected:
            raise MCPClientError("Client not connected to server")
        
        try:
            params = {"name": name}
            if arguments:
                params["arguments"] = arguments
            
            response = await self._send_request(
                method="prompts/get",
                params=params,
                timeout=timeout or self.timeout
            )
            
            if response.error:
                raise MCPClientError(
                    f"Failed to get prompt: {response.error.get('message', 'Unknown error')}",
                    error_code=response.error.get('code')
                )
            
            return response.result
            
        except Exception as e:
            if isinstance(e, MCPClientError):
                raise
            else:
                raise MCPClientError(f"Get prompt error: {e}")
    
    async def list_prompts(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        List available prompts from the MCP server.
        
        Args:
            timeout: Request timeout
            
        Returns:
            List of prompt definitions
        """
        if not self.connected:
            raise MCPClientError("Client not connected to server")
        
        try:
            response = await self._send_request(
                method="prompts/list",
                params={},
                timeout=timeout or self.timeout
            )
            
            if response.error:
                raise MCPClientError(
                    f"Failed to list prompts: {response.error.get('message', 'Unknown error')}",
                    error_code=response.error.get('code')
                )
            
            prompts = response.result.get("prompts", [])
            logger.debug(f"Listed {len(prompts)} prompts from {self.server_name}")
            return prompts
            
        except Exception as e:
            if isinstance(e, MCPClientError):
                raise
            else:
                raise MCPClientError(f"List prompts error: {e}")
    
    async def ping(self, timeout: Optional[float] = None) -> bool:
        """
        Ping the MCP server to check connectivity.
        
        Args:
            timeout: Request timeout
            
        Returns:
            True if server responds
        """
        if not self.connected:
            return False
        
        try:
            response = await self._send_request(
                method="ping",
                params={},
                timeout=timeout or 5.0
            )
            
            return response.error is None
            
        except Exception:
            return False
    
    def add_notification_handler(self, method: str, handler: Callable):
        """
        Add a handler for server notifications.
        
        Args:
            method: Notification method name
            handler: Handler function
        """
        self.notification_handlers[method] = handler
    
    def remove_notification_handler(self, method: str):
        """
        Remove a notification handler.
        
        Args:
            method: Notification method name
        """
        if method in self.notification_handlers:
            del self.notification_handlers[method]
    
    async def _initialize_connection(self):
        """Initialize the MCP connection with handshake."""
        try:
            response = await self._send_request(
                method="initialize",
                params={
                    "protocolVersion": "2025-06-18",
                    "capabilities": {
                        "experimental": {},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "ai-core-scaffolding",
                        "version": "0.1.0"
                    }
                },
                timeout=10.0
            )
            
            if response.error:
                raise MCPClientError(f"Initialization failed: {response.error}")
            
            await self._send_notification("notifications/initialized", {})
            
            logger.debug(f"Initialized connection to {self.server_name}")
            
        except Exception as e:
            raise MCPClientError(f"Connection initialization failed: {e}")
    
    async def _send_request(
        self,
        method: str,
        params: Dict[str, Any],
        timeout: float
    ) -> MCPResponse:
        """
        Send a JSON-RPC request and wait for response.
        
        Args:
            method: RPC method name
            params: Method parameters
            timeout: Request timeout
            
        Returns:
            MCP response
        """
        request_id = self._generate_request_id()
        
        request = MCPRequest(
            id=request_id,
            method=method,
            params=params,
            timestamp=datetime.now(),
            timeout=timeout
        )
        
        future = asyncio.Future()
        self.response_futures[request_id] = future
        self.pending_requests[request_id] = request
        
        try:
            message = MCPMessage(
                id=request_id,
                method=method,
                params=params
            )
            
            if not await self.transport.send_message(message):
                raise MCPClientError("Failed to send request")
            
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            raise MCPClientError(f"Request timeout after {timeout}s")
        finally:
            if request_id in self.response_futures:
                del self.response_futures[request_id]
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
    
    async def _send_notification(self, method: str, params: Dict[str, Any]):
        """
        Send a JSON-RPC notification (no response expected).
        
        Args:
            method: Notification method
            params: Method parameters
        """
        message = MCPMessage(
            method=method,
            params=params
        )
        
        await self.transport.send_message(message)
    
    def _handle_message(self, message: MCPMessage):
        """
        Handle incoming messages from the transport.
        
        Args:
            message: Received MCP message
        """
        try:
            if message.id is not None:
                self._handle_response(message)
            else:
                self._handle_notification(message)
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _handle_response(self, message: MCPMessage):
        """Handle response messages."""
        request_id = message.id
        
        if request_id in self.response_futures:
            future = self.response_futures[request_id]
            
            if not future.done():
                response = MCPResponse(
                    id=request_id,
                    result=message.result,
                    error=message.error,
                    timestamp=datetime.now()
                )
                future.set_result(response)
        else:
            logger.warning(f"Received response for unknown request ID: {request_id}")
    
    def _handle_notification(self, message: MCPMessage):
        """Handle notification messages."""
        method = message.method
        
        if method in self.notification_handlers:
            try:
                self.notification_handlers[method](message.params)
            except Exception as e:
                logger.error(f"Error in notification handler for {method}: {e}")
        else:
            logger.debug(f"No handler for notification: {method}")
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        self._request_id_counter += 1
        return f"{self.server_name}-{self._request_id_counter}-{uuid.uuid4().hex[:8]}"
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status information.
        
        Returns:
            Connection status details
        """
        return {
            "server_name": self.server_name,
            "connected": self.connected,
            "transport_connected": self.transport.connected,
            "pending_requests": len(self.pending_requests),
            "notification_handlers": list(self.notification_handlers.keys())
        }


class MCPClientManager:
    """
    Manager for multiple MCP clients.
    
    Provides centralized management of connections to multiple MCP servers
    with load balancing and failover capabilities.
    """
    
    def __init__(self, registry: Optional[MCPRegistry] = None):
        """
        Initialize MCP client manager.
        
        Args:
            registry: Optional registry for server discovery
        """
        self.registry = registry
        self.clients: Dict[str, MCPClient] = {}
        self.connection_pool: Dict[str, List[MCPClient]] = {}
    
    async def add_client(
        self,
        server_name: str,
        transport_config: Dict[str, Any],
        auto_connect: bool = True
    ) -> bool:
        """
        Add a new MCP client.
        
        Args:
            server_name: Name of the MCP server
            transport_config: Transport configuration
            auto_connect: Whether to connect immediately
            
        Returns:
            True if client added successfully
        """
        try:
            transport = TransportFactory.create_transport(**transport_config)
            
            client = MCPClient(
                server_name=server_name,
                transport=transport,
                registry=self.registry
            )
            
            self.clients[server_name] = client
            
            if auto_connect:
                await client.connect()
            
            logger.info(f"Added MCP client for {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add client for {server_name}: {e}")
            return False
    
    async def remove_client(self, server_name: str) -> bool:
        """
        Remove an MCP client.
        
        Args:
            server_name: Name of the server
            
        Returns:
            True if client removed successfully
        """
        if server_name in self.clients:
            client = self.clients[server_name]
            await client.disconnect()
            del self.clients[server_name]
            
            logger.info(f"Removed MCP client for {server_name}")
            return True
        
        return False
    
    async def get_client(self, server_name: str) -> Optional[MCPClient]:
        """
        Get an MCP client by server name.
        
        Args:
            server_name: Name of the server
            
        Returns:
            MCP client or None if not found
        """
        return self.clients.get(server_name)
    
    async def call_tool_on_any_server(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        preferred_servers: Optional[List[str]] = None
    ) -> Any:
        """
        Call a tool on any available server that supports it.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            preferred_servers: Preferred server order
            
        Returns:
            Tool execution result
            
        Raises:
            MCPClientError: If no server can execute the tool
        """
        candidate_servers = []
        
        if self.registry:
            tools = await self.registry.find_tools_by_capability(tool_name)
            candidate_servers = [tool.server_name for tool in tools]
        
        if preferred_servers:
            ordered_candidates = []
            for preferred in preferred_servers:
                if preferred in candidate_servers:
                    ordered_candidates.append(preferred)
            for candidate in candidate_servers:
                if candidate not in ordered_candidates:
                    ordered_candidates.append(candidate)
            candidate_servers = ordered_candidates
        
        last_error = None
        for server_name in candidate_servers:
            if server_name in self.clients:
                client = self.clients[server_name]
                if client.connected:
                    try:
                        result = await client.call_tool(tool_name, arguments)
                        return result
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Tool call failed on {server_name}: {e}")
                        continue
        
        if last_error:
            raise MCPClientError(f"No server could execute tool '{tool_name}': {last_error}")
        else:
            raise MCPClientError(f"No available server supports tool '{tool_name}'")
    
    async def disconnect_all(self):
        """Disconnect all clients."""
        disconnect_tasks = [
            client.disconnect() for client in self.clients.values()
        ]
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        self.clients.clear()
        logger.info("Disconnected all MCP clients")
    
    async def get_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available tools from all connected servers.
        
        Returns:
            Dictionary mapping server names to their tool lists
        """
        all_tools = {}
        
        for server_name, client in self.clients.items():
            if client.connected:
                try:
                    tools = await client.list_tools()
                    all_tools[server_name] = tools
                except Exception as e:
                    logger.error(f"Failed to list tools from {server_name}: {e}")
                    all_tools[server_name] = []
        
        return all_tools
    
    async def get_all_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all available resources from all connected servers.
        
        Returns:
            Dictionary mapping server names to their resource lists
        """
        all_resources = {}
        
        for server_name, client in self.clients.items():
            if client.connected:
                try:
                    resources = await client.list_resources()
                    all_resources[server_name] = resources
                except Exception as e:
                    logger.error(f"Failed to list resources from {server_name}: {e}")
                    all_resources[server_name] = []
        
        return all_resources
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health check on all clients.
        
        Returns:
            Dictionary mapping server names to their health status
        """
        health_status = {}
        
        for server_name, client in self.clients.items():
            try:
                is_healthy = await client.ping()
                health_status[server_name] = is_healthy
            except Exception as e:
                logger.error(f"Health check failed for {server_name}: {e}")
                health_status[server_name] = False
        
        return health_status
    
    def get_manager_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the client manager.
        
        Returns:
            Manager status information
        """
        connected_clients = sum(1 for client in self.clients.values() if client.connected)
        
        client_details = {}
        for server_name, client in self.clients.items():
            client_details[server_name] = client.get_connection_status()
        
        return {
            "total_clients": len(self.clients),
            "connected_clients": connected_clients,
            "client_details": client_details,
            "registry_available": self.registry is not None
        }
