"""
MCP Transport layer supporting stdio, SSE, and streamable HTTP.

This module implements the latest MCP protocol transport mechanisms
including the new streamable HTTP transport and enhanced SSE support.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Callable, AsyncIterator, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import aiohttp
from aiohttp import web, WSMsgType
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class MCPMessage:
    """Represents an MCP protocol message."""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        data = {"jsonrpc": self.jsonrpc}
        
        if self.id is not None:
            data["id"] = self.id
        if self.method is not None:
            data["method"] = self.method
        if self.params is not None:
            data["params"] = self.params
        if self.result is not None:
            data["result"] = self.result
        if self.error is not None:
            data["error"] = self.error
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPMessage":
        """Create message from dictionary."""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            id=data.get("id"),
            method=data.get("method"),
            params=data.get("params"),
            result=data.get("result"),
            error=data.get("error")
        )


class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations."""
    
    def __init__(self):
        """Initialize transport."""
        self.message_handler: Optional[Callable[[MCPMessage], None]] = None
        self.connected = False
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the transport."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the transport."""
        pass
    
    @abstractmethod
    async def send_message(self, message: MCPMessage) -> bool:
        """Send a message through the transport."""
        pass
    
    def set_message_handler(self, handler: Callable[[MCPMessage], None]):
        """Set the message handler for incoming messages."""
        self.message_handler = handler


class StdioTransport(MCPTransport):
    """
    Standard I/O transport for MCP servers.
    
    Communicates with MCP servers through stdin/stdout pipes.
    """
    
    def __init__(self, command: list[str], working_dir: Optional[str] = None, env: Optional[Dict[str, str]] = None):
        """
        Initialize stdio transport.
        
        Args:
            command: Command to start the MCP server
            working_dir: Working directory for the server process
            env: Environment variables for the server process
        """
        super().__init__()
        self.command = command
        self.working_dir = working_dir
        self.env = env
        self.process: Optional[subprocess.Popen] = None
        self.read_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> bool:
        """Start the stdio transport and server process."""
        try:
            logger.info(f"Starting stdio transport with command: {' '.join(self.command)}")
            
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
                env=self.env
            )
            
            self.read_task = asyncio.create_task(self._read_messages())
            self.connected = True
            
            logger.info("Stdio transport started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stdio transport: {e}")
            return False
    
    async def stop(self):
        """Stop the stdio transport and server process."""
        logger.info("Stopping stdio transport")
        
        self._shutdown_event.set()
        self.connected = False
        
        if self.read_task:
            self.read_task.cancel()
            try:
                await self.read_task
            except asyncio.CancelledError:
                pass
        
        if self.process:
            self.process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process()),
                    timeout=5
                )
            except asyncio.TimeoutError:
                logger.warning("Force killing stdio process")
                self.process.kill()
                await asyncio.create_task(self._wait_for_process())
            
            self.process = None
        
        logger.info("Stdio transport stopped")
    
    async def send_message(self, message: MCPMessage) -> bool:
        """Send a message to the server via stdin."""
        if not self.connected or not self.process or not self.process.stdin:
            logger.error("Cannot send message: transport not connected")
            return False
        
        try:
            message_json = json.dumps(message.to_dict())
            self.process.stdin.write(message_json + "\n")
            self.process.stdin.flush()
            
            logger.debug(f"Sent message: {message_json}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def _read_messages(self):
        """Read messages from server stdout."""
        if not self.process or not self.process.stdout:
            return
        
        try:
            while not self._shutdown_event.is_set() and self.process.poll() is None:
                line = await asyncio.create_task(self._read_line())
                if line:
                    try:
                        message_data = json.loads(line.strip())
                        message = MCPMessage.from_dict(message_data)
                        
                        if self.message_handler:
                            self.message_handler(message)
                        
                        logger.debug(f"Received message: {line.strip()}")
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON message: {e}")
                        logger.debug(f"Raw message: {line}")
                
        except Exception as e:
            logger.error(f"Error reading messages: {e}")
        finally:
            self.connected = False
    
    async def _read_line(self) -> Optional[str]:
        """Read a line from stdout asynchronously."""
        if not self.process or not self.process.stdout:
            return None
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process.stdout.readline)
    
    async def _wait_for_process(self):
        """Wait for process to terminate."""
        if self.process:
            while self.process.poll() is None:
                await asyncio.sleep(0.1)


class SSETransport(MCPTransport):
    """
    Server-Sent Events transport for MCP servers.
    
    Provides HTTP-based communication using SSE for server-to-client
    messages and regular HTTP POST for client-to-server messages.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080, path: str = "/mcp"):
        """
        Initialize SSE transport.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            path: Base path for MCP endpoints
        """
        super().__init__()
        self.host = host
        self.port = port
        self.path = path
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.clients: Dict[str, web.StreamResponse] = {}
    
    async def start(self) -> bool:
        """Start the SSE transport server."""
        try:
            logger.info(f"Starting SSE transport on {self.host}:{self.port}")
            
            self.app = web.Application()
            self.app.router.add_get(f"{self.path}/events", self._handle_sse)
            self.app.router.add_post(f"{self.path}/messages", self._handle_message)
            self.app.router.add_options(f"{self.path}/messages", self._handle_options)
            
            self.app.middlewares.append(self._cors_middleware)
            
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            
            self.connected = True
            logger.info(f"SSE transport started on http://{self.host}:{self.port}{self.path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start SSE transport: {e}")
            return False
    
    async def stop(self):
        """Stop the SSE transport server."""
        logger.info("Stopping SSE transport")
        
        self.connected = False
        
        for client_id, response in self.clients.items():
            try:
                await response.write_eof()
            except Exception as e:
                logger.debug(f"Error closing client {client_id}: {e}")
        
        self.clients.clear()
        
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("SSE transport stopped")
    
    async def send_message(self, message: MCPMessage) -> bool:
        """Send a message to all connected clients via SSE."""
        if not self.connected:
            logger.error("Cannot send message: transport not connected")
            return False
        
        try:
            message_json = json.dumps(message.to_dict())
            data = f"data: {message_json}\n\n"
            
            disconnected_clients = []
            for client_id, response in self.clients.items():
                try:
                    await response.write(data.encode())
                except Exception as e:
                    logger.debug(f"Failed to send to client {client_id}: {e}")
                    disconnected_clients.append(client_id)
            
            for client_id in disconnected_clients:
                del self.clients[client_id]
            
            logger.debug(f"Sent SSE message to {len(self.clients)} clients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send SSE message: {e}")
            return False
    
    async def _handle_sse(self, request: web.Request) -> web.StreamResponse:
        """Handle SSE connection requests."""
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'MCP-Protocol-Version': '2025-06-18'
            }
        )
        
        await response.prepare(request)
        
        client_id = f"client_{datetime.now().timestamp()}"
        self.clients[client_id] = response
        
        logger.info(f"SSE client connected: {client_id}")
        
        welcome_message = MCPMessage(
            method="connection/established",
            params={"client_id": client_id, "protocol_version": "2025-06-18"}
        )
        await self._send_sse_message(response, welcome_message)
        
        try:
            while True:
                await asyncio.sleep(30)  # Send keepalive every 30 seconds
                await response.write(b"data: {\"type\": \"keepalive\"}\n\n")
        except Exception as e:
            logger.debug(f"SSE client {client_id} disconnected: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
        
        return response
    
    async def _handle_message(self, request: web.Request) -> web.Response:
        """Handle incoming messages from clients."""
        try:
            data = await request.json()
            message = MCPMessage.from_dict(data)
            
            if self.message_handler:
                self.message_handler(message)
            
            return web.json_response({"status": "received"})
            
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
            return web.json_response({"error": str(e)}, status=400)
    
    async def _handle_options(self, request: web.Request) -> web.Response:
        """Handle CORS preflight requests."""
        return web.Response(
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, MCP-Protocol-Version'
            }
        )
    
    async def _cors_middleware(self, request: web.Request, handler):
        """CORS middleware for all requests."""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    
    async def _send_sse_message(self, response: web.StreamResponse, message: MCPMessage):
        """Send a single SSE message to a specific response."""
        message_json = json.dumps(message.to_dict())
        data = f"data: {message_json}\n\n"
        await response.write(data.encode())


class StreamableHTTPTransport(MCPTransport):
    """
    Streamable HTTP transport for MCP servers (latest protocol feature).
    
    Implements the new streamable HTTP transport mechanism introduced
    in MCP protocol version 2025-06-18.
    """
    
    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        """
        Initialize streamable HTTP transport.
        
        Args:
            base_url: Base URL of the MCP server
            auth_token: Optional authentication token
        """
        super().__init__()
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.session: Optional[aiohttp.ClientSession] = None
        self.stream_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self) -> bool:
        """Start the streamable HTTP transport."""
        try:
            logger.info(f"Starting streamable HTTP transport to {self.base_url}")
            
            headers = {
                'MCP-Protocol-Version': '2025-06-18',
                'Content-Type': 'application/json'
            }
            
            if self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'
            
            self.session = aiohttp.ClientSession(headers=headers)
            
            self.stream_task = asyncio.create_task(self._stream_messages())
            self.connected = True
            
            logger.info("Streamable HTTP transport started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streamable HTTP transport: {e}")
            return False
    
    async def stop(self):
        """Stop the streamable HTTP transport."""
        logger.info("Stopping streamable HTTP transport")
        
        self._shutdown_event.set()
        self.connected = False
        
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Streamable HTTP transport stopped")
    
    async def send_message(self, message: MCPMessage) -> bool:
        """Send a message via HTTP POST."""
        if not self.connected or not self.session:
            logger.error("Cannot send message: transport not connected")
            return False
        
        try:
            url = f"{self.base_url}/mcp/messages"
            data = message.to_dict()
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    logger.debug(f"Sent HTTP message: {json.dumps(data)}")
                    return True
                else:
                    logger.error(f"HTTP message failed with status {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send HTTP message: {e}")
            return False
    
    async def _stream_messages(self):
        """Stream messages from the server."""
        if not self.session:
            return
        
        try:
            url = f"{self.base_url}/mcp/stream"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Stream connection failed with status {response.status}")
                    return
                
                logger.info("Streamable HTTP connection established")
                
                async for line in response.content:
                    if self._shutdown_event.is_set():
                        break
                    
                    try:
                        line_str = line.decode().strip()
                        if line_str:
                            message_data = json.loads(line_str)
                            message = MCPMessage.from_dict(message_data)
                            
                            if self.message_handler:
                                self.message_handler(message)
                            
                            logger.debug(f"Received streamed message: {line_str}")
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse streamed message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing streamed message: {e}")
                        
        except Exception as e:
            logger.error(f"Stream connection error: {e}")
        finally:
            self.connected = False


class TransportFactory:
    """Factory for creating MCP transport instances."""
    
    @staticmethod
    def create_transport(
        transport_type: str,
        **kwargs
    ) -> MCPTransport:
        """
        Create a transport instance.
        
        Args:
            transport_type: Type of transport ("stdio", "sse", "http")
            **kwargs: Transport-specific configuration
            
        Returns:
            Transport instance
            
        Raises:
            ValueError: If transport type is not supported
        """
        if transport_type.lower() == "stdio":
            return StdioTransport(
                command=kwargs.get("command", []),
                working_dir=kwargs.get("working_dir"),
                env=kwargs.get("env")
            )
        elif transport_type.lower() == "sse":
            return SSETransport(
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 8080),
                path=kwargs.get("path", "/mcp")
            )
        elif transport_type.lower() in ["http", "streamable_http"]:
            return StreamableHTTPTransport(
                base_url=kwargs["base_url"],
                auth_token=kwargs.get("auth_token")
            )
        else:
            raise ValueError(f"Unsupported transport type: {transport_type}")
