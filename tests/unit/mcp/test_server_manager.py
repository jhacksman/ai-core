"""
Unit tests for MCP Server Manager.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.mcp.server_manager import MCPServerManager, ServerInstance, ServerStatus
from src.mcp.transport import MCPTransport


class TestMCPServerManager:
    """Test cases for MCPServerManager."""
    
    @pytest.fixture
    def server_manager(self):
        """Create MCPServerManager instance for testing."""
        config = {
            "max_concurrent_servers": 10,
            "server_timeout": 30,
            "health_check_interval": 60
        }
        return MCPServerManager(config=config)
    
    @pytest.mark.asyncio
    async def test_server_manager_initialization(self, server_manager):
        """Test server manager initializes correctly."""
        await server_manager.initialize()
        
        assert server_manager.max_concurrent_servers == 10
        assert server_manager.server_timeout == 30
        assert server_manager.health_check_interval == 60
        assert len(server_manager.servers) == 0
    
    @pytest.mark.asyncio
    async def test_start_server_success(self, server_manager):
        """Test successful server startup."""
        await server_manager.initialize()
        
        server_config = {
            "module": "test_module",
            "class": "TestServer",
            "transport": "stdio",
            "timeout": 30
        }
        
        with patch('src.mcp.server_manager.MCPServerManager._create_server_instance') as mock_create:
            mock_instance = Mock(spec=ServerInstance)
            mock_instance.server_id = "test-server"
            mock_instance.status = ServerStatus.RUNNING
            mock_instance.start = AsyncMock()
            mock_create.return_value = mock_instance
            
            result = await server_manager.start_server("test-server", server_config)
            
            assert result["success"] is True
            assert result["server_id"] == "test-server"
            assert "test-server" in server_manager.servers
            mock_instance.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_server_already_exists(self, server_manager):
        """Test starting server that already exists."""
        await server_manager.initialize()
        
        existing_instance = Mock(spec=ServerInstance)
        existing_instance.server_id = "test-server"
        existing_instance.status = ServerStatus.RUNNING
        server_manager.servers["test-server"] = existing_instance
        
        server_config = {"module": "test_module", "class": "TestServer"}
        
        result = await server_manager.start_server("test-server", server_config)
        
        assert result["success"] is False
        assert "already exists" in result["error"]
    
    @pytest.mark.asyncio
    async def test_stop_server_success(self, server_manager):
        """Test successful server shutdown."""
        await server_manager.initialize()
        
        mock_instance = Mock(spec=ServerInstance)
        mock_instance.server_id = "test-server"
        mock_instance.status = ServerStatus.RUNNING
        mock_instance.stop = AsyncMock()
        server_manager.servers["test-server"] = mock_instance
        
        result = await server_manager.stop_server("test-server")
        
        assert result["success"] is True
        mock_instance.stop.assert_called_once()
        assert "test-server" not in server_manager.servers
    
    @pytest.mark.asyncio
    async def test_stop_server_not_found(self, server_manager):
        """Test stopping non-existent server."""
        await server_manager.initialize()
        
        result = await server_manager.stop_server("non-existent")
        
        assert result["success"] is False
        assert "not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_get_server_status(self, server_manager):
        """Test getting server status."""
        await server_manager.initialize()
        
        mock_instance = Mock(spec=ServerInstance)
        mock_instance.server_id = "test-server"
        mock_instance.status = ServerStatus.RUNNING
        mock_instance.get_status.return_value = {
            "status": "running",
            "uptime": 3600,
            "tools_count": 5
        }
        server_manager.servers["test-server"] = mock_instance
        
        status = await server_manager.get_server_status("test-server")
        
        assert status["status"] == "running"
        assert status["uptime"] == 3600
        assert status["tools_count"] == 5
    
    @pytest.mark.asyncio
    async def test_list_servers(self, server_manager):
        """Test listing all servers."""
        await server_manager.initialize()
        
        for i in range(3):
            mock_instance = Mock(spec=ServerInstance)
            mock_instance.server_id = f"test-server-{i}"
            mock_instance.status = ServerStatus.RUNNING
            mock_instance.get_status.return_value = {"status": "running"}
            server_manager.servers[f"test-server-{i}"] = mock_instance
        
        servers = await server_manager.list_servers()
        
        assert len(servers) == 3
        assert all(f"test-server-{i}" in servers for i in range(3))
    
    @pytest.mark.asyncio
    async def test_health_check(self, server_manager):
        """Test server health checking."""
        await server_manager.initialize()
        
        healthy_server = Mock(spec=ServerInstance)
        healthy_server.server_id = "healthy-server"
        healthy_server.status = ServerStatus.RUNNING
        healthy_server.health_check = AsyncMock(return_value=True)
        
        unhealthy_server = Mock(spec=ServerInstance)
        unhealthy_server.server_id = "unhealthy-server"
        unhealthy_server.status = ServerStatus.RUNNING
        unhealthy_server.health_check = AsyncMock(return_value=False)
        unhealthy_server.stop = AsyncMock()
        
        server_manager.servers["healthy-server"] = healthy_server
        server_manager.servers["unhealthy-server"] = unhealthy_server
        
        await server_manager._perform_health_checks()
        
        healthy_server.health_check.assert_called_once()
        unhealthy_server.health_check.assert_called_once()
        unhealthy_server.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_max_concurrent_servers_limit(self, server_manager):
        """Test maximum concurrent servers limit."""
        server_manager.max_concurrent_servers = 2
        await server_manager.initialize()
        
        for i in range(2):
            mock_instance = Mock(spec=ServerInstance)
            mock_instance.server_id = f"test-server-{i}"
            mock_instance.status = ServerStatus.RUNNING
            server_manager.servers[f"test-server-{i}"] = mock_instance
        
        server_config = {"module": "test_module", "class": "TestServer"}
        result = await server_manager.start_server("test-server-3", server_config)
        
        assert result["success"] is False
        assert "maximum number" in result["error"]
    
    @pytest.mark.asyncio
    async def test_cleanup(self, server_manager):
        """Test cleanup functionality."""
        await server_manager.initialize()
        
        for i in range(3):
            mock_instance = Mock(spec=ServerInstance)
            mock_instance.server_id = f"test-server-{i}"
            mock_instance.status = ServerStatus.RUNNING
            mock_instance.stop = AsyncMock()
            server_manager.servers[f"test-server-{i}"] = mock_instance
        
        await server_manager.cleanup()
        
        for i in range(3):
            server_manager.servers[f"test-server-{i}"].stop.assert_called_once()
        
        assert len(server_manager.servers) == 0


class TestServerInstance:
    """Test cases for ServerInstance."""
    
    @pytest.fixture
    def mock_transport(self):
        """Mock transport for testing."""
        transport = Mock(spec=MCPTransport)
        transport.start = AsyncMock()
        transport.stop = AsyncMock()
        transport.is_connected = Mock(return_value=True)
        transport.call_tool = AsyncMock()
        return transport
    
    @pytest.fixture
    def server_instance(self, mock_transport):
        """Create ServerInstance for testing."""
        config = {
            "module": "test_module",
            "class": "TestServer",
            "transport": "stdio",
            "timeout": 30
        }
        return ServerInstance(
            server_id="test-server",
            config=config,
            transport=mock_transport
        )
    
    @pytest.mark.asyncio
    async def test_server_instance_start(self, server_instance, mock_transport):
        """Test server instance startup."""
        await server_instance.start()
        
        assert server_instance.status == ServerStatus.RUNNING
        assert server_instance.start_time is not None
        mock_transport.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_server_instance_stop(self, server_instance, mock_transport):
        """Test server instance shutdown."""
        await server_instance.start()
        
        await server_instance.stop()
        
        assert server_instance.status == ServerStatus.STOPPED
        mock_transport.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_server_instance_health_check(self, server_instance, mock_transport):
        """Test server instance health check."""
        await server_instance.start()
        
        mock_transport.is_connected.return_value = True
        
        is_healthy = await server_instance.health_check()
        
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_server_instance_call_tool(self, server_instance, mock_transport):
        """Test calling tool through server instance."""
        await server_instance.start()
        
        mock_transport.call_tool.return_value = {"result": "success"}
        
        result = await server_instance.call_tool("test_tool", {"arg": "value"})
        
        assert result == {"result": "success"}
        mock_transport.call_tool.assert_called_once_with("test_tool", {"arg": "value"})
    
    def test_server_instance_get_status(self, server_instance):
        """Test getting server instance status."""
        status = server_instance.get_status()
        
        assert status["server_id"] == "test-server"
        assert status["status"] == "stopped"  # Initial state
        assert "uptime" in status
        assert "tools_count" in status
