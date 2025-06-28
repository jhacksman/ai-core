"""
Unit tests for Meta MCP Server.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.mcp.meta_server import MetaMCPServer
from src.venice.client import VeniceClient
from src.memory.long_term_memory import LongTermMemory


class TestMetaMCPServer:
    """Test cases for MetaMCPServer."""
    
    @pytest.fixture
    def mock_venice_client(self):
        """Mock Venice.ai client for testing."""
        client = Mock(spec=VeniceClient)
        client.structured_completion = AsyncMock()
        client.simple_completion = AsyncMock()
        client.analyze_with_context = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_long_term_memory(self):
        """Mock long-term memory for testing."""
        memory = Mock(spec=LongTermMemory)
        memory.store_memory = AsyncMock()
        memory.retrieve_memories = AsyncMock()
        return memory
    
    @pytest.fixture
    def meta_server(self, mock_venice_client, mock_long_term_memory):
        """Create MetaMCPServer instance for testing."""
        config = {
            "server_output_dir": "/tmp/test_servers",
            "max_history_size": 100,
            "research_depth": "medium"
        }
        return MetaMCPServer(
            venice_client=mock_venice_client,
            long_term_memory=mock_long_term_memory,
            config=config
        )
    
    @pytest.mark.asyncio
    async def test_meta_server_initialization(self, meta_server):
        """Test meta server initializes correctly."""
        await meta_server.initialize()
        
        assert meta_server.server_output_dir == "/tmp/test_servers"
        assert meta_server.max_history_size == 100
        assert meta_server.research_depth == "medium"
        assert len(meta_server.generation_history) == 0
    
    @pytest.mark.asyncio
    async def test_discover_api_tool(self, meta_server, mock_venice_client):
        """Test discover_api tool functionality."""
        await meta_server.initialize()
        
        with patch('src.mcp.meta_server.MetaMCPServer._discover_with_openapi_explorer') as mock_discover:
            mock_discover.return_value = {
                "openapi": "3.0.0",
                "info": {"title": "Test API", "version": "1.0"},
                "paths": {
                    "/users": {"get": {"summary": "Get users"}},
                    "/posts": {"get": {"summary": "Get posts"}}
                }
            }
            
            mock_venice_client.structured_completion.return_value = {
                "api_name": "Test API",
                "endpoints": [
                    {"path": "/users", "method": "GET", "description": "Retrieve user list"},
                    {"path": "/posts", "method": "GET", "description": "Retrieve posts"}
                ],
                "authentication": "Bearer token",
                "rate_limits": "100 requests/minute"
            }
            
            result = await meta_server.discover_api(
                api_url="https://api.test.com/v1",
                api_name="Test API"
            )
            
            assert result["api_name"] == "Test API"
            assert len(result["endpoints"]) == 2
            assert result["authentication"] == "Bearer token"
            assert result["rate_limits"] == "100 requests/minute"
    
    @pytest.mark.asyncio
    async def test_generate_mcp_server_tool(self, meta_server, mock_venice_client):
        """Test generate_mcp_server tool functionality."""
        await meta_server.initialize()
        
        mock_venice_client.structured_completion.return_value = {
            "server_code": """
from mcp.server.lowlevel import Server

class TestMCPServer:
    def __init__(self):
        self.server = Server("test-server")
    
    @self.server.call_tool()
    async def test_tool(self, args):
        return {"result": "success"}
""",
            "tools": ["test_tool", "another_tool"],
            "dependencies": ["mcp", "requests"],
            "configuration": {
                "api_key_required": True,
                "rate_limit": 100
            }
        }
        
        with patch('src.mcp.meta_server.MetaMCPServer._write_server_file') as mock_write:
            mock_write.return_value = "/tmp/test_servers/test_server.py"
            
            result = await meta_server.generate_mcp_server(
                purpose="Test API integration",
                api_url="https://api.test.com/v1",
                tools_needed=["test_tool", "another_tool"]
            )
            
            assert result["server_creation"]["tools"] == ["test_tool", "another_tool"]
            assert result["server_creation"]["file_path"] == "/tmp/test_servers/test_server.py"
            assert "dependencies" in result["server_creation"]
            assert "configuration" in result["server_creation"]
    
    @pytest.mark.asyncio
    async def test_research_problem_domain_tool(self, meta_server, mock_venice_client):
        """Test research_problem_domain tool functionality."""
        await meta_server.initialize()
        
        mock_venice_client.structured_completion.return_value = {
            "research_findings": [
                "Social media APIs provide sentiment analysis capabilities",
                "Twitter API v2 offers comprehensive data access",
                "Reddit API allows subreddit monitoring"
            ],
            "recommended_apis": [
                {
                    "name": "Twitter API",
                    "url": "https://api.twitter.com/2",
                    "relevance": 0.9
                },
                {
                    "name": "Reddit API", 
                    "url": "https://www.reddit.com/api",
                    "relevance": 0.8
                }
            ],
            "tools_needed": ["analyze_sentiment", "monitor_mentions", "extract_trends"],
            "complexity_assessment": "medium"
        }
        
        result = await meta_server.research_problem_domain(
            problem_description="Monitor social media sentiment about our brand",
            research_depth="deep"
        )
        
        assert len(result["research_findings"]) == 3
        assert len(result["recommended_apis"]) == 2
        assert result["recommended_apis"][0]["name"] == "Twitter API"
        assert result["complexity_assessment"] == "medium"
    
    @pytest.mark.asyncio
    async def test_install_mcp_server_tool(self, meta_server):
        """Test install_mcp_server tool functionality."""
        await meta_server.initialize()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Successfully installed test-server",
                stderr=""
            )
            
            result = await meta_server.install_mcp_server(
                server_name="test-server",
                source="github:user/test-server"
            )
            
            assert result["success"] is True
            assert result["server_name"] == "test-server"
            assert result["source"] == "github:user/test-server"
            assert "installation_output" in result
    
    @pytest.mark.asyncio
    async def test_list_available_servers_tool(self, meta_server):
        """Test list_available_servers tool functionality."""
        await meta_server.initialize()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="""
Available MCP Servers:
- slack-server: Slack integration
- discord-server: Discord integration  
- weather-server: Weather data API
""",
                stderr=""
            )
            
            result = await meta_server.list_available_servers()
            
            assert result["success"] is True
            assert "available_servers" in result
            assert len(result["available_servers"]) >= 3
    
    @pytest.mark.asyncio
    async def test_research_and_create_server(self, meta_server, mock_venice_client):
        """Test research_and_create_server method."""
        await meta_server.initialize()
        
        mock_venice_client.structured_completion.side_effect = [
            {
                "research_findings": ["Finding 1", "Finding 2"],
                "recommended_apis": [{"name": "Test API", "url": "https://api.test.com"}],
                "tools_needed": ["tool1", "tool2"]
            },
            {
                "server_code": "# Generated server code",
                "tools": ["tool1", "tool2"],
                "dependencies": ["mcp", "requests"]
            }
        ]
        
        with patch('src.mcp.meta_server.MetaMCPServer._write_server_file') as mock_write:
            mock_write.return_value = "/tmp/test_servers/research_server.py"
            
            result = await meta_server.research_and_create_server(
                problem_description="Test problem",
                research_depth="medium"
            )
            
            assert "research_findings" in result
            assert "server_creation" in result
            assert len(result["research_findings"]) == 2
            assert result["server_creation"]["tools"] == ["tool1", "tool2"]
    
    @pytest.mark.asyncio
    async def test_get_generation_history(self, meta_server):
        """Test get_generation_history method."""
        await meta_server.initialize()
        
        meta_server.generation_history.append({
            "timestamp": datetime.now(),
            "purpose": "Test server 1",
            "server_id": "test-1",
            "status": "success"
        })
        
        meta_server.generation_history.append({
            "timestamp": datetime.now(),
            "purpose": "Test server 2", 
            "server_id": "test-2",
            "status": "success"
        })
        
        history = meta_server.get_generation_history()
        
        assert len(history) == 2
        assert history[0]["purpose"] == "Test server 1"
        assert history[1]["purpose"] == "Test server 2"
    
    @pytest.mark.asyncio
    async def test_cleanup_old_servers(self, meta_server):
        """Test cleanup_old_servers method."""
        await meta_server.initialize()
        
        with patch('os.listdir') as mock_listdir, \
             patch('os.path.getmtime') as mock_getmtime, \
             patch('os.remove') as mock_remove:
            
            mock_listdir.return_value = ["old_server.py", "new_server.py"]
            mock_getmtime.side_effect = [
                datetime.now().timestamp() - 86400 * 8,  # 8 days old
                datetime.now().timestamp() - 3600        # 1 hour old
            ]
            
            cleaned_count = await meta_server.cleanup_old_servers(max_age_days=7)
            
            assert cleaned_count == 1
            mock_remove.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_in_api_discovery(self, meta_server, mock_venice_client):
        """Test error handling in API discovery."""
        await meta_server.initialize()
        
        with patch('src.mcp.meta_server.MetaMCPServer._discover_with_openapi_explorer') as mock_discover:
            mock_discover.side_effect = Exception("Discovery failed")
            
            result = await meta_server.discover_api(
                api_url="https://invalid-api.com",
                api_name="Invalid API"
            )
            
            assert result["success"] is False
            assert "error" in result
            assert "Discovery failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_server_generation_with_memory_integration(self, meta_server, mock_long_term_memory):
        """Test server generation with memory integration."""
        await meta_server.initialize()
        
        mock_long_term_memory.retrieve_memories.return_value = [
            {
                "content": "Previous API integration experience",
                "metadata": {"api_type": "REST", "success": True}
            }
        ]
        
        meta_server.venice_client.structured_completion.return_value = {
            "server_code": "# Generated with memory context",
            "tools": ["enhanced_tool"],
            "dependencies": ["mcp"]
        }
        
        with patch('src.mcp.meta_server.MetaMCPServer._write_server_file') as mock_write:
            mock_write.return_value = "/tmp/test_servers/memory_enhanced_server.py"
            
            result = await meta_server.generate_mcp_server(
                purpose="Enhanced API integration",
                use_memory_context=True
            )
            
            mock_long_term_memory.retrieve_memories.assert_called_once()
            
            assert result["server_creation"]["tools"] == ["enhanced_tool"]
            assert result["server_creation"]["file_path"] == "/tmp/test_servers/memory_enhanced_server.py"
