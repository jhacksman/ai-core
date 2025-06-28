"""
Integration tests for MCP framework components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.mcp.integration import MCPIntegrationManager
from src.mcp.meta_server import MetaMCPServer
from src.venice.client import VeniceClient
from src.memory.long_term_memory import LongTermMemory
from src.agent.manager import AgentManager


class TestMCPIntegration:
    """Integration tests for MCP framework."""
    
    @pytest.fixture
    def mock_venice_client(self):
        """Mock Venice.ai client for testing."""
        client = Mock(spec=VeniceClient)
        client.simple_completion = AsyncMock()
        client.structured_completion = AsyncMock()
        client.analyze_with_context = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_long_term_memory(self):
        """Mock long-term memory for testing."""
        memory = Mock(spec=LongTermMemory)
        memory.store_memory = AsyncMock()
        memory.retrieve_memories = AsyncMock()
        memory.initialize = AsyncMock()
        return memory
    
    @pytest.fixture
    def mock_agent_manager(self):
        """Mock agent manager for testing."""
        manager = Mock(spec=AgentManager)
        manager.submit_task = AsyncMock()
        manager.get_task_status = AsyncMock()
        manager.initialize = AsyncMock()
        return manager
    
    @pytest.fixture
    def integration_manager(self, mock_venice_client, mock_long_term_memory, mock_agent_manager):
        """Create MCPIntegrationManager for testing."""
        config = {
            "server_manager": {"max_concurrent_servers": 10},
            "registry": {"storage_path": "/tmp/test_registry"},
            "meta_server": {"research_depth": "medium"}
        }
        return MCPIntegrationManager(
            venice_client=mock_venice_client,
            long_term_memory=mock_long_term_memory,
            agent_manager=mock_agent_manager,
            config=config
        )
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialization(self, integration_manager):
        """Test integration manager initializes all components."""
        with patch.multiple(
            integration_manager,
            server_manager=Mock(initialize=AsyncMock()),
            registry=Mock(initialize=AsyncMock()),
            client_manager=Mock(initialize=AsyncMock()),
            server_factory=Mock(initialize=AsyncMock()),
            template_engine=Mock(initialize=AsyncMock()),
            capability_analyzer=Mock(initialize=AsyncMock()),
            meta_server=Mock(initialize=AsyncMock())
        ):
            await integration_manager.initialize()
            
            integration_manager.server_manager.initialize.assert_called_once()
            integration_manager.registry.initialize.assert_called_once()
            integration_manager.client_manager.initialize.assert_called_once()
            integration_manager.meta_server.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_foundational_servers(self, integration_manager):
        """Test starting foundational MCP servers."""
        with patch.object(integration_manager.server_manager, 'start_server') as mock_start:
            mock_start.return_value = {"success": True, "server_id": "test-server"}
            
            with patch.object(integration_manager, '_get_server_tools') as mock_tools:
                mock_tools.return_value = ["tool1", "tool2", "tool3"]
                
                result = await integration_manager.start_foundational_servers(["slack", "discord"])
                
                assert result["started_servers"] == 2
                assert result["failed_servers"] == 0
                assert mock_start.call_count == 2
    
    @pytest.mark.asyncio
    async def test_create_dynamic_server_with_api(self, integration_manager):
        """Test creating dynamic server with API integration."""
        with patch.object(integration_manager.meta_server, 'generate_mcp_server') as mock_generate:
            mock_generate.return_value = {
                "server_creation": {
                    "server_id": "dynamic-123",
                    "server_name": "weather_api_server",
                    "tools": ["get_weather", "get_forecast"],
                    "file_path": "/tmp/weather_server.py"
                }
            }
            
            result = await integration_manager.create_dynamic_server(
                purpose="Weather data integration",
                api_url="https://api.weather.com/v1"
            )
            
            assert result["server_id"] == "dynamic-123"
            assert result["server_name"] == "weather_api_server"
            assert result["tools_count"] == 2
            assert result["status"] == "created"
            
            assert "dynamic-123" in integration_manager.server_instances
    
    @pytest.mark.asyncio
    async def test_create_dynamic_server_research_based(self, integration_manager):
        """Test creating dynamic server through research."""
        with patch.object(integration_manager.meta_server, 'research_and_create_server') as mock_research:
            mock_research.return_value = {
                "server_id": "research-456",
                "server_name": "social_media_analyzer",
                "tools": ["analyze_sentiment", "extract_trends", "generate_report"],
                "file_path": "/tmp/social_analyzer.py"
            }
            
            result = await integration_manager.create_dynamic_server(
                purpose="Social media sentiment analysis",
                research_depth="deep"
            )
            
            assert result["server_id"] == "research-456"
            assert result["server_name"] == "social_media_analyzer"
            assert result["tools_count"] == 3
            
            mock_research.assert_called_once_with(
                problem_description="Social media sentiment analysis",
                research_depth="deep"
            )
    
    @pytest.mark.asyncio
    async def test_execute_tool_across_servers(self, integration_manager):
        """Test executing tools across multiple servers."""
        from src.mcp.integration import IntegratedServerInstance
        
        server1 = IntegratedServerInstance(
            server_id="server1",
            server_type="foundational",
            name="slack",
            status="running",
            tools=["send_message", "get_history"],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata={}
        )
        
        server2 = IntegratedServerInstance(
            server_id="server2",
            server_type="generated",
            name="weather",
            status="running",
            tools=["get_weather", "send_message"],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata={}
        )
        
        integration_manager.server_instances["server1"] = server1
        integration_manager.server_instances["server2"] = server2
        
        with patch.object(integration_manager.client_manager, 'call_tool') as mock_call:
            mock_call.return_value = {"result": "Message sent successfully"}
            
            result = await integration_manager.execute_tool_across_servers(
                tool_name="send_message",
                arguments={"channel": "general", "message": "Hello"},
                preferred_server="server1"
            )
            
            assert result["server_used"] == "server1"
            assert result["tool_name"] == "send_message"
            assert result["result"]["result"] == "Message sent successfully"
            assert "server1" in result["available_servers"]
            assert "server2" in result["available_servers"]
    
    @pytest.mark.asyncio
    async def test_analyze_capability_gaps(self, integration_manager):
        """Test capability gap analysis."""
        integration_manager.server_instances = {
            "slack": Mock(tools=["send_message", "get_history"]),
            "discord": Mock(tools=["send_message", "manage_roles"])
        }
        
        with patch.object(integration_manager.capability_analyzer, 'analyze_capability_gaps') as mock_analyze:
            mock_analyze.return_value = {
                "missing_capabilities": ["file_upload", "image_processing"],
                "recommended_servers": ["file_manager", "image_processor"],
                "confidence": 0.85
            }
            
            result = await integration_manager.analyze_capability_gaps(
                "Need to process uploaded images and store them"
            )
            
            assert "missing_capabilities" in result
            assert "file_upload" in result["missing_capabilities"]
            assert "image_processing" in result["missing_capabilities"]
            assert result["confidence"] == 0.85
    
    @pytest.mark.asyncio
    async def test_server_status_monitoring(self, integration_manager):
        """Test server status monitoring."""
        from src.mcp.integration import IntegratedServerInstance
        
        foundational_server = IntegratedServerInstance(
            server_id="slack",
            server_type="foundational",
            name="slack_server",
            status="running",
            tools=["send_message"],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata={}
        )
        
        generated_server = IntegratedServerInstance(
            server_id="weather-123",
            server_type="generated",
            name="weather_server",
            status="running",
            tools=["get_weather"],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata={}
        )
        
        integration_manager.server_instances["slack"] = foundational_server
        integration_manager.server_instances["weather-123"] = generated_server
        
        with patch.object(integration_manager.registry, 'get_status') as mock_registry_status:
            mock_registry_status.return_value = {"registered_servers": 2}
            
            status = await integration_manager.get_server_status()
            
            assert "foundational_servers" in status
            assert "generated_servers" in status
            assert "statistics" in status
            assert "registry_status" in status
            
            assert "slack" in status["foundational_servers"]
            assert "weather-123" in status["generated_servers"]
    
    @pytest.mark.asyncio
    async def test_integration_manager_shutdown(self, integration_manager):
        """Test integration manager shutdown."""
        from src.mcp.integration import IntegratedServerInstance
        
        server = IntegratedServerInstance(
            server_id="test-server",
            server_type="foundational",
            name="test",
            status="running",
            tools=["test_tool"],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata={}
        )
        
        integration_manager.server_instances["test-server"] = server
        
        with patch.multiple(
            integration_manager,
            server_manager=Mock(stop_server=AsyncMock(), cleanup=AsyncMock()),
            meta_server=Mock(cleanup=AsyncMock()),
            client_manager=Mock(cleanup=AsyncMock())
        ):
            await integration_manager.shutdown()
            
            integration_manager.server_manager.stop_server.assert_called_once_with("test-server")
            integration_manager.meta_server.cleanup.assert_called_once()
            integration_manager.client_manager.cleanup.assert_called_once()
            
            assert len(integration_manager.server_instances) == 0


class TestMetaServerIntegration:
    """Integration tests for meta-server capabilities."""
    
    @pytest.fixture
    def mock_venice_client(self):
        """Mock Venice.ai client for meta-server testing."""
        client = Mock(spec=VeniceClient)
        client.structured_completion = AsyncMock()
        return client
    
    @pytest.fixture
    def mock_long_term_memory(self):
        """Mock long-term memory for meta-server testing."""
        memory = Mock(spec=LongTermMemory)
        memory.store_memory = AsyncMock()
        memory.retrieve_memories = AsyncMock()
        return memory
    
    @pytest.fixture
    def meta_server(self, mock_venice_client, mock_long_term_memory):
        """Create MetaMCPServer for testing."""
        config = {
            "server_output_dir": "/tmp/test_servers",
            "research_depth": "medium"
        }
        return MetaMCPServer(
            venice_client=mock_venice_client,
            long_term_memory=mock_long_term_memory,
            config=config
        )
    
    @pytest.mark.asyncio
    async def test_api_discovery_integration(self, meta_server, mock_venice_client):
        """Test API discovery integration."""
        mock_venice_client.structured_completion.return_value = {
            "api_name": "Weather API",
            "endpoints": [
                {"path": "/weather", "method": "GET", "description": "Get current weather"},
                {"path": "/forecast", "method": "GET", "description": "Get weather forecast"}
            ],
            "authentication": "API key",
            "rate_limits": "1000 requests/hour"
        }
        
        with patch('src.mcp.meta_server.MetaMCPServer._discover_with_openapi_explorer') as mock_discover:
            mock_discover.return_value = {
                "openapi": "3.0.0",
                "info": {"title": "Weather API", "version": "1.0"},
                "paths": {
                    "/weather": {"get": {"summary": "Get weather"}},
                    "/forecast": {"get": {"summary": "Get forecast"}}
                }
            }
            
            result = await meta_server.discover_api(
                api_url="https://api.weather.com/v1",
                api_name="Weather API"
            )
            
            assert result["api_name"] == "Weather API"
            assert len(result["endpoints"]) == 2
            assert result["authentication"] == "API key"
    
    @pytest.mark.asyncio
    async def test_server_generation_integration(self, meta_server, mock_venice_client):
        """Test MCP server generation integration."""
        mock_venice_client.structured_completion.return_value = {
            "server_code": "# Generated MCP Server\nclass WeatherServer:\n    pass",
            "tools": ["get_weather", "get_forecast"],
            "dependencies": ["requests", "mcp"],
            "configuration": {"api_key_required": True}
        }
        
        with patch('src.mcp.meta_server.MetaMCPServer._write_server_file') as mock_write:
            mock_write.return_value = "/tmp/weather_server.py"
            
            result = await meta_server.generate_mcp_server(
                purpose="Weather data integration",
                api_url="https://api.weather.com/v1"
            )
            
            assert result["server_creation"]["tools"] == ["get_weather", "get_forecast"]
            assert result["server_creation"]["file_path"] == "/tmp/weather_server.py"
            assert "dependencies" in result["server_creation"]
    
    @pytest.mark.asyncio
    async def test_research_and_create_integration(self, meta_server, mock_venice_client):
        """Test research-based server creation integration."""
        mock_venice_client.structured_completion.side_effect = [
            {
                "research_findings": [
                    "Social media APIs provide sentiment analysis",
                    "Twitter API v2 offers comprehensive data access",
                    "Reddit API allows subreddit monitoring"
                ],
                "recommended_apis": [
                    {"name": "Twitter API", "url": "https://api.twitter.com/2"},
                    {"name": "Reddit API", "url": "https://www.reddit.com/api"}
                ]
            },
            {
                "server_code": "# Social Media MCP Server\nclass SocialMediaServer:\n    pass",
                "tools": ["analyze_sentiment", "monitor_mentions", "extract_trends"],
                "dependencies": ["tweepy", "praw", "textblob"]
            }
        ]
        
        with patch('src.mcp.meta_server.MetaMCPServer._write_server_file') as mock_write:
            mock_write.return_value = "/tmp/social_media_server.py"
            
            result = await meta_server.research_and_create_server(
                problem_description="Monitor social media sentiment about our brand",
                research_depth="deep"
            )
            
            assert "research_findings" in result
            assert "server_creation" in result
            assert len(result["research_findings"]) == 3
            assert result["server_creation"]["tools"] == ["analyze_sentiment", "monitor_mentions", "extract_trends"]
