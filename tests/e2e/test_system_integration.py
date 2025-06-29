"""
End-to-end tests for the complete PDX Hackerspace AI Agent system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.mcp.integration import MCPIntegrationManager
from src.venice.client import VeniceClient
from src.memory.long_term_memory import LongTermMemory
from src.agent.manager import AgentManager


class TestSystemIntegration:
    """End-to-end tests for the complete system."""
    
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
    def integration_system(self, mock_venice_client, mock_long_term_memory, mock_agent_manager):
        """Create complete integration system for testing."""
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
    async def test_complete_system_startup(self, integration_system):
        """Test complete system startup sequence."""
        with patch.multiple(
            integration_system,
            server_manager=Mock(initialize=AsyncMock()),
            registry=Mock(initialize=AsyncMock()),
            client_manager=Mock(initialize=AsyncMock()),
            server_factory=Mock(initialize=AsyncMock()),
            template_engine=Mock(initialize=AsyncMock()),
            capability_analyzer=Mock(initialize=AsyncMock()),
            meta_server=Mock(initialize=AsyncMock())
        ):
            await integration_system.initialize()
            
            with patch.object(integration_system.server_manager, 'start_server') as mock_start:
                mock_start.return_value = {"success": True, "server_id": "test-server"}
                
                startup_result = await integration_system.start_foundational_servers([
                    "slack", "discord", "infrastructure", "automation"
                ])
                
                assert startup_result["started_servers"] == 4
                assert startup_result["failed_servers"] == 0
                assert mock_start.call_count == 4
    
    @pytest.mark.asyncio
    async def test_end_to_end_problem_solving_workflow(self, integration_system, mock_venice_client, mock_agent_manager):
        """Test complete problem-solving workflow from research to tool creation."""
        await integration_system.initialize()
        
        mock_agent_manager.submit_task.return_value = "task-123"
        mock_agent_manager.get_task_status.return_value = {
            "status": "completed",
            "result": {
                "findings": ["Need weather data integration", "Slack notifications required"],
                "recommended_actions": ["Create weather MCP server", "Integrate with Slack"]
            }
        }
        
        task_id = await mock_agent_manager.submit_task(
            description="Research weather monitoring solution for hackerspace",
            task_type="research"
        )
        
        with patch.object(integration_system.meta_server, 'research_and_create_server') as mock_research_create:
            mock_research_create.return_value = {
                "research_findings": ["Weather APIs available", "OpenWeatherMap recommended"],
                "server_creation": {
                    "server_id": "weather-monitor-123",
                    "server_name": "weather_monitor",
                    "tools": ["get_current_weather", "get_forecast", "send_weather_alert"],
                    "file_path": "/tmp/weather_monitor.py"
                }
            }
            
            server_result = await integration_system.create_dynamic_server(
                purpose="Weather monitoring for hackerspace",
                research_depth="deep"
            )
            
            assert server_result["server_id"] == "weather-monitor-123"
            assert "get_current_weather" in server_result["tools"]
        
        with patch.object(integration_system.client_manager, 'call_tool') as mock_call_tool:
            mock_call_tool.side_effect = [
                {"temperature": 72, "condition": "sunny", "humidity": 45},  # Weather data
                {"success": True, "message_id": "msg-456"}  # Slack notification
            ]
            
            weather_result = await integration_system.execute_tool_across_servers(
                tool_name="get_current_weather",
                arguments={"location": "Portland, OR"}
            )
            
            slack_result = await integration_system.execute_tool_across_servers(
                tool_name="send_message",
                arguments={
                    "channel": "general",
                    "message": f"Current weather: {weather_result['result']['temperature']}°F, {weather_result['result']['condition']}"
                }
            )
            
            assert weather_result["result"]["temperature"] == 72
            assert slack_result["result"]["success"] is True
    
    @pytest.mark.asyncio
    async def test_system_shutdown_and_cleanup(self, integration_system):
        """Test graceful system shutdown and cleanup."""
        await integration_system.initialize()
        
        from src.mcp.integration import IntegratedServerInstance
        
        server = IntegratedServerInstance(
            server_id="test-server",
            server_type="foundational",
            name="test_server",
            status="running",
            tools=["test_tool"],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            metadata={}
        )
        
        integration_system.server_instances["test-server"] = server
        
        with patch.multiple(
            integration_system,
            server_manager=Mock(stop_server=AsyncMock(), cleanup=AsyncMock()),
            meta_server=Mock(cleanup=AsyncMock()),
            client_manager=Mock(cleanup=AsyncMock()),
            registry=Mock(cleanup=AsyncMock())
        ):
            await integration_system.shutdown()
            
            integration_system.server_manager.stop_server.assert_called_once_with("test-server")
            integration_system.server_manager.cleanup.assert_called_once()
            integration_system.meta_server.cleanup.assert_called_once()
            integration_system.client_manager.cleanup.assert_called_once()
            integration_system.registry.cleanup.assert_called_once()
            
            assert len(integration_system.server_instances) == 0
