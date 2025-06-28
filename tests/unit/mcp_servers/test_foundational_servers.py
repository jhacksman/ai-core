"""
Unit tests for foundational MCP servers (Slack, Discord, Infrastructure, Automation).
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.mcp_servers.slack_server import SlackMCPServer
from src.mcp_servers.discord_server import DiscordMCPServer
from src.mcp_servers.infrastructure_server import InfrastructureMCPServer
from src.mcp_servers.automation_server import AutomationMCPServer


class TestSlackMCPServer:
    """Test cases for SlackMCPServer."""
    
    @pytest.fixture
    def mock_venice_client(self):
        """Mock Venice.ai client for testing."""
        client = Mock()
        client.simple_completion = AsyncMock()
        client.analyze_with_context = AsyncMock()
        return client
    
    @pytest.fixture
    def slack_server(self, mock_venice_client):
        """Create SlackMCPServer instance for testing."""
        config = {
            "workspace_id": "test-workspace",
            "channels": ["general", "testing"],
            "rate_limit": 100,
            "timeout": 30
        }
        return SlackMCPServer(venice_client=mock_venice_client, config=config)
    
    @pytest.mark.asyncio
    async def test_slack_server_initialization(self, slack_server):
        """Test Slack server initializes correctly."""
        await slack_server.initialize()
        
        assert slack_server.workspace_id == "test-workspace"
        assert "general" in slack_server.channels
        assert "testing" in slack_server.channels
        assert slack_server.rate_limit == 100
    
    @pytest.mark.asyncio
    async def test_send_message_tool(self, slack_server):
        """Test send_message tool functionality."""
        await slack_server.initialize()
        
        with patch('slack_sdk.WebClient.chat_postMessage') as mock_post:
            mock_post.return_value = {"ok": True, "ts": "1234567890.123456"}
            
            result = await slack_server.send_message(
                channel="general",
                message="Test message",
                thread_ts=None
            )
            
            assert result["success"] is True
            assert result["timestamp"] == "1234567890.123456"
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_channel_history_tool(self, slack_server):
        """Test get_channel_history tool functionality."""
        await slack_server.initialize()
        
        with patch('slack_sdk.WebClient.conversations_history') as mock_history:
            mock_history.return_value = {
                "ok": True,
                "messages": [
                    {"text": "Message 1", "ts": "1234567890.123456", "user": "U123"},
                    {"text": "Message 2", "ts": "1234567891.123456", "user": "U456"}
                ]
            }
            
            result = await slack_server.get_channel_history(
                channel="general",
                limit=10
            )
            
            assert result["success"] is True
            assert len(result["messages"]) == 2
            assert result["messages"][0]["text"] == "Message 1"
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_tool(self, slack_server, mock_venice_client):
        """Test analyze_sentiment tool functionality."""
        await slack_server.initialize()
        
        mock_venice_client.analyze_with_context.return_value = "positive"
        
        result = await slack_server.analyze_sentiment(
            text="This is a great day!",
            context="slack_message"
        )
        
        assert result["sentiment"] == "positive"
        assert result["text"] == "This is a great day!"
        mock_venice_client.analyze_with_context.assert_called_once()


class TestDiscordMCPServer:
    """Test cases for DiscordMCPServer."""
    
    @pytest.fixture
    def mock_venice_client(self):
        """Mock Venice.ai client for testing."""
        client = Mock()
        client.simple_completion = AsyncMock()
        client.analyze_with_context = AsyncMock()
        return client
    
    @pytest.fixture
    def discord_server(self, mock_venice_client):
        """Create DiscordMCPServer instance for testing."""
        config = {
            "guild_id": "test-guild",
            "channels": ["general", "bot-testing"],
            "rate_limit": 100,
            "timeout": 30
        }
        return DiscordMCPServer(venice_client=mock_venice_client, config=config)
    
    @pytest.mark.asyncio
    async def test_discord_server_initialization(self, discord_server):
        """Test Discord server initializes correctly."""
        await discord_server.initialize()
        
        assert discord_server.guild_id == "test-guild"
        assert "general" in discord_server.channels
        assert "bot-testing" in discord_server.channels
        assert discord_server.rate_limit == 100
    
    @pytest.mark.asyncio
    async def test_send_message_tool(self, discord_server):
        """Test send_message tool functionality."""
        await discord_server.initialize()
        
        with patch('discord.TextChannel.send') as mock_send:
            mock_message = Mock()
            mock_message.id = 123456789
            mock_send.return_value = mock_message
            
            result = await discord_server.send_message(
                channel_id="987654321",
                content="Test message"
            )
            
            assert result["success"] is True
            assert result["message_id"] == 123456789
    
    @pytest.mark.asyncio
    async def test_get_channel_messages_tool(self, discord_server):
        """Test get_channel_messages tool functionality."""
        await discord_server.initialize()
        
        with patch('discord.TextChannel.history') as mock_history:
            mock_message1 = Mock()
            mock_message1.content = "Message 1"
            mock_message1.author.name = "User1"
            mock_message1.created_at = datetime.now()
            
            mock_message2 = Mock()
            mock_message2.content = "Message 2"
            mock_message2.author.name = "User2"
            mock_message2.created_at = datetime.now()
            
            mock_history.return_value = [mock_message1, mock_message2]
            
            result = await discord_server.get_channel_messages(
                channel_id="987654321",
                limit=10
            )
            
            assert result["success"] is True
            assert len(result["messages"]) == 2
            assert result["messages"][0]["content"] == "Message 1"
    
    @pytest.mark.asyncio
    async def test_manage_roles_tool(self, discord_server):
        """Test manage_roles tool functionality."""
        await discord_server.initialize()
        
        with patch('discord.Member.add_roles') as mock_add_roles:
            mock_add_roles.return_value = None
            
            result = await discord_server.manage_roles(
                user_id="123456789",
                action="add",
                role_name="Moderator"
            )
            
            assert result["success"] is True
            assert result["action"] == "add"
            assert result["role_name"] == "Moderator"


class TestInfrastructureMCPServer:
    """Test cases for InfrastructureMCPServer."""
    
    @pytest.fixture
    def mock_venice_client(self):
        """Mock Venice.ai client for testing."""
        client = Mock()
        client.analyze_with_context = AsyncMock()
        return client
    
    @pytest.fixture
    def infrastructure_server(self, mock_venice_client):
        """Create InfrastructureMCPServer instance for testing."""
        config = {
            "monitored_services": ["nginx", "docker", "ssh"],
            "alert_thresholds": {
                "cpu_percent": 80.0,
                "memory_percent": 85.0,
                "disk_usage": 90.0
            },
            "allowed_commands": ["systemctl", "ps", "df", "free"]
        }
        return InfrastructureMCPServer(venice_client=mock_venice_client, config=config)
    
    @pytest.mark.asyncio
    async def test_infrastructure_server_initialization(self, infrastructure_server):
        """Test Infrastructure server initializes correctly."""
        await infrastructure_server.initialize()
        
        assert "nginx" in infrastructure_server.monitored_services
        assert "docker" in infrastructure_server.monitored_services
        assert infrastructure_server.alert_thresholds["cpu_percent"] == 80.0
    
    @pytest.mark.asyncio
    async def test_get_system_metrics_tool(self, infrastructure_server):
        """Test get_system_metrics tool functionality."""
        await infrastructure_server.initialize()
        
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_cpu.return_value = 45.2
            mock_memory.return_value = Mock(percent=67.8)
            mock_disk.return_value = Mock(percent=23.4)
            
            result = await infrastructure_server.get_system_metrics()
            
            assert result["cpu_percent"] == 45.2
            assert result["memory_percent"] == 67.8
            assert result["disk_percent"] == 23.4
            assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_check_service_status_tool(self, infrastructure_server):
        """Test check_service_status tool functionality."""
        await infrastructure_server.initialize()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="active (running)"
            )
            
            result = await infrastructure_server.check_service_status("nginx")
            
            assert result["service"] == "nginx"
            assert result["status"] == "active"
            assert result["is_running"] is True
    
    @pytest.mark.asyncio
    async def test_execute_system_command_tool(self, infrastructure_server):
        """Test execute_system_command tool functionality."""
        await infrastructure_server.initialize()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Command output",
                stderr=""
            )
            
            result = await infrastructure_server.execute_system_command(
                command="ps aux",
                timeout=30
            )
            
            assert result["success"] is True
            assert result["output"] == "Command output"
            assert result["return_code"] == 0


class TestAutomationMCPServer:
    """Test cases for AutomationMCPServer."""
    
    @pytest.fixture
    def mock_venice_client(self):
        """Mock Venice.ai client for testing."""
        client = Mock()
        client.analyze_with_context = AsyncMock()
        return client
    
    @pytest.fixture
    def automation_server(self, mock_venice_client):
        """Create AutomationMCPServer instance for testing."""
        config = {
            "headless": True,
            "browser_type": "chromium",
            "download_dir": "/tmp/downloads",
            "search_engines": ["duckduckgo", "bing"]
        }
        return AutomationMCPServer(venice_client=mock_venice_client, config=config)
    
    @pytest.mark.asyncio
    async def test_automation_server_initialization(self, automation_server):
        """Test Automation server initializes correctly."""
        await automation_server.initialize()
        
        assert automation_server.headless is True
        assert automation_server.browser_type == "chromium"
        assert automation_server.download_dir == "/tmp/downloads"
        assert "duckduckgo" in automation_server.search_engines
    
    @pytest.mark.asyncio
    async def test_create_browser_session_tool(self, automation_server):
        """Test create_browser_session tool functionality."""
        await automation_server.initialize()
        
        with patch('playwright.async_api.async_playwright') as mock_playwright:
            mock_browser = Mock()
            mock_context = Mock()
            mock_page = Mock()
            
            mock_playwright.return_value.__aenter__.return_value.chromium.launch.return_value = mock_browser
            mock_browser.new_context.return_value = mock_context
            mock_context.new_page.return_value = mock_page
            
            result = await automation_server.create_browser_session(
                session_id="test-session",
                user_agent="Test Agent"
            )
            
            assert result["success"] is True
            assert result["session_id"] == "test-session"
            assert "test-session" in automation_server.browser_sessions
    
    @pytest.mark.asyncio
    async def test_navigate_to_url_tool(self, automation_server):
        """Test navigate_to_url tool functionality."""
        await automation_server.initialize()
        
        mock_page = Mock()
        mock_page.goto = AsyncMock()
        mock_page.title.return_value = "Test Page"
        automation_server.browser_sessions["test-session"] = {
            "page": mock_page,
            "context": Mock(),
            "browser": Mock()
        }
        
        result = await automation_server.navigate_to_url(
            session_id="test-session",
            url="https://example.com"
        )
        
        assert result["success"] is True
        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page"
        mock_page.goto.assert_called_once_with("https://example.com")
    
    @pytest.mark.asyncio
    async def test_extract_page_content_tool(self, automation_server):
        """Test extract_page_content tool functionality."""
        await automation_server.initialize()
        
        mock_page = Mock()
        mock_page.content.return_value = "<html><body>Test content</body></html>"
        mock_page.title.return_value = "Test Page"
        automation_server.browser_sessions["test-session"] = {
            "page": mock_page,
            "context": Mock(),
            "browser": Mock()
        }
        
        result = await automation_server.extract_page_content(
            session_id="test-session",
            content_type="text"
        )
        
        assert result["success"] is True
        assert result["title"] == "Test Page"
        assert "content" in result
    
    @pytest.mark.asyncio
    async def test_perform_web_search_tool(self, automation_server, mock_venice_client):
        """Test perform_web_search tool functionality."""
        await automation_server.initialize()
        
        mock_page = Mock()
        mock_page.goto = AsyncMock()
        mock_page.fill = AsyncMock()
        mock_page.click = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.content.return_value = "<html><body>Search results</body></html>"
        
        automation_server.browser_sessions["test-session"] = {
            "page": mock_page,
            "context": Mock(),
            "browser": Mock()
        }
        
        mock_venice_client.analyze_with_context.return_value = "Relevant search results found"
        
        result = await automation_server.perform_web_search(
            session_id="test-session",
            query="test search",
            search_engine="duckduckgo"
        )
        
        assert result["success"] is True
        assert result["query"] == "test search"
        assert result["search_engine"] == "duckduckgo"
        assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_take_screenshot_tool(self, automation_server):
        """Test take_screenshot tool functionality."""
        await automation_server.initialize()
        
        mock_page = Mock()
        mock_page.screenshot = AsyncMock(return_value=b"fake_screenshot_data")
        
        automation_server.browser_sessions["test-session"] = {
            "page": mock_page,
            "context": Mock(),
            "browser": Mock()
        }
        
        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            result = await automation_server.take_screenshot(
                session_id="test-session",
                filename="test_screenshot.png"
            )
            
            assert result["success"] is True
            assert result["filename"] == "test_screenshot.png"
            assert "file_path" in result
            mock_page.screenshot.assert_called_once()
