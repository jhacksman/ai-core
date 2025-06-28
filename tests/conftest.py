"""
Pytest configuration and shared fixtures for PDX Hackerspace AI Agent tests.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, Generator

pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        "venice": {
            "api_url": "https://api.venice.ai/v1",
            "model_preferences": {
                "default": "claude-3-5-sonnet-20241022"
            },
            "timeout_seconds": 30,
            "max_retries": 3
        },
        "memory": {
            "vector_store": {
                "provider": "chroma",
                "persist_directory": "/tmp/test_chroma",
                "collection_name": "test_collection"
            },
            "long_term_memory": {
                "retention_days": 30,
                "importance_threshold": 0.3
            }
        },
        "mcp_servers": {
            "slack": {
                "enabled": True,
                "transport": "stdio",
                "rate_limit": 100
            },
            "discord": {
                "enabled": True,
                "transport": "stdio", 
                "rate_limit": 100
            },
            "infrastructure": {
                "enabled": True,
                "transport": "stdio",
                "rate_limit": 50
            },
            "automation": {
                "enabled": True,
                "transport": "stdio",
                "rate_limit": 20
            }
        },
        "agent": {
            "manager": {
                "max_concurrent_tasks": 5,
                "task_timeout_minutes": 30,
                "retry_attempts": 3
            }
        }
    }


@pytest.fixture
def mock_venice_client():
    """Mock Venice.ai client for testing."""
    client = Mock()
    client.simple_completion = AsyncMock()
    client.structured_completion = AsyncMock()
    client.analyze_with_context = AsyncMock()
    client.get_usage_stats = Mock(return_value={
        "total_requests": 0,
        "total_tokens": 0,
        "successful_requests": 0,
        "failed_requests": 0
    })
    return client


@pytest.fixture
def mock_long_term_memory():
    """Mock long-term memory for testing."""
    memory = Mock()
    memory.initialize = AsyncMock()
    memory.store_memory = AsyncMock()
    memory.retrieve_memories = AsyncMock()
    memory.update_memory_importance = AsyncMock()
    memory.delete_memory = AsyncMock()
    memory.cleanup_old_memories = AsyncMock()
    memory.get_memory_stats = AsyncMock()
    memory.backup_memories = AsyncMock()
    return memory


@pytest.fixture
def mock_agent_manager():
    """Mock agent manager for testing."""
    manager = Mock()
    manager.initialize = AsyncMock()
    manager.submit_task = AsyncMock()
    manager.get_task_status = AsyncMock()
    manager.cancel_task = AsyncMock()
    manager.list_active_tasks = AsyncMock()
    manager.get_stats = Mock()
    return manager


@pytest.fixture
def mock_server_manager():
    """Mock MCP server manager for testing."""
    manager = Mock()
    manager.initialize = AsyncMock()
    manager.start_server = AsyncMock()
    manager.stop_server = AsyncMock()
    manager.get_server_status = AsyncMock()
    manager.list_servers = AsyncMock()
    manager.cleanup = AsyncMock()
    return manager


@pytest.fixture
def mock_meta_server():
    """Mock meta MCP server for testing."""
    server = Mock()
    server.initialize = AsyncMock()
    server.discover_api = AsyncMock()
    server.generate_mcp_server = AsyncMock()
    server.research_problem_domain = AsyncMock()
    server.research_and_create_server = AsyncMock()
    server.install_mcp_server = AsyncMock()
    server.list_available_servers = AsyncMock()
    server.cleanup = AsyncMock()
    return server


@pytest.fixture
def sample_api_response():
    """Sample API response for testing."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "version": "1.0.0",
            "description": "A test API for unit testing"
        },
        "paths": {
            "/users": {
                "get": {
                    "summary": "Get users",
                    "responses": {
                        "200": {
                            "description": "List of users"
                        }
                    }
                }
            },
            "/posts": {
                "get": {
                    "summary": "Get posts",
                    "responses": {
                        "200": {
                            "description": "List of posts"
                        }
                    }
                }
            }
        }
    }


@pytest.fixture
def sample_memory_entries():
    """Sample memory entries for testing."""
    from datetime import datetime
    return [
        {
            "id": "memory-1",
            "content": "Successfully integrated Slack API",
            "metadata": {
                "memory_type": "experience",
                "importance": 0.8,
                "timestamp": datetime.now().isoformat(),
                "tags": ["slack", "integration", "success"]
            }
        },
        {
            "id": "memory-2",
            "content": "Weather API rate limit encountered",
            "metadata": {
                "memory_type": "observation",
                "importance": 0.6,
                "timestamp": datetime.now().isoformat(),
                "tags": ["weather", "api", "rate_limit"]
            }
        },
        {
            "id": "memory-3",
            "content": "Research findings on MCP server architecture",
            "metadata": {
                "memory_type": "research",
                "importance": 0.9,
                "timestamp": datetime.now().isoformat(),
                "tags": ["mcp", "architecture", "research"]
            }
        }
    ]


@pytest.fixture
def sample_server_configs():
    """Sample server configurations for testing."""
    return {
        "slack": {
            "module": "src.mcp_servers.slack_server",
            "class": "SlackMCPServer",
            "transport": "stdio",
            "config": {
                "workspace_id": "test-workspace",
                "channels": ["general", "testing"]
            }
        },
        "discord": {
            "module": "src.mcp_servers.discord_server",
            "class": "DiscordMCPServer",
            "transport": "stdio",
            "config": {
                "guild_id": "test-guild",
                "channels": ["general", "bot-testing"]
            }
        },
        "infrastructure": {
            "module": "src.mcp_servers.infrastructure_server",
            "class": "InfrastructureMCPServer",
            "transport": "stdio",
            "config": {
                "monitored_services": ["nginx", "docker"],
                "alert_thresholds": {
                    "cpu_percent": 80.0,
                    "memory_percent": 85.0
                }
            }
        }
    }


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        if "test_complete_system" in item.name or "test_end_to_end" in item.name:
            item.add_marker(pytest.mark.slow)
        
        if any(keyword in item.name.lower() for keyword in ["api", "web", "http", "network"]):
            item.add_marker(pytest.mark.requires_network)
