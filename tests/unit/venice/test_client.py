"""
Unit tests for Venice.ai client integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from src.venice.client import VeniceClient
from src.venice.models import VeniceRequest, VeniceResponse, VeniceError
from src.venice.tokens import TokenManager


class TestVeniceClient:
    """Test cases for VeniceClient."""
    
    @pytest.fixture
    def mock_token_manager(self):
        """Mock token manager for testing."""
        token_manager = Mock(spec=TokenManager)
        token_manager.get_token.return_value = "test-token"
        token_manager.refresh_token = AsyncMock()
        return token_manager
    
    @pytest.fixture
    def venice_client(self, mock_token_manager):
        """Create VeniceClient instance for testing."""
        config = {
            "api_url": "https://api.venice.ai/v1",
            "model_preferences": {
                "default": "claude-3-5-sonnet-20241022"
            },
            "timeout_seconds": 30,
            "max_retries": 3
        }
        return VeniceClient(config=config, token_manager=mock_token_manager)
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, venice_client):
        """Test client initializes correctly."""
        assert venice_client.api_url == "https://api.venice.ai/v1"
        assert venice_client.default_model == "claude-3-5-sonnet-20241022"
        assert venice_client.timeout == 30
        assert venice_client.max_retries == 3
    
    @pytest.mark.asyncio
    async def test_simple_completion_success(self, venice_client):
        """Test successful simple completion."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "Test response"}}],
                "usage": {"total_tokens": 100}
            })
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await venice_client.simple_completion("Test prompt")
            
            assert result == "Test response"
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_simple_completion_error(self, venice_client):
        """Test simple completion with API error."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 400
            mock_response.json = AsyncMock(return_value={
                "error": {"message": "Bad request"}
            })
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with pytest.raises(VeniceError):
                await venice_client.simple_completion("Test prompt")
    
    @pytest.mark.asyncio
    async def test_structured_completion_success(self, venice_client):
        """Test successful structured completion."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "choices": [{"message": {"content": '{"result": "success"}'}}],
                "usage": {"total_tokens": 150}
            })
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await venice_client.structured_completion(
                prompt="Test prompt",
                schema={"type": "object", "properties": {"result": {"type": "string"}}}
            )
            
            assert result == {"result": "success"}
    
    @pytest.mark.asyncio
    async def test_analyze_with_context_success(self, venice_client):
        """Test successful analysis with context."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "Analysis result"}}],
                "usage": {"total_tokens": 200}
            })
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await venice_client.analyze_with_context(
                content="Test content",
                context={"key": "value"},
                analysis_type="sentiment"
            )
            
            assert result == "Analysis result"
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, venice_client):
        """Test retry mechanism on temporary failures."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response_fail = Mock()
            mock_response_fail.status = 500
            mock_response_fail.json = AsyncMock(return_value={
                "error": {"message": "Internal server error"}
            })
            
            mock_response_success = Mock()
            mock_response_success.status = 200
            mock_response_success.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "Success after retry"}}],
                "usage": {"total_tokens": 100}
            })
            
            mock_post.return_value.__aenter__.side_effect = [
                mock_response_fail,
                mock_response_success
            ]
            
            result = await venice_client.simple_completion("Test prompt")
            
            assert result == "Success after retry"
            assert mock_post.call_count == 2
    
    @pytest.mark.asyncio
    async def test_token_refresh_on_auth_error(self, venice_client, mock_token_manager):
        """Test token refresh on authentication error."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response_auth_error = Mock()
            mock_response_auth_error.status = 401
            mock_response_auth_error.json = AsyncMock(return_value={
                "error": {"message": "Unauthorized"}
            })
            
            mock_response_success = Mock()
            mock_response_success.status = 200
            mock_response_success.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "Success after token refresh"}}],
                "usage": {"total_tokens": 100}
            })
            
            mock_post.return_value.__aenter__.side_effect = [
                mock_response_auth_error,
                mock_response_success
            ]
            
            result = await venice_client.simple_completion("Test prompt")
            
            assert result == "Success after token refresh"
            mock_token_manager.refresh_token.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self, venice_client):
        """Test rate limiting handling."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 429
            mock_response.json = AsyncMock(return_value={
                "error": {"message": "Rate limit exceeded"}
            })
            mock_response.headers = {"Retry-After": "1"}
            mock_post.return_value.__aenter__.return_value = mock_response
            
            with patch('asyncio.sleep') as mock_sleep:
                with pytest.raises(VeniceError):
                    await venice_client.simple_completion("Test prompt")
                
                mock_sleep.assert_called_with(1.0)
    
    @pytest.mark.asyncio
    async def test_get_usage_stats(self, venice_client):
        """Test usage statistics tracking."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "choices": [{"message": {"content": "Test response"}}],
                "usage": {"total_tokens": 100}
            })
            mock_post.return_value.__aenter__.return_value = mock_response
            
            await venice_client.simple_completion("Test prompt 1")
            await venice_client.simple_completion("Test prompt 2")
            
            stats = venice_client.get_usage_stats()
            
            assert stats["total_requests"] == 2
            assert stats["total_tokens"] == 200
            assert stats["successful_requests"] == 2
            assert stats["failed_requests"] == 0


class TestVeniceModels:
    """Test cases for Venice models."""
    
    def test_venice_request_creation(self):
        """Test VeniceRequest creation."""
        request = VeniceRequest(
            prompt="Test prompt",
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7
        )
        
        assert request.prompt == "Test prompt"
        assert request.model == "claude-3-5-sonnet-20241022"
        assert request.max_tokens == 1000
        assert request.temperature == 0.7
    
    def test_venice_response_creation(self):
        """Test VeniceResponse creation."""
        response = VeniceResponse(
            content="Test response",
            model="claude-3-5-sonnet-20241022",
            usage={"total_tokens": 100},
            timestamp=datetime.now()
        )
        
        assert response.content == "Test response"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.usage["total_tokens"] == 100
        assert isinstance(response.timestamp, datetime)
    
    def test_venice_error_creation(self):
        """Test VeniceError creation."""
        error = VeniceError("Test error", status_code=400)
        
        assert str(error) == "Test error"
        assert error.status_code == 400


class TestTokenManager:
    """Test cases for TokenManager."""
    
    @pytest.fixture
    def token_manager(self):
        """Create TokenManager instance for testing."""
        return TokenManager(api_key="test-api-key")
    
    def test_token_manager_initialization(self, token_manager):
        """Test token manager initializes correctly."""
        assert token_manager.api_key == "test-api-key"
        assert token_manager.get_token() == "test-api-key"
    
    @pytest.mark.asyncio
    async def test_token_refresh(self, token_manager):
        """Test token refresh functionality."""
        await token_manager.refresh_token()
        assert token_manager.get_token() == "test-api-key"
    
    def test_token_validation(self, token_manager):
        """Test token validation."""
        assert token_manager.is_valid() is True
        
        empty_token_manager = TokenManager(api_key="")
        assert empty_token_manager.is_valid() is False
