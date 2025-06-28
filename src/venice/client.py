"""
Venice.ai API client for LLM operations and scaffolding integration.

This module provides an async client for interacting with Venice.ai models,
supporting the scaffolding system's research, tool creation, and execution needs.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import aiohttp
from pydantic import BaseModel, Field
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class VeniceMessage(BaseModel):
    """Message structure for Venice.ai API calls."""
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)


class VeniceResponse(BaseModel):
    """Response structure from Venice.ai API."""
    content: str = Field(..., description="Generated response content")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")


class VeniceAPIError(Exception):
    """Exception raised for Venice.ai API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)


class VeniceClient:
    """
    Async client for Venice.ai API integration.
    
    Provides methods for LLM operations, embeddings, and scaffolding-specific
    functionality like research coordination and tool creation assistance.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize Venice.ai client.
        
        Args:
            api_key: Venice.ai API key (defaults to VENICE_AI_API_KEY env var)
            base_url: API base URL (defaults to VENICE_AI_BASE_URL env var)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key or os.getenv("VENICE_AI_API_KEY")
        self.base_url = base_url or os.getenv("VENICE_AI_BASE_URL", "https://api.venice.ai")
        self.timeout = timeout
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError("Venice.ai API key is required")
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ai-core-scaffolding/0.1.0"
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                headers=self._headers,
                timeout=timeout
            )
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Venice.ai API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            VeniceAPIError: If request fails after retries
        """
        await self._ensure_session()
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params
                ) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        return response_data
                    elif response.status == 429 and attempt < self.max_retries:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise VeniceAPIError(
                            f"API request failed: {response.status}",
                            status_code=response.status,
                            response_data=response_data
                        )
                        
            except aiohttp.ClientError as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise VeniceAPIError(f"Network error after {self.max_retries} retries: {e}")
        
        raise VeniceAPIError(f"Request failed after {self.max_retries} retries")
    
    async def chat_completion(
        self,
        messages: List[VeniceMessage],
        model: str = "llama-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> VeniceResponse:
        """
        Generate chat completion using Venice.ai models.
        
        Args:
            messages: List of conversation messages
            model: Model name to use for generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional model parameters
            
        Returns:
            VeniceResponse with generated content
        """
        data = {
            "model": model,
            "messages": [msg.dict() for msg in messages],
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        
        response_data = await self._make_request("POST", "/v1/chat/completions", data=data)
        
        choice = response_data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = response_data.get("usage", {})
        
        return VeniceResponse(
            content=content,
            model=model,
            usage=usage,
            metadata=response_data
        )
    
    async def analyze_problem(self, problem_description: str) -> Dict[str, Any]:
        """
        Analyze a problem to understand domain, requirements, and complexity.
        
        This is a scaffolding-specific method that helps the system understand
        what tools and capabilities might be needed to solve a problem.
        
        Args:
            problem_description: Description of the problem to analyze
            
        Returns:
            Analysis results including domain, actions, complexity
        """
        system_prompt = """You are an expert problem analyzer for an AI scaffolding system. 
        Analyze the given problem and provide structured output about:
        1. Domain: What field/area does this problem belong to?
        2. Required Actions: What types of actions would be needed to solve this?
        3. Complexity: Rate complexity from 1-10 and explain why
        4. Tools Needed: What types of tools or capabilities would be helpful?
        5. Research Areas: What should be researched to better understand this problem?
        
        Provide your analysis in a structured format."""
        
        messages = [
            VeniceMessage(role="system", content=system_prompt),
            VeniceMessage(role="user", content=f"Analyze this problem: {problem_description}")
        ]
        
        response = await self.chat_completion(messages, model="qwen-qwq-32b", temperature=0.3)
        
        return {
            "domain": "general",  # Would extract from response
            "actions": [],  # Would extract from response
            "complexity": 5,  # Would extract from response
            "analysis_text": response.content
        }
    
    async def synthesize_tool_design(
        self,
        requirements: Dict[str, Any],
        research_context: Dict[str, Any],
        existing_patterns: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize tool design based on requirements and research.
        
        This scaffolding method helps create specifications for new MCP servers
        and tools based on problem analysis and research findings.
        
        Args:
            requirements: Tool requirements from problem analysis
            research_context: Research findings about the problem domain
            existing_patterns: Successful tool patterns from experience database
            
        Returns:
            Tool design specification
        """
        system_prompt = """You are an expert tool designer for an AI scaffolding system.
        Based on the requirements, research context, and existing successful patterns,
        design a new MCP server tool that can help solve the identified problem.
        
        Provide a detailed specification including:
        1. Tool name and description
        2. Input schema (JSON schema format)
        3. Implementation strategy
        4. Integration points with existing systems
        5. Success criteria and testing approach"""
        
        context = f"""
        Requirements: {requirements}
        Research Context: {research_context}
        Existing Patterns: {existing_patterns}
        """
        
        messages = [
            VeniceMessage(role="system", content=system_prompt),
            VeniceMessage(role="user", content=f"Design a tool based on this context: {context}")
        ]
        
        response = await self.chat_completion(messages, model="llama-4", temperature=0.5)
        
        return {
            "name": "generated_tool",  # Would extract from response
            "description": response.content[:200],  # Would extract from response
            "schema": {},  # Would extract from response
            "strategy": response.content,
            "integrations": []
        }
    
    async def get_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """
        Generate embeddings for text using Venice.ai embedding models.
        
        Args:
            texts: List of texts to embed
            model: Embedding model name
            
        Returns:
            List of embedding vectors
        """
        data = {
            "model": model,
            "input": texts
        }
        
        response_data = await self._make_request("POST", "/v1/embeddings", data=data)
        
        embeddings = []
        for item in response_data.get("data", []):
            embeddings.append(item.get("embedding", []))
        
        return embeddings
    
    async def health_check(self) -> bool:
        """
        Check if Venice.ai API is accessible and responding.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            await self._make_request("GET", "/v1/models")
            return True
        except VeniceAPIError:
            return False
