"""
Venice.ai API models and request/response handling.

This module provides data models for Venice.ai API integration and VRAM optimization.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VeniceRequest(BaseModel):
    """Request model for Venice.ai API calls."""
    model: str = Field(..., description="Model name to use")
    messages: List[Dict[str, str]] = Field(..., description="Conversation messages")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class VeniceResponse(BaseModel):
    """Response model for Venice.ai API calls."""
    id: str = Field(..., description="Response ID")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage")
    created: Optional[int] = Field(None, description="Creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")


class VeniceError(BaseModel):
    """Error model for Venice.ai API errors."""
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")


class ModelType(Enum):
    """Supported Venice.ai model types."""
    LLAMA_4 = "llama-4"
    QWEN_QWQ_32B = "qwen-qwq-32b"
    TEXT_EMBEDDING = "text-embedding-ada-002"


class ModelCapability(Enum):
    """Model capabilities for different scaffolding tasks."""
    CHAT_COMPLETION = "chat_completion"
    PROBLEM_ANALYSIS = "problem_analysis"
    TOOL_DESIGN = "tool_design"
    CODE_GENERATION = "code_generation"
    RESEARCH_SYNTHESIS = "research_synthesis"
    EMBEDDINGS = "embeddings"


@dataclass
class ModelSpec:
    """Specification for a Venice.ai model."""
    name: str
    model_type: ModelType
    capabilities: List[ModelCapability]
    vram_usage_mb: int
    context_window: int
    max_tokens: int
    cost_per_token: float
    recommended_temperature: float
    description: str


@dataclass
class ModelUsage:
    """Track model usage for VRAM management."""
    model_name: str
    start_time: datetime
    estimated_duration: timedelta
    vram_allocated: int
    task_type: str
    priority: int = 1  # 1 = high, 5 = low


class VRAMManager:
    """
    Manages VRAM allocation across Venice.ai models.
    
    Ensures the 64GB VRAM limit is respected while optimizing
    for performance and task priority.
    """
    
    def __init__(self, total_vram_gb: int = 64):
        """
        Initialize VRAM manager.
        
        Args:
            total_vram_gb: Total available VRAM in GB
        """
        self.total_vram_mb = total_vram_gb * 1024
        self.allocated_vram_mb = 0
        self.active_models: Dict[str, ModelUsage] = {}
        self.allocation_lock = asyncio.Lock()
    
    async def can_allocate(self, model_spec: ModelSpec) -> bool:
        """
        Check if VRAM can be allocated for a model.
        
        Args:
            model_spec: Model specification requiring VRAM
            
        Returns:
            True if allocation is possible
        """
        async with self.allocation_lock:
            available_vram = self.total_vram_mb - self.allocated_vram_mb
            return available_vram >= model_spec.vram_usage_mb
    
    async def allocate_vram(
        self,
        model_spec: ModelSpec,
        task_type: str,
        estimated_duration: timedelta,
        priority: int = 1
    ) -> bool:
        """
        Allocate VRAM for a model.
        
        Args:
            model_spec: Model specification
            task_type: Type of task (for tracking)
            estimated_duration: Expected duration of usage
            priority: Task priority (1=high, 5=low)
            
        Returns:
            True if allocation successful
        """
        async with self.allocation_lock:
            if not await self.can_allocate(model_spec):
                if not await self._try_evict_models(model_spec.vram_usage_mb, priority):
                    return False
            
            usage = ModelUsage(
                model_name=model_spec.name,
                start_time=datetime.now(),
                estimated_duration=estimated_duration,
                vram_allocated=model_spec.vram_usage_mb,
                task_type=task_type,
                priority=priority
            )
            
            self.active_models[model_spec.name] = usage
            self.allocated_vram_mb += model_spec.vram_usage_mb
            
            logger.info(f"Allocated {model_spec.vram_usage_mb}MB VRAM for {model_spec.name}")
            return True
    
    async def deallocate_vram(self, model_name: str):
        """
        Deallocate VRAM for a model.
        
        Args:
            model_name: Name of the model to deallocate
        """
        async with self.allocation_lock:
            if model_name in self.active_models:
                usage = self.active_models[model_name]
                self.allocated_vram_mb -= usage.vram_allocated
                del self.active_models[model_name]
                logger.info(f"Deallocated {usage.vram_allocated}MB VRAM for {model_name}")
    
    async def _try_evict_models(self, required_vram: int, requesting_priority: int) -> bool:
        """
        Try to evict lower-priority models to free VRAM.
        
        Args:
            required_vram: Amount of VRAM needed
            requesting_priority: Priority of the requesting task
            
        Returns:
            True if enough VRAM was freed
        """
        eviction_candidates = [
            (name, usage) for name, usage in self.active_models.items()
            if usage.priority > requesting_priority
        ]
        eviction_candidates.sort(key=lambda x: x[1].priority, reverse=True)
        
        freed_vram = 0
        evicted_models = []
        
        for model_name, usage in eviction_candidates:
            if freed_vram >= required_vram:
                break
            
            freed_vram += usage.vram_allocated
            evicted_models.append(model_name)
        
        if freed_vram >= required_vram:
            for model_name in evicted_models:
                await self.deallocate_vram(model_name)
                logger.warning(f"Evicted model {model_name} to free VRAM")
            return True
        
        return False
    
    def get_vram_status(self) -> Dict[str, Any]:
        """
        Get current VRAM allocation status.
        
        Returns:
            Dictionary with VRAM usage information
        """
        return {
            "total_vram_mb": self.total_vram_mb,
            "allocated_vram_mb": self.allocated_vram_mb,
            "available_vram_mb": self.total_vram_mb - self.allocated_vram_mb,
            "utilization_percent": (self.allocated_vram_mb / self.total_vram_mb) * 100,
            "active_models": len(self.active_models),
            "model_details": {
                name: {
                    "vram_mb": usage.vram_allocated,
                    "task_type": usage.task_type,
                    "priority": usage.priority,
                    "duration": str(datetime.now() - usage.start_time)
                }
                for name, usage in self.active_models.items()
            }
        }


class ModelManager:
    """
    Manages Venice.ai models for scaffolding operations.
    
    Provides intelligent model selection, VRAM management,
    and optimization for different task types.
    """
    
    def __init__(self, vram_manager: Optional[VRAMManager] = None):
        """
        Initialize model manager.
        
        Args:
            vram_manager: VRAM manager instance (creates default if None)
        """
        self.vram_manager = vram_manager or VRAMManager()
        self.model_specs = self._initialize_model_specs()
        self.model_cache: Dict[str, Any] = {}
    
    def _initialize_model_specs(self) -> Dict[str, ModelSpec]:
        """Initialize specifications for supported models."""
        return {
            "llama-4": ModelSpec(
                name="llama-4",
                model_type=ModelType.LLAMA_4,
                capabilities=[
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.TOOL_DESIGN,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.RESEARCH_SYNTHESIS
                ],
                vram_usage_mb=8192,  # 8GB estimated
                context_window=128000,
                max_tokens=4096,
                cost_per_token=0.00001,
                recommended_temperature=0.7,
                description="General-purpose large language model for complex reasoning"
            ),
            "qwen-qwq-32b": ModelSpec(
                name="qwen-qwq-32b",
                model_type=ModelType.QWEN_QWQ_32B,
                capabilities=[
                    ModelCapability.CHAT_COMPLETION,
                    ModelCapability.PROBLEM_ANALYSIS,
                    ModelCapability.RESEARCH_SYNTHESIS
                ],
                vram_usage_mb=16384,  # 16GB estimated for 32B model
                context_window=32768,
                max_tokens=2048,
                cost_per_token=0.00002,
                recommended_temperature=0.3,
                description="Specialized model for analytical and reasoning tasks"
            ),
            "text-embedding-ada-002": ModelSpec(
                name="text-embedding-ada-002",
                model_type=ModelType.TEXT_EMBEDDING,
                capabilities=[ModelCapability.EMBEDDINGS],
                vram_usage_mb=2048,  # 2GB estimated
                context_window=8192,
                max_tokens=0,  # Embeddings don't generate tokens
                cost_per_token=0.0000001,
                recommended_temperature=0.0,
                description="Text embedding model for semantic similarity"
            )
        }
    
    def get_model_for_capability(
        self,
        capability: ModelCapability,
        priority: int = 1
    ) -> Optional[ModelSpec]:
        """
        Get the best model for a specific capability.
        
        Args:
            capability: Required model capability
            priority: Task priority for VRAM allocation
            
        Returns:
            Best model spec for the capability, or None if unavailable
        """
        candidates = [
            spec for spec in self.model_specs.values()
            if capability in spec.capabilities
        ]
        
        if not candidates:
            return None
        
        candidates.sort(key=lambda x: x.vram_usage_mb)
        
        for candidate in candidates:
            if asyncio.run(self.vram_manager.can_allocate(candidate)):
                return candidate
        
        return None
    
    async def allocate_model(
        self,
        model_name: str,
        task_type: str,
        estimated_duration: timedelta = timedelta(minutes=5),
        priority: int = 1
    ) -> bool:
        """
        Allocate a model for use.
        
        Args:
            model_name: Name of the model to allocate
            task_type: Type of task (for tracking)
            estimated_duration: Expected usage duration
            priority: Task priority
            
        Returns:
            True if allocation successful
        """
        if model_name not in self.model_specs:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model_spec = self.model_specs[model_name]
        return await self.vram_manager.allocate_vram(
            model_spec, task_type, estimated_duration, priority
        )
    
    async def deallocate_model(self, model_name: str):
        """
        Deallocate a model.
        
        Args:
            model_name: Name of the model to deallocate
        """
        await self.vram_manager.deallocate_vram(model_name)
    
    def get_model_recommendations(self, task_description: str) -> List[str]:
        """
        Get model recommendations for a task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            List of recommended model names, ordered by preference
        """
        recommendations = []
        
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["analyze", "research", "understand"]):
            recommendations.append("qwen-qwq-32b")
        
        if any(word in task_lower for word in ["generate", "create", "design", "code"]):
            recommendations.append("llama-4")
        
        if any(word in task_lower for word in ["embed", "similarity", "search"]):
            recommendations.append("text-embedding-ada-002")
        
        if "llama-4" not in recommendations:
            recommendations.append("llama-4")
        
        return recommendations
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of model management.
        
        Returns:
            Status information including VRAM usage and model availability
        """
        vram_status = self.vram_manager.get_vram_status()
        
        model_availability = {}
        for name, spec in self.model_specs.items():
            can_allocate = asyncio.run(self.vram_manager.can_allocate(spec))
            model_availability[name] = {
                "available": can_allocate,
                "vram_required_mb": spec.vram_usage_mb,
                "capabilities": [cap.value for cap in spec.capabilities],
                "context_window": spec.context_window
            }
        
        return {
            "vram_status": vram_status,
            "model_availability": model_availability,
            "total_models": len(self.model_specs)
        }
