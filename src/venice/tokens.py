"""
Token management and context window optimization for Venice.ai models.

This module provides utilities for managing token usage, optimizing context windows,
and ensuring efficient use of model capabilities within token limits.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import tiktoken
import json

logger = logging.getLogger(__name__)


class TokenizationStrategy(Enum):
    """Strategies for handling token limits."""
    TRUNCATE_OLDEST = "truncate_oldest"
    TRUNCATE_NEWEST = "truncate_newest"
    SUMMARIZE_OLDEST = "summarize_oldest"
    PRIORITIZE_SYSTEM = "prioritize_system"
    ADAPTIVE = "adaptive"


@dataclass
class TokenUsage:
    """Track token usage for a request."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cost_estimate: float


@dataclass
class ContextWindow:
    """Represents a model's context window constraints."""
    max_tokens: int
    reserved_for_completion: int
    effective_prompt_limit: int
    
    def __post_init__(self):
        """Calculate effective prompt limit."""
        self.effective_prompt_limit = self.max_tokens - self.reserved_for_completion


class TokenManager:
    """
    Manages token usage and context window optimization for Venice.ai models.
    
    Provides intelligent truncation, summarization, and context management
    to maximize the effectiveness of model interactions within token limits.
    """
    
    def __init__(self):
        """Initialize token manager with model configurations."""
        self.encoders = {}
        self.context_windows = {
            "llama-4": ContextWindow(
                max_tokens=128000,
                reserved_for_completion=4096,
                effective_prompt_limit=123904
            ),
            "qwen-qwq-32b": ContextWindow(
                max_tokens=32768,
                reserved_for_completion=2048,
                effective_prompt_limit=30720
            ),
            "text-embedding-ada-002": ContextWindow(
                max_tokens=8192,
                reserved_for_completion=0,
                effective_prompt_limit=8192
            )
        }
        
        self.token_costs = {
            "llama-4": {"prompt": 0.01, "completion": 0.02},
            "qwen-qwq-32b": {"prompt": 0.02, "completion": 0.04},
            "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0}
        }
    
    def get_encoder(self, model: str) -> tiktoken.Encoding:
        """
        Get or create tokenizer encoder for a model.
        
        Args:
            model: Model name
            
        Returns:
            Tiktoken encoder instance
        """
        if model not in self.encoders:
            try:
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Could not load encoder for {model}, using default: {e}")
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        
        return self.encoders[model]
    
    def count_tokens(self, text: str, model: str) -> int:
        """
        Count tokens in text for a specific model.
        
        Args:
            text: Text to count tokens for
            model: Model name for tokenization
            
        Returns:
            Number of tokens
        """
        encoder = self.get_encoder(model)
        return len(encoder.encode(text))
    
    def count_message_tokens(self, messages: List[Dict[str, str]], model: str) -> int:
        """
        Count tokens in a list of messages.
        
        Args:
            messages: List of message dictionaries
            model: Model name for tokenization
            
        Returns:
            Total number of tokens
        """
        encoder = self.get_encoder(model)
        total_tokens = 0
        
        for message in messages:
            total_tokens += 4  # Every message has role, content, and formatting tokens
            
            for key, value in message.items():
                if isinstance(value, str):
                    total_tokens += len(encoder.encode(value))
        
        total_tokens += 2  # Add tokens for assistant response priming
        return total_tokens
    
    def optimize_context(
        self,
        messages: List[Dict[str, str]],
        model: str,
        strategy: TokenizationStrategy = TokenizationStrategy.ADAPTIVE,
        preserve_system: bool = True
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Optimize message context to fit within model's token limits.
        
        Args:
            messages: List of message dictionaries
            model: Model name
            strategy: Tokenization strategy to use
            preserve_system: Whether to always preserve system messages
            
        Returns:
            Tuple of (optimized_messages, optimization_info)
        """
        context_window = self.context_windows.get(model)
        if not context_window:
            logger.warning(f"Unknown model {model}, using default context window")
            context_window = ContextWindow(max_tokens=4096, reserved_for_completion=512, effective_prompt_limit=3584)
        
        current_tokens = self.count_message_tokens(messages, model)
        
        if current_tokens <= context_window.effective_prompt_limit:
            return messages, {"tokens_used": current_tokens, "optimization_applied": False}
        
        optimized_messages = self._apply_optimization_strategy(
            messages, model, context_window, strategy, preserve_system
        )
        
        final_tokens = self.count_message_tokens(optimized_messages, model)
        
        optimization_info = {
            "original_tokens": current_tokens,
            "final_tokens": final_tokens,
            "tokens_saved": current_tokens - final_tokens,
            "optimization_applied": True,
            "strategy_used": strategy.value,
            "messages_removed": len(messages) - len(optimized_messages)
        }
        
        return optimized_messages, optimization_info
    
    def _apply_optimization_strategy(
        self,
        messages: List[Dict[str, str]],
        model: str,
        context_window: ContextWindow,
        strategy: TokenizationStrategy,
        preserve_system: bool
    ) -> List[Dict[str, str]]:
        """Apply specific optimization strategy to messages."""
        
        if strategy == TokenizationStrategy.TRUNCATE_OLDEST:
            return self._truncate_oldest(messages, model, context_window, preserve_system)
        elif strategy == TokenizationStrategy.TRUNCATE_NEWEST:
            return self._truncate_newest(messages, model, context_window, preserve_system)
        elif strategy == TokenizationStrategy.SUMMARIZE_OLDEST:
            return self._summarize_oldest(messages, model, context_window, preserve_system)
        elif strategy == TokenizationStrategy.PRIORITIZE_SYSTEM:
            return self._prioritize_system(messages, model, context_window)
        elif strategy == TokenizationStrategy.ADAPTIVE:
            return self._adaptive_optimization(messages, model, context_window, preserve_system)
        else:
            return self._truncate_oldest(messages, model, context_window, preserve_system)
    
    def _truncate_oldest(
        self,
        messages: List[Dict[str, str]],
        model: str,
        context_window: ContextWindow,
        preserve_system: bool
    ) -> List[Dict[str, str]]:
        """Truncate oldest messages to fit context window."""
        optimized = messages.copy()
        
        system_messages = [msg for msg in messages if msg.get("role") == "system"] if preserve_system else []
        non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        while self.count_message_tokens(system_messages + non_system_messages, model) > context_window.effective_prompt_limit:
            if non_system_messages:
                non_system_messages.pop(0)
            else:
                break
        
        return system_messages + non_system_messages
    
    def _truncate_newest(
        self,
        messages: List[Dict[str, str]],
        model: str,
        context_window: ContextWindow,
        preserve_system: bool
    ) -> List[Dict[str, str]]:
        """Truncate newest messages to fit context window."""
        optimized = messages.copy()
        
        while self.count_message_tokens(optimized, model) > context_window.effective_prompt_limit:
            if len(optimized) > 1:
                if preserve_system and optimized[-1].get("role") == "system":
                    for i in range(len(optimized) - 2, -1, -1):
                        if optimized[i].get("role") != "system":
                            optimized.pop(i)
                            break
                else:
                    optimized.pop()
            else:
                break
        
        return optimized
    
    def _summarize_oldest(
        self,
        messages: List[Dict[str, str]],
        model: str,
        context_window: ContextWindow,
        preserve_system: bool
    ) -> List[Dict[str, str]]:
        """Summarize oldest messages to fit context window."""
        return self._truncate_oldest(messages, model, context_window, preserve_system)
    
    def _prioritize_system(
        self,
        messages: List[Dict[str, str]],
        model: str,
        context_window: ContextWindow
    ) -> List[Dict[str, str]]:
        """Prioritize system messages and recent user/assistant exchanges."""
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]
        
        optimized = system_messages.copy()
        
        for msg in reversed(other_messages):
            test_messages = optimized + [msg]
            if self.count_message_tokens(test_messages, model) <= context_window.effective_prompt_limit:
                optimized.append(msg)
            else:
                break
        
        non_system_in_optimized = [msg for msg in optimized if msg.get("role") != "system"]
        non_system_in_optimized.reverse()
        
        return system_messages + non_system_in_optimized
    
    def _adaptive_optimization(
        self,
        messages: List[Dict[str, str]],
        model: str,
        context_window: ContextWindow,
        preserve_system: bool
    ) -> List[Dict[str, str]]:
        """Apply adaptive optimization based on message content and importance."""
        total_tokens = self.count_message_tokens(messages, model)
        overflow_ratio = total_tokens / context_window.effective_prompt_limit
        
        if overflow_ratio < 1.2:
            return self._truncate_oldest(messages, model, context_window, preserve_system)
        elif overflow_ratio < 2.0:
            return self._prioritize_system(messages, model, context_window)
        else:
            return self._truncate_oldest(messages, model, context_window, preserve_system)
    
    def estimate_cost(self, token_usage: TokenUsage) -> float:
        """
        Estimate cost for token usage.
        
        Args:
            token_usage: Token usage information
            
        Returns:
            Estimated cost in USD
        """
        model_costs = self.token_costs.get(token_usage.model, {"prompt": 0.01, "completion": 0.02})
        
        prompt_cost = (token_usage.prompt_tokens / 1000) * model_costs["prompt"]
        completion_cost = (token_usage.completion_tokens / 1000) * model_costs["completion"]
        
        return prompt_cost + completion_cost
    
    def get_context_window_info(self, model: str) -> Dict[str, Any]:
        """
        Get context window information for a model.
        
        Args:
            model: Model name
            
        Returns:
            Context window information
        """
        context_window = self.context_windows.get(model)
        if not context_window:
            return {"error": f"Unknown model: {model}"}
        
        return {
            "max_tokens": context_window.max_tokens,
            "reserved_for_completion": context_window.reserved_for_completion,
            "effective_prompt_limit": context_window.effective_prompt_limit,
            "utilization_strategies": [strategy.value for strategy in TokenizationStrategy]
        }
    
    def create_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> TokenUsage:
        """
        Create a TokenUsage object with cost estimation.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
            
        Returns:
            TokenUsage object with cost estimate
        """
        total_tokens = prompt_tokens + completion_tokens
        
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model,
            cost_estimate=0.0
        )
        
        usage.cost_estimate = self.estimate_cost(usage)
        return usage
