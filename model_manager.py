"""
Model management for T.O.M. CLI - Updated with PromptCacheManager
"""

import gc
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm import generate, load, sample_utils

from config import (
    MAX_GENERATION_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_GC_FREQUENCY
)
from context_manager import ContextManager, TokenCounter
from tools import TOOLS_DEFINITIONS
from prompt_cache_manager import PromptCacheManager

logger = logging.getLogger("tom_cli")


class ModelManager:
    """Manages model loading, caching, and generation"""
    
    def __init__(
        self,
        model_path: Path,
        context_manager: ContextManager,
        cache_path: Optional[str] = None,
        enable_cache: bool = True,
        max_kv_size: Optional[int] = None,
        auto_size_cache: bool = True,
        kv_bits: Optional[int] = None,
        prewarm: bool = True,
        auto_gc: bool = True,
        gc_frequency: int = DEFAULT_GC_FREQUENCY
    ):
        self.model_path = model_path
        self.context_manager = context_manager
        self.enable_cache = enable_cache
        self.prewarm = prewarm
        self.auto_gc = auto_gc
        self.gc_frequency = gc_frequency
        
        self.model = None
        self.tokenizer = None
        self.cache_manager = None
        self.generation_count = 0
        
        # Cache configuration
        self.cache_path = cache_path or str(self.model_path.parent / "prompt_cache.safetensors")
        self.max_kv_size = max_kv_size
        self.auto_size_cache = auto_size_cache
        self.kv_bits = kv_bits
    
    def load_model(self):
        """Load the MLX model and initialize prompt cache"""
        try:
            self.model, self.tokenizer = load(str(self.model_path))
            self.context_manager.set_tokenizer(self.tokenizer)
            
            logger.info(f"Model loaded from {self.model_path}")
            logger.debug(f"Has chat_template: {hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None}")
            
            if self.enable_cache:
                self._initialize_cache_manager()
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def _initialize_cache_manager(self):
        """Initialize the cache manager with intelligent sizing"""
        # Estimate token counts for cache sizing
        system_tokens = TokenCounter.estimate_tokens(
            self.context_manager.system_prompt,
            self.tokenizer
        )
        
        import json
        tools_tokens = TokenCounter.estimate_tokens(
            json.dumps(TOOLS_DEFINITIONS),
            self.tokenizer
        )
        
        self.cache_manager = PromptCacheManager(
            model=self.model,
            cache_path=self.cache_path,
            max_kv_size=self.max_kv_size,
            auto_size=self.auto_size_cache,
            kv_bits=self.kv_bits,
            system_prompt_tokens=system_tokens,
            tools_tokens=tools_tokens
        )
        
        # Initialize cache (load existing or create new)
        cache_loaded = self.cache_manager.initialize()
        
        # Prewarm if needed and cache wasn't loaded
        if self.prewarm and not cache_loaded:
            self.cache_manager.prewarm(
                self.tokenizer,
                self.context_manager.system_prompt,
                TOOLS_DEFINITIONS
            )
    
    def generate_response(self, include_tools: bool = False) -> tuple[str, str]:
        """
        Generate a single response from the model.
        Returns (thinking_content, content) tuple.
        """
        prompt = self.context_manager.build_prompt(self.tokenizer, include_tools=include_tools)
        
        try:
            sampler = sample_utils.make_sampler(
                temp=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                top_k=DEFAULT_TOP_K
            )
            logits_processors = sample_utils.make_logits_processors(
                repetition_penalty=DEFAULT_REPETITION_PENALTY
            )
            
            generation_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "prompt": prompt,
                "sampler": sampler,
                "logits_processors": logits_processors,
                "max_tokens": MAX_GENERATION_TOKENS
            }
            
            # Add cache kwargs if enabled
            if self.enable_cache and self.cache_manager:
                generation_kwargs.update(self.cache_manager.get_generation_kwargs())
            
            full_response = generate(**generation_kwargs)
            
            # Record cache usage
            if self.cache_manager:
                self.cache_manager.record_generation(hit=True)
            
            self.generation_count += 1
            if self.auto_gc and self.generation_count % self.gc_frequency == 0:
                self.run_gc()
            
            # DO NOT save cache during conversation
            # This prevents conversation history from persisting across sessions
            # Cache is only saved after prewarm (static content only)
            # Save cache after successful generation
            #if self.cache_manager:
            #    self.cache_manager.save()
            
            # Parse thinking content from actual content
            thinking_content, content = self._parse_thinking_and_content(full_response)
            
            return thinking_content, content
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            
            # Record cache miss on error
            if self.cache_manager:
                self.cache_manager.record_generation(hit=False)
            
            return "", "Sorry, I encountered an error generating a response."
    
    def _parse_thinking_and_content(self, full_response: str) -> tuple[str, str]:
        """
        Parse thinking content from actual response content.
        Returns (thinking_content, content) tuple.
        """
        try:
            tokens = self.tokenizer.encode(full_response)
            
            # Find the last occurrence of 151668 (</think>)
            try:
                reversed_tokens = tokens[::-1]
                index = len(tokens) - reversed_tokens.index(151668)
            except ValueError:
                # No </think> token found
                return "", full_response
            
            # Decode thinking content
            thinking_content = self.tokenizer.decode(
                tokens[:index],
                skip_special_tokens=True
            ).strip("\n")
            
            if thinking_content.startswith("<think>"):
                thinking_content = thinking_content[7:].strip()
            
            # Decode actual content
            content = self.tokenizer.decode(
                tokens[index:],
                skip_special_tokens=True
            ).strip("\n")
            
            return thinking_content, content
            
        except Exception as e:
            logger.debug(f"Error parsing thinking content: {e}")
            return "", full_response
    
    def handle_context_trim(self, trim_percentage: float):
        """
        Handle cache after context trimming.
        
        Args:
            trim_percentage: Fraction of messages trimmed (0.0 to 1.0)
        """
        if not self.cache_manager:
            return
        
        should_reset = self.cache_manager.should_reset_on_trim(trim_percentage)
        
        if should_reset:
            self.cache_manager.reset()
            
            # Re-prewarm after reset
            if self.prewarm:
                self.cache_manager.prewarm(
                    self.tokenizer,
                    self.context_manager.system_prompt,
                    TOOLS_DEFINITIONS
                )
    
    def reset_cache(self):
        """Manually reset the prompt cache"""
        if self.cache_manager:
            self.cache_manager.reset()
            self.run_gc()
            logger.info("Cache reset")
    
    def run_gc(self):
        """Force garbage collection and MLX memory cleanup"""
        gc.collect()
        mx.clear_cache()
    
    def get_cache_info(self) -> dict:
        """Get cache information"""
        if not self.enable_cache or not self.cache_manager:
            return {
                "enabled": False,
                "path": self.cache_path
            }
        
        return self.cache_manager.get_stats()