"""
Model management for T.O.M. CLI - Updated with streaming support
"""

import gc
import logging
from pathlib import Path
from typing import Optional, Generator, Dict, Any

import mlx.core as mx
from mlx_lm import generate, stream_generate, load, sample_utils

from config import (
    MAX_GENERATION_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_GC_FREQUENCY,
    ENABLE_STREAMING
)
from context_manager import ContextManager, TokenCounter
from tools import TOOLS_DEFINITIONS
from prompt_cache_manager import PromptCacheManager

logger = logging.getLogger("tom_cli")


class ModelManager:
    """Manages model loading, caching, and generation with streaming support"""
    
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
    
    def stream_response(
        self, 
        include_tools: bool = False
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream a response from the model token by token.
        
        Yields dictionaries with:
            {
                'type': 'thinking' | 'content' | 'done',
                'text': str,  # Full accumulated text so far
                'delta': str,  # New text in this chunk
                'tokens': list  # Token IDs generated so far
            }
        
        The 'done' type is yielded at the end with the complete response.
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
            
            # Stream generation
            full_response_text = ""
            thinking_complete = False
            thinking_text = ""
            content_text = ""
            
            for response in stream_generate(**generation_kwargs):
                # stream_generate yields response objects where .text is the NEW token text (delta)
                # We need to accumulate it ourselves
                delta = response.text
                full_response_text += delta
                
                # Check if we're transitioning out of thinking mode
                # Look for </think> marker in the accumulated text
                if not thinking_complete and '</think>' in full_response_text:
                    thinking_complete = True
                    
                    # Split at </think> to separate thinking from content
                    parts = full_response_text.split('</think>', 1)
                    thinking_raw = parts[0]
                    content_text = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Clean up thinking text
                    if thinking_raw.startswith('<think>'):
                        thinking_text = thinking_raw[7:].strip()
                    else:
                        thinking_text = thinking_raw.strip()
                    
                    # Yield complete thinking
                    yield {
                        'type': 'thinking',
                        'text': thinking_text,
                        'delta': '',  # No more thinking deltas
                        'tokens': [],
                        'complete': True
                    }
                    
                    # If there's already content, yield it
                    if content_text:
                        yield {
                            'type': 'content',
                            'text': content_text,
                            'delta': content_text,
                            'tokens': [],
                            'complete': False
                        }
                
                elif not thinking_complete:
                    # Still in thinking mode - yield the delta
                    yield {
                        'type': 'thinking',
                        'text': full_response_text,
                        'delta': delta,
                        'tokens': [],
                        'complete': False
                    }
                
                else:
                    # In content mode - extract content from full text
                    if '</think>' in full_response_text:
                        parts = full_response_text.split('</think>', 1)
                        current_content = parts[1].strip() if len(parts) > 1 else ""
                    else:
                        current_content = full_response_text
                    
                    # Calculate delta for content only
                    content_delta = current_content[len(content_text):]
                    content_text = current_content
                    
                    yield {
                        'type': 'content',
                        'text': current_content,
                        'delta': content_delta,
                        'tokens': [],
                        'complete': False
                    }
            
            # Final yield with complete response
            # Parse final thinking and content
            if '</think>' in full_response_text:
                parts = full_response_text.split('</think>', 1)
                thinking_raw = parts[0]
                final_content = parts[1].strip() if len(parts) > 1 else ""
                
                if thinking_raw.startswith('<think>'):
                    final_thinking = thinking_raw[7:].strip()
                else:
                    final_thinking = thinking_raw.strip()
            else:
                final_thinking = ""
                final_content = full_response_text.strip()
            
            yield {
                'type': 'done',
                'thinking': final_thinking,
                'content': final_content,
                'full_text': full_response_text,
                'tokens': [],
                'complete': True
            }
            
            # Record cache usage
            if self.cache_manager:
                self.cache_manager.record_generation(hit=True)
            
            self.generation_count += 1
            if self.auto_gc and self.generation_count % self.gc_frequency == 0:
                self.run_gc()
                
        except Exception as e:
            logger.error(f"Streaming generation error: {e}", exc_info=True)
            
            # Record cache miss on error
            if self.cache_manager:
                self.cache_manager.record_generation(hit=False)
            
            # Yield error
            yield {
                'type': 'error',
                'text': "Sorry, I encountered an error generating a response.",
                'delta': '',
                'tokens': [],
                'error': str(e)
            }
    
    def generate_response(self, include_tools: bool = False) -> tuple[str, str]:
        """
        Generate a single response from the model (non-streaming fallback).
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
            # Look for </think> marker
            if '</think>' in full_response:
                parts = full_response.split('</think>', 1)
                thinking_raw = parts[0]
                content = parts[1].strip() if len(parts) > 1 else ""
                
                # Clean up thinking text
                if thinking_raw.startswith('<think>'):
                    thinking_content = thinking_raw[7:].strip()
                else:
                    thinking_content = thinking_raw.strip()
                
                return thinking_content, content
            else:
                # No thinking content
                return "", full_response.strip()
            
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