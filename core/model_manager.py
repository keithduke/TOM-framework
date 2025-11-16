"""
Model management for T.O.M. CLI
Handles model loading, generation, and caching
"""

import gc
import logging
from pathlib import Path
from typing import Optional, Generator, Dict, Any

import mlx.core as mx
from mlx_lm import generate, stream_generate, load, sample_utils

from .config import (
    MAX_GENERATION_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_GC_FREQUENCY,
)
from .context_manager import ContextManager, TokenCounter
from .tools import TOOLS_DEFINITIONS
from .prompt_cache_manager import PromptCacheManager

logger = logging.getLogger("tom_cli")


class ModelManager:
    """
    Manages model loading, caching, and generation.
    
    Features:
    - Model and tokenizer loading
    - Prompt cache management
    - Streaming and non-streaming generation
    - Automatic garbage collection
    - Thinking/content separation
    """
    
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
        
        # Cache config
        self.cache_path = cache_path or str(self.model_path.parent / "prompt_cache.safetensors")
        self.max_kv_size = max_kv_size
        self.auto_size_cache = auto_size_cache
        self.kv_bits = kv_bits
    
    def load_model(self):
        """Load model and initialize cache"""
        try:
            self.model, self.tokenizer = load(str(self.model_path))
            self.context_manager.set_tokenizer(self.tokenizer)
            
            logger.info(f"Model loaded from {self.model_path}")
            
            if self.enable_cache:
                self._initialize_cache()
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def _initialize_cache(self):
        """Initialize prompt cache with intelligent sizing"""
        # Estimate static content tokens
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
        
        # Load or create cache
        cache_loaded = self.cache_manager.initialize()
        
        # Prewarm if needed
        if self.prewarm and not cache_loaded:
            self.cache_manager.prewarm(
                self.tokenizer,
                self.context_manager.system_prompt,
                TOOLS_DEFINITIONS
            )
    
    def stream_response(self, include_tools: bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Stream response token by token.
        
        Yields dicts with:
            - type: 'thinking' | 'content' | 'done' | 'error'
            - text: accumulated text
            - delta: new text in this chunk
            - complete: whether this segment is complete
        """
        prompt = self.context_manager.build_prompt(self.tokenizer, include_tools=include_tools)
        
        try:
            # Setup generation
            sampler = sample_utils.make_sampler(
                temp=DEFAULT_TEMPERATURE,
                top_p=DEFAULT_TOP_P,
                top_k=DEFAULT_TOP_K
            )
            logits_processors = sample_utils.make_logits_processors(
                repetition_penalty=DEFAULT_REPETITION_PENALTY
            )
            
            gen_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "prompt": prompt,
                "sampler": sampler,
                "logits_processors": logits_processors,
                "max_tokens": MAX_GENERATION_TOKENS
            }
            
            # Add cache if enabled
            if self.enable_cache and self.cache_manager:
                gen_kwargs.update(self.cache_manager.get_generation_kwargs())
            
            # Stream and parse
            full_text = ""
            thinking_complete = False
            thinking_text = ""
            content_text = ""
            
            for response in stream_generate(**gen_kwargs):
                delta = response.text
                full_text += delta
                
                # Check for thinking completion
                if not thinking_complete and '</think>' in full_text:
                    thinking_complete = True
                    
                    # Split thinking from content
                    parts = full_text.split('</think>', 1)
                    thinking_raw = parts[0]
                    content_text = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Clean thinking
                    thinking_text = thinking_raw.replace('<think>', '').strip()
                    
                    # Yield complete thinking
                    yield {
                        'type': 'thinking',
                        'text': thinking_text,
                        'delta': '',
                        'complete': True
                    }
                    
                    # Yield initial content if any
                    if content_text:
                        yield {
                            'type': 'content',
                            'text': content_text,
                            'delta': content_text,
                            'complete': False
                        }
                
                elif not thinking_complete:
                    # Still thinking
                    yield {
                        'type': 'thinking',
                        'text': full_text,
                        'delta': delta,
                        'complete': False
                    }
                
                else:
                    # Content mode - calculate new content
                    current_content = full_text.split('</think>', 1)[1].strip() if '</think>' in full_text else full_text
                    content_delta = current_content[len(content_text):]
                    content_text = current_content
                    
                    yield {
                        'type': 'content',
                        'text': content_text,
                        'delta': content_delta,
                        'complete': False
                    }
            
            # Parse final response
            final_thinking, final_content = self._parse_thinking_and_content(full_text)
            
            yield {
                'type': 'done',
                'thinking': final_thinking,
                'content': final_content,
                'complete': True
            }
            
            # Record stats and GC
            if self.cache_manager:
                self.cache_manager.record_generation(hit=True)
            
            self.generation_count += 1
            if self.auto_gc and self.generation_count % self.gc_frequency == 0:
                self.run_gc()
                
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            
            if self.cache_manager:
                self.cache_manager.record_generation(hit=False)
            
            yield {
                'type': 'error',
                'text': f"Error: {str(e)}",
                'delta': '',
                'complete': True
            }
    
    def generate_response(self, include_tools: bool = False) -> tuple[str, str]:
        """
        Generate complete response (non-streaming).
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
            
            gen_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "prompt": prompt,
                "sampler": sampler,
                "logits_processors": logits_processors,
                "max_tokens": MAX_GENERATION_TOKENS
            }
            
            if self.enable_cache and self.cache_manager:
                gen_kwargs.update(self.cache_manager.get_generation_kwargs())
            
            full_response = generate(**gen_kwargs)
            
            if self.cache_manager:
                self.cache_manager.record_generation(hit=True)
            
            self.generation_count += 1
            if self.auto_gc and self.generation_count % self.gc_frequency == 0:
                self.run_gc()
            
            return self._parse_thinking_and_content(full_response)
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            
            if self.cache_manager:
                self.cache_manager.record_generation(hit=False)
            
            return "", f"Error: {str(e)}"
    
    def _parse_thinking_and_content(self, full_response: str) -> tuple[str, str]:
        """
        Parse thinking from content.
        Returns (thinking, content) tuple.
        """
        if '</think>' in full_response:
            parts = full_response.split('</think>', 1)
            thinking_raw = parts[0]
            content = parts[1].strip() if len(parts) > 1 else ""
            
            thinking = thinking_raw.replace('<think>', '').strip()
            return thinking, content
        
        return "", full_response.strip()
    
    def reset_cache(self):
        """Manually reset cache"""
        if self.cache_manager:
            self.cache_manager.reset()
            self.run_gc()
            logger.info("Cache reset")
    
    def run_gc(self):
        """Force garbage collection"""
        gc.collect()
        mx.clear_cache()
        logger.debug("GC complete")
    
    def get_cache_info(self) -> dict:
        """Get cache statistics"""
        if not self.enable_cache or not self.cache_manager:
            return {
                "enabled": False,
                "path": self.cache_path
            }
        
        return self.cache_manager.get_stats()
