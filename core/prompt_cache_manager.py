"""
Prompt Cache Manager for T.O.M. CLI
Handles cache lifecycle with intelligent sizing
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import mlx.core as mx
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

logger = logging.getLogger("tom_cli")


class PromptCacheManager:
    """
    Manages prompt cache with automatic sizing and monitoring.
    
    Features:
    - Automatic cache sizing based on static content (system + tools)
    - Cache persistence across sessions
    - Hit/miss tracking for monitoring
    - Optional quantization for memory efficiency
    """
    
    def __init__(
        self,
        model,
        cache_path: str,
        max_kv_size: Optional[int] = None,
        auto_size: bool = True,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        system_prompt_tokens: int = 0,
        tools_tokens: int = 0
    ):
        """
        Initialize cache manager.
        
        Args:
            model: MLX model instance
            cache_path: Path to save/load cache file
            max_kv_size: Maximum tokens to cache (None = unlimited)
            auto_size: Automatically size cache based on static content
            kv_bits: Quantize cache (4, 8, or None)
            kv_group_size: Quantization group size
            system_prompt_tokens: Estimated system prompt tokens
            tools_tokens: Estimated tools definition tokens
        """
        self.model = model
        self.cache_path = Path(cache_path)
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        
        # Calculate cache size
        if max_kv_size is not None:
            self.max_kv_size = max_kv_size
        elif auto_size:
            # Size cache for static content + small conversation window
            static_tokens = system_prompt_tokens + tools_tokens
            conversation_buffer = 2000
            self.max_kv_size = static_tokens + conversation_buffer
        else:
            self.max_kv_size = None
        
        # Statistics
        self.cache = None
        self.generations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Cache initialized: max_kv_size={self.max_kv_size}, kv_bits={self.kv_bits}")
    
    def initialize(self) -> bool:
        """
        Initialize or load cache.
        Returns True if existing cache was loaded.
        """
        # Try to load existing cache
        if self.cache_path.exists():
            try:
                logger.info(f"Loading cache from {self.cache_path}")
                self.cache = load_prompt_cache(str(self.cache_path))
                return True
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Create new cache
        logger.info("Creating new cache")
        cache_kwargs = {}
        if self.max_kv_size:
            cache_kwargs["max_kv_size"] = self.max_kv_size
        
        self.cache = make_prompt_cache(self.model, **cache_kwargs)
        return False
    
    def prewarm(self, tokenizer, system_prompt: str, tools_definitions: list):
        """
        Prewarm cache with static content (system prompt + tools).
        This ensures the static parts are always cached.
        """
        if not self.cache:
            logger.warning("Cannot prewarm: cache not initialized")
            return
        
        try:
            logger.info("Prewarming cache...")
            
            # Build minimal prompt with system + tools
            chat_messages = [{"role": "system", "content": system_prompt}]
            
            prompt = None
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                try:
                    prompt = tokenizer.apply_chat_template(
                        chat_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        tools=tools_definitions
                    )
                except Exception:
                    pass
            
            if not prompt:
                import json
                tools_str = json.dumps(tools_definitions)
                prompt = f"System: {system_prompt}\n\nTools: {tools_str}"
            
            # Generate 1 token to populate cache
            from mlx_lm import generate, sample_utils
            
            gen_kwargs = {
                "model": self.model,
                "tokenizer": tokenizer,
                "prompt": prompt,
                "sampler": sample_utils.make_sampler(temp=0.7),
                "max_tokens": 1,
                "prompt_cache": self.cache
            }
            
            if self.kv_bits:
                gen_kwargs["kv_bits"] = self.kv_bits
                gen_kwargs["kv_group_size"] = self.kv_group_size
            
            _ = generate(**gen_kwargs)
            
            # Save prewarmed cache
            self.save()
            logger.info("Cache prewarmed successfully")
            
        except Exception as e:
            logger.warning(f"Cache prewarm failed: {e}")
    
    def save(self):
        """Save cache to disk"""
        if not self.cache:
            return
        
        try:
            save_prompt_cache(str(self.cache_path), self.cache)
            logger.debug(f"Cache saved to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def reset(self):
        """Reset cache completely"""
        cache_kwargs = {}
        if self.max_kv_size:
            cache_kwargs["max_kv_size"] = self.max_kv_size
        
        self.cache = make_prompt_cache(self.model, **cache_kwargs)
        mx.clear_cache()
        
        logger.info("Cache reset")
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for generate() to use cache"""
        kwargs = {}
        
        if self.cache:
            kwargs["prompt_cache"] = self.cache
        
        if self.kv_bits:
            kwargs["kv_bits"] = self.kv_bits
            kwargs["kv_group_size"] = self.kv_group_size
        
        return kwargs
    
    def record_generation(self, hit: bool = True):
        """Record cache hit/miss for statistics"""
        self.generations += 1
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "enabled": self.cache is not None,
            "path": str(self.cache_path),
            "max_kv_size": self.max_kv_size or "unlimited",
            "kv_bits": self.kv_bits or "no quantization",
            "generations": self.generations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": (self.cache_hits / self.generations * 100) if self.generations > 0 else 0.0
        }
        
        if self.cache_path.exists():
            stats["size_mb"] = self.cache_path.stat().st_size / (1024 * 1024)
        
        return stats
    
    def should_reset_on_trim(self, trim_percentage: float) -> bool:
        """
        Decide if cache should reset after context trimming.
        Only reset if we have a size limit and major trim occurred.
        """
        if not self.max_kv_size:
            return False
        
        return trim_percentage > 0.5