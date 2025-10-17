"""
Prompt Cache Manager for T.O.M. CLI
Handles intelligent cache sizing, monitoring, and lifecycle management
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import mlx.core as mx
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

logger = logging.getLogger("tom_cli")


class PromptCacheManager:
    """
    Manages prompt cache with intelligent sizing and monitoring.
    
    Key features:
    - Automatic cache sizing based on model and system constraints
    - Cache hit/miss tracking
    - Memory-efficient quantization options
    - Selective cache reset (partial vs full)
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
            model: The MLX model
            cache_path: Path to save/load cache
            max_kv_size: Maximum tokens to cache (None = unlimited)
            auto_size: Automatically determine optimal cache size
            kv_bits: Quantize cache to N bits (4, 8, or None for no quantization)
            kv_group_size: Quantization group size
            system_prompt_tokens: Estimated system prompt token count
            tools_tokens: Estimated tools definition token count
        """
        self.model = model
        self.cache_path = Path(cache_path)
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        
        # Calculate optimal cache size
        self.max_kv_size = self._calculate_cache_size(
            max_kv_size, auto_size, system_prompt_tokens, tools_tokens
        )
        
        # Initialize cache
        self.cache = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.generations = 0
        
        logger.info(f"PromptCacheManager initialized:")
        logger.info(f"  max_kv_size: {self.max_kv_size if self.max_kv_size else 'unlimited'}")
        logger.info(f"  kv_bits: {self.kv_bits if self.kv_bits else 'no quantization'}")
        logger.info(f"  cache_path: {self.cache_path}")
    
    def _calculate_cache_size(
        self,
        max_kv_size: Optional[int],
        auto_size: bool,
        system_prompt_tokens: int,
        tools_tokens: int
    ) -> Optional[int]:
        """
        Calculate optimal cache size.
        
        Strategy:
        1. If max_kv_size provided, use it
        2. If auto_size=True, calculate based on:
           - System prompt + tools (always cached)
           - Recent conversation window (last ~10 messages)
           - Buffer for tool results
        3. Otherwise, unlimited
        """
        if max_kv_size is not None:
            return max_kv_size
        
        if not auto_size:
            return None
        
        # Auto-size strategy:
        # System + Tools + ~2000 tokens for recent conversation + 1000 buffer
        static_tokens = system_prompt_tokens + tools_tokens
        conversation_window = 2000
        buffer = 1000
        
        optimal_size = static_tokens + conversation_window + buffer
        
        logger.info(f"Auto-sized cache: {optimal_size} tokens")
        logger.info(f"  Static (system+tools): {static_tokens}")
        logger.info(f"  Conversation window: {conversation_window}")
        logger.info(f"  Buffer: {buffer}")
        
        return optimal_size
    
    def initialize(self) -> bool:
        """
        Initialize or load the prompt cache.
        Returns True if existing cache was loaded.
        """
        if self.cache_path.exists():
            try:
                logger.info(f"Loading existing prompt cache from {self.cache_path}")
                self.cache = load_prompt_cache(str(self.cache_path))
                return True
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, creating new one")
        
        logger.info("Creating new prompt cache")
        cache_kwargs = {}
        if self.max_kv_size:
            cache_kwargs["max_kv_size"] = self.max_kv_size
        
        self.cache = make_prompt_cache(self.model, **cache_kwargs)
        return False
    
    def prewarm(self, tokenizer, system_prompt: str, tools_definitions: list):
        """
        Prewarm cache with system prompt and tools.
        This ensures static context is always cached.
        """
        if not self.cache:
            logger.warning("Cannot prewarm: cache not initialized")
            return
        
        try:
            logger.info("Prewarming cache with system prompt and tools...")
            
            # Build minimal prompt with just system + tools
            chat_messages = [{"role": "system", "content": system_prompt}]
            
            if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=tools_definitions
                )
            else:
                prompt = f"System: {system_prompt}"
            
            # Do a minimal generation to populate cache
            from mlx_lm import generate, sample_utils
            sampler = sample_utils.make_sampler(temp=0.7, top_p=0.9)
            
            gen_kwargs = {
                "model": self.model,
                "tokenizer": tokenizer,
                "prompt": prompt,
                "sampler": sampler,
                "max_tokens": 1,
                "prompt_cache": self.cache
            }
            
            if self.kv_bits:
                gen_kwargs["kv_bits"] = self.kv_bits
                gen_kwargs["kv_group_size"] = self.kv_group_size
            
            _ = generate(**gen_kwargs)
            
            self.save()
            logger.info("Cache prewarmed successfully")
            
        except Exception as e:
            logger.warning(f"Failed to prewarm cache: {e}")
    
    def save(self):
        """Save cache to disk."""
        if not self.cache:
            return
        
        try:
            save_prompt_cache(str(self.cache_path), self.cache)
            logger.debug(f"Cache saved to {self.cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def reset(self, partial: bool = False):
        """
        Reset the cache.
        
        Args:
            partial: If True, try to preserve system prompt in cache.
                    If False, completely reset cache.
        """
        if partial and self.cache:
            # For partial reset, we could implement logic to:
            # 1. Extract cached system prompt KV pairs
            # 2. Create new cache
            # 3. Restore system prompt KV pairs
            # This is complex and may not be worth it - just full reset for now
            logger.info("Partial cache reset requested, performing full reset")
        
        cache_kwargs = {}
        if self.max_kv_size:
            cache_kwargs["max_kv_size"] = self.max_kv_size
        
        self.cache = make_prompt_cache(self.model, **cache_kwargs)
        
        # Clear MLX memory
        mx.clear_cache()
        
        logger.info("Cache reset complete")
    
    def get_generation_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs to pass to generate() for cache usage.
        """
        kwargs = {}
        
        if self.cache:
            kwargs["prompt_cache"] = self.cache
        
        if self.kv_bits:
            kwargs["kv_bits"] = self.kv_bits
            kwargs["kv_group_size"] = self.kv_group_size
        
        return kwargs
    
    def record_generation(self, hit: bool = True):
        """
        Record cache hit/miss statistics.
        
        Args:
            hit: True if cache was useful, False if cache miss
        """
        self.generations += 1
        
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "enabled": self.cache is not None,
            "path": str(self.cache_path),
            "max_kv_size": self.max_kv_size if self.max_kv_size else "unlimited",
            "kv_bits": self.kv_bits if self.kv_bits else "no quantization",
            "generations": self.generations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        }
        
        if self.generations > 0:
            stats["hit_rate"] = (self.cache_hits / self.generations) * 100
        else:
            stats["hit_rate"] = 0.0
        
        if self.cache_path.exists():
            stats["size_mb"] = self.cache_path.stat().st_size / (1024 * 1024)
        
        return stats
    
    def should_reset_on_trim(self, trim_percentage: float) -> bool:
        """
        Decide if cache should be reset based on context trimming.
        
        Args:
            trim_percentage: Percentage of messages trimmed (0.0 to 1.0)
        
        Returns:
            True if cache should be reset
        
        Strategy:
        - If >50% of messages trimmed and we have max_kv_size, reset
        - Otherwise keep cache (recent messages likely still cached)
        """
        if not self.max_kv_size:
            # Unlimited cache - no need to reset
            return False
        
        # Reset if major trim occurred
        should_reset = trim_percentage > 0.5
        
        if should_reset:
            logger.info(f"Major trim ({trim_percentage*100:.0f}%), resetting cache")
        
        return should_reset