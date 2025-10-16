"""
Model management for T.O.M. CLI
"""

import gc
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm import generate, load, sample_utils
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

from config import (
    MAX_GENERATION_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_GC_FREQUENCY
)
from context_manager import ContextManager
from tools import TOOLS_DEFINITIONS

logger = logging.getLogger("tom_cli")


class ModelManager:
    """Manages model loading, caching, and generation"""
    
    def __init__(
        self,
        model_path: Path,
        context_manager: ContextManager,
        cache_path: Optional[str] = None,
        enable_cache: bool = True,
        prewarm: bool = True,
        auto_gc: bool = True,
        gc_frequency: int = DEFAULT_GC_FREQUENCY
    ):
        self.model_path = model_path
        self.context_manager = context_manager
        self.cache_path = cache_path or str(self.model_path.parent / "prompt_cache.safetensors")
        self.enable_cache = enable_cache
        self.prewarm = prewarm
        self.auto_gc = auto_gc
        self.gc_frequency = gc_frequency
        
        self.model = None
        self.tokenizer = None
        self.prompt_cache = None
        self.generation_count = 0
    
    def load_model(self):
        """Load the MLX model and initialize prompt cache"""
        try:
            self.model, self.tokenizer = load(str(self.model_path))
            self.context_manager.set_tokenizer(self.tokenizer)
            
            logger.info(f"Model loaded from {self.model_path}")
            logger.debug(f"Has chat_template: {hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None}")
            
            if self.enable_cache:
                self._initialize_prompt_cache()
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise
    
    def _initialize_prompt_cache(self):
        """Initialize or load the prompt cache"""
        cache_file = Path(self.cache_path)
        
        if cache_file.exists():
            try:
                logger.info(f"Loading existing prompt cache")
                self.prompt_cache = load_prompt_cache(self.cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache, creating new one: {e}")
                self.prompt_cache = make_prompt_cache(self.model)
                if self.prewarm:
                    self._prewarm_cache()
        else:
            self.prompt_cache = make_prompt_cache(self.model)
            if self.prewarm:
                self._prewarm_cache()
    
    def _prewarm_cache(self):
        """Prewarm the cache by processing the system prompt"""
        if not self.prompt_cache:
            return
        
        try:
            logger.info("Prewarming cache...")
            chat_messages = [{"role": "system", "content": self.context_manager.system_prompt}]
            
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        chat_messages, tokenize=False, add_generation_prompt=False, tools=TOOLS_DEFINITIONS
                    )
                except Exception:
                    prompt = f"System: {self.context_manager.system_prompt}"
            else:
                prompt = f"System: {self.context_manager.system_prompt}"
            
            sampler = sample_utils.make_sampler(temp=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, top_k=DEFAULT_TOP_K)
            _ = generate(
                model=self.model, tokenizer=self.tokenizer, prompt=prompt,
                sampler=sampler, max_tokens=1, prompt_cache=self.prompt_cache
            )
            
            self.save_prompt_cache()
            logger.info("Cache prewarmed")
        except Exception as e:
            logger.warning(f"Failed to prewarm cache: {e}")
    
    def save_prompt_cache(self):
        """Save the prompt cache to disk"""
        if self.prompt_cache and self.enable_cache:
            try:
                save_prompt_cache(self.cache_path, self.prompt_cache)
            except Exception as e:
                logger.warning(f"Failed to save prompt cache: {e}")
    
    def reset_cache(self):
        """Reset the prompt cache"""
        if self.enable_cache:
            self.prompt_cache = make_prompt_cache(self.model)
            self.run_gc()
            logger.info("Cache reset")
    
    def generate_response(self, include_tools: bool = False) -> tuple[str, str]:
        """
        Generate a single response from the model.
        Returns (thinking_content, content) tuple.
        """
        prompt = self.context_manager.build_prompt(self.tokenizer, include_tools=include_tools)
        
        try:
            sampler = sample_utils.make_sampler(temp=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P, top_k=DEFAULT_TOP_K)
            logits_processors = sample_utils.make_logits_processors(repetition_penalty=DEFAULT_REPETITION_PENALTY)
            
            generation_kwargs = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "prompt": prompt,
                "sampler": sampler,
                "logits_processors": logits_processors,
                "max_tokens": MAX_GENERATION_TOKENS
            }
            
            if self.enable_cache and self.prompt_cache is not None:
                generation_kwargs["prompt_cache"] = self.prompt_cache
            
            full_response = generate(**generation_kwargs)
            
            self.generation_count += 1
            if self.auto_gc and self.generation_count % self.gc_frequency == 0:
                self.run_gc()
            
            if self.enable_cache:
                self.save_prompt_cache()
            
            # Parse thinking content from actual content
            thinking_content, content = self._parse_thinking_and_content(full_response)
            
            return thinking_content, content
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            return "", "Sorry, I encountered an error generating a response."
    
    def _parse_thinking_and_content(self, full_response: str) -> tuple[str, str]:
        """
        Parse thinking content from actual response content.
        Returns (thinking_content, content) tuple.
        """
        # Token ID for </think> is 151668
        try:
            tokens = self.tokenizer.encode(full_response)
            
            # Find the last occurrence of 151668 (</think>)
            try:
                reversed_tokens = tokens[::-1]
                index = len(tokens) - reversed_tokens.index(151668)
            except ValueError:
                # No </think> token found, no thinking content
                return "", full_response
            
            # Decode thinking content (everything before </think>)
            thinking_content = self.tokenizer.decode(tokens[:index], skip_special_tokens=True).strip("\n")
            # Remove opening <think> tag if present
            if thinking_content.startswith("<think>"):
                thinking_content = thinking_content[7:].strip()
            
            # Decode actual content (everything after </think>)
            content = self.tokenizer.decode(tokens[index:], skip_special_tokens=True).strip("\n")
            
            return thinking_content, content
            
        except Exception as e:
            logger.debug(f"Error parsing thinking content: {e}, returning full response")
            return "", full_response
    
    def run_gc(self):
        """Force garbage collection and MLX memory cleanup"""
        gc.collect()
        mx.clear_cache()
    
    def get_cache_info(self) -> dict:
        """Get cache information"""
        cache_file = Path(self.cache_path)
        
        info = {
            "enabled": self.enable_cache,
            "path": self.cache_path,
            "loaded": self.prompt_cache is not None,
            "exists": cache_file.exists()
        }
        
        if cache_file.exists():
            info["size_mb"] = cache_file.stat().st_size / (1024 * 1024)
        
        return info
