"""
Context management for T.O.M. CLI
"""

import json
import logging
from typing import Dict, List

from config import DEFAULT_SYSTEM_PROMPT
from tools import TOOLS_DEFINITIONS

logger = logging.getLogger("tom_cli")


class TokenCounter:
    """Utility class for token counting"""
    
    @staticmethod
    def estimate_tokens(text: str, tokenizer=None) -> int:
        if tokenizer is not None:
            try:
                tokens = tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                logger.debug(f"Tokenizer encode failed: {e}, using heuristic")
        return len(text) // 4  # Rough estimation
    
    @staticmethod
    def count_tokens_in_messages(messages: List[Dict[str, str]], tokenizer=None) -> int:
        """Count tokens in a list of messages"""
        return sum(TokenCounter.estimate_tokens(str(msg), tokenizer) for msg in messages)


class ContextManager:
    """Manages conversation context with intelligent trimming"""
    
    def __init__(self, max_context_tokens: int, tokenizer=None):
        self.max_context_tokens = max_context_tokens
        self.tokenizer = tokenizer
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for accurate token counting"""
        self.tokenizer = tokenizer
        logger.debug("Tokenizer set for accurate token counting")
    
    def add_message(self, role: str, content: str) -> bool:
        """Add message and trim if necessary. Returns True if significant trimming occurred."""
        self.messages.append({"role": role, "content": content})
        logger.debug(f"Added {role} message: {content[:100]}...")
        return self._trim_if_needed()
    
    def _trim_if_needed(self) -> bool:
        """Trim older messages if context exceeds limit. Returns True if >25% trimmed."""
        current_tokens = self._count_total_tokens()
        
        if current_tokens > self.max_context_tokens:
            original_count = len(self.messages)
            target_tokens = int(self.max_context_tokens * 0.8)
            
            while current_tokens > target_tokens and len(self.messages) > 2:
                removed = self.messages.pop(0)
                current_tokens = self._count_total_tokens()
                logger.debug(f"Trimmed message: {removed['role']}")
            
            trimmed_count = original_count - len(self.messages)
            if trimmed_count > 0:
                logger.info(f"Trimmed {trimmed_count} messages. Context: {current_tokens:,} tokens")
                return trimmed_count > (original_count * 0.25)
        return False
    
    def _count_total_tokens(self) -> int:
        system_tokens = TokenCounter.estimate_tokens(self.system_prompt, self.tokenizer)
        message_tokens = TokenCounter.count_tokens_in_messages(self.messages, self.tokenizer)
        tool_tokens = TokenCounter.estimate_tokens(json.dumps(TOOLS_DEFINITIONS), self.tokenizer)
        return system_tokens + message_tokens + tool_tokens
    
    def build_prompt(self, tokenizer, include_tools: bool = False) -> str:
        """Build the complete prompt using proper chat template"""
        chat_messages = [{"role": "system", "content": self.system_prompt}] + self.messages
        
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            try:
                kwargs = {"tokenize": False, "add_generation_prompt": True}
                if include_tools:
                    kwargs["tools"] = TOOLS_DEFINITIONS
                prompt = tokenizer.apply_chat_template(chat_messages, **kwargs)
                logger.debug("Using tokenizer's chat template")
            except Exception as e:
                logger.debug(f"Chat template failed: {e}")
                prompt = self._build_fallback_prompt(chat_messages)
        else:
            prompt = self._build_fallback_prompt(chat_messages)
        
        return prompt
    
    def _build_fallback_prompt(self, chat_messages: List[Dict[str, str]]) -> str:
        parts = []
        for msg in chat_messages:
            role = msg["role"]
            if role == "system":
                parts.append(f"System: {msg['content']}")
            elif role == "user":
                parts.append(f"User: {msg['content']}")
            elif role == "assistant":
                parts.append(f"Assistant: {msg['content']}")
            elif role == "tool":
                parts.append(f"Tool Result: {msg['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    def get_stats(self) -> Dict[str, any]:
        """Get context statistics"""
        total_tokens = self._count_total_tokens()
        msg_count = len(self.messages)
        
        return {
            "message_count": msg_count,
            "total_tokens": total_tokens,
            "max_tokens": self.max_context_tokens,
            "usage_percent": (total_tokens / self.max_context_tokens) * 100
        }
