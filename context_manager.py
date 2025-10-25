"""
Context management for T.O.M. CLI
"""

import json
import logging
from typing import Dict, List, Optional

from config import DEFAULT_SYSTEM_PROMPT
from tools import TOOLS_DEFINITIONS

logger = logging.getLogger("tom_cli")


class TokenCounter:
    """Utility class for token counting with fallback estimation"""
    
    @staticmethod
    def estimate_tokens(text: str, tokenizer=None) -> int:
        """
        Count tokens in text.
        Uses tokenizer if available, otherwise estimates 1 token ≈ 4 characters.
        """
        if tokenizer is not None:
            try:
                tokens = tokenizer.encode(text)
                return len(tokens)
            except Exception as e:
                logger.debug(f"Tokenizer encode failed: {e}, using estimation")
        
        # Fallback estimation - rough but consistent
        return max(1, len(text) // 4)
    
    @staticmethod
    def count_tokens_in_messages(messages: List[Dict[str, str]], tokenizer=None) -> int:
        """Count total tokens across all messages"""
        total = 0
        for msg in messages:
            # Count role + content
            msg_str = f"{msg['role']}: {msg['content']}"
            total += TokenCounter.estimate_tokens(msg_str, tokenizer)
        return total


class ContextManager:
    """
    Manages conversation context with intelligent trimming.
    
    Handles:
    - Message history with role validation
    - Token counting and context limits
    - Intelligent trimming when limits exceeded
    - Prompt construction with chat templates
    """
    
    def __init__(self, max_context_tokens: int, tokenizer=None):
        self.max_context_tokens = max_context_tokens
        self.tokenizer = tokenizer
        self.messages: List[Dict[str, str]] = []
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        
        logger.debug(f"ContextManager initialized with max {max_context_tokens:,} tokens")
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for accurate token counting"""
        self.tokenizer = tokenizer
        logger.debug("Tokenizer set for accurate token counting")
    
    def add_message(self, role: str, content: str) -> bool:
        """
        Add message to context and trim if necessary.
        
        Args:
            role: Message role (user, assistant, tool)
            content: Message content
        
        Returns:
            True if significant trimming occurred (>25% of messages removed)
        """
        # Validate role
        valid_roles = {"user", "assistant", "tool", "system"}
        if role not in valid_roles:
            logger.warning(f"Invalid message role: {role}")
            return False
        
        # Add message
        self.messages.append({"role": role, "content": content})
        logger.debug(f"Added {role} message ({len(content)} chars)")
        
        # Trim if needed
        return self._trim_if_needed()
    
    def _trim_if_needed(self) -> bool:
        """
        Trim older messages if context exceeds limit.
        
        Returns:
            True if >25% of messages were trimmed
        """
        current_tokens = self._count_total_tokens()
        
        if current_tokens <= self.max_context_tokens:
            return False
        
        original_count = len(self.messages)
        target_tokens = int(self.max_context_tokens * 0.7)  # More aggressive trim
        
        logger.warning(f"Context overflow: {current_tokens:,} > {self.max_context_tokens:,} tokens")
        
        # Remove oldest messages until under target
        while current_tokens > target_tokens and len(self.messages) > 2:
            removed = self.messages.pop(0)
            current_tokens = self._count_total_tokens()
            logger.debug(f"Trimmed {removed['role']} message")
        
        trimmed_count = original_count - len(self.messages)
        
        if trimmed_count > 0:
            trim_percentage = trimmed_count / original_count
            logger.info(
                f"Trimmed {trimmed_count}/{original_count} messages ({trim_percentage*100:.0f}%). "
                f"Context now: {current_tokens:,} tokens"
            )
            return trim_percentage > 0.25
        
        return False
    
    def _count_total_tokens(self) -> int:
        """Count total tokens: system + messages + tools"""
        system_tokens = TokenCounter.estimate_tokens(self.system_prompt, self.tokenizer)
        message_tokens = TokenCounter.count_tokens_in_messages(self.messages, self.tokenizer)
        tools_tokens = TokenCounter.estimate_tokens(json.dumps(TOOLS_DEFINITIONS), self.tokenizer)
        
        total = system_tokens + message_tokens + tools_tokens
        return total
    
    def build_prompt(self, tokenizer, include_tools: bool = False) -> str:
        """
        Build the complete prompt using chat template.
        
        Args:
            tokenizer: Model tokenizer
            include_tools: Whether to include tool definitions
        
        Returns:
            Formatted prompt string ready for generation
        """
        # Build messages list with system prompt
        chat_messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation messages
        chat_messages.extend(self.messages)
        
        # Try to use tokenizer's chat template
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            try:
                kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True
                }
                
                # Add tools if requested
                if include_tools and TOOLS_DEFINITIONS:
                    kwargs["tools"] = TOOLS_DEFINITIONS
                
                prompt = tokenizer.apply_chat_template(chat_messages, **kwargs)
                logger.debug(f"Built prompt using chat template ({len(prompt)} chars)")
                return prompt
                
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback")
        
        # Fallback prompt construction
        return self._build_fallback_prompt(chat_messages, include_tools)
    
    def _build_fallback_prompt(self, chat_messages: List[Dict[str, str]], include_tools: bool) -> str:
        """
        Fallback prompt builder when chat template unavailable.
        Simple but functional format.
        """
        parts = []
        
        # Add system prompt
        if chat_messages and chat_messages[0]["role"] == "system":
            parts.append(f"System: {chat_messages[0]['content']}\n")
            chat_messages = chat_messages[1:]
        
        # Add tools if requested
        if include_tools and TOOLS_DEFINITIONS:
            tools_str = json.dumps(TOOLS_DEFINITIONS, indent=2)
            parts.append(f"Available Tools:\n{tools_str}\n")
        
        # Add conversation
        for msg in chat_messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "tool":
                parts.append(f"Tool Result: {content}")
        
        # Add generation prompt
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    def get_stats(self) -> Dict[str, any]:
        """Get context statistics for monitoring"""
        total_tokens = self._count_total_tokens()
        msg_count = len(self.messages)
        
        return {
            "message_count": msg_count,
            "total_tokens": total_tokens,
            "max_tokens": self.max_context_tokens,
            "usage_percent": (total_tokens / self.max_context_tokens) * 100
        }
    
    def clear_messages(self):
        """Clear all messages while keeping system prompt"""
        self.messages.clear()
        logger.info("Context cleared")