"""
CLI interface for T.O.M. with streaming support
FIXED: Tool result handling to prevent context pollution
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import psutil
import mlx.core as mx
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter, PathCompleter, merge_completers

from config import (
    CONTEXT_USAGE_RATIO,
    TOOL_RESULT_CONTEXT_RATIO,
    MAX_TOOL_RESULT_TOKENS,
    LOW_MEMORY_THRESHOLD_GB,
    ENABLE_STREAMING
)
from context_manager import ContextManager, TokenCounter
from model_manager import ModelManager
from tools import (
    execute_tool_call, 
    extract_tool_calls, 
    strip_tool_calls,
    truncate_tool_result, 
    TOOLS_DEFINITIONS
)
from utils import load_model_config

# Initialize Rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("tom_cli")


def clear_cache_file(model_path: str, cache_path: str = None, force: bool = False):
    """Utility to clear cache file"""
    resolved_cache = cache_path or str(Path(model_path).parent / "prompt_cache.safetensors")
    cache_file = Path(resolved_cache)
    
    if not cache_file.exists():
        console.print("[yellow]No cache file found[/yellow]")
        return
    
    cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
    
    if not force:
        console.print(f"[yellow]Cache: {resolved_cache} ({cache_size_mb:.2f} MB)[/yellow]")
        confirm = Prompt.ask("Delete?", choices=["yes", "no"], default="no")
        
        if confirm.lower() != "yes":
            console.print("[dim]Cancelled[/dim]")
            return
    
    try:
        cache_file.unlink()
        console.print(f"[green]✓ Cache deleted ({cache_size_mb:.2f} MB freed)[/green]")
    except Exception as e:
        console.print(f"[red]Failed to delete: {e}[/red]")


class ChatInterface:
    """Interactive chat interface with streaming support"""
    
    def __init__(
        self,
        model_path: Path,
        cache_path: str = None,
        enable_cache: bool = True,
        prewarm: bool = True,
        max_context_override: int = None,
        auto_gc: bool = True,
        gc_frequency: int = 3
    ):
        self.model_path = model_path
        
        # Load model config
        config = load_model_config(self.model_path)
        model_max_context = config.get("max_position_embeddings", 32768)
        
        # Set max context
        max_context_tokens = max_context_override or int(model_max_context * CONTEXT_USAGE_RATIO)
        logger.info(f"Max context: {max_context_tokens:,} tokens (model: {model_max_context:,})")
        
        # Calculate max tool result size
        self.max_tool_result_tokens = min(
            int(max_context_tokens * TOOL_RESULT_CONTEXT_RATIO), 
            MAX_TOOL_RESULT_TOKENS
        )
        self.max_tool_result_chars = self.max_tool_result_tokens * 4
        
        # Initialize managers
        self.context_manager = ContextManager(max_context_tokens=max_context_tokens)
        self.model_manager = ModelManager(
            model_path=model_path,
            context_manager=self.context_manager,
            cache_path=cache_path,
            enable_cache=enable_cache,
            prewarm=prewarm,
            auto_gc=auto_gc,
            gc_frequency=gc_frequency
        )
        
        # Setup prompt session
        self.prompt_session = PromptSession(
            history=FileHistory('.tom_history'),
            auto_suggest=AutoSuggestFromHistory(),
            completer=merge_completers([
                WordCompleter(
                    ['/help', '/stats', '/cache', '/memory', '/gc', 
                     '/context', '/raw-prompt', '/clear-cache', '/exit', '/quit'],
                    ignore_case=True,
                    sentence=True
                ),
                PathCompleter(expanduser=True)
            ]),
            complete_while_typing=True,
            enable_history_search=True,
        )
    
    def load_model(self):
        """Load the model"""
        with Status("Loading model...", console=console):
            self.model_manager.load_model()
    
    def run(self):
        """Main interactive loop"""
        cache_status = "Caching enabled" if self.model_manager.enable_cache else "Caching disabled"
        streaming_status = "Streaming enabled" if ENABLE_STREAMING else "Streaming disabled"
        
        console.print(Panel.fit(
            f"[bold blue]T.O.M. CLI[/bold blue]\n"
            f"{cache_status} | {streaming_status}\n"
            f"Max context: {self.context_manager.max_context_tokens:,} tokens\n"
            f"Max tool result: {self.max_tool_result_chars:,} chars\n"
            "Commands: /help, /stats, /cache, /memory, /gc, /exit",
            border_style="blue"
        ))
        
        try:
            while True:
                try:
                    user_input = self.prompt_session.prompt("\nYou> ")
                    
                except KeyboardInterrupt:
                    console.print("[dim]Use /exit or Ctrl+D to quit[/dim]")
                    continue
                except EOFError:
                    console.print("\nGoodbye!")
                    break
                
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.lower() in ['/exit', '/quit']:
                    break
                elif user_input.lower() == '/help':
                    self._show_help()
                    continue
                elif user_input.lower() == '/stats':
                    self._show_stats()
                    continue
                elif user_input.lower() == '/cache':
                    self._show_cache_info()
                    continue
                elif user_input.lower() == '/memory':
                    self._show_memory_stats()
                    continue
                elif user_input.lower() == '/gc':
                    console.print("[dim]Running GC...[/dim]")
                    self.model_manager.run_gc()
                    console.print("[green]✓ GC complete[/green]")
                    continue
                elif user_input.lower() == '/context':
                    self._show_context()
                    continue
                elif user_input.lower() == '/raw-prompt':
                    self._show_raw_prompt()
                    continue
                elif user_input.lower() == '/clear-cache':
                    self._clear_cache()
                    continue
                
                # Add user message
                should_reset = self.context_manager.add_message("user", user_input)
                
                if should_reset and self.model_manager.enable_cache:
                    logger.warning("Significant trimming, resetting cache")
                    self.model_manager.reset_cache()
                
                # Generate response
                if ENABLE_STREAMING:
                    self._generate_streaming()
                else:
                    self._generate_legacy()
                
        except KeyboardInterrupt:
            console.print("\nGoodbye!")
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
    
    def _generate_streaming(self):
        """Generate response with streaming - FIXED tool handling"""
        start_time = time.time()
        console.print()
        
        # First generation with tools
        thinking, content, tool_calls = self._stream_and_parse()
        
        if tool_calls:
            logger.info(f"Detected {len(tool_calls)} tool call(s)")
            
            # CRITICAL FIX: Strip tool call XML from content before adding to context
            clean_content = strip_tool_calls(content)
            logger.info(f"Adding assistant message to context: '{clean_content}' (original had {len(content)} chars)")
            self.context_manager.add_message("assistant", clean_content)
            
            # Execute tools and add results
            for tc in tool_calls:
                try:
                    result = execute_tool_call(tc)
                    truncated = truncate_tool_result(
                        result, 
                        tc["name"], 
                        self.max_tool_result_chars
                    )
                    
                    # Add tool result with clear formatting
                    tool_msg = f"Tool: {tc['name']}\nResult: {truncated}"
                    self.context_manager.add_message("tool", tool_msg)
                    
                except Exception as e:
                    logger.error(f"Tool error: {e}", exc_info=True)
                    error_msg = f"Tool: {tc['name']}\nError: {str(e)}"
                    self.context_manager.add_message("tool", error_msg)
            
            # Second generation without tools
            print("\n")  # Spacing
            follow_thinking, follow_content, _ = self._stream_and_parse(include_tools=False)
            
            # Add final response
            print("\n")
            self.context_manager.add_message("assistant", follow_content)
        else:
            # No tools, just add response
            print("\n")
            self.context_manager.add_message("assistant", content)
        
        elapsed = time.time() - start_time
        console.print(f"[dim]{elapsed:.2f}s[/dim]")
    
    def _stream_and_parse(self, include_tools: bool = True) -> tuple[str, str, list]:
        """
        Stream response and parse thinking/content/tools.
        Returns (thinking, content, tool_calls)
        """
        thinking = ""
        content = ""
        tool_calls = []
        
        in_thinking = False
        in_content = False
        
        for chunk in self.model_manager.stream_response(include_tools=include_tools):
            chunk_type = chunk.get('type')
            
            if chunk_type == 'thinking':
                if not in_thinking and chunk.get('delta'):
                    console.print("[dim italic]💭 Thinking:[/dim italic] ", end="")
                    in_thinking = True
                
                delta = chunk.get('delta', '')
                if delta:
                    print(delta, end="", flush=True)
                
                if chunk.get('complete'):
                    thinking = chunk.get('text', '')
                    if in_thinking:
                        print()
            
            elif chunk_type == 'content':
                if not in_content and chunk.get('delta'):
                    console.print("\n[bold cyan]T.O.M.[/bold cyan]: ", end="")
                    in_content = True
                
                delta = chunk.get('delta', '')
                if delta:
                    # Check for tool call markers
                    if '<tool_call>' in delta:
                        print(" [🔧 Tool call]", end="", flush=True)
                    else:
                        print(delta, end="", flush=True)
            
            elif chunk_type == 'done':
                thinking = chunk.get('thinking', thinking)
                content = chunk.get('content', content)
                
                # Extract tool calls from complete content
                tool_calls = extract_tool_calls(content)
                break
            
            elif chunk_type == 'error':
                console.print(f"\n[red]Error: {chunk.get('text', 'Unknown error')}[/red]")
                content = chunk.get('text', '')
                break
        
        return thinking, content, tool_calls
    
    def _generate_legacy(self):
        """Non-streaming generation (fallback)"""
        start_time = time.time()
        
        with Status("Thinking...", console=console):
            thinking, content = self.model_manager.generate_response(include_tools=True)
        
        if thinking:
            console.print(f"\n[dim italic]💭 Thinking: {thinking}[/dim italic]")
        
        # Extract and execute tools
        tool_calls = extract_tool_calls(content)
        
        if tool_calls:
            logger.info(f"Detected {len(tool_calls)} tool call(s)")
            
            # Strip tool XML and add to context
            clean_content = strip_tool_calls(content)
            self.context_manager.add_message("assistant", clean_content)
            
            # Execute tools
            for tc in tool_calls:
                try:
                    result = execute_tool_call(tc)
                    truncated = truncate_tool_result(
                        result,
                        tc["name"],
                        self.max_tool_result_chars
                    )
                    tool_msg = f"Tool: {tc['name']}\nResult: {truncated}"
                    self.context_manager.add_message("tool", tool_msg)
                except Exception as e:
                    logger.error(f"Tool error: {e}", exc_info=True)
                    error_msg = f"Tool: {tc['name']}\nError: {str(e)}"
                    self.context_manager.add_message("tool", error_msg)
            
            # Follow-up generation
            with Status("Processing results...", console=console):
                follow_thinking, follow_content = self.model_manager.generate_response(include_tools=False)
            
            if follow_thinking:
                console.print(f"\n[dim italic]💭 Thinking: {follow_thinking}[/dim italic]")
            
            final_response = follow_content
            self.context_manager.add_message("assistant", final_response)
        else:
            final_response = content
            self.context_manager.add_message("assistant", final_response)
        
        elapsed = time.time() - start_time
        console.print(f"\n[bold cyan]T.O.M.[/bold cyan]: {final_response}")
        console.print(f"[dim]{elapsed:.2f}s[/dim]")
    
    def _show_help(self):
        """Display help"""
        console.print("\n[bold cyan]T.O.M. CLI - Help[/bold cyan]\n")
        
        commands = Table(show_header=False, box=None, padding=(0, 2))
        commands.add_column(style="cyan", no_wrap=True)
        commands.add_column(style="white")
        
        commands.add_row("/help", "Show this help")
        commands.add_row("/stats", "Context statistics")
        commands.add_row("/cache", "Cache information")
        commands.add_row("/memory", "Memory usage")
        commands.add_row("/gc", "Force garbage collection")
        commands.add_row("/context", "Show conversation context")
        commands.add_row("/raw-prompt", "Show formatted prompt")
        commands.add_row("/clear-cache", "Reset cache")
        commands.add_row("/exit, /quit", "Exit")
        
        console.print(commands)
        console.print()
    
    def _show_stats(self):
        """Show context statistics"""
        stats = self.context_manager.get_stats()
        
        table = Table(title="Context Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Messages", str(stats["message_count"]))
        table.add_row("Tokens", f"{stats['total_tokens']:,}")
        table.add_row("Max", f"{stats['max_tokens']:,}")
        table.add_row("Usage", f"{stats['usage_percent']:.1f}%")
        
        console.print(table)
    
    def _show_cache_info(self):
        """Show cache info"""
        cache_info = self.model_manager.get_cache_info()
        
        table = Table(title="Prompt Cache")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Enabled", "Yes" if cache_info["enabled"] else "No")
        table.add_row("Path", cache_info["path"])
        
        if cache_info["enabled"]:
            table.add_row("Max KV Size", str(cache_info.get("max_kv_size", "unlimited")))
            table.add_row("KV Bits", str(cache_info.get("kv_bits", "no quantization")))
            table.add_row("Generations", str(cache_info.get("generations", 0)))
            table.add_row("Hits", str(cache_info.get("cache_hits", 0)))
            table.add_row("Misses", str(cache_info.get("cache_misses", 0)))
            
            if cache_info.get("generations", 0) > 0:
                table.add_row("Hit Rate", f"{cache_info.get('hit_rate', 0):.1f}%")
        
        if "size_mb" in cache_info:
            table.add_row("File Size", f"{cache_info['size_mb']:.2f} MB")
        
        console.print(table)
    
    def _show_memory_stats(self):
        """Show memory usage"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024)
        
        sys_mem = psutil.virtual_memory()
        sys_total_gb = sys_mem.total / (1024 ** 3)
        sys_available_gb = sys_mem.available / (1024 ** 3)
        sys_percent = sys_mem.percent
        
        try:
            mlx_mem = mx.get_active_memory() / (1024 * 1024)
            mlx_peak = mx.get_peak_memory() / (1024 * 1024)
            mlx_cache = mx.get_cache_memory() / (1024 * 1024)
        except:
            mlx_mem = mlx_peak = mlx_cache = 0
        
        table = Table(title="Memory Usage")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("System Total", f"{sys_total_gb:.2f} GB")
        table.add_row("System Available", f"{sys_available_gb:.2f} GB")
        table.add_row("System Usage", f"{sys_percent:.1f}%")
        table.add_row("", "")
        table.add_row("Process RSS", f"{mem_mb:.2f} MB")
        
        if mlx_mem > 0:
            table.add_row("", "")
            table.add_row("MLX Active", f"{mlx_mem:.2f} MB")
            table.add_row("MLX Peak", f"{mlx_peak:.2f} MB")
            table.add_row("MLX Cache", f"{mlx_cache:.2f} MB")
        
        console.print(table)
        
        if sys_available_gb < LOW_MEMORY_THRESHOLD_GB:
            console.print("[yellow]⚠ System memory low![/yellow]")
    
    def _show_context(self):
        """Show conversation context"""
        console.print("\n[bold cyan]SYSTEM PROMPT[/bold cyan]")
        console.print(Panel(self.context_manager.system_prompt, border_style="cyan"))
        
        if self.context_manager.messages:
            console.print("\n[bold cyan]MESSAGES[/bold cyan]")
            for idx, msg in enumerate(self.context_manager.messages, 1):
                role = msg["role"]
                content = msg["content"]
                
                style = "green" if role == "user" else "blue" if role == "assistant" else "yellow"
                console.print(Panel(content, title=f"{idx}. {role.title()}", border_style=style))
        
        stats = self.context_manager.get_stats()
        console.print(f"\n[cyan]Total: {stats['message_count']} messages, "
                     f"{stats['total_tokens']:,} tokens ({stats['usage_percent']:.1f}%)[/cyan]\n")
    
    def _show_raw_prompt(self):
        """Show raw formatted prompt"""
        console.print("\n[bold cyan]RAW PROMPT (WITH TOOLS)[/bold cyan]")
        prompt = self.context_manager.build_prompt(self.model_manager.tokenizer, include_tools=True)
        console.print(Panel(prompt, border_style="magenta"))
        
        console.print("\n[bold cyan]RAW PROMPT (WITHOUT TOOLS)[/bold cyan]")
        prompt_no_tools = self.context_manager.build_prompt(self.model_manager.tokenizer, include_tools=False)
        console.print(Panel(prompt_no_tools, border_style="yellow"))
        console.print()
    
    def _clear_cache(self):
        """Clear cache"""
        if not self.model_manager.enable_cache:
            console.print("[yellow]Caching disabled[/yellow]")
            return
        
        cache_file = Path(self.model_manager.cache_path)
        
        if cache_file.exists():
            confirm = Prompt.ask("[yellow]Clear cache?[/yellow]", choices=["yes", "no"], default="no")
            
            if confirm.lower() == "yes":
                try:
                    cache_file.unlink()
                    console.print("[green]Cache file deleted[/green]")
                except Exception as e:
                    console.print(f"[red]Failed: {e}[/red]")
                    return
        
        self.model_manager.reset_cache()
        console.print("[green]Cache reset[/green]")