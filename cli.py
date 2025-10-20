"""
CLI interface for T.O.M. using prompt_toolkit
"""

import json
import logging
import os
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
    LOW_MEMORY_THRESHOLD_GB
)
from context_manager import ContextManager, TokenCounter
from model_manager import ModelManager
from tools import execute_tool_call, extract_tool_calls, truncate_tool_result, TOOLS_DEFINITIONS
from utils import load_model_config

# Initialize Rich console and logging
console = Console()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("tom_cli")


def clear_cache_file(model_path: str, cache_path: str = None, force: bool = False):
    """Utility function to clear cache file"""
    resolved_cache = cache_path or str(Path(model_path).parent / "prompt_cache.safetensors")
    cache_file = Path(resolved_cache)
    
    if not cache_file.exists():
        console.print(f"[yellow]No cache file found[/yellow]")
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
    """Handles the interactive chat interface"""
    
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
        
        # Load model config to get actual context size
        config = load_model_config(self.model_path)
        model_max_context = config.get("max_position_embeddings", 32768)
        
        # Use override if provided, otherwise use 80% of model's max
        max_context_tokens = max_context_override or int(model_max_context * CONTEXT_USAGE_RATIO)
        logger.info(f"Using max context: {max_context_tokens:,} tokens (model supports {model_max_context:,})")
        
        # Calculate max tool result size
        self.max_tool_result_tokens = min(int(max_context_tokens * TOOL_RESULT_CONTEXT_RATIO), MAX_TOOL_RESULT_TOKENS)
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

        # Setup prompt-toolkit session
        command_completer = WordCompleter(
            ['/help', '/stats', '/cache', '/memory', '/gc', 
             '/context', '/raw-prompt', '/clear-cache', '/exit', '/quit'],
            ignore_case=True,
            sentence=True
        )
        
        self.prompt_session = PromptSession(
            history=FileHistory('.tom_history'),
            auto_suggest=AutoSuggestFromHistory(),
            completer=merge_completers([
                command_completer,
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
        """Main interactive chat loop"""
        cache_status = "Caching enabled" if self.model_manager.enable_cache else "Caching disabled"
        console.print(Panel.fit(
            f"[bold blue]T.O.M. CLI[/bold blue]\n"
            f"{cache_status}\n"
            f"Max context: {self.context_manager.max_context_tokens:,} tokens\n"
            f"Max tool result: {self.max_tool_result_chars:,} chars\n"
            "Commands: /stats, /cache, /memory, /gc, /context, /raw-prompt, /clear-cache, /exit\n"
            "[dim]History: ↑/↓ arrows, Ctrl+R to search, Tab for completion[/dim]",
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
                
                # Handle empty input
                if not user_input.strip():
                    continue
                
                # Handle commands
                if user_input.lower() in ['/exit', '/quit']:
                    break
                
                if user_input.lower() == '/help':
                    self._show_help()
                    continue
                if user_input.lower() == '/stats':
                    self._show_stats()
                    continue
                if user_input.lower() == '/cache':
                    self._show_cache_info()
                    continue
                if user_input.lower() == '/memory':
                    self._show_memory_stats()
                    continue
                if user_input.lower() == '/gc':
                    console.print("[dim]Running garbage collection...[/dim]")
                    self.model_manager.run_gc()
                    console.print("[green]✓ GC complete[/green]")
                    continue
                if user_input.lower() == '/context':
                    self._show_context()
                    continue
                if user_input.lower() == '/raw-prompt':
                    self._show_raw_prompt()
                    continue
                if user_input.lower() == '/clear-cache':
                    self._clear_cache()
                    continue
                
                # Process user message
                should_reset = self.context_manager.add_message("user", user_input)
                
                if should_reset and self.model_manager.enable_cache:
                    logger.warning("Significant context trimming, resetting cache")
                    self.model_manager.reset_cache()
                
                self._generate_and_display_response()
                
        except KeyboardInterrupt:
            console.print("\nGoodbye!")
        except Exception as e:
            logger.error(f"Chat error: {e}", exc_info=True)
    
    def _generate_and_display_response(self):
        """Generate and display AI response with tool call processing"""
        start_time = time.time()
        
        with Status("Thinking...", console=console):
            thinking_content, content = self.model_manager.generate_response(include_tools=True)
        
        # Display thinking content if present
        if thinking_content:
            console.print(f"\n[dim italic]💭 Thinking: {thinking_content}[/dim italic]")
        
        # Extract tool calls from content only
        tool_calls = extract_tool_calls(content)

        if tool_calls:
            logger.info(f"Found {len(tool_calls)} tool call(s)")
            self.context_manager.add_message("assistant", content)

            for tc in tool_calls:
                try:
                    result = execute_tool_call(tc)
                    truncated_result = truncate_tool_result(result, tc["name"], self.max_tool_result_chars)
                    self.context_manager.add_message("tool", truncated_result)
                except Exception as e:
                    logger.error(f"Error executing tool: {e}", exc_info=True)
                    self.context_manager.add_message("tool", f"Tool error: {str(e)}")

            with Status("Processing results...", console=console):
                follow_up_thinking, follow_up_content = self.model_manager.generate_response(include_tools=False)

            if follow_up_thinking:
                console.print(f"\n[dim italic]💭 Thinking: {follow_up_thinking}[/dim italic]")

            final_response = follow_up_content
            self.context_manager.add_message("assistant", final_response)
        else:
            final_response = content
            self.context_manager.add_message("assistant", final_response)
        
        generation_time = time.time() - start_time
        console.print(f"\n[bold cyan]T.O.M.[/bold cyan]: {final_response}")
        console.print(f"[dim]{generation_time:.2f}s[/dim]")
    
    def _show_help(self):
        """Display comprehensive help information"""
        console.print("\n[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]")
        console.print("[bold cyan]                    T.O.M. CLI HELP[/bold cyan]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")
        
        console.print("[bold yellow]OVERVIEW[/bold yellow]")
        console.print("T.O.M. (Thinking Optimized Model) is an interactive AI assistant built on")
        console.print("Qwen3-4B-Thinking-2507 with tool-calling capabilities and efficient prompt caching.\n")
        
        console.print("[bold yellow]COMMANDS[/bold yellow]")
        commands_table = Table(show_header=False, box=None, padding=(0, 2))
        commands_table.add_column(style="cyan", no_wrap=True)
        commands_table.add_column(style="white")
        
        commands_table.add_row("/help", "Show this help message")
        commands_table.add_row("/stats", "Display context usage statistics")
        commands_table.add_row("/cache", "Show prompt cache information")
        commands_table.add_row("/memory", "Display system and MLX memory usage")
        commands_table.add_row("/gc", "Force garbage collection to free memory")
        commands_table.add_row("/context", "Show complete conversation context")
        commands_table.add_row("/raw-prompt", "Show raw formatted prompt")
        commands_table.add_row("/clear-cache", "Clear and reset the prompt cache")
        commands_table.add_row("/exit, /quit", "Exit the application")
        
        console.print(commands_table)
        console.print()
        
        console.print("[bold yellow]FEATURES[/bold yellow]")
        
        console.print("[cyan]• Thinking Mode:[/cyan] T.O.M. can show its reasoning process")
        console.print("  Look for 💭 Thinking messages to see how it approaches problems\n")
        
        console.print("[cyan]• Tool Calling:[/cyan] Built-in tools that T.O.M. can use:")
        console.print("  - get_datetime: Get current date and time")
        console.print("  - read: Read content from files on your system")
        console.print("  T.O.M. will automatically call tools when needed\n")
        
        console.print("[cyan]• Prompt Caching:[/cyan] Speeds up responses by caching context")
        console.print("  The cache is saved between sessions for faster startup\n")
        
        console.print("[cyan]• Context Management:[/cyan] Automatically manages conversation history")
        console.print("  - Keeps conversations within token limits")
        console.print("  - Intelligently trims old messages when needed")
        console.print("  - Preserves recent context for coherent responses\n")
        
        console.print("[bold yellow]USAGE TIPS[/bold yellow]")
        
        console.print("[cyan]1. File Operations:[/cyan]")
        console.print("   Ask T.O.M. to read files: 'Can you read the file at ./example.txt?'")
        console.print("   T.O.M. can handle text files up to 10MB\n")
        
        console.print("[cyan]2. Context Awareness:[/cyan]")
        console.print("   Check /stats regularly to monitor context usage")
        console.print("   When context is full, older messages are automatically trimmed\n")
        
        console.print("[cyan]3. Performance:[/cyan]")
        console.print("   Use /gc if you notice slowdowns or high memory usage")
        console.print("   The system auto-runs GC every few generations by default\n")
        
        console.print("[cyan]4. Cache Management:[/cyan]")
        console.print("   The prompt cache speeds up responses significantly")
        console.print("   Use /clear-cache if you want to start fresh")
        console.print("   Cache is automatically saved between sessions\n")
        
        stats = self.context_manager.get_stats()
        console.print("[bold yellow]CURRENT CONFIGURATION[/bold yellow]")
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column(style="cyan", no_wrap=True)
        config_table.add_column(style="white")
        
        config_table.add_row("Model Path:", str(self.model_path))
        config_table.add_row("Max Context:", f"{self.context_manager.max_context_tokens:,} tokens")
        config_table.add_row("Max Tool Result:", f"{self.max_tool_result_chars:,} characters")
        config_table.add_row("Caching:", "Enabled" if self.model_manager.enable_cache else "Disabled")
        config_table.add_row("Auto GC:", "Enabled" if self.model_manager.auto_gc else "Disabled")
        config_table.add_row("GC Frequency:", f"Every {self.model_manager.gc_frequency} generations")
        config_table.add_row("", "")
        config_table.add_row("Current Messages:", str(stats["message_count"]))
        config_table.add_row("Current Tokens:", f"{stats['total_tokens']:,}")
        config_table.add_row("Context Usage:", f"{stats['usage_percent']:.1f}%")
        
        console.print(config_table)
        console.print()
        
        console.print("[bold yellow]GETTING STARTED[/bold yellow]")
        console.print("Just type your message and press Enter. T.O.M. will respond naturally.")
        console.print("Try asking questions, requesting file reads, or having a conversation!\n")
        
        console.print("[dim]For more information, see the README.md in the project directory.[/dim]")
        console.print("[bold cyan]═══════════════════════════════════════════════════════════════[/bold cyan]\n")

    def _show_stats(self):
        """Show context statistics"""
        stats = self.context_manager.get_stats()
        
        table = Table(title="Context Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Messages", str(stats["message_count"]))
        table.add_row("Estimated Tokens", f"{stats['total_tokens']:,}")
        table.add_row("Max Context", f"{stats['max_tokens']:,}")
        table.add_row("Usage", f"{stats['usage_percent']:.1f}%")
        
        console.print(table)
    
    def _show_memory_stats(self):
        """Show memory usage statistics"""
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
        table.add_row("", "")
        
        if mlx_mem > 0:
            table.add_row("MLX Active", f"{mlx_mem:.2f} MB")
            table.add_row("MLX Peak", f"{mlx_peak:.2f} MB")
            table.add_row("MLX Cache", f"{mlx_cache:.2f} MB")
        
        console.print(table)
        
        if sys_available_gb < LOW_MEMORY_THRESHOLD_GB:
            console.print("[yellow]⚠ System memory is low![/yellow]")
    
    def _show_context(self):
        """Show the complete conversation context"""
        # Display system prompt
        console.print("\n[bold cyan]═══ SYSTEM PROMPT ═══[/bold cyan]")
        system_tokens = TokenCounter.estimate_tokens(
            self.context_manager.system_prompt, 
            self.context_manager.tokenizer
        )
        console.print(Panel(
            self.context_manager.system_prompt,
            title=f"System ({system_tokens:,} tokens)",
            border_style="cyan"
        ))
        
        # Display conversation messages
        if self.context_manager.messages:
            console.print("\n[bold cyan]═══ CONVERSATION HISTORY ═══[/bold cyan]")
            
            for idx, msg in enumerate(self.context_manager.messages, 1):
                role = msg["role"]
                content = msg["content"]
                msg_tokens = TokenCounter.estimate_tokens(
                    str(msg), 
                    self.context_manager.tokenizer
                )
                
                if role == "user":
                    style = "green"
                    icon = "👤"
                elif role == "assistant":
                    style = "blue"
                    icon = "🤖"
                elif role == "tool":
                    style = "yellow"
                    icon = "🔧"
                else:
                    style = "white"
                    icon = "•"
                
                console.print(Panel(
                    content,
                    title=f"{icon} Message {idx}: {role.title()} ({msg_tokens:,} tokens)",
                    border_style=style
                ))
        else:
            console.print("\n[dim]No messages in context yet[/dim]")
        
        # Display tools info
        console.print("\n[bold cyan]═══ TOOLS DEFINITIONS ═══[/bold cyan]")
        tools_str = json.dumps(TOOLS_DEFINITIONS, indent=2)
        tools_tokens = TokenCounter.estimate_tokens(
            tools_str,
            self.context_manager.tokenizer
        )
        console.print(f"[dim]Tools registered: {len(TOOLS_DEFINITIONS)}[/dim]")
        console.print(f"[dim]Tools definition tokens: {tools_tokens:,}[/dim]")
        
        # Display summary
        stats = self.context_manager.get_stats()
        console.print("\n[bold cyan]═══ CONTEXT SUMMARY ═══[/bold cyan]")
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Messages", str(stats["message_count"]))
        table.add_row("Estimated Total Tokens", f"{stats['total_tokens']:,}")
        table.add_row("Max Context Tokens", f"{stats['max_tokens']:,}")
        table.add_row("Context Usage", f"{stats['usage_percent']:.1f}%")
        
        console.print(table)
        console.print()
    
    def _show_raw_prompt(self):
        """Show the actual formatted prompt sent to the LLM"""
        console.print("\n[bold cyan]═══ RAW PROMPT (WITH TOOLS) ═══[/bold cyan]")
        console.print("[dim]This is the exact formatted string the LLM processes[/dim]\n")
        
        raw_prompt = self.context_manager.build_prompt(self.model_manager.tokenizer, include_tools=True)
        prompt_tokens = TokenCounter.estimate_tokens(raw_prompt, self.context_manager.tokenizer)
        
        console.print(Panel(
            raw_prompt,
            title=f"Formatted Prompt ({prompt_tokens:,} tokens)",
            border_style="magenta",
            subtitle="[dim]Includes special tokens and chat template formatting[/dim]"
        ))
        
        console.print(f"\n[dim]Prompt length: {len(raw_prompt):,} characters[/dim]")
        console.print(f"[dim]Estimated tokens: {prompt_tokens:,}[/dim]")
        
        console.print("\n[bold yellow]═══ RAW PROMPT (WITHOUT TOOLS) ═══[/bold yellow]")
        raw_prompt_no_tools = self.context_manager.build_prompt(self.model_manager.tokenizer, include_tools=False)
        no_tools_tokens = TokenCounter.estimate_tokens(raw_prompt_no_tools, self.context_manager.tokenizer)
        
        console.print(Panel(
            raw_prompt_no_tools,
            title=f"Formatted Prompt Without Tools ({no_tools_tokens:,} tokens)",
            border_style="yellow",
            subtitle="[dim]Same prompt but without tool definitions[/dim]"
        ))
        
        console.print(f"\n[dim]Tools overhead: {prompt_tokens - no_tools_tokens:,} tokens[/dim]")
        console.print()
    
    def _show_cache_info(self):
        """Show prompt cache information"""
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
            table.add_row("Cache Hits", str(cache_info.get("cache_hits", 0)))
            table.add_row("Cache Misses", str(cache_info.get("cache_misses", 0)))
            
            if cache_info.get("generations", 0) > 0:
                table.add_row("Hit Rate", f"{cache_info.get('hit_rate', 0):.1f}%")
        
        if "size_mb" in cache_info:
            table.add_row("File Size", f"{cache_info['size_mb']:.2f} MB")
        
        console.print(table)
    
    def _clear_cache(self):
        """Clear the prompt cache"""
        if not self.model_manager.enable_cache:
            console.print("[yellow]Caching is disabled[/yellow]")
            return
        
        cache_file = Path(self.model_manager.cache_path)
        
        if cache_file.exists():
            confirm = Prompt.ask(
                f"[yellow]Clear cache file?[/yellow]",
                choices=["yes", "no"],
                default="no"
            )
            
            if confirm.lower() == "yes":
                try:
                    cache_file.unlink()
                    console.print("[green]Cache file deleted[/green]")
                except Exception as e:
                    console.print(f"[red]Failed to delete: {e}[/red]")
                    return
        
        self.model_manager.reset_cache()
        console.print("[green]Cache cleared and reset[/green]")