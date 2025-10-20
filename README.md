# T.O.M. CLI

**T**erminal **O**rchestrated **M**odel - A production-ready agentic framework built on Qwen3-4B-Thinking-2507 with MLX optimization for Apple Silicon.

T.O.M. is an interactive AI assistant featuring intelligent prompt caching, tool-calling capabilities, and transparent reasoning processes. Built with modularity and performance in mind.

---

## Installation

### Prerequisites

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.11** (installed via Homebrew recommended)
- **Git** for cloning the repository

### Step 1: Clone the Repository

```bash
git clone https://github.com/keithduke/TOM-framework tom-cli
cd tom-cli
```

### Step 2: Set Up Python Environment

Using Python 3.11 installed via Homebrew:

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt, indicating the virtual environment is active. All subsequent commands assume you're working within this environment.

> **Note:** If you're unfamiliar with virtual environments, see the [official venv documentation](https://docs.python.org/3/library/venv.html).

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies installed:**
- `mlx` and `mlx-lm` - Apple's MLX framework for efficient ML on Apple Silicon
- `prompt_toolkit` - Advanced interactive prompt functionality
- `rich` - Python library for rich text and beautiful formatting in the terminal
- `psutil` - System and process utilities
- `PySide6` - Qt-based GUI framework for the launcher interface (optional)

### Step 4: Download and Convert the Model

T.O.M. uses Qwen3-4B-Thinking-2507, which must be downloaded from Hugging Face and converted to MLX format for optimal performance on Apple Silicon.

```bash
python -m mlx_lm.convert \
  --hf-path Qwen/Qwen3-4B-Thinking-2507 \
  --mlx-path ./Qwen3-4B-Thinking-2507-8bit \
  -q --q-bits 8
```

**What this does:**
- `--hf-path`: Downloads the model from [Hugging Face](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
- `--mlx-path`: Saves the converted model locally
- `-q --q-bits 8`: Applies 8-bit quantization for efficient memory usage

> **Note:** This process takes several minutes depending on your internet connection (typically under 5 minutes).

### Step 5: Launch T.O.M.

T.O.M. can be launched in two modes:

#### Terminal Mode (CLI)
```bash
python main.py
```

On first launch, T.O.M. will initialize the prompt cache and be ready for interaction.

#### Launcher Mode (GUI)
```bash
python launcher.py
```

The launcher provides a persistent GUI window with:
- **System tray integration** - T.O.M. stays in your menu bar
- **Dedicated app window** - Terminal-style interface with modern styling
- **Always accessible** - Click the tray icon or menu item to show/hide
- **Same functionality** - Full CLI capabilities in a GUI wrapper

The launcher is a thin wrapper around the CLI, showcasing T.O.M.'s modular architecture. All core functionality remains in the CLI modules.

---

## Using T.O.M.

T.O.M. offers two interface modes to suit different workflows:

### Terminal Mode vs Launcher Mode

**Terminal Mode (`python main.py`)**
- Traditional command-line interface
- Runs in your terminal window
- Best for: Quick interactions, scripting, integration with other CLI tools

**Launcher Mode (`python launcher.py`)**
- GUI application with system tray integration
- Dedicated window with terminal styling
- Persistent menu bar presence
- Best for: Extended sessions, always-on availability, desktop workflow integration

Both modes provide identical functionality and share the same codebase.

### Launcher Mode Features

When using `launcher.py`, you get:

#### System Tray Integration
- **Menu bar icon**: T.O.M. appears in your system tray/menu bar
- **Show/Hide**: Click the icon to toggle the window
- **Quick access**: Right-click for context menu
- **Background operation**: Minimize to tray without closing

#### Window Management
- **Persistent window**: Dedicated app window with terminal aesthetics
- **Close to minimize**: Clicking close button minimizes to tray
- **Full quit**: Use `/quit`, `/exit`, or select "Quit" from tray menu
- **Keyboard shortcuts**: 
  - `Ctrl+C` - Clear current input
  - `Ctrl+D` or `/exit` - Quit application

#### Visual Design
- **Dark theme**: Professional dark color scheme
- **Monospace font**: Clear, readable terminal-style text
- **Syntax highlighting**: Color-coded output (thinking, errors, responses)
- **Status bar**: Real-time connection status indicator
- **Styled input**: Prominent input field with focus indication

### Interactive Interface (Both Modes)

T.O.M. provides a sophisticated interactive command-line interface powered by `prompt_toolkit`, offering features typically found in modern development tools:

#### Command History
- **Navigate history**: Use `↑` and `↓` arrow keys to cycle through previous commands
- **Search history**: Press `Ctrl+R` to search through your command history interactively
- **Persistent history**: Your session history is saved to `.tom_history` and persists across sessions

#### Auto-completion
- **Command completion**: Type `/` and press `Tab` to see available commands
- **Path completion**: When typing file paths, press `Tab` for intelligent path completion
- **Real-time suggestions**: Previous commands appear as gray suggestions as you type

#### Input Controls
- **Multi-line editing**: Standard text editing with cursor movement
- **Cancel input**: Press `Ctrl+C` to clear the current input without exiting
- **Exit application**: Press `Ctrl+D` or type `/exit` or `/quit`

### Basic Usage Examples

```bash
# Simple conversation
You> What is machine learning?

# Request file analysis
You> Can you read and summarize ./data/report.txt?

# Get current time
You> What time is it?

# Multi-turn conversation with context
You> Explain recursion
You> Can you give me a Python example?
You> Now show me how to optimize it
```

### Built-in Commands

T.O.M. includes several commands for monitoring and managing the system:

| Command | Description |
|---------|-------------|
| `/help` | Display comprehensive help information |
| `/stats` | Show context usage statistics (messages, tokens, usage percentage) |
| `/cache` | Display prompt cache information and hit rates |
| `/memory` | Show system and MLX memory usage |
| `/gc` | Force garbage collection to free memory |
| `/context` | View complete conversation history with token counts |
| `/raw-prompt` | Inspect the formatted prompt sent to the LLM |
| `/clear-cache` | Clear and reset the prompt cache |
| `/exit`, `/quit` | Exit the application |

### Thinking Mode

T.O.M. features transparent reasoning through "thinking mode". When the model processes complex queries, you'll see its internal reasoning:

```
💭 Thinking: To answer this question, I need to first understand the user's 
technical background, then explain the concept in appropriate detail...

T.O.M.: [Actual response to user]
```

This feature provides insight into the model's decision-making process and helps users understand how conclusions are reached.

### Tool System

T.O.M. can autonomously use tools to extend its capabilities:

#### Available Tools

**`get_datetime`**
- Returns current system date and time
- Automatically called when time-related queries are detected
- Format: "H:MM AM/PM on Month Dth, YYYY"

**`read`**
- Reads content from files on your system
- Supports text files up to 10MB
- Handles UTF-8 encoded files
- Example: "Can you read the file at ./config.py?"

#### Tool Call Process

When T.O.M. determines a tool is needed:
1. The model generates a tool call in its response
2. T.O.M. executes the tool and captures the result
3. The tool result is added to the conversation context
4. The model generates a follow-up response incorporating the tool result

You'll see this happen seamlessly in the conversation flow.

---

## Architecture

### Design Philosophy

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Testability**: Modules can be tested independently
3. **Maintainability**: Changes are localized to specific modules
4. **Reusability**: Components can be imported and used by other projects

### Module Structure

```
├── main.py                 # CLI application entry point
├── launcher.py             # GUI launcher with tray integration
├── cli.py                  # CLI interface and interactive loop
├── model_manager.py        # Model loading, caching, and generation
├── context_manager.py      # Conversation context and prompt building
├── prompt_cache_manager.py # Intelligent prompt cache lifecycle
├── tools.py                # Tool system and built-in tools
├── utils.py                # Shared utility functions
└── config.py               # Configuration constants
```

### Module Responsibilities

#### `main.py`
- CLI application entry point
- Command-line argument parsing
- Launches the terminal interface

#### `launcher.py`
- GUI application entry point
- Qt-based window and system tray management
- Wraps CLI process for GUI presentation
- Terminal emulation with styled output
- Showcases modular architecture - zero changes to core CLI

#### `config.py`
- Centralized configuration management
- Generation parameters (temperature, top_p, top_k, repetition_penalty)
- Context management settings (max tokens, trim ratios)
- Memory management defaults
- System prompts and limits

#### `utils.py`
- Shared utility functions
- `ordinal()`: Date formatting helper for natural language dates
- `load_model_config()`: Reads model configuration from config.json

#### `tools.py`
- Complete tool registration and execution system
- `@tool` decorator for registering new tools
- `TOOLS_REGISTRY` and `TOOLS_DEFINITIONS` for tool management
- Built-in tools: `get_datetime`, `read_file`
- `execute_tool_call()`: Executes tools by name with argument parsing
- `extract_tool_calls()`: Parses tool calls from model responses
- `truncate_tool_result()`: Intelligently truncates large tool outputs

#### `context_manager.py`
- Conversation state management
- `TokenCounter`: Accurate token counting with fallback estimation
- `ContextManager`: Message history, intelligent trimming, prompt building
- `get_stats()`: Context usage statistics and monitoring

#### `prompt_cache_manager.py`
- Sophisticated prompt cache lifecycle management
- Automatic cache sizing based on system constraints
- Cache hit/miss tracking and statistics
- Memory-efficient quantization options (4-bit, 8-bit)
- Intelligent cache reset strategies
- Persistent cache storage between sessions

#### `model_manager.py`
- Model and tokenizer loading
- Integration with `PromptCacheManager`
- Core generation with thinking/content separation
- `generate_response()`: Main generation endpoint
- `_parse_thinking_and_content()`: Separates reasoning from output
- Garbage collection management
- Cache lifecycle coordination

#### `cli.py`
- User interface and interaction
- `ChatInterface`: Main chat loop and state management
- CLI app and command definitions
- Rich console display methods with formatting
- Interactive command handling
- Response generation orchestration
- Tool call processing loop
- Integration with `prompt_toolkit` for advanced input handling

### Data Flow

```
User Input
    ↓
ChatInterface (cli.py)
    ↓
ContextManager.add_message() (context_manager.py)
    ↓
ModelManager.generate_response() (model_manager.py)
    ├─→ ContextManager.build_prompt()
    ├─→ PromptCacheManager.get_generation_kwargs()
    └─→ MLX generate()
    ↓
extract_tool_calls() (tools.py)
    ↓
[For each tool call]
    ↓
execute_tool_call() (tools.py)
    ↓
truncate_tool_result() (tools.py)
    ↓
ContextManager.add_message("tool", result)
    ↓
ModelManager.generate_response() [follow-up]
    ↓
Display to User
```

### Import Dependencies

```
main.py (CLI entry)
  └─→ cli.py
        ├─→ config.py
        ├─→ context_manager.py
        │     ├─→ config.py
        │     └─→ tools.py
        ├─→ model_manager.py
        │     ├─→ config.py
        │     ├─→ context_manager.py
        │     ├─→ tools.py
        │     └─→ prompt_cache_manager.py
        ├─→ tools.py
        │     ├─→ config.py
        │     └─→ utils.py
        └─→ utils.py
              └─→ config.py

launcher.py (GUI entry)
  └─→ main.py (spawned as subprocess)
```

---

## Advanced Features

### Prompt Caching

T.O.M. implements intelligent prompt caching to significantly accelerate response times:

**How it works:**
- Static content (system prompt, tool definitions) is cached and reused
- Cache is quantized (8-bit by default) for memory efficiency
- Cache persists between sessions via `prompt_cache.safetensors`
- Automatic cache sizing based on model constraints
- Hit/miss tracking for performance monitoring

**Benefits:**
- 2-5x faster response generation after initial cache warmup
- Reduced redundant computation
- Lower memory footprint with quantization

**Cache Management:**
```bash
# View cache statistics
You> /cache

# Clear cache if needed
You> /clear-cache

# Disable caching (not recommended)
python main.py --no-cache
```

### Context Management

T.O.M. automatically manages conversation context to stay within model limits:

**Features:**
- Monitors total token count across system prompt, messages, and tools
- Intelligently trims older messages when context limit approached
- Preserves recent conversation for coherence
- Resets cache when significant trimming occurs
- Provides detailed context statistics via `/context` command

**Configuration:**
- Default max context: 80% of model's maximum (26,214 tokens for 32K model)
- Trim target: 80% of max when limit exceeded
- Tool results: Limited to 20% of context (max 8,192 tokens)

```bash
# Override max context
python main.py --max-context 16000

# View current context usage
You> /stats
```

### Memory Management

T.O.M. includes automatic garbage collection to maintain performance:

**Features:**
- Automatic GC every N generations (default: 3)
- Manual GC via `/gc` command
- MLX cache clearing integrated with GC
- Memory monitoring via `/memory` command
- Low memory warnings when system memory drops below 2GB

**Configuration:**
```bash
# Adjust GC frequency
python main.py --gc-frequency 5

# Disable automatic GC
python main.py --no-auto-gc
```

---

## Command-Line Options

### Terminal Mode (main.py)

```bash
# Basic usage
python main.py

# Specify model path
python main.py --model ./path/to/model

# Custom cache location
python main.py --cache ./my_cache.safetensors

# Override max context tokens
python main.py --max-context 20000

# Disable prompt caching
python main.py --no-cache

# Skip cache prewarming (faster startup, slower first response)
python main.py --no-prewarm

# Disable automatic garbage collection
python main.py --no-auto-gc

# Adjust GC frequency
python main.py --gc-frequency 5

# Enable debug logging
python main.py --debug

# Combine options
python main.py -m ./model --max-context 16000 --gc-frequency 10 --debug
```

### Launcher Mode (launcher.py)

```bash
# Launch GUI application
python launcher.py
```

The launcher automatically starts `main.py` as a subprocess with default settings. To use custom CLI options with the launcher, modify the model path or other settings directly in `launcher.py` or run `main.py` in terminal mode instead.

### Utility Commands

```bash
# Clear cache file without launching interactive mode
python main.py clear-cache

# Clear cache for specific model
python main.py clear-cache --model ./custom-model

# Clear with custom cache path
python main.py clear-cache --cache ./my_cache.safetensors

# Force delete without confirmation
python main.py clear-cache --force
```

---

## Configuration

All configuration constants are centralized in `config.py`:

### Generation Parameters
```python
MAX_GENERATION_TOKENS = 2048          # Maximum tokens per generation
DEFAULT_TEMPERATURE = 0.7             # Sampling temperature
DEFAULT_TOP_P = 0.9                   # Nucleus sampling threshold
DEFAULT_TOP_K = 50                    # Top-k sampling limit
DEFAULT_REPETITION_PENALTY = 1.0      # Repetition penalty multiplier
```

### Context Management
```python
DEFAULT_MODEL_MAX_CONTEXT = 32768     # Default model context size
CONTEXT_USAGE_RATIO = 0.8             # Use 80% of model's max context
CONTEXT_TRIM_TARGET_RATIO = 0.8       # Trim to 80% when exceeded
TOOL_RESULT_CONTEXT_RATIO = 0.2       # Max 20% of context for tool results
MAX_TOOL_RESULT_TOKENS = 8192         # Hard cap on tool result size
```

### Memory Management
```python
DEFAULT_GC_FREQUENCY = 3              # Run GC every N generations
LOW_MEMORY_THRESHOLD_GB = 2.0         # Warn when system memory low
```

### File Operations
```python
MAX_FILE_SIZE_MB = 10                 # Maximum file size for read tool
```

### System Prompt
```python
DEFAULT_SYSTEM_PROMPT = "You are here."
```

Modify these values in `config.py` to customize T.O.M.'s behavior to your needs.

---

## Extending T.O.M.

### Adding Custom Tools

T.O.M.'s tool system is designed for easy extension. Here's how to add a new tool:

```python
# In tools.py

@tool(
    "calculator",
    "Perform mathematical calculations",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Use ast.literal_eval for safety
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
```

The `@tool` decorator automatically:
- Registers the tool in `TOOLS_REGISTRY`
- Adds the tool definition to `TOOLS_DEFINITIONS`
- Makes the tool available to the model

### Modifying the System Prompt

Edit `DEFAULT_SYSTEM_PROMPT` in `config.py`:

```python
DEFAULT_SYSTEM_PROMPT = """You are T.O.M., a helpful AI assistant.

You have access to tools and should use them when appropriate.
Always explain your reasoning when making decisions.
Be concise but thorough in your responses."""
```

### Customizing Context Limits

Adjust context management in `config.py`:

```python
# Use 90% of model's context instead of 80%
CONTEXT_USAGE_RATIO = 0.9

# Trim to 70% when limit exceeded (more aggressive)
CONTEXT_TRIM_TARGET_RATIO = 0.7

# Allow larger tool results
TOOL_RESULT_CONTEXT_RATIO = 0.3
MAX_TOOL_RESULT_TOKENS = 12000
```

---

## Performance Tips

### Optimal Configuration

For best performance on Apple Silicon:

1. **Enable prompt caching** (default): Significant speedup after warmup
2. **Use quantized cache** (8-bit default): Balances speed and memory
3. **Allow auto-GC** (default): Prevents memory accumulation
4. **Monitor context usage**: Use `/stats` regularly to avoid excessive trimming

### Memory Optimization

If experiencing memory issues:

```bash
# Reduce context window
python main.py --max-context 16000

# Increase GC frequency
python main.py --gc-frequency 2

# Manual GC during long sessions
You> /gc
```

### Troubleshooting Slow Performance

1. Check cache status: `You> /cache`
   - Low hit rate? Cache may need reset: `You> /clear-cache`
2. Check memory: `You> /memory`
   - High MLX cache? Run: `You> /gc`
3. Check context: `You> /stats`
   - Near limit? Context trimming causes slowdowns

---

## Technical Details

### Model Specifications

- **Base Model**: Qwen3-4B-Thinking-2507
- **Parameters**: 4 billion
- **Architecture**: Transformer-based with thinking capabilities
- **Context Window**: 32,768 tokens
- **Quantization**: 8-bit (default), configurable to 4-bit or none
- **Special Tokens**: 
  - Thinking delimiters: `<think>` (151667) and `</think>` (151668)
  - Tool call markers: `<tool_call>` and `</tool_call>`

### MLX Optimization

T.O.M. leverages Apple's MLX framework for optimal performance on Apple Silicon:

- **Unified Memory**: Efficient memory sharing between CPU and GPU
- **Metal Backend**: Direct GPU acceleration without framework overhead
- **Dynamic Computation Graphs**: Flexible model execution
- **Lazy Evaluation**: Computation deferred until needed

### Token Counting

T.O.M. uses a hybrid token counting approach:

1. **Accurate counting**: Uses tokenizer when available
2. **Fallback estimation**: 1 token ≈ 4 characters when tokenizer unavailable
3. **Context tracking**: Monitors system prompt, messages, and tool definitions
4. **Trimming logic**: Preserves recent context when limits approached

---

## Project Status

- ✅ Architecture with separation of concerns
- ✅ Advanced LLM optimization techniques (prompt caching, quantization)
- ✅ Extensible tool system for agentic capabilities
- ✅ Modular design enabling multiple interface modes
- ✅ Interactive CLI with modern UX patterns
- ✅ Optional GUI launcher with system tray integration
- ✅ Documentation and code organization

**Future Enhancements:**
- Additional tools 
- Memory
- Enhanced GUI features
- Integration with external APIs

---

## Requirements

```
mlx
mlx-lm
prompt-toolkit
psutil
PySide6 # For launcher.py GUI (optional)
Rich
```

**Note**: PySide6 is only required for the launcher GUI (`launcher.py`). The core CLI (`main.py`) works without it.

---

## License

MIT License

Copyright (c) 2025 Keith Duke

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Acknowledgments

- **Qwen Team** for the Qwen3-4B-Thinking-2507 model
- **Apple MLX Team** for the MLX framework
- **Open source community** for the excellent libraries that make this possible