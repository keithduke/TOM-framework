# T.O.M. CLI

**T**erminal **O**rchestrated **M**odel - A production-ready agentic framework built on Qwen3-4B-Thinking-2507 with MLX optimization for Apple Silicon.

T.O.M. is a **local-first** AI assistant: every component (MLX runtime, FastAPI service, CLI/GUI/web clients) runs on your own machine and talks over loopback. There is no hosted backend, so your data, cache, and tools never leave your device. The framework layers intelligent prompt caching, tool-calling, and transparent reasoning on top of that local runtime while keeping the modules modular and ergonomic.

---

## What's New (Sprint 1)

- **API-first runtime:** `services/api` now owns orchestration while every adapter (CLI, PySide6, and the web UI) connects over HTTP + SSE for identical behavior.
- **Unified entry point:** `python main.py` launches the FastAPI server and web UI by default, while `--cli` and `--pyside` reuse the same backend. `launcher.py` simply proxies to PySide mode.
- **Live streaming everywhere:** CLI, GUI, and web clients subscribe to `/chat/stream`, so thinking/tool events display live without duplicated final responses.
- **Desktop polish:** The PySide shell keeps the input focused after launch and each turn, mirrors streaming output exactly once, and integrates with the system tray.
- **Web client stability:** The browser UI handles incremental streaming safely and busts caches so new assets load immediately.
- **Tokenizers tuning:** `main.py` sets `TOKENIZERS_PARALLELISM=true` before any workers spawn, eliminating Hugging Face fork warnings without throttling throughput.

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
- `fastapi[standard]` - Installs FastAPI plus the recommended extras: `email-validator` (Pydantic email validation), `httpx` (Starlette TestClient), `jinja2` (templating), `python-multipart` (form parsing), `uvicorn[standard]` (server with uvloop and friends), and `fastapi-cli[standard]`/`fastapi-cloud-cli` (CLI tooling & deploys)
- `pytest` - Lightweight test runner used by our `pytest -q` workflow
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

`main.py` is now the single entry point for every adapter:

- **Default** – `python main.py` starts the FastAPI service on `http://127.0.0.1:8000`, serves the static client at `/web/`, and opens your browser automatically.
- **CLI** – `python main.py --cli` runs the original terminal client (all CLI flags still apply, e.g. `python main.py --cli --model ./Qwen3-4B`).
- **PySide desktop** – `python main.py --pyside` spins up the API server in the background and launches the Qt GUI with tray integration and always-focused input.
- **Legacy launcher** – `python launcher.py` continues to work but now simply proxies to `python main.py --pyside`.

This keeps every workflow on the same code path: FastAPI owns orchestration, and each UI is just a thin adapter layered on top.

---

## Using T.O.M.

T.O.M. keeps three presentation layers in sync via the FastAPI backend:

- **Web UI (default)**: `python main.py` starts the API and opens the browser at `http://127.0.0.1:8000/web/`. Chat, inspect tool calls, and monitor thinking directly in the browser.
- **Terminal CLI**: `python main.py --cli [extra CLI flags]` launches the original prompt-toolkit experience. Main automatically hosts the FastAPI backend (unless `--api-base`/`TOM_API_BASE` is provided), so the CLI shares the same runtime as PySide/web. For legacy direct-core mode, run `python ui/cli/main.py` explicitly.
- **PySide6 desktop shell**: `python main.py --pyside` runs the Qt window with tray integration while transparently hosting the API in the background. Input focus stays in the composer after the app launches and after every turn, so you can continue typing immediately. The legacy `python launcher.py` command simply forwards to this mode.

Every adapter shares the same cache, model runtime, and tooling—they only differ in UI.
All of them now listen to the FastAPI server’s SSE stream so “thinking”, tool calls, and final responses appear live while a turn is executing.

### PySide (Desktop) Features

When using `python main.py --pyside` (or the compatibility `launcher.py`), you get:

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
- **Always-ready input**: Composer keeps focus after launch and after each response.

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

---

## Testing

Before opening a PR (and after large refactors), verify the following:

1. `pytest -q` – exercises API/CLI integration, prompt construction, and tool execution with the stubbed model manager.
2. `python main.py` – confirms the FastAPI server boots, the browser UI streams responses (including tool calls), and no console errors appear.
3. `python main.py --cli` – validates the terminal UI against the shared API server (try a tool call and a non-tool question to confirm no duplicate assistant output).
4. `python main.py --pyside` – ensures the desktop shell launches, the input remains focused after each prompt, and SSE streaming mirrors the CLI/web behavior.

Document any deviations plus logs/screenshots in your PR so reviewers can reproduce issues quickly.

## Troubleshooting

- **“huggingface/tokenizers: The current process just got forked…”** – `main.py` now sets `TOKENIZERS_PARALLELISM=true` before spawning any workers, so the warning should be gone when you use the standard entry points. If you embed the runtime elsewhere, export `TOKENIZERS_PARALLELISM=true` yourself before forking.
- **Web client shows stale JavaScript** – Hard-refresh the browser or bump the cache-busting query (`/web/app.js?v=…`) if you tweak static assets while the server is running.
- **Duplicate assistant replies** – All adapters now deduplicate SSE `assistant` events. If you observe repeats, confirm you are hitting `/chat/stream` and aren’t replaying cached responses.
- **PySide input loses focus** – The composer auto-focuses after launch and every submission. If it doesn’t, ensure you’re on the latest code and no global Qt shortcut is stealing focus.
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

### Local FastAPI Service

T.O.M. also exposes the same orchestration stack through a FastAPI app that runs entirely on your machine. Start it with:

```bash
uvicorn services.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Endpoints:
- `GET /health` – readiness + model information
- `POST /sessions` – create a conversation session (override system prompt or context size as needed)
- `POST /sessions/{id}/chat` – submit a user message; the service handles thinking, tool execution, and the assistant reply in a single payload
- `POST /sessions/{id}/messages` and `DELETE /sessions/{id}` – manually inspect or manage session history

To point the CLI at the API instead of loading the local model, launch it with `python main.py --api-base http://127.0.0.1:8000`. If you’ve set `TOM_API_KEY` on the server, pass `--api-key` (or set `TOM_API_KEY` in the CLI environment) so requests include the `X-TOM-API-Key` header. Use `python main.py --pyside` for the desktop client (it talks to the same API). The web client arriving in Phase 3 will reuse these endpoints as well.

---

## Diagnostics & Troubleshooting

The repository ships with a few diagnostic harnesses that double as pytest suites and standalone scripts. When you `pytest -q` they act like normal tests, but running the files directly surfaces richer walkthroughs:
- `python test_prompt_building.py` prints how prompts are assembled with/without tools and inspects tokenizer chat templates.
- `python test_chat_template_issue.py` simulates the known chat-template failure modes (rejects, ignores, or honors the `tools` parameter) and suggests mitigations.
- `python test_end_to_end.py` replays the entire tool-call pipeline, summarizing each phase and highlighting failures.
- `python test_tool_system.py` exercises individual tool utilities with verbose output.

Use these scripts whenever you need to diagnose prompt wiring or tool execution issues beyond the terse pytest output.

---

## Architecture

> See `ROADMAP.md` for the multi-phase plan that follows the Sprint 1 refactor.

### Design Philosophy

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Testability**: Modules can be tested independently
3. **Maintainability**: Changes are localized to specific modules
4. **Reusability**: Components can be imported and used by other projects

### Module Structure

```
├── main.py                 # Unified entry point (web/API, CLI, PySide)
├── launcher.py             # Back-compat shim for PySide launcher
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
- Unified entry point
- Parses global flags (`--cli`, `--pyside`, `--host`, `--port`, etc.)
- Starts FastAPI + web UI by default or dispatches to CLI/PySide adapters

#### `launcher.py`
- Legacy Qt launcher entry point
- Calls `python main.py --pyside` for compatibility
- Maintained for existing shortcuts/scripts pointing directly at `launcher.py`

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
main.py (Unified entry)
  ├─→ services/api/main.py (default web/API mode)
  ├─→ ui/cli/main.py (when --cli)
  └─→ ui/pyside6/launcher.py (when --pyside)

ui/cli/main.py
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

launcher.py (legacy entry)
  └─→ python main.py --pyside
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

### PySide Desktop Mode

```bash
# Launch GUI application (with local API)
python main.py --pyside

# Compatibility shim
python launcher.py
```

The PySide launcher automatically starts the FastAPI server (if it is not already running) and then opens the Qt window. The `launcher.py` script remains solely for back-compat; new workflows should call `python main.py --pyside` instead.

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
