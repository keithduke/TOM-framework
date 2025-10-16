# T.O.M. CLI

A minimal, hackable agent framework for running Qwen3-4B-Thinking locally on Apple Silicon. A starting point for building your own AI tools without API costs, entirely on your Mac.

**What it does:** Runs a capable small language model with tool calling, long-context management, and prompt caching. The agent can read files, check system time, and maintain multi-turn conversations - all processed locally using MLX.

**Who it's for:** Developers who want to learn agentic design, experiment without burning API credits, or need a clean template to extend. This isn't a kitchen-sink, it's tightly focused on getting the best inference possible from one specific model, but could be easily extended to others.

## Why This Exists

Most agent frameworks try to support every model and every use case. That flexibility comes at a cost - complexity, bloat, and performance compromises.

T.O.M. takes the opposite approach: hyper-focused on Qwen3-4B-Thinking running via MLX on Apple Silicon. By optimizing for one model and one platform, we can make it fast, efficient, and simple enough to actually understand and modify.

This is a **template**, not a product. Take it, extend it, make it do what you need. The codebase is deliberately small and modular so you can see how everything works and change what you don't like.

I built this after two more complex iterations (including one with a web UI) taught me that simple, terminal-focused tools are easier to reason about and extend. This is version 3 - the one that actually shipped.

## Module Structure
1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **Testability**: Modules can be tested independently
3. **Maintainability**: Changes are localized to specific modules
4. **Reusability**: Modules can be imported by other projects

```
├── main.py              # Entry point
├── cli.py               # CLI interface and interactive loop
├── model_manager.py     # Model loading, caching, and generation
├── context_manager.py   # Conversation context and prompt building
├── tools.py             # Tool system and built-in tools
├── utils.py             # Helper functions
└── config.py            # Configuration constants
```

## Module Responsibilities

### `main.py`
- **Purpose**: Application entry point
- **Responsibilities**: Import and run the Typer app

### `config.py`
- **Purpose**: Centralized configuration
- **Contains**:
  - Generation parameters (temperature, top_p, etc.)
  - Context management settings
  - Memory management defaults
  - System prompt

### `utils.py`
- **Purpose**: Shared utility functions
- **Contains**:
  - `ordinal()` - Date formatting helper
  - `load_model_config()` - Read model config.json

### `tools.py`
- **Purpose**: Complete tool system
- **Contains**:
  - `@tool` decorator for registration
  - `TOOLS_REGISTRY` and `TOOLS_DEFINITIONS`
  - Built-in tools (`get_datetime`, `read_file`)
  - `execute_tool_call()` - Execute tool by name
  - `extract_tool_calls()` - Parse tool calls from response
  - `truncate_tool_result()` - Intelligent result truncation

### `context_manager.py`
- **Purpose**: Conversation state management
- **Contains**:
  - `TokenCounter` - Token counting utilities
  - `ContextManager` - Message history, trimming, prompt building
  - `get_stats()` - Context statistics

### `model_manager.py`
- **Purpose**: Model lifecycle and generation
- **Contains**:
  - `ModelManager` class
  - Model/tokenizer loading
  - Prompt cache initialization, prewarming, save/load
  - `generate_response()` - Core generation with thinking parsing
  - `_parse_thinking_and_content()` - Separates thinking from output
  - Garbage collection management
  - Cache info and reset methods

### `cli.py`
- **Purpose**: User interface and interaction
- **Contains**:
  - `ChatInterface` - Main chat loop
  - Typer app and commands
  - Rich console display methods
  - Interactive commands (/stats, /cache, /memory, etc.)
  - Response generation and display orchestration
  - Tool call processing loop

## Data Flow

```
User Input
    ↓
ChatInterface (cli.py)
    ↓
ContextManager.add_message() (context_manager.py)
    ↓
ModelManager.generate_response() (model_manager.py)
    ├─→ ContextManager.build_prompt()
    └─→ MLX generate()
    ↓
extract_tool_calls() (tools.py)
    ↓
execute_tool_call() (tools.py)
    ↓
ModelManager.generate_response() [follow-up]
    ↓
Display to User
```

## Import Dependencies

```
main.py
  └─→ cli.py
        ├─→ config.py
        ├─→ context_manager.py
        │     ├─→ config.py
        │     └─→ tools.py
        ├─→ model_manager.py
        │     ├─→ config.py
        │     ├─→ context_manager.py
        │     └─→ tools.py
        ├─→ tools.py
        │     ├─→ config.py
        │     └─→ utils.py
        └─→ utils.py
              └─→ config.py
```

## Usage

```bash
# Interactive mode
python main.py
python main.py --model ./path/to/model

# With options
python main.py -m ./model --max-context 16000 --debug

# Clear cache
python main.py clear-cache --model ./model --force
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url> my-agent
cd my-agent
```

I use python 3.11, installed via homebrew to create and activate a virtual environment to isolate python dependencies:

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your terminal prompt, indicating the virtual environment is active. If unfamiliar with venv see [venv documentation](https://docs.python.org/3/library/venv.html)
From this step forward we are working in the virtual environment.

#### Install dependencies
```bash
pip install -r requirements.txt
```
#### Download and Convert AI Model

The framework uses Qwen3-4B-Thinking-2507, which needs to be downloaded from Hugging Face and converted to MLX format optimized for Apple Silicon. Here is the huggingface page for the model [huggingface](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
```bash
python -m mlx_lm convert --hf-path Qwen/Qwen3-4B-Thinking-2507 \
  --mlx-path ./Qwen3-4B-Thinking-2507-8bit \
  -q --q-bits 8
```
What this command does:

--hf-path: Downloads the model from Hugging Face

--mlx-path: Saves the converted model locally

--q --q-bits 8: Converts to 8bit 

Note: This process may take several ( < 5 for me ) minutes depending mostly on your internet connection. You can also use --q-bits 4 to convert to 4bit if you need smaller. 