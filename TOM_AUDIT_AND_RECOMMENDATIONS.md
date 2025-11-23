# TOM Framework: Comprehensive Code Audit & Enhancement Roadmap

**Audit Date:** 2025-11-23
**Auditor:** Claude (Sonnet 4.5)
**Framework Version:** Post-Sprint 1 (API-First Architecture)

---

## Executive Summary

TOM (Terminal Orchestrated Model) is a **well-architected, production-quality agent framework** with clean separation of concerns, comprehensive testing, and professional engineering practices. The codebase demonstrates solid fundamentals with 29 Python files implementing a local-first AI assistant on MLX-optimized Qwen3-4B-Thinking-2507.

**Key Findings:**
- ✅ **Architecture**: Excellent separation between core, services, and UI layers
- ✅ **Code Quality**: Clean, well-documented, follows best practices
- ⚠️ **Minor Issues**: Some unused code, missing security hardening for tools
- 🚀 **Major Opportunity**: Shell/Bash access would dramatically expand capabilities

**Recommendation:** TOM is ready for shell access with appropriate sandboxing. This single enhancement would unlock enormous potential while maintaining the framework's solid foundation.

---

## 1. Architecture Assessment

### 1.1 Strengths

**Separation of Concerns (Excellent)**
```
core/           → Business logic (tools, context, model management)
services/api/   → API layer (FastAPI, sessions, streaming)
ui/             → Presentation adapters (CLI, PySide6, Web)
```

This design enables:
- Independent testing of each layer
- Multiple UI adapters sharing same backend
- Easy extensibility without breaking existing code

**API-First Design (Production-Ready)**
- FastAPI with proper async/await patterns
- SSE streaming for real-time updates
- Session isolation with thread-safe context swapping
- Clean REST endpoints following conventions

**Context Management (Sophisticated)**
- Intelligent token counting with fallback estimation
- Automatic trimming with cache invalidation
- Support for tool/function roles in conversation
- Robust prompt construction with template fallbacks

**Caching System (Advanced)**
- Automatic sizing based on static content
- Persistent cache with quantization support
- Hit/miss tracking and statistics
- Smart reset strategies on major context changes

### 1.2 Design Patterns Observed

1. **Decorator Pattern**: `@tool` for registration
2. **Dependency Injection**: ModelManager owns ContextManager
3. **Factory Pattern**: Cache and model initialization
4. **Strategy Pattern**: Fallback prompt building
5. **Observer Pattern**: SSE event streaming

### 1.3 Code Quality Metrics

- **Documentation**: Excellent docstrings and inline comments
- **Error Handling**: Comprehensive try-except with logging
- **Logging**: Consistent use of Python logging framework
- **Type Hints**: Partial coverage (could be improved)
- **Testing**: Multiple test files with both pytest and standalone modes

---

## 2. Issues Identified

### 2.1 Unused/Deprecated Code

**Critical Finding:** `truncate_tool_result()` is deprecated but still imported

**Location:** `core/tools.py:217-223`
```python
def truncate_tool_result(result: str, tool_name: str, max_chars: int) -> str:
    """
    DEPRECATED: No-op function kept for backward compatibility.
    Returns result unchanged.
    """
    return result
```

**Impact:**
- Still imported in `ui/cli/cli.py:43` and `services/api/runtime.py` (indirectly)
- Called in two places in CLI code (lines 280, 390)
- Creates confusion about actual truncation behavior

**Recommendation:**
- Remove the function entirely
- Use direct truncation in `services/api/runtime.py:227-228` pattern:
  ```python
  if len(result) > max_chars:
      result = result[:max_chars] + "…"
  ```
- Update CLI to match API runtime pattern
- Remove from `core/tools.py` exports

### 2.2 Security Considerations

**File Reading Tool** (`core/tools.py:66-109`)

Current implementation is reasonably safe but could be hardened:

**Existing Protections:**
- ✅ File size limit (10MB)
- ✅ Character limit (15,000 chars)
- ✅ UTF-8 encoding check
- ✅ Permission error handling
- ✅ Path resolution with `expanduser()` and `resolve()`

**Missing Protections:**
- ⚠️ No path traversal validation (could read `/etc/passwd` if user allows)
- ⚠️ No blocklist for sensitive files (.env, credentials, private keys)
- ⚠️ No sandboxing to limit accessible directories

**Recommendations:**
1. Add configurable allowed directories:
   ```python
   ALLOWED_READ_PATHS = [
       Path.home() / "Documents",
       Path.cwd(),
       # User configurable
   ]
   ```

2. Implement path validation:
   ```python
   def is_path_allowed(file_path: Path) -> bool:
       resolved = file_path.resolve()
       return any(
           resolved.is_relative_to(allowed)
           for allowed in ALLOWED_READ_PATHS
       )
   ```

3. Add sensitive file blocklist:
   ```python
   BLOCKED_FILES = {
       ".env", "credentials.json", "id_rsa",
       "id_ed25519", ".aws/credentials"
   }
   ```

### 2.3 Missing Type Hints

While the codebase has some type hints, coverage is incomplete:

**Examples:**
- `context_manager.py:211` - `Dict[str, any]` should be `Dict[str, Any]`
- Many function parameters lack type hints
- Return types sometimes omitted

**Impact:** Reduced IDE support and potential runtime errors

**Recommendation:** Add mypy to CI/CD and gradually improve coverage

### 2.4 Error Handling Gaps

**Tool Execution** (`core/tools.py:136-158`)

Current implementation catches broad exceptions but could be more specific:

```python
except TypeError as e:
    # Good - specific exception
except Exception as e:
    # Too broad - masks different error types
```

**Recommendation:**
- Add specific exception types (FileNotFoundError, PermissionError, etc.)
- Create custom exception classes for tool errors
- Provide more context in error messages for debugging

### 2.5 Testing Gaps

**Current Coverage:**
- ✅ Unit tests for tool system
- ✅ Integration tests for API endpoints
- ✅ End-to-end conversation flow tests
- ✅ Prompt building tests

**Missing:**
- ⚠️ No tests for security edge cases (path traversal, injection)
- ⚠️ No load/stress tests for concurrent sessions
- ⚠️ No tests for cache corruption/recovery
- ⚠️ No tests for memory leak scenarios

---

## 3. Shell Access: Deep Dive Analysis

### 3.1 Why Shell Access Transforms TOM

**Current Capability:** 2 tools (datetime, file reading)

**With Shell Access:** Hundreds of capabilities instantly available:
- File system operations (ls, find, grep, sed, awk)
- Git operations (status, diff, log, commit)
- Development tools (npm, pip, cargo, make)
- System monitoring (ps, top, df, netstat)
- Text processing (cat, head, tail, wc, sort)
- Network operations (curl, wget, ping)
- And literally every CLI tool installed on the system

**Strategic Value:**
- TOM becomes a **true coding agent** like Claude Code
- Zero maintenance for new capabilities (inherits system tools)
- Users can extend via their own scripts
- Natural integration with developer workflows

### 3.2 Implementation Architecture

**Recommended Approach: Sandboxed Subprocess Execution**

```python
# core/tools.py

@tool(
    "shell",
    "Execute a shell command in a sandboxed environment. Use for file operations, git commands, system queries, etc.",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute"
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory (must be within allowed paths)"
            },
            "timeout": {
                "type": "integer",
                "description": "Command timeout in seconds (default: 30)",
                "default": 30
            }
        },
        "required": ["command"]
    }
)
def shell_execute(
    command: str,
    working_dir: str = None,
    timeout: int = 30
) -> str:
    """
    Execute shell command with safety constraints.
    """
    # Validation
    if not is_command_allowed(command):
        return f"Error: Command blocked by security policy: {command}"

    # Set working directory
    cwd = Path(working_dir or Path.cwd()).resolve()
    if not is_path_allowed(cwd):
        return f"Error: Directory not allowed: {cwd}"

    # Execute with timeout
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            timeout=timeout,
            capture_output=True,
            text=True,
            env=get_safe_environment()
        )

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"

        if result.returncode != 0:
            output = f"Exit code {result.returncode}\n{output}"

        # Truncate if too large
        max_output = 10000  # chars
        if len(output) > max_output:
            output = output[:max_output] + f"\n... (truncated, {len(output)} total chars)"

        return output

    except subprocess.TimeoutExpired:
        return f"Error: Command timeout after {timeout}s"
    except Exception as e:
        return f"Error executing command: {e}"
```

### 3.3 Security Model: Multi-Layer Defense

**Layer 1: Command Allowlist/Blocklist**

```python
# core/shell_security.py

# Approach 1: Allowlist (most secure)
ALLOWED_COMMANDS = {
    # File operations
    "ls", "cat", "head", "tail", "find", "grep", "wc", "sort", "uniq",

    # Git operations
    "git",

    # Development
    "npm", "pip", "python", "node", "cargo", "make",

    # System info
    "ps", "df", "du", "which", "type",

    # Text processing
    "sed", "awk", "cut", "tr", "jq"
}

# Approach 2: Blocklist (more permissive)
BLOCKED_COMMANDS = {
    # Destructive
    "rm", "rmdir", "dd", "mkfs", "fdisk",

    # System modification
    "sudo", "su", "chown", "chmod",

    # Network attacks
    "nmap", "nc", "netcat",

    # Process control
    "kill", "killall", "pkill",

    # Package managers (if you want to prevent installs)
    "apt-get", "yum", "brew"
}

def is_command_allowed(command: str) -> bool:
    """
    Validate command against security policy.
    """
    # Extract base command
    base_cmd = command.strip().split()[0].split("|")[0]

    # Remove path if present
    base_cmd = Path(base_cmd).name

    # Check against policy (choose one approach)
    # Option 1: Allowlist
    if base_cmd not in ALLOWED_COMMANDS:
        logger.warning(f"Command blocked (not in allowlist): {base_cmd}")
        return False

    # Option 2: Blocklist
    # if base_cmd in BLOCKED_COMMANDS:
    #     logger.warning(f"Command blocked (in blocklist): {base_cmd}")
    #     return False

    # Additional checks
    if "&&" in command or ";" in command:
        # Allow pipes but be cautious with command chaining
        logger.warning(f"Command chaining detected: {command}")
        # Could allow with additional validation

    return True
```

**Layer 2: Path Restrictions**

```python
# core/config.py

# Configurable allowed paths
SHELL_ALLOWED_PATHS = [
    Path.home(),                    # User home directory
    Path.cwd(),                     # Current working directory
    Path("/tmp"),                   # Temp directory
    # User can add more via config file
]

# Blocked paths (overrides allowed)
SHELL_BLOCKED_PATHS = [
    Path("/etc"),
    Path("/var"),
    Path("/usr/bin"),
    Path("/System"),                # macOS system
    Path.home() / ".ssh",           # SSH keys
    Path.home() / ".aws",           # AWS credentials
]
```

**Layer 3: Environment Sanitization**

```python
def get_safe_environment() -> dict:
    """
    Provide sanitized environment variables for shell execution.
    """
    safe_env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(Path.home()),
        "USER": os.environ.get("USER", ""),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        # Add more as needed
    }

    # Explicitly exclude sensitive vars
    # (AWS_*, API_KEY*, TOKEN*, etc.)

    return safe_env
```

**Layer 4: Resource Limits**

```python
import resource

def set_resource_limits():
    """
    Limit resources for subprocess execution.
    """
    # Limit CPU time
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))

    # Limit memory
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))

    # Limit file size
    resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))

# Use in subprocess.run with preexec_fn
subprocess.run(
    command,
    preexec_fn=set_resource_limits,
    ...
)
```

### 3.4 Configuration System

**User Configuration File** (`~/.tom/security.yaml`)

```yaml
shell:
  enabled: true

  # Security mode: "allowlist" | "blocklist" | "disabled"
  mode: allowlist

  allowed_commands:
    - git
    - ls
    - cat
    - grep
    - npm
    - python
    # User adds more

  blocked_commands:
    - rm
    - sudo
    - kill

  allowed_paths:
    - ~/Documents
    - ~/Projects
    - /tmp

  blocked_paths:
    - ~/.ssh
    - ~/.aws
    - ~/secrets

  limits:
    timeout: 30
    max_output_chars: 10000
    max_memory_mb: 512
    max_cpu_seconds: 30
```

**Loading Configuration:**

```python
# core/shell_config.py

import yaml
from pathlib import Path
from typing import Dict, Any

DEFAULT_CONFIG = {
    "shell": {
        "enabled": False,  # Opt-in by default
        "mode": "allowlist",
        "allowed_commands": ["ls", "cat", "git", "grep"],
        "blocked_commands": ["rm", "sudo", "kill"],
        "allowed_paths": [str(Path.home()), str(Path.cwd())],
        "blocked_paths": [str(Path.home() / ".ssh")],
        "limits": {
            "timeout": 30,
            "max_output_chars": 10000,
        }
    }
}

def load_shell_config() -> Dict[str, Any]:
    """Load shell security configuration."""
    config_path = Path.home() / ".tom" / "security.yaml"

    if not config_path.exists():
        logger.info("No security config found, using defaults")
        return DEFAULT_CONFIG

    with open(config_path) as f:
        user_config = yaml.safe_load(f)

    # Merge with defaults
    config = {**DEFAULT_CONFIG, **user_config}

    logger.info(f"Loaded shell config: mode={config['shell']['mode']}")
    return config
```

### 3.5 Audit Logging

**Track All Shell Commands:**

```python
# core/shell_audit.py

import json
from datetime import datetime
from pathlib import Path

AUDIT_LOG = Path.home() / ".tom" / "shell_audit.log"

def log_shell_command(
    command: str,
    working_dir: str,
    allowed: bool,
    result: str = None,
    error: str = None
):
    """Log shell command execution for security audit."""

    entry = {
        "timestamp": datetime.now().isoformat(),
        "command": command,
        "working_dir": working_dir,
        "allowed": allowed,
        "success": error is None,
        "result_length": len(result) if result else 0,
        "error": error
    }

    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
```

### 3.6 User Control & Transparency

**Permission Prompts (Optional Feature):**

```python
def shell_execute_with_approval(command: str, **kwargs) -> str:
    """
    Execute shell command with optional user approval.
    """
    # Check if approval required (configurable)
    if requires_approval(command):
        approval = prompt_user_approval(command)
        if not approval:
            return "Command execution cancelled by user"

    return shell_execute(command, **kwargs)

def prompt_user_approval(command: str) -> bool:
    """
    Ask user to approve command execution.
    Could be implemented in UI layer.
    """
    # In CLI: Use rich.prompt.Confirm
    # In Web: WebSocket message to browser
    # In PySide: QMessageBox
    pass
```

**Shell History Viewer:**

```python
# Add to CLI commands

def cmd_shell_history(self):
    """Show recent shell commands executed by TOM."""

    if not AUDIT_LOG.exists():
        console.print("[dim]No shell history[/dim]")
        return

    # Read last N entries
    with open(AUDIT_LOG) as f:
        entries = [json.loads(line) for line in f.readlines()[-50:]]

    table = Table(title="Shell Command History")
    table.add_column("Time", style="cyan")
    table.add_column("Command", style="yellow")
    table.add_column("Status", style="green")

    for entry in entries:
        status = "✓" if entry["success"] else "✗"
        table.add_row(
            entry["timestamp"],
            entry["command"][:60],
            status
        )

    console.print(table)
```

### 3.7 Phased Rollout Plan

**Phase 1: Basic Implementation (Week 1)**
- [ ] Implement `shell_execute()` with basic security
- [ ] Add allowlist/blocklist validation
- [ ] Add path restrictions
- [ ] Add timeout and output truncation
- [ ] Write comprehensive tests

**Phase 2: Configuration System (Week 2)**
- [ ] Create `~/.tom/security.yaml` config
- [ ] Implement config loading and merging
- [ ] Add CLI command to edit config safely
- [ ] Add config validation

**Phase 3: Audit & Monitoring (Week 3)**
- [ ] Implement audit logging
- [ ] Add `/shell-history` command
- [ ] Add shell statistics to `/stats`
- [ ] Create shell security documentation

**Phase 4: Advanced Features (Week 4)**
- [ ] Optional user approval prompts
- [ ] Resource limits (CPU, memory)
- [ ] Environment variable sanitization
- [ ] Interactive shell sessions (optional)

**Phase 5: Testing & Hardening (Week 5)**
- [ ] Security penetration testing
- [ ] Performance benchmarking
- [ ] Edge case handling
- [ ] User acceptance testing

### 3.8 Example Use Cases

**1. Git Operations**
```
User: What files changed in the last commit?
TOM: <tool_call>{"name": "shell", "arguments": {"command": "git diff HEAD~1 --name-only"}}</tool_call>
→ src/app.py
→ tests/test_app.py
TOM: The last commit modified src/app.py and tests/test_app.py.

User: Show me the git log
TOM: <tool_call>{"name": "shell", "arguments": {"command": "git log --oneline -10"}}</tool_call>
```

**2. File System Exploration**
```
User: Find all Python files in the project that import numpy
TOM: <tool_call>{"name": "shell", "arguments": {"command": "find . -name '*.py' -exec grep -l 'import numpy' {} \\;"}}</tool_call>
→ ./src/data_processor.py
→ ./src/analytics.py
TOM: I found 2 files importing numpy: src/data_processor.py and src/analytics.py.
```

**3. Development Workflow**
```
User: Run the tests
TOM: <tool_call>{"name": "shell", "arguments": {"command": "pytest -v"}}</tool_call>
→ test_api.py::test_health PASSED
→ test_api.py::test_sessions PASSED
TOM: All tests passed! ✓

User: Check the test coverage
TOM: <tool_call>{"name": "shell", "arguments": {"command": "pytest --cov=core --cov-report=term-missing"}}</tool_call>
```

**4. System Monitoring**
```
User: How much disk space is left?
TOM: <tool_call>{"name": "shell", "arguments": {"command": "df -h ."}}</tool_call>
→ Filesystem      Size  Used Avail Use% Mounted on
→ /dev/disk1s1   500G  350G  150G  71% /

User: What Python processes are running?
TOM: <tool_call>{"name": "shell", "arguments": {"command": "ps aux | grep python"}}</tool_call>
```

---

## 4. Additional Enhancement Opportunities

### 4.1 More Built-in Tools

**Write File Tool**
```python
@tool(
    "write_file",
    "Write or update a file with given content",
    parameters={...}
)
def write_file(path: str, content: str, mode: str = "w") -> str:
    """Write content to file with safety checks."""
    # Validate path
    # Check if overwriting existing file
    # Write with proper encoding
    # Return confirmation
```

**HTTP Request Tool**
```python
@tool(
    "http_request",
    "Make HTTP requests to APIs",
    parameters={...}
)
def http_request(url: str, method: str = "GET", **kwargs) -> str:
    """Make HTTP request with rate limiting."""
    # Validate URL (no localhost unless configured)
    # Rate limiting
    # Timeout enforcement
    # Return formatted response
```

**Code Analysis Tool**
```python
@tool(
    "analyze_code",
    "Analyze Python code for issues using pylint or flake8",
    parameters={...}
)
def analyze_code(file_path: str, tool: str = "pylint") -> str:
    """Run static analysis on code."""
    # Could use shell tool internally
    # Or use ast module for Python files
```

### 4.2 Context Improvements

**Semantic Trimming**

Current trimming is FIFO (first-in-first-out). Could implement:

```python
def trim_by_importance(messages: List[Dict]) -> List[Dict]:
    """
    Trim messages based on importance score.

    Keep:
    - Recent messages (last 5)
    - Messages with tool calls
    - Messages with errors
    - User corrections

    Remove:
    - Old casual conversation
    - Redundant confirmations
    """
    # Score each message
    # Sort by score
    # Keep until token budget
```

**Conversation Summarization**

```python
@tool(
    "summarize_context",
    "Summarize older parts of conversation to compress context",
    parameters={}
)
def summarize_context() -> str:
    """
    Have model summarize old messages into a compact summary.
    Replace old messages with summary to save tokens.
    """
    # Extract messages to summarize
    # Generate summary with model
    # Replace in context with single summary message
```

### 4.3 Multi-Modal Capabilities

**Image Understanding** (if model supports)

```python
@tool(
    "view_image",
    "Analyze an image file",
    parameters={...}
)
def view_image(image_path: str) -> str:
    """
    Load image and describe contents.
    Requires vision-capable model.
    """
```

**Code Screenshot Analysis**

```python
@tool(
    "analyze_screenshot",
    "Extract code or text from screenshot",
    parameters={...}
)
def analyze_screenshot(image_path: str) -> str:
    """
    Use OCR to extract text from screenshots.
    Could use pytesseract or similar.
    """
```

### 4.4 Memory & Persistence

**Long-Term Memory**

```python
# core/memory.py

class LongTermMemory:
    """
    Store important facts across sessions.
    """

    def __init__(self, storage_path: Path):
        self.storage = storage_path / "memory.json"
        self.facts = self.load()

    def remember(self, key: str, value: Any):
        """Store a fact for future sessions."""
        self.facts[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        self.save()

    def recall(self, key: str) -> Optional[Any]:
        """Retrieve a stored fact."""
        return self.facts.get(key, {}).get("value")

    def search(self, query: str) -> List[tuple]:
        """Search memory for relevant facts."""
        # Simple keyword search
        # Could use embedding similarity later
```

**Session Persistence**

Currently sessions are in-memory. Could add:

```python
# services/api/persistence.py

def save_session(session: SessionData):
    """Save session to disk for resumption."""
    session_file = Path(f"~/.tom/sessions/{session.session_id}.json")
    data = {
        "session_id": session.session_id,
        "created": datetime.now().isoformat(),
        "messages": session.context.messages,
        "system_prompt": session.context.system_prompt
    }
    with open(session_file, "w") as f:
        json.dump(data, f, indent=2)

def load_session(session_id: str) -> Optional[SessionData]:
    """Resume a saved session."""
    session_file = Path(f"~/.tom/sessions/{session_id}.json")
    if not session_file.exists():
        return None

    with open(session_file) as f:
        data = json.load(f)

    # Reconstruct session
    context = ContextManager(...)
    context.system_prompt = data["system_prompt"]
    for msg in data["messages"]:
        context.add_message(msg["role"], msg["content"])

    return SessionData(session_id=session_id, context=context)
```

### 4.5 Advanced Tool Features

**Tool Chaining**

```python
# Allow model to plan multi-step tool sequences

Example:
User: Clone the react repo and show me the README
TOM:
  1. <tool_call>{"name": "shell", "arguments": {"command": "git clone https://github.com/facebook/react.git /tmp/react"}}</tool_call>
  2. <tool_call>{"name": "read", "arguments": {"location": "/tmp/react/README.md"}}</tool_call>
```

**Parallel Tool Execution**

```python
# Execute multiple independent tools concurrently

async def execute_tools_parallel(tool_calls: List[Dict]) -> List[str]:
    """Execute tools concurrently when safe."""
    tasks = [
        asyncio.create_task(execute_tool_async(call))
        for call in tool_calls
    ]
    return await asyncio.gather(*tasks)
```

**Tool Error Recovery**

```python
# Automatically retry or fall back on tool errors

def execute_with_retry(tool_call: Dict, max_retries: int = 3) -> str:
    """Execute tool with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return execute_tool_call(tool_call)
        except RetryableError as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            time.sleep(wait)
    ```

---

## 5. Performance Optimizations

### 5.1 Current Performance

**Strengths:**
- ✅ Prompt caching delivers 2-5x speedup
- ✅ MLX optimization for Apple Silicon
- ✅ Automatic GC prevents memory leaks
- ✅ SSE streaming for responsive UI

**Bottlenecks:**
- Context trimming is CPU-intensive
- Token counting can be slow on large contexts
- No batching for multi-tool execution

### 5.2 Optimization Opportunities

**Async Tool Execution**

```python
# Currently sequential
for tool_call in tool_calls:
    result = execute_tool_call(tool_call)  # Blocking
    # Process result

# Could be async
results = await asyncio.gather(*[
    execute_tool_async(call) for call in tool_calls
])
```

**Lazy Token Counting**

```python
class LazyTokenCounter:
    """Cache token counts to avoid repeated calculation."""

    def __init__(self):
        self._cache = {}

    def count(self, text: str, tokenizer) -> int:
        cache_key = hash(text)
        if cache_key not in self._cache:
            self._cache[cache_key] = TokenCounter.estimate_tokens(text, tokenizer)
        return self._cache[cache_key]
```

**Streaming Context Building**

```python
# Build prompts incrementally instead of rebuilding each time
class IncrementalPromptBuilder:
    """Build prompts by appending deltas instead of rebuilding."""

    def __init__(self):
        self.static_part = ""  # System + tools (cached)
        self.dynamic_part = ""  # Messages (incremental)

    def add_message(self, role: str, content: str):
        """Append message to existing prompt."""
        self.dynamic_part += f"\n{role}: {content}"

    def get_full_prompt(self) -> str:
        return self.static_part + self.dynamic_part
```

---

## 6. Documentation Improvements

### 6.1 Current State

**Strengths:**
- ✅ Comprehensive README with examples
- ✅ Architecture documentation
- ✅ Code comments and docstrings
- ✅ Testing documentation

**Gaps:**
- ⚠️ No API documentation (OpenAPI/Swagger)
- ⚠️ No deployment guide
- ⚠️ No contribution guidelines
- ⚠️ No security best practices guide

### 6.2 Recommended Documentation

**API Documentation**

```python
# services/api/main.py

from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="TOM API",
        version="1.0.0",
        description="Terminal Orchestrated Model API",
        routes=app.routes,
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

Access at: `http://localhost:8000/docs`

**Security Guide** (`SECURITY.md`)

```markdown
# Security Best Practices

## Tool Security

### File Reading
- Only enable in trusted environments
- Configure allowed paths in `~/.tom/security.yaml`
- Review audit logs regularly

### Shell Access
- **Disabled by default** - opt-in required
- Use allowlist mode for maximum security
- Never run TOM with sudo/root privileges
- Review command history in `~/.tom/shell_audit.log`

## Network Security

- TOM runs on localhost by default
- Do not expose port 8000 to internet
- Use API keys if deploying remotely

## Data Privacy

- All processing is local (no external API calls)
- Conversation history stored in memory only
- Cache files contain model state, not user data
```

**Contribution Guide** (`CONTRIBUTING.md`)

```markdown
# Contributing to TOM

## Development Setup

1. Fork and clone
2. Create virtual environment
3. Install dev dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `pytest -v`

## Code Style

- Follow PEP 8
- Add type hints to new code
- Write docstrings for public functions
- Keep functions focused and small

## Adding Tools

1. Add tool function to `core/tools.py`
2. Use `@tool` decorator
3. Write tests in `test_tool_system.py`
4. Update README with example usage
5. Consider security implications

## Submitting PRs

- Create feature branch
- Write tests
- Update documentation
- Run full test suite
- Submit PR with clear description
```

---

## 7. Recommended Priorities

### 7.1 Immediate (This Week)

**Priority 1: Clean Up Unused Code**
- [ ] Remove `truncate_tool_result()` function
- [ ] Update all imports
- [ ] Run tests to ensure nothing breaks

**Priority 2: Security Hardening**
- [ ] Add path validation to `read_file` tool
- [ ] Implement sensitive file blocklist
- [ ] Add allowed paths configuration

**Priority 3: Type Hints**
- [ ] Fix `Dict[str, any]` → `Dict[str, Any]`
- [ ] Add mypy configuration
- [ ] Run mypy and fix critical issues

### 7.2 Short Term (This Month)

**Priority 4: Shell Access - Phase 1**
- [ ] Implement basic `shell_execute()` tool
- [ ] Add allowlist/blocklist validation
- [ ] Add path restrictions
- [ ] Write comprehensive tests
- [ ] Document security model

**Priority 5: Configuration System**
- [ ] Create `~/.tom/security.yaml` structure
- [ ] Implement config loading
- [ ] Add validation

**Priority 6: Documentation**
- [ ] Add OpenAPI/Swagger docs
- [ ] Write SECURITY.md
- [ ] Write CONTRIBUTING.md
- [ ] Create tool development guide

### 7.3 Medium Term (Next 3 Months)

**Priority 7: Shell Access - Advanced**
- [ ] Audit logging
- [ ] Resource limits
- [ ] User approval prompts
- [ ] Shell history viewer

**Priority 8: Additional Tools**
- [ ] Write file tool
- [ ] HTTP request tool
- [ ] Code analysis tool

**Priority 9: Context Improvements**
- [ ] Semantic trimming
- [ ] Conversation summarization
- [ ] Long-term memory

**Priority 10: Performance**
- [ ] Async tool execution
- [ ] Lazy token counting
- [ ] Benchmark and optimize hot paths

### 7.4 Long Term (6+ Months)

**Priority 11: Advanced Features**
- [ ] Multi-modal support (if model allows)
- [ ] Tool marketplace/registry
- [ ] Session persistence
- [ ] Distributed deployment

**Priority 12: Enterprise Features**
- [ ] Role-based access control
- [ ] Audit logging and compliance
- [ ] Multi-tenancy
- [ ] SSO integration

---

## 8. Risk Assessment

### 8.1 Current Risks

**Low Risk:**
- Code quality issues (minor, easily fixed)
- Missing type hints (gradual improvement)
- Documentation gaps (no blocking issues)

**Medium Risk:**
- Path traversal in file reading (mitigated by user control)
- No security audits performed (recommend before production)
- Session data not persistent (could lose work)

**High Risk (if Shell Access Added Without Safeguards):**
- Command injection
- Destructive operations (rm -rf)
- Privilege escalation
- Resource exhaustion

### 8.2 Mitigation Strategies

**For Shell Access:**
1. **Default Deny**: Shell access disabled by default, opt-in required
2. **Multi-Layer Security**: Allowlist + path restrictions + resource limits + audit logging
3. **User Control**: Clear configuration, easy to disable, transparent logging
4. **Graceful Degradation**: If shell disabled, TOM still fully functional
5. **Documentation**: Clear security guidelines, warning about risks
6. **Testing**: Comprehensive security testing before release

**For General Security:**
1. **Input Validation**: Validate all user inputs and tool parameters
2. **Principle of Least Privilege**: Run with minimal permissions
3. **Audit Logging**: Track all potentially dangerous operations
4. **Regular Updates**: Keep dependencies updated
5. **Security Review**: Regular code audits, especially for tool system

---

## 9. Conclusion

### 9.1 Summary

TOM is a **production-quality framework** with excellent architecture and clean code. The identified issues are minor and easily addressable. The codebase demonstrates:

- Strong separation of concerns
- Comprehensive testing
- Good documentation
- Professional engineering practices

### 9.2 Key Recommendations

**1. Shell Access is the Killer Feature**

Adding shell access with proper sandboxing would transform TOM from a "promising framework" to a "production-ready coding agent" comparable to Claude Code. The multi-layer security model outlined in this document makes this achievable with acceptable risk.

**2. Security First**

Before adding shell access:
- Implement all security layers (allowlist, path restrictions, resource limits, audit logging)
- Create comprehensive test suite for security scenarios
- Write clear security documentation
- Default to disabled (opt-in)

**3. Incremental Rollout**

Don't try to build everything at once:
- Phase 1: Basic shell with allowlist
- Phase 2: Configuration system
- Phase 3: Audit logging
- Phase 4: Advanced features
- Phase 5: Hardening and testing

**4. Maintain the Foundation**

TOM's architecture is solid. As you add features:
- Keep the clean separation of concerns
- Maintain test coverage
- Document everything
- Don't sacrifice simplicity for features

### 9.3 Final Verdict

**TOM is ready for shell access.** The framework has the architectural foundation to support powerful capabilities safely. With the security model outlined in this document, you can give TOM an enormous toolbox while maintaining user control and system security.

**Next Step:** Start with Priority 1-3 (cleanup and hardening), then move to Priority 4 (basic shell implementation with security). With careful execution, TOM can become a personal coding agent that rivals commercial alternatives while remaining fully local and user-controlled.

---

## Appendix A: Quick Wins Checklist

These can be implemented in a few hours each:

- [ ] Remove `truncate_tool_result()` and update callers (30 min)
- [ ] Fix `Dict[str, any]` → `Dict[str, Any]` (10 min)
- [ ] Add `.env` to sensitive files blocklist in read_file (15 min)
- [ ] Add OpenAPI documentation (30 min)
- [ ] Create SECURITY.md (1 hour)
- [ ] Add mypy configuration (30 min)
- [ ] Write contribution guide (1 hour)
- [ ] Add pre-commit hooks for linting (30 min)

Total time investment: ~5 hours for significant quality improvements

---

## Appendix B: Testing Checklist for Shell Access

Before releasing shell access feature:

**Functionality Tests:**
- [ ] Execute simple command (ls, pwd)
- [ ] Execute with working directory
- [ ] Execute with timeout
- [ ] Handle command errors gracefully
- [ ] Truncate long output correctly

**Security Tests:**
- [ ] Blocked command rejected (rm, sudo)
- [ ] Path traversal prevented
- [ ] Timeout enforced
- [ ] Output size limited
- [ ] Audit log written correctly

**Edge Cases:**
- [ ] Empty command
- [ ] Command with special characters
- [ ] Very long command
- [ ] Command that hangs
- [ ] Command that produces binary output

**Integration Tests:**
- [ ] Shell tool called by model
- [ ] Result processed correctly
- [ ] Multiple tool calls in sequence
- [ ] Shell + other tools in same conversation

**Performance Tests:**
- [ ] Long-running command (within timeout)
- [ ] Large output handling
- [ ] Concurrent tool executions

---

## Appendix C: Example Shell Security Config

**Conservative (Recommended for Most Users):**

```yaml
shell:
  enabled: true
  mode: allowlist

  allowed_commands:
    # Read-only file operations
    - ls
    - cat
    - head
    - tail
    - find
    - grep
    - less
    - wc

    # Git (read-only operations are safe)
    - git

    # Development tools
    - python
    - node
    - npm
    - pip

    # System info
    - ps
    - df
    - du
    - which

  allowed_paths:
    - ~/Documents
    - ~/Projects
    - /tmp

  blocked_paths:
    - ~/.ssh
    - ~/.aws
    - ~/.config

  limits:
    timeout: 30
    max_output_chars: 10000
```

**Permissive (For Advanced Users):**

```yaml
shell:
  enabled: true
  mode: blocklist

  blocked_commands:
    # Destructive
    - rm
    - rmdir
    - dd

    # System modification
    - sudo
    - su
    - chown
    - chmod

    # Process control
    - kill
    - killall

  blocked_paths:
    - ~/.ssh
    - ~/.aws
    - /etc
    - /var

  limits:
    timeout: 60
    max_output_chars: 50000
```

**Locked Down (For Shared/Public Machines):**

```yaml
shell:
  enabled: false  # Completely disabled
```

---

**End of Audit Report**

Generated: 2025-11-23
Document Version: 1.0
Framework Version: TOM Post-Sprint 1
