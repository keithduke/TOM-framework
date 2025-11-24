# Sprint 5 — Advanced Tools & Improvements

## Objective
Add additional tools, performance optimizations, and quality-of-life improvements. Complete Sprint 5 with TOM as a fully-featured, production-ready personal coding agent.

## Sprint 5 Deliverables

### 1. Write File Tool
**Priority:** High | **Effort:** 3 days

Add ability to create and modify files.

Create `write_file` tool in `core/tools.py`:

```python
@tool(
    "write_file",
    "Write content to a file. Creates new file or overwrites existing file. "
    "Use with caution as this modifies the filesystem.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to write to",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
            "mode": {
                "type": "string",
                "description": "Write mode: 'w' (overwrite) or 'a' (append). Default: 'w'",
                "enum": ["w", "a"],
            },
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str, mode: str = "w") -> str:
    """
    Write content to a file with security checks.

    Args:
        path: File path to write
        content: Content to write
        mode: Write mode ('w' or 'a')

    Returns:
        Success message or error
    """
    from .audit_logger import get_audit_logger
    from .security import is_path_allowed

    audit = get_audit_logger()

    try:
        file_path = Path(path).expanduser().resolve()

        # Security check: Path allowed?
        if not is_path_allowed(file_path):
            audit.log_file_access(
                file_path=str(file_path),
                operation="write",
                allowed=False,
                error="Path not allowed",
            )
            return f"Error: Access denied. Path not in allowed directories: {path}"

        # Security check: Don't overwrite sensitive files
        if is_sensitive_file(file_path):
            audit.log_file_access(
                file_path=str(file_path),
                operation="write",
                allowed=False,
                error="Sensitive file",
            )
            return f"Error: Cannot write to sensitive file: {path}"

        # Check if file exists
        exists = file_path.exists()
        if exists and mode == "w":
            logger.warning(f"Overwriting existing file: {file_path}")

        # Write file
        with open(file_path, mode, encoding="utf-8") as f:
            f.write(content)

        # Log successful write
        audit.log_file_access(
            file_path=str(file_path),
            operation="write",
            allowed=True,
            success=True,
            size=len(content),
        )

        action = "Updated" if exists and mode == "a" else "Created" if not exists else "Overwrote"
        logger.info(f"{action} {file_path}: {len(content)} chars")

        return f"{action} file: {file_path.name} ({len(content)} characters)"

    except PermissionError:
        audit.log_file_access(
            file_path=str(file_path),
            operation="write",
            allowed=True,
            success=False,
            error="Permission denied",
        )
        return f"Error: Permission denied writing to: {path}"

    except Exception as e:
        audit.log_file_access(
            file_path=str(file_path),
            operation="write",
            allowed=True,
            success=False,
            error=str(e),
        )
        logger.error(f"Error writing {path}: {e}")
        return f"Error writing file: {str(e)}"
```

**Example usage:**
```
User: Create a README file for this project
TOM: <tool_call>{
  "name": "write_file",
  "arguments": {
    "path": "./README.md",
    "content": "# Project Name\n\nDescription here..."
  }
}</tool_call>

User: Add a note to that file
TOM: <tool_call>{
  "name": "write_file",
  "arguments": {
    "path": "./README.md",
    "content": "\n\n## Note\nAdded via TOM",
    "mode": "a"
  }
}</tool_call>
```

**Security Considerations:**
- Use same path validation as `read_file`
- Don't allow overwriting sensitive files
- Log all writes to audit log
- Warn when overwriting existing files
- Consider adding approval prompts (future)

**Tasks:**
- [ ] Implement `write_file` tool
- [ ] Add security validation
- [ ] Add audit logging
- [ ] Write comprehensive tests
- [ ] Update documentation

---

### 2. HTTP Request Tool
**Priority:** Medium | **Effort:** 3 days

Add ability to make HTTP requests to APIs.

```python
import httpx
from typing import Optional, Dict

@tool(
    "http_request",
    "Make HTTP request to an API. Supports GET and POST methods. "
    "Use for fetching data from web services, APIs, etc.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to request (must be valid HTTP/HTTPS URL)",
            },
            "method": {
                "type": "string",
                "description": "HTTP method: GET or POST. Default: GET",
                "enum": ["GET", "POST"],
            },
            "headers": {
                "type": "object",
                "description": "Optional HTTP headers",
            },
            "body": {
                "type": "string",
                "description": "Request body (for POST requests)",
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds. Default: 10",
            },
        },
        "required": ["url"],
    },
)
def http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[str] = None,
    timeout: int = 10,
) -> str:
    """
    Make HTTP request with security controls.

    Args:
        url: Target URL
        method: HTTP method
        headers: Optional headers
        body: Optional request body
        timeout: Request timeout

    Returns:
        Response body or error message
    """
    from .config_loader import get_security_config
    from .audit_logger import get_audit_logger

    config = get_security_config()
    audit = get_audit_logger()

    # Validate URL
    if not url.startswith(("http://", "https://")):
        return "Error: URL must start with http:// or https://"

    # Check allowed domains (if configured)
    # allowed_domains = config.http_allowed_domains
    # if allowed_domains and domain not in allowed_domains:
    #     return f"Error: Domain not allowed: {domain}"

    # Block localhost/internal IPs (security)
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
        return "Error: Cannot request localhost URLs"

    # Log request
    audit.log_event("http_request", {
        "url": url,
        "method": method,
        "has_body": body is not None,
    })

    try:
        with httpx.Client(timeout=timeout) as client:
            if method == "GET":
                response = client.get(url, headers=headers)
            elif method == "POST":
                response = client.post(url, headers=headers, content=body)
            else:
                return f"Error: Unsupported method: {method}"

            # Check response size
            max_size = 100000  # 100KB
            content = response.text
            if len(content) > max_size:
                content = content[:max_size] + f"\n\n... (truncated, {len(content)} total chars)"

            # Format response
            result = f"Status: {response.status_code}\n"
            if response.status_code >= 400:
                result += f"Error response:\n{content}"
            else:
                result += f"\n{content}"

            return result

    except httpx.TimeoutException:
        return f"Error: Request timeout after {timeout}s"
    except httpx.RequestError as e:
        return f"Error: Request failed - {e}"
    except Exception as e:
        logger.error(f"HTTP request error: {e}")
        return f"Error: {str(e)}"
```

**Example usage:**
```
User: Check if the GitHub API is up
TOM: <tool_call>{
  "name": "http_request",
  "arguments": {
    "url": "https://api.github.com/status"
  }
}</tool_call>

User: Get information about the Python requests library
TOM: <tool_call>{
  "name": "http_request",
  "arguments": {
    "url": "https://pypi.org/pypi/requests/json"
  }
}</tool_call>
```

**Security Considerations:**
- Block localhost/internal IPs
- Rate limiting (future)
- Allowed domain whitelist (future)
- Response size limits
- Timeout enforcement
- No authentication secrets in logs

**Tasks:**
- [ ] Implement `http_request` tool
- [ ] Add URL validation
- [ ] Add domain restrictions
- [ ] Add rate limiting (future)
- [ ] Test with various APIs
- [ ] Document usage and security

---

### 3. Code Analysis Tool
**Priority:** Medium | **Effort:** 2 days

Add ability to analyze Python code for issues.

```python
@tool(
    "analyze_python",
    "Analyze Python code for style issues, potential bugs, and complexity. "
    "Returns analysis results from pylint or flake8.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to Python file to analyze",
            },
            "tool": {
                "type": "string",
                "description": "Analysis tool: pylint or flake8. Default: flake8",
                "enum": ["pylint", "flake8"],
            },
        },
        "required": ["file_path"],
    },
)
def analyze_python(file_path: str, tool: str = "flake8") -> str:
    """
    Analyze Python code for issues.

    Args:
        file_path: Path to Python file
        tool: Analysis tool to use

    Returns:
        Analysis results or error
    """
    from .security import is_path_allowed

    file_path = Path(file_path).expanduser().resolve()

    # Security check
    if not is_path_allowed(file_path):
        return f"Error: Access denied: {file_path}"

    if not file_path.exists():
        return f"Error: File not found: {file_path}"

    if file_path.suffix != ".py":
        return f"Error: Not a Python file: {file_path}"

    # Use shell tool if available, otherwise subprocess
    from .tools import shell_execute, get_shell_executor

    executor = get_shell_executor()

    if executor:
        # Use shell tool
        if tool == "pylint":
            command = f"pylint {file_path}"
        else:  # flake8
            command = f"flake8 {file_path}"

        result = shell_execute(command, working_dir=str(file_path.parent))
        return result
    else:
        # Fallback to direct subprocess
        import subprocess

        try:
            if tool == "pylint":
                cmd = ["pylint", str(file_path)]
            else:
                cmd = ["flake8", str(file_path)]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}"

            return output or "No issues found ✓"

        except FileNotFoundError:
            return f"Error: {tool} not installed. Install with: pip install {tool}"
        except subprocess.TimeoutExpired:
            return "Error: Analysis timeout"
        except Exception as e:
            return f"Error: {str(e)}"
```

**Tasks:**
- [ ] Implement `analyze_python` tool
- [ ] Support pylint and flake8
- [ ] Add mypy support (future)
- [ ] Test with various code samples
- [ ] Document usage

---

### 4. Performance Optimizations
**Priority:** Medium | **Effort:** 1 week

#### 4.1 Async Tool Execution

Currently tools execute sequentially. Add async support:

```python
# core/async_tools.py

import asyncio
from typing import List, Dict, Any

async def execute_tool_async(tool_call: Dict[str, Any]) -> str:
    """Execute single tool asynchronously."""
    # Wrapper around synchronous execute_tool_call
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        execute_tool_call,
        tool_call
    )

async def execute_tools_parallel(tool_calls: List[Dict[str, Any]]) -> List[str]:
    """
    Execute multiple tools concurrently.

    Args:
        tool_calls: List of tool call dictionaries

    Returns:
        List of results in same order
    """
    tasks = [execute_tool_async(call) for call in tool_calls]
    return await asyncio.gather(*tasks)
```

Update `services/api/runtime.py` to use async execution:

```python
async def execute_tools_concurrent(self, tool_calls: List[Dict]) -> List[str]:
    """Execute tools concurrently when safe."""
    from core.async_tools import execute_tools_parallel

    # Check if tools can run in parallel
    # (e.g., all read-only operations)
    can_parallelize = all(
        call.get("name") in {"read", "get_datetime", "http_request"}
        for call in tool_calls
    )

    if can_parallelize:
        logger.info(f"Executing {len(tool_calls)} tools in parallel")
        return await execute_tools_parallel(tool_calls)
    else:
        # Sequential execution for safety
        results = []
        for call in tool_calls:
            result = execute_tool_call(call)
            results.append(result)
        return results
```

**Benefits:**
- Faster multi-tool execution
- Better resource utilization
- Improved user experience

#### 4.2 Lazy Token Counting

Cache token counts to avoid repeated calculation:

```python
# core/token_cache.py

from functools import lru_cache

class TokenCache:
    """Cache token counts for frequently-accessed strings."""

    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size

    def count_tokens(self, text: str, tokenizer) -> int:
        """Get cached count or compute."""
        # Use hash for cache key
        key = hash(text)

        if key in self.cache:
            return self.cache[key]

        # Compute
        count = TokenCounter.estimate_tokens(text, tokenizer)

        # Cache if space available
        if len(self.cache) < self.max_size:
            self.cache[key] = count

        return count

    def clear(self):
        """Clear cache."""
        self.cache.clear()
```

#### 4.3 Context Building Optimization

Build prompts incrementally instead of rebuilding:

```python
# core/incremental_prompt.py

class IncrementalPromptBuilder:
    """Build prompts incrementally for performance."""

    def __init__(self, system_prompt: str, tools: list):
        self.system_prompt = system_prompt
        self.tools = tools

        # Build static part once
        self.static_part = self._build_static()
        self.messages = []

    def _build_static(self) -> str:
        """Build static part (system + tools)."""
        import json

        parts = [f"System: {self.system_prompt}"]

        if self.tools:
            tools_str = json.dumps(self.tools, indent=2)
            parts.append(f"Available Tools:\n{tools_str}")

        return "\n\n".join(parts) + "\n\n"

    def add_message(self, role: str, content: str):
        """Add message incrementally."""
        self.messages.append((role, content))

    def get_prompt(self) -> str:
        """Get full prompt."""
        message_parts = []

        for role, content in self.messages:
            if role == "user":
                message_parts.append(f"User: {content}")
            elif role == "assistant":
                message_parts.append(f"Assistant: {content}")
            elif role == "tool":
                message_parts.append(f"User:\n<tool_response>\n{content}\n</tool_response>")

        message_parts.append("Assistant:")

        return self.static_part + "\n\n".join(message_parts)
```

**Tasks:**
- [ ] Implement async tool execution
- [ ] Add token count caching
- [ ] Optimize prompt building
- [ ] Benchmark improvements
- [ ] Test for correctness

**Expected Improvements:**
- 30-50% faster multi-tool execution
- 20% reduction in token counting time
- Smoother user experience

---

### 5. Quality of Life Improvements
**Priority:** Low | **Effort:** 1 week

#### 5.1 Session Persistence

Save and resume sessions:

```python
# core/session_store.py

import json
from pathlib import Path
from datetime import datetime
from typing import Optional

class SessionStore:
    """Persist and restore conversation sessions."""

    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            store_path = Path.home() / ".tom" / "sessions"

        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)

    def save_session(self, session_id: str, context: ContextManager):
        """Save session to disk."""
        session_file = self.store_path / f"{session_id}.json"

        data = {
            "session_id": session_id,
            "saved_at": datetime.now().isoformat(),
            "system_prompt": context.system_prompt,
            "max_context_tokens": context.max_context_tokens,
            "messages": context.messages,
        }

        with open(session_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved session {session_id}")

    def load_session(self, session_id: str) -> Optional[ContextManager]:
        """Load session from disk."""
        session_file = self.store_path / f"{session_id}.json"

        if not session_file.exists():
            return None

        with open(session_file) as f:
            data = json.load(f)

        # Reconstruct context
        context = ContextManager(
            max_context_tokens=data["max_context_tokens"]
        )
        context.system_prompt = data["system_prompt"]

        for msg in data["messages"]:
            context.add_message(msg["role"], msg["content"])

        logger.info(f"Loaded session {session_id}")
        return context

    def list_sessions(self) -> List[Dict]:
        """List all saved sessions."""
        sessions = []

        for session_file in self.store_path.glob("*.json"):
            try:
                with open(session_file) as f:
                    data = json.load(f)

                sessions.append({
                    "session_id": data["session_id"],
                    "saved_at": data["saved_at"],
                    "message_count": len(data["messages"]),
                })
            except Exception as e:
                logger.error(f"Failed to read session {session_file}: {e}")

        return sorted(sessions, key=lambda s: s["saved_at"], reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Delete saved session."""
        session_file = self.store_path / f"{session_id}.json"

        if session_file.exists():
            session_file.unlink()
            logger.info(f"Deleted session {session_id}")
            return True

        return False
```

Add CLI commands:

```python
def cmd_save_session(self):
    """Save current session."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    store = SessionStore()
    store.save_session(session_id, self.context_manager)
    console.print(f"[green]✓ Saved session: {session_id}[/green]")

def cmd_load_session(self):
    """Load a saved session."""
    store = SessionStore()
    sessions = store.list_sessions()

    if not sessions:
        console.print("[dim]No saved sessions[/dim]")
        return

    # Show sessions and prompt for selection
    # ... interactive selection ...

def cmd_list_sessions(self):
    """List saved sessions."""
    store = SessionStore()
    sessions = store.list_sessions()

    # Display table of sessions
```

#### 5.2 Conversation Summarization

Summarize old messages to save context:

```python
@tool(
    "summarize_context",
    "Summarize older parts of our conversation to save space. Internal tool.",
    parameters={"type": "object", "properties": {}}
)
def summarize_context() -> str:
    """
    Trigger context summarization.
    Model will generate summary of old messages.
    """
    # This tool signals to the system to perform summarization
    return "Summarization requested"

# In context manager
def summarize_old_messages(self, model_manager):
    """Summarize messages to compress context."""

    # Get messages to summarize (oldest 50%)
    split_point = len(self.messages) // 2
    to_summarize = self.messages[:split_point]

    if len(to_summarize) < 5:
        return  # Not worth summarizing

    # Build summary prompt
    summary_prompt = "Summarize this conversation:\n\n"
    for msg in to_summarize:
        summary_prompt += f"{msg['role']}: {msg['content'][:500]}\n"

    summary_prompt += "\nProvide a concise summary in 2-3 sentences."

    # Generate summary
    _, summary = model_manager.generate_response(...)

    # Replace old messages with summary
    self.messages = [
        {"role": "assistant", "content": f"[Summary of earlier conversation: {summary}]"}
    ] + self.messages[split_point:]

    logger.info(f"Summarized {len(to_summarize)} messages")
```

#### 5.3 Multi-Model Support (Future)

Prepare for multiple model backends:

```python
# core/model_registry.py

class ModelRegistry:
    """Manage multiple model backends."""

    def __init__(self):
        self.models = {}
        self.default_model = None

    def register(self, name: str, model_class, **config):
        """Register a model backend."""
        self.models[name] = {
            "class": model_class,
            "config": config,
        }

    def get_model(self, name: Optional[str] = None):
        """Get model by name or default."""
        name = name or self.default_model

        if name not in self.models:
            raise ValueError(f"Unknown model: {name}")

        # Instantiate if needed
        # ...

# Example:
registry = ModelRegistry()
registry.register("qwen-4b", ModelManager, model_path="...")
registry.register("llama-8b", ModelManager, model_path="...")
registry.default_model = "qwen-4b"
```

**Tasks:**
- [ ] Implement session persistence
- [ ] Add conversation summarization
- [ ] Prepare multi-model architecture
- [ ] Test and document features

---

## Sprint 5 Timeline

**Total Effort:** 3-4 weeks

- **Week 1:** Write file tool, HTTP request tool
- **Week 2:** Code analysis tool, performance optimizations
- **Week 3:** QoL improvements (sessions, summarization)
- **Week 4:** Testing, documentation, polish

## Testing Checklist

### New Tools
- [ ] `write_file` creates files correctly
- [ ] `write_file` appends correctly
- [ ] `write_file` security checks work
- [ ] `http_request` makes requests
- [ ] `http_request` handles errors
- [ ] `http_request` blocks localhost
- [ ] `analyze_python` runs linters
- [ ] `analyze_python` handles errors

### Performance
- [ ] Async tool execution works
- [ ] Parallel execution faster than sequential
- [ ] Token caching improves performance
- [ ] No correctness regressions

### QoL Features
- [ ] Sessions save correctly
- [ ] Sessions load correctly
- [ ] Session list accurate
- [ ] Summarization reduces context

## Documentation Tasks

- [ ] Document all new tools with examples
- [ ] Add performance tuning guide
- [ ] Update README with all features
- [ ] Create advanced usage guide
- [ ] Document session management
- [ ] Add troubleshooting section

## Success Criteria

- [ ] All new tools functional
- [ ] Performance measurably improved
- [ ] Sessions can be saved/restored
- [ ] Documentation complete
- [ ] All tests passing
- [ ] Ready for production use

## Future Enhancements (Post-Sprint 5)

### Tools
- [ ] Database query tool (SQLite, PostgreSQL)
- [ ] Image viewing/analysis (if model supports)
- [ ] Screenshot OCR
- [ ] Environment variable management
- [ ] Process management

### Performance
- [ ] Streaming token generation (when MLX supports)
- [ ] Model quantization options
- [ ] Batch request handling
- [ ] Resource quotas (CPU/memory limits)

### Features
- [ ] User approval prompts for dangerous operations
- [ ] Tool marketplace/plugin system
- [ ] Custom tool creation via config
- [ ] Interactive shell sessions
- [ ] Multi-agent coordination

### Infrastructure
- [ ] Containerization (Docker)
- [ ] Cloud deployment guides
- [ ] Metrics and observability
- [ ] API authentication
- [ ] Rate limiting
- [ ] Log rotation

---

**Status:** 🚧 Ready to Start
**Dependencies:** Sprint 4 (configuration and audit)
**Outcome:** Production-ready TOM with comprehensive capabilities

---

## Post-Sprint 5: TOM 1.0 Release

After completing Sprints 2-5, TOM will have:

✅ **Security:** Multi-layer protection, audit logging, user configuration
✅ **Capabilities:** Shell access, file I/O, HTTP requests, code analysis
✅ **Performance:** Async execution, token caching, optimized prompts
✅ **Usability:** Session management, clear documentation, multiple UIs
✅ **Production-Ready:** Comprehensive testing, error handling, monitoring

**TOM becomes a fully-featured personal coding agent—local, private, and powerful.**
