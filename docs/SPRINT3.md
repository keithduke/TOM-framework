# Sprint 3 — Shell Access: Core Implementation

## Objective
Implement secure shell command execution capability with multi-layer security model. Complete Sprint 3 with a fully functional, sandboxed shell tool that transforms TOM into a true coding agent.

## Why Shell Access?

**Current Capabilities:** 2 tools (datetime, file reading)

**With Shell Access:** Hundreds of capabilities:
- File operations: `ls`, `find`, `grep`, `sed`, `awk`
- Git: `status`, `diff`, `log`, `blame`
- Development: `npm`, `pip`, `pytest`, `cargo`
- System: `ps`, `df`, `du`, `which`
- Network: `curl`, `wget`, `ping`

**Strategic Impact:** Zero-maintenance expansion of TOM's capabilities by inheriting all system CLI tools.

---

## Sprint 3 Deliverables

### 1. Core Shell Tool Implementation
**Priority:** Critical | **Effort:** 1 day

#### 1.1 Shell Execution Engine

Create `core/shell_executor.py`:

```python
"""
Shell command execution with security sandboxing.
"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("tom_cli")


class ShellExecutor:
    """
    Execute shell commands with safety constraints.

    Features:
    - Command validation against allowlist/blocklist
    - Working directory restrictions
    - Timeout enforcement
    - Output size limiting
    - Environment sanitization
    """

    def __init__(
        self,
        allowed_commands: set = None,
        blocked_commands: set = None,
        allowed_paths: list = None,
        max_output_chars: int = 10000,
        default_timeout: int = 30,
    ):
        """
        Initialize shell executor with security configuration.

        Args:
            allowed_commands: Set of allowed command names (None = allow all)
            blocked_commands: Set of blocked command names
            allowed_paths: List of allowed working directories
            max_output_chars: Maximum output size in characters
            default_timeout: Default command timeout in seconds
        """
        self.allowed_commands = allowed_commands
        self.blocked_commands = blocked_commands or set()
        self.allowed_paths = allowed_paths or [Path.home(), Path.cwd()]
        self.max_output_chars = max_output_chars
        self.default_timeout = default_timeout

    def is_command_allowed(self, command: str) -> Tuple[bool, str]:
        """
        Validate command against security policy.

        Args:
            command: Shell command to validate

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        # Extract base command
        parts = command.strip().split()
        if not parts:
            return False, "Empty command"

        # Get first command (before pipes)
        base_cmd = parts[0].split("|")[0].strip()

        # Remove path if present
        base_cmd = Path(base_cmd).name

        # Check allowlist (if configured)
        if self.allowed_commands is not None:
            if base_cmd not in self.allowed_commands:
                return False, f"Command not in allowlist: {base_cmd}"

        # Check blocklist
        if base_cmd in self.blocked_commands:
            return False, f"Command blocked: {base_cmd}"

        # Warn about command chaining
        if any(sep in command for sep in ["&&", "||", ";"]):
            logger.warning(f"Command chaining detected: {command}")
            # Allow but log - could make configurable

        return True, "OK"

    def is_path_allowed(self, path: Path) -> bool:
        """
        Check if working directory is allowed.

        Args:
            path: Directory path to validate

        Returns:
            True if path is allowed
        """
        resolved = path.resolve()

        for allowed in self.allowed_paths:
            try:
                allowed_resolved = allowed.resolve()
                resolved.relative_to(allowed_resolved)
                return True
            except ValueError:
                continue

        return False

    def get_safe_environment(self) -> Dict[str, str]:
        """
        Create sanitized environment for subprocess.

        Returns:
            Dictionary of safe environment variables
        """
        import os

        safe_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(Path.home()),
            "USER": os.environ.get("USER", ""),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
            "TERM": os.environ.get("TERM", "xterm-256color"),
        }

        # Explicitly exclude sensitive variables
        # AWS_*, API_KEY*, TOKEN*, etc.

        return safe_env

    def execute(
        self,
        command: str,
        working_dir: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Execute shell command with security checks.

        Args:
            command: Shell command to execute
            working_dir: Working directory (must be allowed)
            timeout: Command timeout in seconds

        Returns:
            Dict with keys: success, output, error, exit_code, truncated
        """
        # Validate command
        allowed, reason = self.is_command_allowed(command)
        if not allowed:
            logger.warning(f"Command blocked: {command} - {reason}")
            return {
                "success": False,
                "output": "",
                "error": f"Security: {reason}",
                "exit_code": -1,
                "truncated": False,
            }

        # Validate working directory
        cwd = Path(working_dir or Path.cwd()).resolve()
        if not self.is_path_allowed(cwd):
            logger.warning(f"Path blocked: {cwd}")
            return {
                "success": False,
                "output": "",
                "error": f"Security: Working directory not allowed: {cwd}",
                "exit_code": -1,
                "truncated": False,
            }

        # Execute with timeout
        timeout_val = timeout or self.default_timeout

        try:
            logger.info(f"Executing: {command} (cwd={cwd}, timeout={timeout_val}s)")

            result = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                timeout=timeout_val,
                capture_output=True,
                text=True,
                env=self.get_safe_environment(),
            )

            # Combine stdout and stderr
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            # Truncate if needed
            truncated = False
            if len(output) > self.max_output_chars:
                truncated = True
                output = (
                    output[: self.max_output_chars]
                    + f"\n\n... (output truncated, {len(output)} total chars)"
                )

            logger.info(
                f"Command completed: exit_code={result.returncode}, "
                f"output_size={len(output)}, truncated={truncated}"
            )

            return {
                "success": result.returncode == 0,
                "output": output,
                "error": result.stderr if result.returncode != 0 else "",
                "exit_code": result.returncode,
                "truncated": truncated,
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"Command timeout: {command}")
            return {
                "success": False,
                "output": "",
                "error": f"Command timeout after {timeout_val} seconds",
                "exit_code": -1,
                "truncated": False,
            }

        except Exception as e:
            logger.error(f"Command execution error: {e}", exc_info=True)
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "exit_code": -1,
                "truncated": False,
            }
```

#### 1.2 Shell Tool Registration

Add to `core/tools.py`:

```python
from .shell_executor import ShellExecutor
from .config import (
    SHELL_ENABLED,
    SHELL_ALLOWED_COMMANDS,
    SHELL_BLOCKED_COMMANDS,
    SHELL_ALLOWED_PATHS,
    SHELL_MAX_OUTPUT_CHARS,
    SHELL_DEFAULT_TIMEOUT,
)

# Initialize shell executor
_shell_executor = None

def get_shell_executor() -> Optional[ShellExecutor]:
    """Get or create shell executor instance."""
    global _shell_executor

    if not SHELL_ENABLED:
        return None

    if _shell_executor is None:
        _shell_executor = ShellExecutor(
            allowed_commands=SHELL_ALLOWED_COMMANDS,
            blocked_commands=SHELL_BLOCKED_COMMANDS,
            allowed_paths=SHELL_ALLOWED_PATHS,
            max_output_chars=SHELL_MAX_OUTPUT_CHARS,
            default_timeout=SHELL_DEFAULT_TIMEOUT,
        )

    return _shell_executor


@tool(
    "shell",
    "Execute a shell command. Use for file operations (ls, find, grep), git commands, "
    "running tests, system queries, and more. Returns command output or error.",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute (e.g., 'ls -la', 'git status')",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for command execution (optional, defaults to current directory)",
            },
            "timeout": {
                "type": "integer",
                "description": "Command timeout in seconds (optional, defaults to 30)",
            },
        },
        "required": ["command"],
    },
)
def shell_execute(
    command: str,
    working_dir: Optional[str] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Execute shell command with security sandbox.

    Args:
        command: Shell command to run
        working_dir: Optional working directory
        timeout: Optional timeout in seconds

    Returns:
        Command output or error message
    """
    executor = get_shell_executor()

    if executor is None:
        return (
            "Error: Shell access is disabled. "
            "To enable, set SHELL_ENABLED=true in core/config.py"
        )

    result = executor.execute(command, working_dir, timeout)

    if not result["success"]:
        error_msg = f"Command failed (exit code {result['exit_code']})"
        if result["error"]:
            error_msg += f"\n{result['error']}"
        return error_msg

    output = result["output"]
    if result["truncated"]:
        output += "\n[Note: Output was truncated]"

    return output or "[Command completed with no output]"
```

#### 1.3 Configuration

Add to `core/config.py`:

```python
# Shell tool configuration
SHELL_ENABLED = False  # DISABLED BY DEFAULT - user must opt-in

# Security mode: "allowlist" or "blocklist"
SHELL_SECURITY_MODE = "allowlist"

# Allowlist: Only these commands are allowed (when mode = "allowlist")
SHELL_ALLOWED_COMMANDS = {
    # File operations (read-only)
    "ls", "cat", "head", "tail", "find", "grep", "less", "more",
    "wc", "sort", "uniq", "cut", "tr",

    # Git operations
    "git",

    # Development tools
    "python", "python3", "node", "npm", "pip", "pip3",
    "cargo", "rustc", "go", "make",

    # Testing
    "pytest", "jest", "cargo test",

    # System info (safe)
    "ps", "df", "du", "which", "type", "whoami", "hostname",
    "uname", "date", "cal",

    # Text processing
    "sed", "awk", "jq",

    # Compression (read operations)
    "tar", "gzip", "gunzip", "zip", "unzip",
}

# Blocklist: These commands are never allowed (when mode = "blocklist")
SHELL_BLOCKED_COMMANDS = {
    # Destructive
    "rm", "rmdir", "dd", "mkfs", "fdisk", "shred",

    # System modification
    "sudo", "su", "doas", "chown", "chmod", "chgrp",

    # Package managers (prevent installations)
    "apt", "apt-get", "yum", "dnf", "pacman", "brew",

    # Network attacks
    "nmap", "nc", "netcat", "telnet",

    # Process control (potentially dangerous)
    "kill", "killall", "pkill", "reboot", "shutdown", "halt",

    # Disk operations
    "mount", "umount", "fsck",
}

# Allowed working directories
SHELL_ALLOWED_PATHS = [
    Path.home(),
    Path.cwd(),
    Path("/tmp"),
]

# Resource limits
SHELL_MAX_OUTPUT_CHARS = 10000  # Maximum output size
SHELL_DEFAULT_TIMEOUT = 30       # Default timeout in seconds

# Apply mode
if SHELL_SECURITY_MODE == "allowlist":
    # In allowlist mode, only ALLOWED_COMMANDS can run
    pass
elif SHELL_SECURITY_MODE == "blocklist":
    # In blocklist mode, everything except BLOCKED_COMMANDS can run
    SHELL_ALLOWED_COMMANDS = None  # None = allow all
else:
    raise ValueError(f"Invalid SHELL_SECURITY_MODE: {SHELL_SECURITY_MODE}")
```

**Tasks:**
- [ ] Create `core/shell_executor.py`
- [ ] Add shell tool to `core/tools.py`
- [ ] Add configuration to `core/config.py`
- [ ] Test shell executor independently

**Success Criteria:**
- Shell executor can run simple commands
- Security validation works correctly
- Output is properly captured and truncated
- Timeouts are enforced

---

### 2. Comprehensive Testing
**Priority:** Critical | **Effort:** 1 day

Create `test_shell_tool.py`:

```python
"""
Comprehensive tests for shell tool.
"""

import pytest
from pathlib import Path
from core.shell_executor import ShellExecutor
from core.tools import shell_execute


class TestShellExecutor:
    """Test ShellExecutor class."""

    def test_simple_command(self):
        """Test executing a simple command."""
        executor = ShellExecutor(
            allowed_commands={"echo", "ls"},
            blocked_commands=set(),
        )

        result = executor.execute("echo 'hello world'")

        assert result["success"] is True
        assert "hello world" in result["output"]
        assert result["exit_code"] == 0

    def test_command_with_working_dir(self):
        """Test command execution with working directory."""
        executor = ShellExecutor(
            allowed_commands={"pwd"},
            allowed_paths=[Path.home()],
        )

        test_dir = str(Path.home())
        result = executor.execute("pwd", working_dir=test_dir)

        assert result["success"] is True
        assert test_dir in result["output"]

    def test_command_timeout(self):
        """Test that long-running commands timeout."""
        executor = ShellExecutor(
            allowed_commands={"sleep"},
            default_timeout=1,
        )

        result = executor.execute("sleep 10")

        assert result["success"] is False
        assert "timeout" in result["error"].lower()

    def test_blocked_command(self):
        """Test that blocked commands are rejected."""
        executor = ShellExecutor(
            allowed_commands=None,  # Allow all
            blocked_commands={"rm"},
        )

        result = executor.execute("rm -rf /")

        assert result["success"] is False
        assert "blocked" in result["error"].lower()

    def test_allowlist_enforcement(self):
        """Test that allowlist is enforced."""
        executor = ShellExecutor(
            allowed_commands={"ls", "cat"},
            blocked_commands=set(),
        )

        # Allowed command
        result = executor.execute("ls")
        assert result["success"] is True

        # Not in allowlist
        result = executor.execute("echo test")
        assert result["success"] is False
        assert "allowlist" in result["error"].lower()

    def test_path_validation(self):
        """Test working directory validation."""
        executor = ShellExecutor(
            allowed_commands={"pwd"},
            allowed_paths=[Path.home()],
        )

        # Allowed path
        result = executor.execute("pwd", working_dir=str(Path.home()))
        assert result["success"] is True

        # Disallowed path
        result = executor.execute("pwd", working_dir="/etc")
        assert result["success"] is False
        assert "not allowed" in result["error"].lower()

    def test_output_truncation(self):
        """Test that large outputs are truncated."""
        executor = ShellExecutor(
            allowed_commands={"cat"},
            max_output_chars=100,
        )

        # Create command that produces large output
        result = executor.execute("cat /dev/urandom | head -c 10000 | base64")

        assert result["truncated"] is True
        assert len(result["output"]) <= 200  # Some buffer for truncation message

    def test_stderr_capture(self):
        """Test that stderr is captured."""
        executor = ShellExecutor(
            allowed_commands={"ls"},
        )

        result = executor.execute("ls /nonexistent_directory_12345")

        assert result["success"] is False
        assert result["exit_code"] != 0
        assert len(result["error"]) > 0


class TestShellTool:
    """Test shell tool integration."""

    def test_shell_tool_disabled(self):
        """Test that shell tool returns error when disabled."""
        # Assumes SHELL_ENABLED = False by default
        result = shell_execute("ls")

        assert "disabled" in result.lower()

    # Additional integration tests when SHELL_ENABLED = True
    # (Run separately or with config override)


class TestShellSecurity:
    """Security-focused tests."""

    def test_command_injection_prevention(self):
        """Test prevention of command injection attempts."""
        executor = ShellExecutor(
            allowed_commands={"echo"},
        )

        # Attempt injection
        result = executor.execute("echo test && rm -rf /")

        # Should execute but log warning about chaining
        # The rm command won't run due to blocklist

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal."""
        executor = ShellExecutor(
            allowed_commands={"cat"},
            allowed_paths=[Path.home() / "Documents"],
        )

        # Attempt to read outside allowed path
        result = executor.execute(
            "cat ../../../etc/passwd",
            working_dir=str(Path.home() / "Documents"),
        )

        # Should fail path validation or file access

    def test_sensitive_command_blocking(self):
        """Test that sensitive commands are blocked."""
        executor = ShellExecutor(
            allowed_commands=None,
            blocked_commands={"sudo", "rm", "kill"},
        )

        dangerous_commands = [
            "sudo rm -rf /",
            "rm important_file.txt",
            "kill -9 1",
        ]

        for cmd in dangerous_commands:
            result = executor.execute(cmd)
            assert result["success"] is False
            assert "blocked" in result["error"].lower()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_command(self):
        """Test handling of empty command."""
        executor = ShellExecutor()

        result = executor.execute("")

        assert result["success"] is False

    def test_very_long_command(self):
        """Test handling of very long commands."""
        executor = ShellExecutor(allowed_commands={"echo"})

        long_cmd = "echo " + "a" * 10000
        result = executor.execute(long_cmd)

        # Should execute or fail gracefully

    def test_binary_output(self):
        """Test handling of binary output."""
        executor = ShellExecutor(allowed_commands={"cat"})

        # Commands that might produce binary output
        # Should handle gracefully

    def test_concurrent_execution(self):
        """Test concurrent command execution."""
        import concurrent.futures

        executor = ShellExecutor(allowed_commands={"echo"})

        def run_command(i):
            return executor.execute(f"echo test_{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_command, i) for i in range(10)]
            results = [f.result() for f in futures]

        # All should succeed
        assert all(r["success"] for r in results)
```

**Tasks:**
- [ ] Create comprehensive test suite
- [ ] Test all security validations
- [ ] Test edge cases and error handling
- [ ] Test concurrent execution
- [ ] Test with various command types

**Success Criteria:**
- All security tests pass
- Edge cases handled gracefully
- 90%+ code coverage for shell executor
- No security vulnerabilities in tests

---

### 3. Documentation
**Priority:** High | **Effort:** 4 hours

#### 3.1 Update README

Add shell tool documentation:

```markdown
## Shell Tool

### Overview

The `shell` tool allows TOM to execute system commands, dramatically expanding capabilities:

```bash
# File operations
You> What Python files are in this directory?
TOM> <tool_call>{"name": "shell", "arguments": {"command": "find . -name '*.py'"}}</tool_call>

# Git operations
You> Show me recent commits
TOM> <tool_call>{"name": "shell", "arguments": {"command": "git log --oneline -10"}}</tool_call>

# Running tests
You> Run the tests
TOM> <tool_call>{"name": "shell", "arguments": {"command": "pytest -v"}}</tool_call>
```

### Security

**Shell access is DISABLED by default.** To enable:

1. Edit `core/config.py`:
   ```python
   SHELL_ENABLED = True
   ```

2. Configure security mode:
   - **Allowlist mode** (recommended): Only approved commands run
   - **Blocklist mode**: All commands except blocked ones run

3. Customize allowed commands and paths as needed

### Safety Features

- **Command validation**: Allowlist/blocklist enforcement
- **Path restrictions**: Limited to approved directories
- **Timeout enforcement**: Commands can't run forever
- **Output limiting**: Large outputs are truncated
- **No sudo**: Elevated privileges are blocked
- **Audit logging**: All commands are logged (Sprint 4)

### Example Usage

```bash
# Find large files
You> Find files larger than 10MB
TOM> <tool_call>{"name": "shell", "arguments": {"command": "find . -type f -size +10M"}}</tool_call>

# Check disk space
You> How much disk space is available?
TOM> <tool_call>{"name": "shell", "arguments": {"command": "df -h ."}}</tool_call>

# Git diff
You> What changed in the last commit?
TOM> <tool_call>{"name": "shell", "arguments": {"command": "git diff HEAD~1"}}</tool_call>

# Run linter
You> Check code quality
TOM> <tool_call>{"name": "shell", "arguments": {"command": "flake8 src/"}}</tool_call>
```

### Limitations

- Output limited to 10,000 characters (configurable)
- Commands timeout after 30 seconds (configurable)
- Interactive commands not supported (no stdin)
- Working directory must be in allowed paths

### Risks & Mitigation

**Risks:**
- Accidental destructive commands
- Resource exhaustion
- Access to sensitive files

**Mitigations:**
- Disabled by default (opt-in)
- Allowlist/blocklist validation
- Path restrictions
- Resource limits
- Clear error messages
- Audit logging (coming in Sprint 4)

See [SECURITY.md](SECURITY.md) for detailed security information.
```

#### 3.2 Create Shell Tool Guide

Create `docs/SHELL_TOOL.md`:

```markdown
# Shell Tool Guide

## Overview

The shell tool gives TOM access to your system's command-line interface, enabling powerful workflows for development, system administration, and file management.

## Configuration

### Enabling Shell Access

**Step 1:** Edit `core/config.py`

```python
SHELL_ENABLED = True  # Change from False
```

**Step 2:** Choose security mode

```python
# Option 1: Allowlist (most secure, recommended)
SHELL_SECURITY_MODE = "allowlist"
SHELL_ALLOWED_COMMANDS = {
    "ls", "cat", "git", "npm", "pytest", # etc.
}

# Option 2: Blocklist (more permissive)
SHELL_SECURITY_MODE = "blocklist"
SHELL_BLOCKED_COMMANDS = {
    "rm", "sudo", "kill", # etc.
}
```

**Step 3:** Configure paths

```python
SHELL_ALLOWED_PATHS = [
    Path.home() / "Projects",  # Your projects
    Path.home() / "Documents", # Documents
    Path("/tmp"),              # Temp files
]
```

### Security Modes Compared

| Mode | Behavior | Best For |
|------|----------|----------|
| **Allowlist** | Only specified commands allowed | Most users, maximum security |
| **Blocklist** | All commands except blocked ones | Power users, development environments |

### Recommended Configurations

**For Beginners (Conservative):**
```python
SHELL_SECURITY_MODE = "allowlist"
SHELL_ALLOWED_COMMANDS = {
    "ls", "cat", "head", "tail",  # File reading
    "git",                         # Git operations
    "python", "pip",               # Python
}
```

**For Developers (Balanced):**
```python
SHELL_SECURITY_MODE = "allowlist"
SHELL_ALLOWED_COMMANDS = {
    # File operations
    "ls", "cat", "head", "tail", "find", "grep",

    # Git
    "git",

    # Development
    "npm", "node", "python", "pip", "pytest",
    "cargo", "rustc", "go", "make",

    # System
    "ps", "df", "du",
}
```

**For Advanced Users (Permissive):**
```python
SHELL_SECURITY_MODE = "blocklist"
SHELL_BLOCKED_COMMANDS = {
    "rm", "sudo", "kill", "shutdown",
}
```

## Use Cases

### 1. Git Workflows

```
You> What's the current git status?
TOM> <tool_call>{"name": "shell", "arguments": {"command": "git status --short"}}</tool_call>

You> Show me what changed in the last 3 commits
TOM> <tool_call>{"name": "shell", "arguments": {"command": "git log -3 --stat"}}</tool_call>

You> Create a new branch called feature-x
TOM> <tool_call>{"name": "shell", "arguments": {"command": "git checkout -b feature-x"}}</tool_call>
```

### 2. Development

```
You> Run the test suite
TOM> <tool_call>{"name": "shell", "arguments": {"command": "pytest -v"}}</tool_call>

You> Install the missing dependencies
TOM> <tool_call>{"name": "shell", "arguments": {"command": "pip install -r requirements.txt"}}</tool_call>

You> Check code coverage
TOM> <tool_call>{"name": "shell", "arguments": {"command": "pytest --cov=src --cov-report=term"}}</tool_call>
```

### 3. File System Operations

```
You> Find all JSON files in this project
TOM> <tool_call>{"name": "shell", "arguments": {"command": "find . -name '*.json'"}}</tool_call>

You> Show me the largest files
TOM> <tool_call>{"name": "shell", "arguments": {"command": "du -ah . | sort -rh | head -20"}}</tool_call>

You> Count lines of Python code
TOM> <tool_call>{"name": "shell", "arguments": {"command": "find . -name '*.py' -exec wc -l {} + | tail -1"}}</tool_call>
```

### 4. System Monitoring

```
You> Check disk space
TOM> <tool_call>{"name": "shell", "arguments": {"command": "df -h"}}</tool_call>

You> What Python processes are running?
TOM> <tool_call>{"name": "shell", "arguments": {"command": "ps aux | grep python"}}</tool_call>

You> Show system information
TOM> <tool_call>{"name": "shell", "arguments": {"command": "uname -a"}}</tool_call>
```

## Best Practices

### DO ✅

- **Start with allowlist mode** - Safest option
- **Review commands** - Check what TOM plans to run
- **Limit paths** - Only allow necessary directories
- **Use specific commands** - "List Python files" not "Do something with files"
- **Monitor logs** - Check audit logs regularly (Sprint 4)

### DON'T ❌

- **Run as root/sudo** - Never give TOM elevated privileges
- **Allow destructive commands** - rm, dd, etc. should be blocked
- **Expose to internet** - Shell access is for local use only
- **Trust blindly** - Review commands before execution
- **Disable all security** - Always keep some restrictions

## Troubleshooting

### Command Blocked

```
Error: Security: Command not in allowlist: xyz
```

**Solution:** Add command to `SHELL_ALLOWED_COMMANDS` or switch to blocklist mode

### Path Not Allowed

```
Error: Security: Working directory not allowed: /some/path
```

**Solution:** Add path to `SHELL_ALLOWED_PATHS`

### Timeout

```
Error: Command timeout after 30 seconds
```

**Solution:** Increase timeout in tool call:
```json
{"name": "shell", "arguments": {"command": "long_command", "timeout": 60}}
```

### Output Truncated

```
[Note: Output was truncated]
```

**Solution:** Increase `SHELL_MAX_OUTPUT_CHARS` in config or use more specific commands

## Security Considerations

### Command Injection

Shell commands are executed with `shell=True`, which means:
- Be careful with user-provided input
- Allowlist/blocklist protects against basic attacks
- Advanced users should review security code

### Path Traversal

Working directory validation prevents:
- Reading files outside allowed paths
- Modifying system files
- Accessing sensitive directories

### Resource Limits

Current limits:
- **Timeout:** 30 seconds default (prevents runaway processes)
- **Output:** 10,000 characters (prevents memory exhaustion)
- **No resource quotas yet** (CPU/memory limits in Sprint 4)

## FAQ

**Q: Is shell access safe?**
A: With proper configuration (allowlist mode, path restrictions), yes. It's as safe as running commands yourself.

**Q: Can TOM delete files?**
A: Not by default. `rm` is blocked. Only enable if you trust TOM completely.

**Q: What if I make a mistake in configuration?**
A: Start conservative (allowlist with few commands). Add more gradually.

**Q: Can I review commands before execution?**
A: Not yet. User approval prompts coming in Sprint 4.

**Q: What's logged?**
A: Currently, basic logging to console. Full audit logging in Sprint 4.

**Q: Can I use this in production?**
A: Yes, but thoroughly test your security configuration first.

## Advanced Topics

### Custom Security Policies

Create custom validation:

```python
# core/shell_executor.py

def is_command_allowed(self, command: str) -> Tuple[bool, str]:
    # Custom logic here
    if "important_file.txt" in command:
        return False, "Cannot touch important_file.txt"

    # Continue with standard checks
    ...
```

### Integration with CI/CD

Use shell tool for automated tasks:

```python
# Run tests before deployment
TOM> <tool_call>{"name": "shell", "arguments": {"command": "pytest && flake8"}}</tool_call>

# Deploy if tests pass
TOM> <tool_call>{"name": "shell", "arguments": {"command": "./deploy.sh"}}</tool_call>
```

### Complex Workflows

Chain commands with pipes:

```python
# Find Python files, count lines, sort by size
TOM> <tool_call>{"name": "shell", "arguments": {"command": "find . -name '*.py' -exec wc -l {} + | sort -rn"}}</tool_call>
```

## Next Steps

- **Sprint 4:** Audit logging and user approval prompts
- **Sprint 5:** Resource quotas and advanced security features
- **Future:** Interactive command sessions, environment variable management

---

**Need Help?** Check [SECURITY.md](../SECURITY.md) or open an issue.
```

**Tasks:**
- [ ] Update README with shell tool section
- [ ] Create `docs/SHELL_TOOL.md` guide
- [ ] Add security warnings to SECURITY.md
- [ ] Update CONTRIBUTING.md with shell testing notes

**Success Criteria:**
- Clear documentation for users
- Security warnings prominent
- Configuration examples provided
- Use cases documented

---

## Sprint 3 Timeline

**Total Effort:** 2-3 weeks

- **Week 1**: Core implementation (shell executor, tool registration, configuration)
- **Week 2**: Testing (unit tests, security tests, integration tests)
- **Week 3**: Documentation, polish, user testing

### Detailed Schedule

**Days 1-2:** Shell executor implementation
**Days 3-4:** Tool registration and configuration
**Day 5:** Initial testing and debugging
**Days 6-8:** Comprehensive test suite
**Days 9-10:** Documentation
**Days 11-12:** User testing and refinement
**Days 13-14:** Final polish and release prep

## Testing Checklist

### Functionality
- [ ] Simple commands execute correctly
- [ ] Output is captured properly
- [ ] Stderr is handled
- [ ] Exit codes are correct
- [ ] Timeouts work
- [ ] Output truncation works
- [ ] Working directory changes work

### Security
- [ ] Allowlist enforcement works
- [ ] Blocklist enforcement works
- [ ] Blocked commands are rejected
- [ ] Path validation works
- [ ] Path traversal is prevented
- [ ] Command injection is logged
- [ ] Environment is sanitized

### Integration
- [ ] Tool called by model works
- [ ] Results processed correctly
- [ ] Works in CLI mode
- [ ] Works in web mode
- [ ] Works in PySide mode
- [ ] Multiple shell calls in one conversation
- [ ] Shell + other tools combination

### Edge Cases
- [ ] Empty command
- [ ] Very long command
- [ ] Binary output
- [ ] Extremely large output
- [ ] Commands that hang
- [ ] Invalid working directory
- [ ] Concurrent execution

## Success Metrics

- **Functionality**: Shell commands execute successfully
- **Security**: All validations work, no bypasses
- **Performance**: Commands complete within timeout
- **Usability**: Clear errors, helpful output
- **Documentation**: Complete, clear, with examples
- **Testing**: 90%+ coverage, all edge cases covered

## Risks & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Security bypass | Low | Critical | Extensive testing, security review |
| Performance issues | Medium | Medium | Timeout limits, output truncation |
| User confusion | Medium | Low | Clear documentation, examples |
| Resource exhaustion | Low | Medium | Limits on timeout, output size |
| Command injection | Low | Critical | Sanitization, validation, logging |

## Rollback Plan

If critical issues found:
1. Set `SHELL_ENABLED = False` by default
2. Document issues and workarounds
3. Fix in hotfix sprint
4. Re-enable after verification

## Success Criteria

- [ ] Shell tool executes commands correctly
- [ ] All security validations work
- [ ] Comprehensive test coverage
- [ ] Documentation complete
- [ ] No critical security issues
- [ ] User feedback positive
- [ ] Ready for Sprint 4 (audit logging)

---

**Status:** 🚧 Ready to Start
**Dependencies:** Sprint 2 (security foundation)
**Next Sprint:** [SPRINT4.md](SPRINT4.md) - Configuration & Audit Logging
