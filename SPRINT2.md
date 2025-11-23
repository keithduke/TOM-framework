# Sprint 2 — Code Cleanup & Security Hardening

## Objective
Clean up technical debt identified in the code audit and harden security for existing tools. Complete Sprint 2 with a clean, type-safe codebase ready for shell access implementation.

## Sprint 2 Deliverables

### 1. Remove Unused Code
**Priority:** High | **Effort:** 30 minutes

- [ ] Delete `truncate_tool_result()` function from `core/tools.py:217-223`
- [ ] Remove import from `ui/cli/cli.py:43`
- [ ] Update direct truncation in `services/api/runtime.py:227-228` to use inline pattern
- [ ] Update CLI calls at `ui/cli/cli.py:280, 390` to use same pattern
- [ ] Run full test suite to verify no breakage
- [ ] Update README documentation references to `truncate_tool_result()`

**Success Criteria:**
- All tests pass
- No references to `truncate_tool_result()` in codebase
- Code still handles large tool results correctly

---

### 2. Fix Type Hints
**Priority:** Medium | **Effort:** 1 hour

- [ ] Fix `Dict[str, any]` → `Dict[str, Any]` in `context_manager.py:211`
- [ ] Add missing return type hints to public functions
- [ ] Add missing parameter type hints
- [ ] Create `mypy.ini` configuration file:
  ```ini
  [mypy]
  python_version = 3.11
  warn_return_any = True
  warn_unused_configs = True
  disallow_untyped_defs = False  # Start permissive
  ```
- [ ] Run `mypy core/ services/` and document baseline
- [ ] Add mypy to GitHub Actions / CI if present

**Success Criteria:**
- No type errors in critical paths
- Mypy configuration in place for future improvements
- Documentation on running mypy

---

### 3. Harden File Reading Security
**Priority:** High | **Effort:** 2 hours

#### 3.1 Add Path Validation

Create `core/security.py`:
```python
from pathlib import Path
from typing import List, Set

# Configurable allowed paths
DEFAULT_ALLOWED_PATHS = [
    Path.home(),
    Path.cwd(),
    Path("/tmp"),
]

# Sensitive files blocklist
SENSITIVE_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.yaml",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    ".pem",
    ".key",
}

# Sensitive directories
SENSITIVE_DIRS = {
    ".ssh",
    ".aws",
    ".azure",
    ".config/gcloud",
}

def is_path_allowed(file_path: Path, allowed_paths: List[Path] = None) -> bool:
    """
    Validate that file_path is within allowed directories.

    Args:
        file_path: Path to validate
        allowed_paths: List of allowed base paths (uses defaults if None)

    Returns:
        True if path is allowed, False otherwise
    """
    if allowed_paths is None:
        allowed_paths = DEFAULT_ALLOWED_PATHS

    resolved = file_path.resolve()

    # Check against allowed paths
    for allowed in allowed_paths:
        try:
            resolved.relative_to(allowed.resolve())
            return True
        except ValueError:
            continue

    return False

def is_sensitive_file(file_path: Path) -> bool:
    """
    Check if file is in sensitive files list.

    Args:
        file_path: Path to check

    Returns:
        True if file is sensitive, False otherwise
    """
    # Check filename
    if file_path.name in SENSITIVE_FILES:
        return True

    # Check suffix
    if file_path.suffix in {".pem", ".key"}:
        return True

    # Check if in sensitive directory
    for part in file_path.parts:
        if part in SENSITIVE_DIRS:
            return True

    return False
```

#### 3.2 Update `read_file` Tool

Modify `core/tools.py`:
```python
from .security import is_path_allowed, is_sensitive_file

@tool(
    "read",
    "Read file content. Returns full file content or error if file is too large or access is denied.",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "File path to read"}
        },
        "required": ["location"]
    }
)
def read_file(location: str) -> str:
    """
    Read file and return complete contents with security checks.
    """
    try:
        file_path = Path(location).expanduser().resolve()

        # Security check: Path allowed?
        if not is_path_allowed(file_path):
            return f"Error: Access denied. Path not in allowed directories: {location}"

        # Security check: Sensitive file?
        if is_sensitive_file(file_path):
            logger.warning(f"Attempted to read sensitive file: {file_path}")
            return f"Error: Cannot read sensitive file: {location}"

        if not file_path.exists():
            return f"Error: File not found: {location}"
        if not file_path.is_file():
            return f"Error: Path is not a file: {location}"

        # Existing size checks...
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return f"Error: File too large ({file_size_mb:.2f} MB). Maximum allowed: {MAX_FILE_SIZE_MB} MB."

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Existing character limit check...
        char_count = len(content)
        if char_count > MAX_FILE_CHARS:
            line_count = content.count('\n') + 1
            return (
                f"Error: File too large to read.\n"
                f"File: {file_path.name}\n"
                f"Size: {char_count:,} characters ({line_count:,} lines)\n"
                f"Limit: {MAX_FILE_CHARS:,} characters\n\n"
                f"File is {char_count - MAX_FILE_CHARS:,} characters over the limit.\n"
                f"Consider reading specific sections or breaking the analysis into parts."
            )

        logger.info(f"Read {location}: {char_count:,} chars ({file_size_mb:.2f} MB)")
        return content

    except UnicodeDecodeError:
        return f"Error: File is not a text file or uses unsupported encoding: {location}"
    except PermissionError:
        return f"Error: Permission denied reading file: {location}"
    except Exception as e:
        logger.error(f"Error reading {location}: {e}")
        return f"Error reading file: {str(e)}"
```

#### 3.3 Add Configuration

Update `core/config.py`:
```python
# File reading security
READ_ALLOWED_PATHS = [
    Path.home(),
    Path.cwd(),
    Path("/tmp"),
]
```

#### 3.4 Testing

Create `test_file_security.py`:
```python
import pytest
from pathlib import Path
from core.security import is_path_allowed, is_sensitive_file

def test_path_allowed_home():
    """Test that files in home directory are allowed."""
    test_file = Path.home() / "test.txt"
    assert is_path_allowed(test_file) is True

def test_path_denied_outside_allowed():
    """Test that files outside allowed paths are denied."""
    test_file = Path("/etc/passwd")
    assert is_path_allowed(test_file) is False

def test_sensitive_file_ssh_key():
    """Test that SSH keys are detected as sensitive."""
    test_file = Path.home() / ".ssh" / "id_rsa"
    assert is_sensitive_file(test_file) is True

def test_sensitive_file_env():
    """Test that .env files are detected as sensitive."""
    test_file = Path.cwd() / ".env"
    assert is_sensitive_file(test_file) is True

def test_non_sensitive_file():
    """Test that normal files are not sensitive."""
    test_file = Path.cwd() / "README.md"
    assert is_sensitive_file(test_file) is False
```

**Tasks:**
- [ ] Create `core/security.py` with validation functions
- [ ] Update `read_file` in `core/tools.py` with security checks
- [ ] Add configuration to `core/config.py`
- [ ] Create `test_file_security.py` with comprehensive tests
- [ ] Update documentation with security notes
- [ ] Run full test suite

**Success Criteria:**
- All security tests pass
- Path traversal attempts are blocked
- Sensitive files cannot be read
- Existing functionality preserved
- Clear error messages for security violations

---

### 4. Documentation Updates
**Priority:** Medium | **Effort:** 2 hours

#### 4.1 Create SECURITY.md

```markdown
# Security Best Practices

## Overview

TOM is designed to be secure by default while remaining flexible for power users.
All processing happens locally on your machine - no data leaves your device.

## File Reading Security

### Allowed Paths

By default, the `read` tool can only access files in:
- Your home directory (`~`)
- Current working directory
- `/tmp` (temporary files)

### Blocked Files

The following file types are automatically blocked:
- Environment files (`.env`, `.env.local`, etc.)
- Credentials (`credentials.json`, `secrets.yaml`)
- SSH keys (`id_rsa`, `id_ed25519`, etc.)
- Private keys (`.pem`, `.key` files)
- AWS/Azure/GCloud credentials

### Path Traversal Protection

TOM validates all file paths and prevents access outside allowed directories.
Even if you try to read `/etc/passwd`, the request will be denied.

## Network Security

### Local-Only by Default

- TOM's API server binds to `127.0.0.1` (localhost only)
- No external network access by default
- All communication over loopback interface

### Exposing the API (Advanced)

If you want to access TOM from other devices:

```bash
# ⚠️ WARNING: Only do this on trusted networks
python main.py --host 0.0.0.0 --port 8000
```

**Security recommendations:**
- Use a firewall to limit access
- Set up API key authentication (future feature)
- Use HTTPS/TLS in production
- Never expose to the internet without authentication

## Data Privacy

### What's Stored

- **Conversation history**: In-memory only (lost on restart)
- **Cache files**: Model KV cache (no user data)
- **Logs**: System events, no message content

### What's NOT Stored

- User messages are never written to disk
- No telemetry or analytics
- No external API calls

## Best Practices

1. **Run with least privilege**: Don't run TOM as root/admin
2. **Review logs**: Check `~/.tom/` for any suspicious activity
3. **Keep updated**: Update dependencies regularly
4. **Trust but verify**: Review tool calls before approving (when that feature is added)

## Reporting Security Issues

If you discover a security vulnerability, please email:
[your-email@example.com]

Do not open public issues for security problems.
```

#### 4.2 Create CONTRIBUTING.md

```markdown
# Contributing to TOM

Thank you for your interest in improving TOM!

## Development Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Git

### Setup Steps

1. Fork and clone:
   ```bash
   git clone https://github.com/YOUR_USERNAME/TOM-framework.git
   cd TOM-framework
   ```

2. Create virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. Download model (if not already done):
   ```bash
   python -m mlx_lm.convert \
     --hf-path Qwen/Qwen3-4B-Thinking-2507 \
     --mlx-path ./Qwen3-4B-Thinking-2507-8bit \
     -q --q-bits 8
   ```

## Code Style

### Python Style

- Follow [PEP 8](https://pep8.org/)
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable names

### Type Hints

- Add type hints to new functions
- Use `from typing import` for complex types
- Run `mypy` before submitting PR

### Documentation

- Add docstrings to public functions (Google style)
- Include parameter descriptions and return types
- Update README if adding user-facing features

### Example

```python
from typing import List, Optional

def process_messages(
    messages: List[str],
    max_length: Optional[int] = None
) -> List[str]:
    """
    Process a list of messages with optional length limiting.

    Args:
        messages: List of message strings to process
        max_length: Maximum length per message (None = unlimited)

    Returns:
        List of processed messages

    Raises:
        ValueError: If messages list is empty
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")

    # Implementation...
    return processed
```

## Testing

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest test_tool_system.py -v

# Run with coverage
pytest --cov=core --cov-report=term-missing
```

### Writing Tests

- Place tests in `test_*.py` files
- Use descriptive test names: `test_tool_execution_with_invalid_args`
- Test both success and failure cases
- Mock external dependencies

### Example Test

```python
import pytest
from core.tools import execute_tool_call

def test_execute_tool_with_valid_args():
    """Test tool execution with valid arguments."""
    tool_call = {
        "name": "get_datetime",
        "arguments": {}
    }
    result = execute_tool_call(tool_call)
    assert "AM" in result or "PM" in result

def test_execute_tool_with_missing_name():
    """Test tool execution fails gracefully with missing name."""
    tool_call = {"arguments": {}}
    result = execute_tool_call(tool_call)
    assert "Error" in result
```

## Adding New Tools

### Tool Development Process

1. **Design the tool interface**
   - What does it do?
   - What parameters does it need?
   - What does it return?

2. **Implement the tool function**
   ```python
   @tool(
       "tool_name",
       "Clear description of what this tool does",
       parameters={
           "type": "object",
           "properties": {
               "param_name": {
                   "type": "string",
                   "description": "What this parameter is for"
               }
           },
           "required": ["param_name"]
       }
   )
   def tool_function(param_name: str) -> str:
       """
       Detailed docstring explaining the tool.

       Args:
           param_name: Description

       Returns:
           Description of return value
       """
       # Implementation
       return result
   ```

3. **Write tests**
   - Test normal operation
   - Test error cases
   - Test edge cases

4. **Update documentation**
   - Add to README's tool list
   - Include example usage
   - Note any security considerations

### Security Checklist for Tools

- [ ] Validates all inputs
- [ ] Handles errors gracefully
- [ ] Doesn't expose sensitive data
- [ ] Limits resource usage (file size, execution time)
- [ ] Logs security-relevant events
- [ ] Documents security implications

## Pull Request Process

### Before Submitting

1. **Run tests**: `pytest -v`
2. **Check types**: `mypy core/ services/`
3. **Format code**: Follow PEP 8
4. **Update docs**: README, docstrings, etc.
5. **Test manually**: Try your changes in CLI/web/PySide

### PR Description Template

```markdown
## Summary
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Bullet list of specific changes
- Keep it concise

## Testing
How did you test this?

## Screenshots
If UI changes, include screenshots

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] Security implications considered
```

### Review Process

1. Submit PR with clear description
2. Respond to reviewer feedback
3. Make requested changes
4. Maintainer will merge when approved

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Welcome newcomers
- Help others learn

## Questions?

- Open a GitHub Discussion
- Check existing issues
- Read the documentation

Thank you for contributing! 🎉
```

#### 4.3 Update README

Add security section:

```markdown
## Security

TOM is designed with security in mind:

- **Local-only processing**: No data leaves your device
- **Path validation**: File reading restricted to allowed directories
- **Sensitive file blocking**: Automatically prevents reading credentials
- **Secure defaults**: All APIs bind to localhost only

See [SECURITY.md](SECURITY.md) for detailed security information.
```

**Tasks:**
- [ ] Create `SECURITY.md`
- [ ] Create `CONTRIBUTING.md`
- [ ] Update README with security section
- [ ] Add security badge to README if applicable

**Success Criteria:**
- Clear security documentation
- Contribution guidelines in place
- README updated with security info

---

### 5. Code Quality Improvements
**Priority:** Low | **Effort:** 1 hour

- [ ] Add `.editorconfig` for consistent formatting:
  ```ini
  root = true

  [*]
  charset = utf-8
  end_of_line = lf
  insert_final_newline = true
  trim_trailing_whitespace = true

  [*.py]
  indent_style = space
  indent_size = 4
  max_line_length = 100

  [*.{yaml,yml}]
  indent_style = space
  indent_size = 2

  [*.md]
  trim_trailing_whitespace = false
  ```

- [ ] Create `requirements-dev.txt`:
  ```
  pytest>=7.0.0
  pytest-cov>=4.0.0
  mypy>=1.0.0
  black>=23.0.0
  flake8>=6.0.0
  ```

- [ ] Add `pyproject.toml` for tool configuration:
  ```toml
  [tool.black]
  line-length = 100
  target-version = ['py311']

  [tool.pytest.ini_options]
  testpaths = [".", "tests"]
  python_files = "test_*.py"
  addopts = "-v --strict-markers"

  [tool.mypy]
  python_version = "3.11"
  warn_return_any = true
  warn_unused_configs = true
  ```

**Success Criteria:**
- Consistent code formatting
- Dev dependencies documented
- Tool configuration in place

---

## Sprint 2 Timeline

**Total Effort:** ~1 week (8-10 hours)

- **Day 1** (2 hours): Remove unused code, fix type hints
- **Day 2** (3 hours): File reading security implementation
- **Day 3** (2 hours): Security testing and validation
- **Day 4** (2 hours): Documentation (SECURITY.md, CONTRIBUTING.md)
- **Day 5** (1 hour): Code quality improvements, final testing

## Testing Checklist

- [ ] All existing tests pass
- [ ] New security tests pass
- [ ] Manual testing of file reading with various paths
- [ ] Attempt to read blocked files (should fail gracefully)
- [ ] Attempt path traversal (should be blocked)
- [ ] CLI, Web, and PySide all work correctly

## Success Metrics

- **Code Quality**: No unused code, proper type hints
- **Security**: Path validation working, sensitive files blocked
- **Documentation**: SECURITY.md and CONTRIBUTING.md complete
- **Testing**: All tests passing, new security tests added
- **Zero Regressions**: All existing functionality works

## Notes

- This sprint sets the foundation for shell access in Sprint 3
- Security patterns established here will be reused for shell tool
- Keep changes backward compatible
- Focus on security without breaking user workflows

---

**Status:** 🚧 Ready to Start
**Dependencies:** None (builds on Sprint 1)
**Next Sprint:** [SPRINT3.md](SPRINT3.md) - Shell Access Implementation
