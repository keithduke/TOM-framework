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
