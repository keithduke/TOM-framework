"""
Tool system for T.O.M. CLI
"""

import json
import logging
import re
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List

from config import MAX_FILE_SIZE_MB
from utils import ordinal

logger = logging.getLogger("tom_cli")

# Tool registry
TOOLS_REGISTRY: Dict[str, Callable] = {}
TOOLS_DEFINITIONS: List[Dict[str, Any]] = []


def tool(name: str, description: str, parameters: Dict[str, Any] = None):
    """Decorator to register tools with optional parameters"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"Executing tool: {name}")
            result = func(*args, **kwargs)
            return result
        
        TOOLS_REGISTRY[name] = wrapper
        tool_parameters = parameters or {"type": "object", "properties": {}, "required": []}
        
        TOOLS_DEFINITIONS.append({
            "type": "function",
            "function": {"name": name, "description": description, "parameters": tool_parameters}
        })
        return wrapper
    return decorator


@tool("get_datetime", "Return the current system date and time as a friendly string: 'H:MM AM/PM on Month Dth, YYYY'")
def get_datetime() -> str:
    """Get the current date and time in a friendly format."""
    now = datetime.now()
    day_with_suffix = ordinal(now.day)
    formatted = now.strftime(f"%-I:%M %p on %B {day_with_suffix}, %Y")
    return formatted


def summarize_code_file(content: str, filename: str) -> str:
    """
    Create structure summary of code files to prevent context overload.
    Shows imports, classes, functions - not full content.
    """
    lines = content.split('\n')
    total_lines = len(lines)
    
    # Small files don't need summarization
    if total_lines <= 200:
        return content
    
    summary_parts = []
    summary_parts.append(f"# FILE: {filename} ({total_lines:,} lines, {len(content):,} chars)")
    summary_parts.append(f"# Showing structure only - use full=true parameter to see complete code\n")
    
    in_docstring = False
    docstring_char = None
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        if not stripped:
            continue
        
        # Track docstrings
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                docstring_char = '"""' if '"""' in line else "'''"
                if line.count(docstring_char) >= 2:
                    in_docstring = False
            elif docstring_char in line:
                in_docstring = False
            continue
        
        if in_docstring:
            continue
        
        # Include structural lines
        if (stripped.startswith(('import ', 'from ')) or
            stripped.startswith('class ') or
            stripped.startswith('def ') or
            stripped.startswith('@') or
            (stripped.startswith('#') and len(line) - len(line.lstrip()) == 0)):
            summary_parts.append(f"{i:4d}: {line.rstrip()}")
    
    summary = '\n'.join(summary_parts)
    summary += f"\n\n# Structure: {len(summary_parts)-2} key lines from {total_lines:,} total"
    
    return summary


@tool(
    "read", 
    "Read file content. Returns structure summary for large code files to keep context manageable.",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "File path to read"}
        },
        "required": ["location"]
    }
)
def read_file(location: str, full: bool = False) -> str:
    """
    Read file with intelligent summarization for large code files.
    Note: full parameter only honored for non-code files or when called internally.
    The model should always receive summaries for large code files.
    """
    try:
        file_path = Path(location).expanduser().resolve()
        
        if not file_path.exists():
            return f"Error: File not found: {location}"
        if not file_path.is_file():
            return f"Error: Path is not a file: {location}"
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return f"Error: File too large ({file_size_mb:.2f} MB). Max: {MAX_FILE_SIZE_MB} MB."
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Read {location}: {len(content):,} chars ({file_size_mb:.2f} MB)")
        
        # Auto-summarize large code files
        code_ext = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.rs', '.go', '.ts', '.jsx', '.tsx'}
        is_code = file_path.suffix.lower() in code_ext
        
        # ALWAYS summarize large code files - don't let model override with full=true
        if is_code and len(content) > 5000:
            summary = summarize_code_file(content, file_path.name)
            logger.info(f"Summarized to {len(summary):,} chars (ignoring full={full})")
            return summary
        
        return content
        
    except UnicodeDecodeError:
        return f"Error: File is not a text file or uses unsupported encoding: {location}"
    except PermissionError:
        return f"Error: Permission denied reading file: {location}"
    except Exception as e:
        logger.error(f"Error reading {location}: {e}")
        return f"Error reading file: {str(e)}"


def parse_tool_arguments(args: Any) -> Dict[str, Any]:
    """Parse tool arguments from string or dict"""
    if isinstance(args, dict):
        return args
    
    if isinstance(args, str):
        # Try JSON first
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            pass
        
        # Try as Python literal
        try:
            import ast
            result = ast.literal_eval(args)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    
    # Return empty dict if parsing fails
    return {}


def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """Execute a tool call and return result as string"""
    try:
        tool_name = tool_call.get("name")
        if not tool_name:
            return "Error: Tool call missing 'name' field"
        
        if tool_name not in TOOLS_REGISTRY:
            return f"Error: Unknown tool '{tool_name}'"
        
        tool_args = parse_tool_arguments(tool_call.get("arguments", {}))
        
        result = TOOLS_REGISTRY[tool_name](**tool_args)
        logger.info(f"Tool '{tool_name}' executed successfully")
        return str(result)

    except TypeError as e:
        logger.error(f"Tool argument error: {e}")
        return f"Error: Invalid arguments for tool - {e}"
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return f"Error executing tool: {e}"


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls from text wrapped in <tool_call>...</tool_call>.
    Returns list of parsed tool call dictionaries.
    """
    if not text or "<tool_call>" not in text:
        return []
    
    tool_calls = []
    
    # Find all tool call blocks
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    
    for match in matches:
        try:
            # Parse the JSON
            tool_call = json.loads(match.strip())
            
            # Validate structure
            if not isinstance(tool_call, dict):
                logger.warning(f"Tool call is not a dict: {match}")
                continue
            
            if "name" not in tool_call:
                logger.warning(f"Tool call missing 'name': {match}")
                continue
            
            # Normalize arguments
            if "arguments" in tool_call:
                tool_call["arguments"] = parse_tool_arguments(tool_call["arguments"])
            else:
                tool_call["arguments"] = {}
            
            tool_calls.append(tool_call)
            logger.debug(f"Extracted tool call: {tool_call['name']}")
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse tool call JSON: {e}")
            continue
        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            continue
    
    return tool_calls


def strip_tool_calls(text: str) -> str:
    """Remove tool call XML tags from text"""
    if not text or "<tool_call>" not in text:
        return text
    
    # Remove all <tool_call>...</tool_call> blocks
    cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def truncate_tool_result(result: str, tool_name: str, max_chars: int) -> str:
    """
    Intelligently truncate large tool results.
    Shows beginning and end with truncation notice in middle.
    """
    if len(result) <= max_chars:
        return result
    
    truncated_chars = len(result) - max_chars
    
    # For very large results, show more from beginning
    if len(result) > max_chars * 3:
        start_size = int(max_chars * 0.7)
        end_size = max_chars - start_size
    else:
        start_size = max_chars // 2
        end_size = max_chars - start_size
    
    truncated = (
        result[:start_size] + 
        f"\n\n... [TRUNCATED {truncated_chars:,} characters - {(truncated_chars/len(result)*100):.1f}% of content] ...\n\n" +
        result[-end_size:]
    )
    
    logger.warning(f"Truncated {tool_name} result: {len(result):,} -> {len(truncated):,} chars")
    return truncated