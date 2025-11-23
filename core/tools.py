"""
Tool system for T.O.M. CLI
"""

import json
import logging
import re
from datetime import datetime
from functools import wraps
from pathlib import Path
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.security import is_path_allowed, is_sensitive_file
from core.config import MAX_FILE_SIZE_MB, READ_ALLOWED_PATHS
from core.utils import ordinal


logger = logging.getLogger("tom_cli")

# Tool registry
TOOLS_REGISTRY: Dict[str, Callable] = {}
TOOLS_DEFINITIONS: List[Dict[str, Any]] = []

# Character limit for file reading - files larger than this will be rejected
MAX_FILE_CHARS = 15000  # ~3,750 tokens at 4 chars/token


def tool(name: str, description: str, parameters: Optional[Dict[str, Any]] = None) -> Callable:
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
        if not is_path_allowed(file_path, allowed_paths=READ_ALLOWED_PATHS):
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


def parse_tool_arguments(args: Any) -> Dict[str, Any]:
    """Parse tool arguments from string or dict"""
    if isinstance(args, dict):
        return args
    
    if isinstance(args, str):
        # Try JSON first
        try:
            parsed_json = json.loads(args)
            if isinstance(parsed_json, dict):
                return parsed_json
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

