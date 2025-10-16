"""
Tool system for T.O.M. CLI
"""

import ast
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


@tool("get_datetime", "Return the current system date and time as a friendly string: 'H:MM:SS AM/PM on Month Dth, YYYY'")
def get_datetime() -> str:
    """Get the current date and time in a friendly format."""
    now = datetime.now()
    day_with_suffix = ordinal(now.day)
    formatted = now.strftime(f"%-I:%M %p on %B {day_with_suffix}, %Y")
    return formatted


@tool(
    "read", 
    "Read content from a specified file path.",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The file path to read from"}
        },
        "required": ["location"]
    }
)
def read_file(location: str) -> str:
    """Read content from a file."""
    try:
        file_path = Path(location)
        
        if not file_path.exists():
            return f"Error: File not found: {location}"
        if not file_path.is_file():
            return f"Error: Path is not a file: {location}"
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.warning(f"Reading large file: {location} ({file_size_mb:.2f} MB)")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        logger.info(f"Read {location}: {len(content):,} chars ({file_size_mb:.2f} MB)")
        return content
        
    except UnicodeDecodeError:
        return f"Error: File is not a text file or uses unsupported encoding: {location}"
    except PermissionError:
        return f"Error: Permission denied reading file: {location}"
    except Exception as e:
        logger.error(f"Error reading {location}: {e}")
        return f"Error reading file: {str(e)}"


def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """Execute a tool call"""
    try:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("arguments", {})
        
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                try:
                    tool_args = ast.literal_eval(tool_args)
                except Exception:
                    tool_args = {}
        
        if not isinstance(tool_args, dict):
            tool_args = {}

        if tool_name not in TOOLS_REGISTRY:
            return f"Error: Unknown tool '{tool_name}'"

        result = TOOLS_REGISTRY[tool_name](**tool_args)
        logger.info(f"Executed {tool_name}")
        return str(result)

    except Exception as e:
        logger.error(f"Error executing tool: {e}", exc_info=True)
        return f"Error executing tool: {e}"


def extract_tool_calls(response: str) -> List[Dict[str, Any]]:
    """Extract tool calls wrapped in <tool_call>...</tool_call>"""
    tool_calls: List[Dict[str, Any]] = []
    
    try:
        pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
        matches = re.findall(pattern, response, flags=re.DOTALL)

        for json_part in matches:
            json_part = json_part.strip()
            parsed = None
            try:
                parsed = json.loads(json_part)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(json_part)
                except Exception:
                    continue

            if not isinstance(parsed, dict):
                continue

            args = parsed.get("arguments", {})
            if isinstance(args, str):
                try:
                    args_parsed = json.loads(args)
                except Exception:
                    try:
                        args_parsed = ast.literal_eval(args)
                    except Exception:
                        args_parsed = {}
                parsed["arguments"] = args_parsed

            tool_calls.append(parsed)

    except Exception as e:
        logger.error(f"Error extracting tool calls: {e}", exc_info=True)

    return tool_calls


def truncate_tool_result(result: Any, tool_name: str, max_chars: int) -> str:
    """Intelligently truncate large tool results"""
    result_str = str(result)
    
    if len(result_str) <= max_chars:
        return result_str
    
    truncated_chars = len(result_str) - max_chars
    
    # For very large files, show more from beginning
    if len(result_str) > max_chars * 3:
        start_size = int(max_chars * 0.7)
        end_size = max_chars - start_size
    else:
        start_size = max_chars // 2
        end_size = max_chars - start_size
    
    truncated = (
        result_str[:start_size] + 
        f"\n\n... [TRUNCATED {truncated_chars:,} characters - {(truncated_chars/len(result_str)*100):.1f}% of file] ...\n\n" +
        result_str[-end_size:]
    )
    
    logger.warning(f"Truncated {tool_name} result: {len(result_str):,} -> {len(truncated):,} chars")
    return truncated
