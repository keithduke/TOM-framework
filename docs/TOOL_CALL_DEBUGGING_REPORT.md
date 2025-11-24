# Tool Call Debugging Report

**Date:** October 27, 2025
**Issue:** Tool calls not working in TOM-framework REPL
**Status:** ✅ RESOLVED

---

## Executive Summary

Tool calls were failing because the model's chat template was **silently ignoring** the `tools` parameter. The template accepted the parameter without error, but didn't include tools in the generated prompt, resulting in a **silent failure** where the model never knew tools existed.

### Fixes Applied

1. **Verification of tool inclusion in prompts** (`context_manager.py:168-176`)
2. **Enhanced system prompt with tool usage instructions** (`config.py:39-45`)

---

## Root Cause Analysis

### The Problem

The tool call flow in `context_manager.py` attempted to use the tokenizer's chat template with a `tools` parameter:

```python
if include_tools and TOOLS_DEFINITIONS:
    kwargs["tools"] = TOOLS_DEFINITIONS

prompt = tokenizer.apply_chat_template(chat_messages, **kwargs)
```

**Three possible scenarios:**

1. ✅ **Template rejects tools** → Exception raised → Fallback works → Tools included
2. ❌ **Template silently ignores tools** → No exception → No fallback → **Tools never sent to model**
3. ✅ **Template properly supports tools** → Tools included → Works correctly

**Scenario 2 was the root cause** - a silent failure where:
- No error was raised
- No fallback was triggered
- Tools parameter was accepted but ignored
- Model never saw the available tools
- Tool calls could never happen

### Why This Is Problematic

Most chat templates in the wild don't support the `tools` parameter. When they receive it:
- Some raise `TypeError` (good - triggers fallback)
- Some accept it but ignore it (bad - silent failure)

The original code only handled the first case.

---

## Solution Implemented

### Fix 1: Verify Tools in Generated Prompts

**File:** `context_manager.py` (lines 168-176)

```python
# CRITICAL FIX: Verify tools are actually present in the prompt
# Some chat templates accept 'tools' parameter but silently ignore it
if include_tools and TOOLS_DEFINITIONS:
    first_tool_name = TOOLS_DEFINITIONS[0]["function"]["name"]
    if first_tool_name not in prompt:
        logger.warning(
            f"Chat template ignored tools parameter (tools not in output), using fallback"
        )
        return self._build_fallback_prompt(chat_messages, include_tools)
```

**How it works:**
1. After calling `apply_chat_template()`, check if tools are actually in the output
2. If tools are missing, log a warning and use the fallback prompt builder
3. The fallback builder reliably includes tools in the prompt

### Fix 2: Enhanced System Prompt

**File:** `config.py` (lines 39-45)

```python
When you have access to tools, you can call them using this exact XML format:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}
</tool_call>

You can make multiple tool calls if needed. After receiving tool results, synthesize them into your response.
Always use tools when they would help answer the user's question accurately.
```

**Why this helps:**
- Explicitly tells the model the expected tool call format
- Provides clear examples of the XML structure
- Encourages the model to use tools when appropriate
- Works regardless of chat template behavior

---

## Complete Tool Call Flow

### Overview

```
1. User Input
   ↓
2. Build Prompt WITH Tools
   ↓
3. Model Generates Response (with <tool_call> XML)
   ↓
4. Extract Tool Calls from Response
   ↓
5. Strip <tool_call> Tags from Content
   ↓
6. Execute Tool Calls
   ↓
7. Add Results to Context
   ↓
8. Build Prompt WITHOUT Tools
   ↓
9. Model Synthesizes Final Response
   ↓
10. Add Final Response to Context
```

### File-by-File Breakdown

#### `cli.py` - REPL Main Loop

**Streaming mode** (`_generate_streaming` at lines 223-271):
1. Call `_stream_and_parse(include_tools=True)` → generates response with tools
2. Extract tool calls from content
3. Strip `<tool_call>` XML tags using `strip_tool_calls()`
4. Add cleaned content as "assistant" message
5. Execute each tool call
6. Add tool results as "tool" messages
7. Call `_stream_and_parse(include_tools=False)` → generates final response
8. Add final response as "assistant" message

**Legacy mode** (`_generate_legacy` at lines 330-381):
- Same flow but using `generate_response()` instead of streaming

#### `tools.py` - Tool System

**Key functions:**

- **`extract_tool_calls(text)`** (lines 160-204)
  - Regex: `r"<tool_call>\s*(\{.*?\})\s*</tool_call>"`
  - Parses JSON inside XML tags
  - Validates structure (has "name" field)
  - Returns list of tool call dicts

- **`execute_tool_call(tool_call)`** (lines 136-157)
  - Looks up tool in `TOOLS_REGISTRY`
  - Parses arguments
  - Executes tool function
  - Returns result as string

- **`strip_tool_calls(text)`** (lines 207-214)
  - Removes all `<tool_call>...</tool_call>` blocks
  - Returns cleaned content

#### `context_manager.py` - Prompt Building

**`build_prompt(tokenizer, include_tools)`** (lines 137-185):

1. Build message list with system prompt
2. If tokenizer has chat template:
   - Try to use it with `tools` parameter
   - **NEW:** Verify tools are actually in the output
   - If missing, use fallback
3. Otherwise use `_build_fallback_prompt()`

**`_build_fallback_prompt()`** (lines 187-208):
- Simple but reliable prompt builder
- Always includes tools when `include_tools=True`
- Format: System → Tools (JSON) → Messages → "Assistant:"

---

## Test Suite

Three comprehensive test files created:

### 1. `test_tool_system.py`
Tests core tool system components:
- ✅ Tool definitions structure
- ✅ Tool call extraction (single, multiple, with args)
- ✅ Tool execution (datetime, file reading, errors)
- ✅ Tool call stripping
- ✅ Argument parsing

**Result:** 5/5 tests passed

### 2. `test_chat_template_issue.py`
Diagnostic test for chat template scenarios:
- ✅ Scenario 1: Template rejects tools → Fallback works
- ✅ Scenario 2: Template ignores tools → **Now fixed with verification**
- ✅ Scenario 3: Template includes tools → Works correctly

**Result:** 3/3 tests passed (after fix)

### 3. `test_end_to_end.py`
Full integration test:
- ✅ Build prompt with tools
- ✅ Extract tool calls from response
- ✅ Strip XML tags
- ✅ Execute tools
- ✅ Add results to context
- ✅ Build follow-up prompt
- ✅ File reading tool (success, not found, too large)

**Result:** All tests passed

---

## Files Modified

### `context_manager.py`
**Lines 168-176:** Added verification that tools are present in prompt
**Impact:** Prevents silent failures when chat template ignores tools

### `config.py`
**Lines 39-45:** Enhanced system prompt with tool usage instructions
**Impact:** Explicitly teaches model how to format tool calls

---

## Remaining Considerations

### 1. Model Training

Even with these fixes, tool calling quality depends on the model:
- Models not trained for tool use may struggle
- Qwen models generally support tool calling well
- May need to fine-tune system prompt for specific models

### 2. Tool Result Truncation

The `truncate_tool_result()` function is currently a no-op (tools.py:217-222):

```python
def truncate_tool_result(result: str, tool_name: str, max_chars: int) -> str:
    """
    DEPRECATED: No-op function kept for backward compatibility.
    Returns result unchanged.
    """
    return result
```

**Recommendation:** Consider re-implementing truncation for very large tool results to prevent context overflow. The previous implementation (commit a65dc27) had intelligent truncation.

### 3. File Reading Limits

Current limits:
- **MAX_FILE_SIZE_MB:** 10 MB
- **MAX_FILE_CHARS:** 15,000 characters (~3,750 tokens)

Files larger than 15k characters are rejected entirely. Consider:
- Restoring the `summarize_code_file()` function for large code files
- Or implementing chunked reading for large files

---

## Verification Steps

To verify tool calls work with your model:

1. **Enable debug logging:**
   ```bash
   python main.py --debug
   ```

2. **Watch for log messages:**
   - "Built prompt using chat template" - using template
   - "Chat template ignored tools parameter" - using fallback
   - "Built prompt using fallback" - using fallback

3. **Test with simple tool call:**
   ```
   You> What time is it?
   ```

   Should trigger `get_datetime` tool.

4. **Check the `/raw-prompt` command:**
   ```
   You> /raw-prompt
   ```

   Verify tools appear in "RAW PROMPT (WITH TOOLS)" output.

---

## Conclusion

The tool call system is now **fully functional** with robust handling of all chat template scenarios. The fix ensures tools are always included in prompts when requested, regardless of whether the chat template supports, ignores, or rejects the `tools` parameter.

All tests pass and the system is ready for use.
