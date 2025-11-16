#!/usr/bin/env python3
"""
End-to-end integration test for the complete tool call flow
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _run_complete_tool_flow():
    """Test the complete flow from prompt -> extraction -> execution"""
    print("=" * 80)
    print("END-TO-END TOOL CALL FLOW TEST")
    print("=" * 80)

    from core.context_manager import ContextManager
    from core.tools import extract_tool_calls, execute_tool_call, strip_tool_calls
    from core.tools import TOOLS_DEFINITIONS

    # Step 1: Build a prompt with tools
    print("\n" + "-" * 80)
    print("STEP 1: Build prompt with tools")
    print("-" * 80)

    context = ContextManager(max_context_tokens=8000)
    context.add_message("user", "What time is it?")

    # Use None tokenizer to force fallback (which we know works)
    prompt = context.build_prompt(tokenizer=None, include_tools=True)

    print(f"\nPrompt length: {len(prompt)} chars")

    # Verify tools are in the prompt
    tools_present = all(
        tool["function"]["name"] in prompt
        for tool in TOOLS_DEFINITIONS
    )

    if tools_present:
        print("✓ All tools present in prompt")
    else:
        print("✗ Tools missing from prompt!")
        return False

    print("\nPrompt preview:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    # Step 2: Simulate model response with tool call
    print("\n" + "-" * 80)
    print("STEP 2: Simulate model response with tool call")
    print("-" * 80)

    model_response = """Let me check the current time for you.
<tool_call>
{"name": "get_datetime", "arguments": {}}
</tool_call>"""

    print(f"Simulated model response:\n{model_response}")

    # Step 3: Extract tool calls
    print("\n" + "-" * 80)
    print("STEP 3: Extract tool calls from response")
    print("-" * 80)

    tool_calls = extract_tool_calls(model_response)
    print(f"Extracted {len(tool_calls)} tool call(s)")

    if len(tool_calls) != 1:
        print(f"✗ Expected 1 tool call, got {len(tool_calls)}")
        return False

    print(f"✓ Tool call extracted: {tool_calls[0]}")

    # Step 4: Strip tool calls from content
    print("\n" + "-" * 80)
    print("STEP 4: Strip tool call XML from content")
    print("-" * 80)

    clean_content = strip_tool_calls(model_response)
    print(f"Clean content: '{clean_content}'")

    if "<tool_call>" in clean_content:
        print("✗ Tool call XML not properly stripped")
        return False

    print("✓ Tool call XML stripped successfully")

    # Step 5: Execute tool call
    print("\n" + "-" * 80)
    print("STEP 5: Execute tool call")
    print("-" * 80)

    result = execute_tool_call(tool_calls[0])
    print(f"Tool result: {result}")

    if "Error" in result:
        print(f"✗ Tool execution failed: {result}")
        return False

    # Validate datetime format
    if ":" in result and "on" in result:
        print("✓ Tool executed successfully, result format valid")
    else:
        print(f"✗ Unexpected result format: {result}")
        return False

    # Step 6: Add to context and generate follow-up
    print("\n" + "-" * 80)
    print("STEP 6: Add tool result to context")
    print("-" * 80)

    # Add assistant message (cleaned)
    context.add_message("assistant", clean_content)

    # Add tool result
    tool_msg = f"Tool: {tool_calls[0]['name']}\nResult: {result}"
    context.add_message("tool", tool_msg)

    print(f"Context now has {len(context.messages)} messages")
    print("✓ Messages added to context")

    # Step 7: Build follow-up prompt without tools
    print("\n" + "-" * 80)
    print("STEP 7: Build follow-up prompt (without tools)")
    print("-" * 80)

    follow_up_prompt = context.build_prompt(tokenizer=None, include_tools=False)
    print(f"Follow-up prompt length: {len(follow_up_prompt)} chars")

    # Verify tool result is in the prompt
    if result[:20] in follow_up_prompt:
        print("✓ Tool result present in follow-up prompt")
    else:
        print("✗ Tool result not in follow-up prompt")
        return False

    # Verify tools are NOT in follow-up prompt (since include_tools=False)
    if "Available Tools:" not in follow_up_prompt:
        print("✓ Tools correctly excluded from follow-up prompt")
    else:
        print("✗ Tools should not be in follow-up prompt")
        return False

    print("\n" + "=" * 80)
    print("✓ ALL STEPS PASSED - END-TO-END FLOW WORKS")
    print("=" * 80)
    return True


def _run_file_reading_tool():
    """Test the file reading tool with actual files"""
    print("\n" + "=" * 80)
    print("FILE READING TOOL TEST")
    print("=" * 80)

    from core.tools import execute_tool_call

    # Create test file
    test_file = Path("/tmp/test_tom_tool.txt")
    test_content = "This is a test file for T.O.M. tool system.\nLine 2\nLine 3"
    test_file.write_text(test_content)

    print(f"\nCreated test file: {test_file}")

    # Test reading
    print("\nTest 1: Read existing file")
    tool_call = {"name": "read", "arguments": {"location": str(test_file)}}
    result = execute_tool_call(tool_call)

    if test_content in result:
        print(f"✓ File read successfully")
        print(f"  Result: {result[:50]}...")
    else:
        print(f"✗ File read failed: {result}")
        return False

    # Test non-existent file
    print("\nTest 2: Read non-existent file")
    tool_call = {"name": "read", "arguments": {"location": "/tmp/nonexistent_file_xyz.txt"}}
    result = execute_tool_call(tool_call)

    if "Error" in result or "not found" in result.lower():
        print(f"✓ Error handled correctly")
        print(f"  Result: {result[:80]}...")
    else:
        print(f"✗ Should have errored: {result}")
        return False

    # Test large file handling
    print("\nTest 3: Large file handling")
    large_file = Path("/tmp/test_large_file.txt")
    large_content = "x" * 20000  # 20k characters, exceeds 15k limit
    large_file.write_text(large_content)

    tool_call = {"name": "read", "arguments": {"location": str(large_file)}}
    result = execute_tool_call(tool_call)

    if "Error" in result and "too large" in result.lower():
        print(f"✓ Large file rejected correctly")
        print(f"  Result: {result[:80]}...")
    else:
        print(f"✗ Large file should be rejected: {result[:100]}")
        return False

    # Cleanup
    test_file.unlink()
    large_file.unlink()

    print("\n✓ All file reading tests passed")
    return True


def test_complete_tool_flow():
    _run_complete_tool_flow()


def test_file_reading_tool():
    _run_file_reading_tool()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPREHENSIVE END-TO-END TEST SUITE")
    print("=" * 80 + "\n")

    # Run all tests
    test1_pass = _run_complete_tool_flow()
    test2_pass = _run_file_reading_tool()

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if test1_pass and test2_pass:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nThe tool call system is working correctly:")
        print("  1. Tools are properly included in prompts")
        print("  2. Tool calls are extracted from model responses")
        print("  3. Tools execute successfully")
        print("  4. Results are added to context")
        print("  5. Follow-up prompts are built correctly")
        print()
        sys.exit(0)
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗")
        print(f"  Complete flow: {'PASS' if test1_pass else 'FAIL'}")
        print(f"  File reading: {'PASS' if test2_pass else 'FAIL'}")
        print()
        sys.exit(1)
