#!/usr/bin/env python3
"""
Comprehensive test suite for tool call system
Tests each component of the tool call flow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.tools import (
    TOOLS_DEFINITIONS,
    TOOLS_REGISTRY,
    extract_tool_calls,
    execute_tool_call,
    strip_tool_calls,
    parse_tool_arguments
)


def test_tool_definitions():
    """Test 1: Verify tool definitions are properly formatted"""
    print("=" * 80)
    print("TEST 1: Tool Definitions")
    print("=" * 80)

    print(f"\nRegistered tools: {list(TOOLS_REGISTRY.keys())}")
    print(f"Tool definitions count: {len(TOOLS_DEFINITIONS)}")

    print("\nTool Definitions (JSON):")
    print(json.dumps(TOOLS_DEFINITIONS, indent=2))

    # Validate structure
    for tool_def in TOOLS_DEFINITIONS:
        assert tool_def["type"] == "function", f"Invalid type: {tool_def}"
        assert "function" in tool_def, f"Missing function key: {tool_def}"
        assert "name" in tool_def["function"], f"Missing name: {tool_def}"
        assert "description" in tool_def["function"], f"Missing description: {tool_def}"
        assert "parameters" in tool_def["function"], f"Missing parameters: {tool_def}"
        print(f"✓ {tool_def['function']['name']}: Valid structure")

    print("\n✓ All tool definitions are valid\n")


def test_tool_extraction():
    """Test 2: Verify tool call extraction from text"""
    print("=" * 80)
    print("TEST 2: Tool Call Extraction")
    print("=" * 80)

    # Test case 1: Single tool call
    text1 = """Let me check the time for you.
<tool_call>
{"name": "get_datetime", "arguments": {}}
</tool_call>"""

    print("\nTest Case 1: Single tool call")
    print(f"Input text:\n{text1}\n")

    tools1 = extract_tool_calls(text1)
    print(f"Extracted: {tools1}")
    assert len(tools1) == 1, f"Expected 1 tool, got {len(tools1)}"
    assert tools1[0]["name"] == "get_datetime", f"Wrong tool name: {tools1[0]}"
    print("✓ Single tool call extraction works")

    # Test case 2: Tool call with arguments
    text2 = """I'll read that file for you.
<tool_call>
{"name": "read", "arguments": {"location": "/tmp/test.txt"}}
</tool_call>"""

    print("\nTest Case 2: Tool call with arguments")
    print(f"Input text:\n{text2}\n")

    tools2 = extract_tool_calls(text2)
    print(f"Extracted: {tools2}")
    assert len(tools2) == 1, f"Expected 1 tool, got {len(tools2)}"
    assert tools2[0]["name"] == "read", f"Wrong tool name: {tools2[0]}"
    assert "location" in tools2[0]["arguments"], f"Missing arguments: {tools2[0]}"
    print("✓ Tool call with arguments extraction works")

    # Test case 3: Multiple tool calls
    text3 = """<tool_call>
{"name": "get_datetime", "arguments": {}}
</tool_call>
<tool_call>
{"name": "read", "arguments": {"location": "/tmp/test.txt"}}
</tool_call>"""

    print("\nTest Case 3: Multiple tool calls")
    print(f"Input text:\n{text3}\n")

    tools3 = extract_tool_calls(text3)
    print(f"Extracted: {tools3}")
    assert len(tools3) == 2, f"Expected 2 tools, got {len(tools3)}"
    print("✓ Multiple tool calls extraction works")

    # Test case 4: No tool calls
    text4 = "Just a regular response with no tools."

    print("\nTest Case 4: No tool calls")
    print(f"Input text:\n{text4}\n")

    tools4 = extract_tool_calls(text4)
    print(f"Extracted: {tools4}")
    assert len(tools4) == 0, f"Expected 0 tools, got {len(tools4)}"
    print("✓ No tool calls handled correctly")

    print("\n✓ All extraction tests passed\n")


def test_tool_execution():
    """Test 3: Verify tool execution"""
    print("=" * 80)
    print("TEST 3: Tool Execution")
    print("=" * 80)

    # Test get_datetime
    print("\nTest Case 1: Execute get_datetime")
    tool_call_1 = {"name": "get_datetime", "arguments": {}}
    result_1 = execute_tool_call(tool_call_1)
    print(f"Result: {result_1}")
    assert "Error" not in result_1, f"Execution failed: {result_1}"
    assert ":" in result_1 and "on" in result_1, f"Invalid datetime format: {result_1}"
    print("✓ get_datetime executes successfully")

    # Test read with non-existent file
    print("\nTest Case 2: Execute read with non-existent file")
    tool_call_2 = {"name": "read", "arguments": {"location": "/tmp/nonexistent_file_12345.txt"}}
    result_2 = execute_tool_call(tool_call_2)
    print(f"Result: {result_2}")
    assert "Error" in result_2 or "not found" in result_2.lower(), f"Should error on missing file: {result_2}"
    print("✓ read handles missing files correctly")

    # Test read with existing file
    print("\nTest Case 3: Execute read with test file")
    test_file = Path("/tmp/test_tool_file.txt")
    test_file.write_text("Test content for tool system\nLine 2\nLine 3")

    tool_call_3 = {"name": "read", "arguments": {"location": str(test_file)}}
    result_3 = execute_tool_call(tool_call_3)
    print(f"Result: {result_3}")
    assert "Test content" in result_3, f"Content not found: {result_3}"
    print("✓ read successfully reads files")

    test_file.unlink()  # Cleanup

    # Test invalid tool name
    print("\nTest Case 4: Execute invalid tool")
    tool_call_4 = {"name": "invalid_tool_xyz", "arguments": {}}
    result_4 = execute_tool_call(tool_call_4)
    print(f"Result: {result_4}")
    assert "Error" in result_4 or "Unknown" in result_4, f"Should error on invalid tool: {result_4}"
    print("✓ Invalid tools handled correctly")

    print("\n✓ All execution tests passed\n")


def test_strip_tool_calls():
    """Test 4: Verify tool call stripping"""
    print("=" * 80)
    print("TEST 4: Tool Call Stripping")
    print("=" * 80)

    text = """Let me help you with that.
<tool_call>
{"name": "get_datetime", "arguments": {}}
</tool_call>
I'll check the time."""

    print(f"Original text:\n{text}\n")

    stripped = strip_tool_calls(text)
    print(f"Stripped text:\n{stripped}\n")

    assert "<tool_call>" not in stripped, "Tool call tags not removed"
    assert "Let me help you" in stripped, "Content was removed"
    assert "I'll check the time" in stripped, "Content was removed"
    print("✓ Tool call stripping works\n")


def test_argument_parsing():
    """Test 5: Verify argument parsing"""
    print("=" * 80)
    print("TEST 5: Argument Parsing")
    print("=" * 80)

    # Test dict input
    args1 = {"location": "/tmp/test.txt"}
    parsed1 = parse_tool_arguments(args1)
    print(f"Dict input: {args1}")
    print(f"Parsed: {parsed1}")
    assert parsed1 == args1, "Dict parsing failed"
    print("✓ Dict parsing works")

    # Test JSON string input
    args2 = '{"location": "/tmp/test.txt"}'
    parsed2 = parse_tool_arguments(args2)
    print(f"\nJSON string input: {args2}")
    print(f"Parsed: {parsed2}")
    assert parsed2 == {"location": "/tmp/test.txt"}, "JSON parsing failed"
    print("✓ JSON string parsing works")

    # Test invalid input
    args3 = "invalid string"
    parsed3 = parse_tool_arguments(args3)
    print(f"\nInvalid input: {args3}")
    print(f"Parsed: {parsed3}")
    assert parsed3 == {}, "Invalid input should return empty dict"
    print("✓ Invalid input handled correctly\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TOOL SYSTEM COMPREHENSIVE TEST SUITE")
    print("=" * 80 + "\n")

    tests = [
        ("Tool Definitions", test_tool_definitions),
        ("Tool Extraction", test_tool_extraction),
        ("Tool Execution", test_tool_execution),
        ("Tool Call Stripping", test_strip_tool_calls),
        ("Argument Parsing", test_argument_parsing),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n✗ {name} FAILED: {e}\n")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed, error in results:
        if passed:
            print(f"✓ {name}: PASSED")
        else:
            print(f"✗ {name}: FAILED - {error}")

    total = len(results)
    passed = sum(1 for _, p, _ in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ ALL TESTS PASSED\n")
        sys.exit(0)
    else:
        print(f"\n✗ {total - passed} TESTS FAILED\n")
        sys.exit(1)
