#!/usr/bin/env python3
"""
Test prompt building and tool integration with chat templates
This tests the CRITICAL part where tools are injected into the prompt
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from context_manager import ContextManager
from tools import TOOLS_DEFINITIONS
from utils import load_model_config


def test_prompt_with_tools():
    """Test how tools are included in the prompt"""
    print("=" * 80)
    print("TEST: Prompt Building with Tools")
    print("=" * 80)

    # Create context manager
    context_mgr = ContextManager(max_context_tokens=8000)

    # Add a simple conversation
    context_mgr.add_message("user", "What time is it?")

    print("\nTest Case 1: Without tokenizer (fallback prompt)")
    print("-" * 80)

    # Build prompt without tokenizer (fallback mode)
    prompt_without_tools = context_mgr.build_prompt(tokenizer=None, include_tools=False)
    prompt_with_tools = context_mgr.build_prompt(tokenizer=None, include_tools=True)

    print("\nPrompt WITHOUT tools:")
    print("-" * 40)
    print(prompt_without_tools)
    print("-" * 40)

    print("\n\nPrompt WITH tools:")
    print("-" * 40)
    print(prompt_with_tools)
    print("-" * 40)

    # Verify tools are present
    assert "get_datetime" in prompt_with_tools, "Tool 'get_datetime' not found in prompt"
    assert "read" in prompt_with_tools, "Tool 'read' not found in prompt"
    assert "get_datetime" not in prompt_without_tools, "Tools should not be in no-tool prompt"

    print("\n✓ Fallback prompt builder includes tools correctly")

    return prompt_with_tools


def test_with_real_tokenizer():
    """Test with actual model tokenizer to see chat template behavior"""
    print("\n" + "=" * 80)
    print("TEST: Prompt Building with Real Tokenizer")
    print("=" * 80)

    # Find model path
    model_path = Path('./Qwen3-4B-Thinking-2507-8bit')
    if not model_path.exists():
        print(f"\n⚠ Model not found at {model_path}")
        print("Skipping tokenizer test - requires actual model")
        return None

    print(f"\nLoading tokenizer from: {model_path}")

    try:
        from mlx_lm import load
        model, tokenizer = load(str(model_path))
        print(f"✓ Tokenizer loaded successfully")

        # Check if chat template exists
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            print(f"✓ Chat template available")
            print(f"\nChat template preview:")
            print("-" * 40)
            template = tokenizer.chat_template
            if len(template) > 500:
                print(template[:500] + "\n... [truncated]")
            else:
                print(template)
            print("-" * 40)
        else:
            print("⚠ No chat template available - will use fallback")

        # Create context and test
        context_mgr = ContextManager(max_context_tokens=8000, tokenizer=tokenizer)
        context_mgr.add_message("user", "What time is it?")

        print("\n" + "-" * 80)
        print("Building prompt WITH tools using chat template:")
        print("-" * 80)

        try:
            prompt_with_tools = context_mgr.build_prompt(tokenizer, include_tools=True)
            print("\n" + prompt_with_tools)
            print("\n" + "-" * 80)

            # Check if tools are in the prompt
            tools_found = "get_datetime" in prompt_with_tools and "read" in prompt_with_tools

            if tools_found:
                print("\n✓ Tools ARE present in the prompt")
            else:
                print("\n✗ WARNING: Tools NOT found in the prompt!")
                print("This is likely the root cause of tool call failures!")

            return prompt_with_tools

        except Exception as e:
            print(f"\n✗ Error building prompt with tools: {e}")
            print(f"Exception type: {type(e).__name__}")
            print(f"This means the chat template doesn't support 'tools' parameter!")

            # Try without tools parameter
            print("\n" + "-" * 80)
            print("Building prompt WITHOUT tools parameter:")
            print("-" * 80)

            prompt_no_tools = context_mgr.build_prompt(tokenizer, include_tools=False)
            print("\n" + prompt_no_tools)
            print("\n" + "-" * 80)

            return None

    except Exception as e:
        print(f"\n✗ Failed to load model/tokenizer: {e}")
        return None


def test_chat_template_tools_support():
    """Test if the chat template actually supports tools"""
    print("\n" + "=" * 80)
    print("TEST: Chat Template Tools Support")
    print("=" * 80)

    model_path = Path('./Qwen3-4B-Thinking-2507-8bit')
    if not model_path.exists():
        print(f"\n⚠ Skipping - model not found at {model_path}")
        return None

    try:
        from mlx_lm import load
        model, tokenizer = load(str(model_path))

        # Test if apply_chat_template accepts tools parameter
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What time is it?"}
        ]

        print("\nTest 1: apply_chat_template WITHOUT tools")
        try:
            result1 = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print("✓ Works without tools parameter")
        except Exception as e:
            print(f"✗ Failed even without tools: {e}")
            return False

        print("\nTest 2: apply_chat_template WITH tools parameter")
        try:
            result2 = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=TOOLS_DEFINITIONS
            )
            print("✓ Works WITH tools parameter")
            print("\nChecking if tools are actually in the output...")

            if "get_datetime" in result2:
                print("✓ Tools ARE included in the output")
                print("\nPrompt with tools:")
                print("-" * 40)
                print(result2)
                print("-" * 40)
                return True
            else:
                print("✗ Tools parameter accepted but tools NOT in output")
                print("This means the chat template ignores the tools parameter!")
                print("\nPrompt:")
                print("-" * 40)
                print(result2)
                print("-" * 40)
                return False

        except TypeError as e:
            if "tools" in str(e) or "unexpected keyword" in str(e):
                print(f"✗ Chat template does NOT support 'tools' parameter")
                print(f"   Error: {e}")
                print("\n⚠ ROOT CAUSE IDENTIFIED:")
                print("   The tokenizer's chat template doesn't support the 'tools' parameter!")
                print("   Tools are never being sent to the model!")
                return False
            else:
                raise

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PROMPT BUILDING & TOOL INTEGRATION TEST SUITE")
    print("=" * 80 + "\n")

    # Run tests
    test_prompt_with_tools()
    test_with_real_tokenizer()
    supports_tools = test_chat_template_tools_support()

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    if supports_tools is None:
        print("\n⚠ Could not complete full testing (model not available)")
    elif supports_tools is False:
        print("\n✗ CRITICAL ISSUE FOUND:")
        print("   The chat template does NOT support the 'tools' parameter")
        print("   or does not include tools in the generated prompt.")
        print("\n   This means the model NEVER sees the available tools!")
        print("   The model cannot make tool calls if it doesn't know tools exist.")
        print("\n   SOLUTION: Need to manually inject tools into the prompt")
        print("   or use a different chat template format.")
    elif supports_tools is True:
        print("\n✓ Chat template correctly supports and includes tools")
        print("   Tools should be visible to the model.")
        print("   If tool calls still fail, the issue is likely:")
        print("   - Model not trained for tool calling")
        print("   - Incorrect tool call format expected by model")
        print("   - System prompt doesn't instruct model to use tools")

    print()
