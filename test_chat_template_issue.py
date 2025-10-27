#!/usr/bin/env python3
"""
Test to diagnose the likely root cause of tool call failures.

HYPOTHESIS: The chat template accepts the 'tools' parameter but silently ignores it,
so tools are never included in the prompt sent to the model.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


class MockTokenizer:
    """Mock tokenizer that simulates the problematic behavior"""

    def __init__(self, supports_tools_param=True, actually_includes_tools=False):
        self.supports_tools_param = supports_tools_param
        self.actually_includes_tools = actually_includes_tools
        self.chat_template = "mock_template"  # Has a template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, tools=None):
        """
        Mock implementation that simulates different chat template behaviors
        """
        # Build basic prompt
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|{role}|>\n{content}<|end|>\n")

        if add_generation_prompt:
            prompt_parts.append("<|assistant|>\n")

        # The CRITICAL part: some templates accept 'tools' but ignore it
        if tools is not None and not self.supports_tools_param:
            raise TypeError("apply_chat_template() got an unexpected keyword argument 'tools'")

        # Even if tools parameter is accepted, they might not be included!
        if tools is not None and self.actually_includes_tools:
            tools_str = json.dumps(tools, indent=2)
            prompt_parts.insert(1, f"<|tools|>\n{tools_str}<|end|>\n")

        return "".join(prompt_parts)


def test_scenario_1_template_rejects_tools():
    """
    Scenario 1: Chat template raises TypeError when given 'tools' parameter
    This is GOOD - it will fall back to our custom prompt builder
    """
    print("=" * 80)
    print("SCENARIO 1: Chat template REJECTS 'tools' parameter")
    print("=" * 80)

    tokenizer = MockTokenizer(supports_tools_param=False)

    from context_manager import ContextManager
    from tools import TOOLS_DEFINITIONS

    context = ContextManager(max_context_tokens=8000, tokenizer=tokenizer)
    context.add_message("user", "What time is it?")

    try:
        prompt = context.build_prompt(tokenizer, include_tools=True)

        # Check if tools are in the prompt
        if "get_datetime" in prompt:
            print("\n✓ RESULT: Tools ARE in the prompt (fallback worked)")
            print("   The exception triggered fallback to _build_fallback_prompt()")
            return True
        else:
            print("\n✗ RESULT: Tools NOT in prompt (something wrong with fallback)")
            return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def test_scenario_2_template_ignores_tools():
    """
    Scenario 2: Chat template ACCEPTS 'tools' parameter but IGNORES it
    This is BAD - no exception, so no fallback, but tools not included
    """
    print("\n" + "=" * 80)
    print("SCENARIO 2: Chat template ACCEPTS but IGNORES 'tools' parameter")
    print("=" * 80)
    print("This is the likely ROOT CAUSE of tool call failures!")

    tokenizer = MockTokenizer(supports_tools_param=True, actually_includes_tools=False)

    from context_manager import ContextManager
    from tools import TOOLS_DEFINITIONS

    context = ContextManager(max_context_tokens=8000, tokenizer=tokenizer)
    context.add_message("user", "What time is it?")

    try:
        prompt = context.build_prompt(tokenizer, include_tools=True)

        print(f"\nGenerated prompt preview:")
        print("-" * 40)
        print(prompt[:500] if len(prompt) > 500 else prompt)
        print("-" * 40)

        # Check if tools are in the prompt
        if "get_datetime" in prompt:
            print("\n✓ RESULT: Tools ARE in the prompt")
            return True
        else:
            print("\n✗ RESULT: Tools NOT in prompt!")
            print("   Chat template accepted 'tools' param but didn't include them")
            print("   No exception was raised, so no fallback happened")
            print("   MODEL NEVER SEES THE TOOLS - this breaks tool calling!")
            return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def test_scenario_3_template_includes_tools():
    """
    Scenario 3: Chat template properly supports and includes tools
    This is IDEAL but rare
    """
    print("\n" + "=" * 80)
    print("SCENARIO 3: Chat template PROPERLY SUPPORTS tools")
    print("=" * 80)

    tokenizer = MockTokenizer(supports_tools_param=True, actually_includes_tools=True)

    from context_manager import ContextManager
    from tools import TOOLS_DEFINITIONS

    context = ContextManager(max_context_tokens=8000, tokenizer=tokenizer)
    context.add_message("user", "What time is it?")

    try:
        prompt = context.build_prompt(tokenizer, include_tools=True)

        # Check if tools are in the prompt
        if "get_datetime" in prompt:
            print("\n✓ RESULT: Tools ARE in the prompt (proper support)")
            return True
        else:
            print("\n✗ RESULT: Tools NOT in prompt (unexpected)")
            return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def show_solution():
    """Show the solution to fix Scenario 2"""
    print("\n" + "=" * 80)
    print("SOLUTION FOR SCENARIO 2")
    print("=" * 80)

    print("""
The fix is to VERIFY that tools are actually in the generated prompt,
and fall back if they're not, even when no exception was raised.

OPTION 1: Always use fallback prompt for tool calls
-------------------------------------------------
Change build_prompt() to always use fallback when include_tools=True:

    def build_prompt(self, tokenizer, include_tools: bool = False) -> str:
        chat_messages = [{"role": "system", "content": self.system_prompt}]
        chat_messages.extend(self.messages)

        # Use fallback when tools are needed (most reliable)
        if include_tools:
            return self._build_fallback_prompt(chat_messages, include_tools)

        # Use chat template only when no tools needed
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            ...

OPTION 2: Verify tools in output and fallback if missing
--------------------------------------------------------
    def build_prompt(self, tokenizer, include_tools: bool = False) -> str:
        chat_messages = [{"role": "system", "content": self.system_prompt}]
        chat_messages.extend(self.messages)

        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            try:
                kwargs = {"tokenize": False, "add_generation_prompt": True}

                if include_tools and TOOLS_DEFINITIONS:
                    kwargs["tools"] = TOOLS_DEFINITIONS

                prompt = tokenizer.apply_chat_template(chat_messages, **kwargs)

                # VERIFY tools are actually present!
                if include_tools and TOOLS_DEFINITIONS:
                    first_tool_name = TOOLS_DEFINITIONS[0]["function"]["name"]
                    if first_tool_name not in prompt:
                        logger.warning(
                            f"Chat template ignored tools parameter, using fallback"
                        )
                        return self._build_fallback_prompt(chat_messages, include_tools)

                return prompt
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback")

        return self._build_fallback_prompt(chat_messages, include_tools)

OPTION 3: Force tool instructions in system prompt
--------------------------------------------------
Add tool usage instructions directly to the system prompt:

    When tools are available, you MUST use this XML format:
    <tool_call>
    {"name": "tool_name", "arguments": {...}}
    </tool_call>

    Available tools:
    [tool definitions here]

This ensures the model knows about tools regardless of chat template.
""")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CHAT TEMPLATE TOOL SUPPORT DIAGNOSTIC")
    print("=" * 80 + "\n")

    results = []

    # Test all scenarios
    results.append(("Scenario 1: Template rejects tools", test_scenario_1_template_rejects_tools()))
    results.append(("Scenario 2: Template ignores tools", test_scenario_2_template_ignores_tools()))
    results.append(("Scenario 3: Template includes tools", test_scenario_3_template_includes_tools()))

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    # Likely diagnosis
    print("\n" + "=" * 80)
    print("LIKELY ROOT CAUSE")
    print("=" * 80)

    if not results[1][1]:  # Scenario 2 failed
        print("""
Your model's chat template likely ACCEPTS the 'tools' parameter
but IGNORES it (Scenario 2).

This means:
1. No exception is raised
2. No fallback happens
3. Tools never appear in the prompt
4. Model has no idea tools exist
5. Tool calls never happen!

This is a silent failure - the code runs without errors but doesn't work.
""")
        show_solution()
    else:
        print("\nAll scenarios passed - this was just a diagnostic test.")
        print("To find the real issue, you need to test with your actual model's tokenizer.")

    print()
