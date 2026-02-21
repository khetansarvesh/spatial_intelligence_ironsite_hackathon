"""
Productivity Agent
LLM-powered agent for natural language productivity queries.

Supports both OpenAI and Anthropic APIs with function calling.
"""

import json
import os
from typing import Optional, Dict, Any, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from temporal.session_aggregator import SessionReport
from .tools import AgentTools, AGENT_TOOL_SCHEMAS
from .prompts import SYSTEM_PROMPT


class ProductivityAgent:
    """
    Natural language agent for querying productivity data.

    Uses LLM function calling to answer questions about worker productivity.
    Supports both OpenAI and Anthropic APIs.
    """

    def __init__(
        self,
        report: SessionReport,
        provider: str = "openai",  # "openai" or "anthropic"
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize productivity agent.

        Args:
            report: SessionReport containing productivity data
            provider: LLM provider ("openai" or "anthropic")
            model: Model name (default: gpt-4o for OpenAI, claude-3-5-sonnet for Anthropic)
            api_key: API key (default: from environment variable)
        """
        self.report = report
        self.provider = provider.lower()
        self.tools = AgentTools(report)

        # Set default model
        if model is None:
            if self.provider == "openai":
                self.model = "gpt-4o"
            elif self.provider == "anthropic":
                self.model = "claude-3-5-sonnet-20241022"
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        else:
            self.model = model

        # Initialize client
        if self.provider == "openai":
            self._init_openai(api_key)
        elif self.provider == "anthropic":
            self._init_anthropic(api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        self.client = OpenAI(api_key=api_key)

    def _init_anthropic(self, api_key: Optional[str]):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")

        self.client = Anthropic(api_key=api_key)

    def chat(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Chat with the agent about productivity.

        Args:
            user_message: User's question
            max_iterations: Maximum number of function calling iterations

        Returns:
            Agent's response
        """
        if self.provider == "openai":
            return self._chat_openai(user_message, max_iterations)
        elif self.provider == "anthropic":
            return self._chat_anthropic(user_message, max_iterations)

    def _chat_openai(self, user_message: str, max_iterations: int) -> str:
        """Chat using OpenAI API."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        # Convert tool schemas to OpenAI format
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
            for tool in AGENT_TOOL_SCHEMAS
        ]

        for iteration in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # If no tool calls, return the response
            if not message.tool_calls:
                return message.content

            # Add assistant message to history
            messages.append(message)

            # Execute tool calls
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                # Execute the function
                result = self._execute_tool(function_name, function_args)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

        # If we hit max iterations, get final response
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return final_response.choices[0].message.content

    def _chat_anthropic(self, user_message: str, max_iterations: int) -> str:
        """Chat using Anthropic API."""
        messages = [
            {"role": "user", "content": user_message}
        ]

        # Convert tool schemas to Anthropic format
        tools = [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"]
            }
            for tool in AGENT_TOOL_SCHEMAS
        ]

        for iteration in range(max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=tools,
            )

            # Check if done
            if response.stop_reason == "end_turn":
                # Extract text content
                for block in response.content:
                    if block.type == "text":
                        return block.text
                return ""

            # Process tool uses
            if response.stop_reason == "tool_use":
                # Add assistant message
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })

                # Execute tools and collect results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result)
                        })

                # Add tool results
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

        # Get final response
        final_response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        for block in final_response.content:
            if block.type == "text":
                return block.text

        return ""

    def _execute_tool(self, function_name: str, function_args: Dict[str, Any]) -> Any:
        """Execute a tool function."""
        # Map function names to AgentTools methods
        method = getattr(self.tools, function_name, None)

        if method is None:
            return {"error": f"Unknown function: {function_name}"}

        try:
            # Call the method with unpacked arguments
            result = method(**function_args)
            return result
        except Exception as e:
            return {"error": str(e)}

    def get_quick_summary(self) -> str:
        """Get a quick text summary without LLM."""
        return self.report.get_summary()


# Quick test / example usage
if __name__ == "__main__":
    """
    Example usage of the productivity agent.

    Note: Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.
    """
    print("ProductivityAgent Example")
    print("=" * 60)

    # Create a mock report for testing
    from temporal.session_aggregator import SessionReport, ActivityBreakdown, ToolUsage
    from temporal.activity_classifier import ActivityState, ActivitySegment

    # Create sample segments
    segments = [
        ActivitySegment(0.0, 100.0, ActivityState.ACTIVE_TOOL_USE, "drill", 1.0, 0.9, 3000),
        ActivitySegment(100.0, 120.0, ActivityState.IDLE, None, 0.0, 0.8, 600),
        ActivitySegment(120.0, 200.0, ActivityState.ACTIVE_TOOL_USE, "hammer", 1.0, 0.9, 2400),
    ]

    # Create mock report
    report = SessionReport(
        session_duration=200.0,
        start_time=0.0,
        end_time=200.0,
        total_frames=6000,
        productivity_score=0.85,
        productive_time=180.0,
        idle_time=20.0,
        idle_percentage=10.0,
        activity_breakdown={
            "ACTIVE_TOOL_USE": ActivityBreakdown(
                ActivityState.ACTIVE_TOOL_USE,
                180.0,
                90.0,
                2,
                90.0,
                1.0
            ),
            "IDLE": ActivityBreakdown(
                ActivityState.IDLE,
                20.0,
                10.0,
                1,
                20.0,
                0.0
            ),
        },
        tool_usage={
            "drill": ToolUsage("drill", 100.0, 1, ["ACTIVE_TOOL_USE"]),
            "hammer": ToolUsage("hammer", 80.0, 1, ["ACTIVE_TOOL_USE"]),
        },
        most_used_tool="drill",
        tool_switches=1,
        segments=segments,
        insights=["High productivity session (85%)"],
        recommendations=["Maintain current workflow"],
    )

    print("\nSession Report Summary:")
    print(report.get_summary())

    # Test agent tools without LLM
    print("\n" + "=" * 60)
    print("Testing Agent Tools (without LLM)")
    print("=" * 60)

    tools = AgentTools(report)

    print("\n1. Productivity Score:")
    print(json.dumps(tools.get_productivity_score(), indent=2))

    print("\n2. Tool Usage:")
    print(json.dumps(tools.get_tool_usage(), indent=2))

    print("\n3. Idle Periods:")
    print(json.dumps(tools.find_idle_periods(10), indent=2))

    # Optionally test with LLM if API key is available
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        print("\n" + "=" * 60)
        print("Testing with LLM Agent")
        print("=" * 60)

        provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
        agent = ProductivityAgent(report, provider=provider)

        questions = [
            "What was the overall productivity?",
            "Which tool was used the most?",
            "How much time was idle?",
        ]

        for question in questions:
            print(f"\nQ: {question}")
            response = agent.chat(question)
            print(f"A: {response}")
    else:
        print("\n" + "=" * 60)
        print("Skipping LLM test - no API key found")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test with LLM")
        print("=" * 60)
