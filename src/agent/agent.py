from openai import OpenAI
from .tools import ProductivityTools
from .insights import InsightEngine
from .prompts import SYSTEM_PROMPT
import json

class ProductivityAgent:
    def __init__(self, activity_segments, api_key: str):
        self.insight_engine = InsightEngine(activity_segments)
        self.tools = ProductivityTools(activity_segments)
        self.client = OpenAI(api_key=api_key)

        self.function_definitions = [
            {
                "name": "get_activity_summary",
                "description": "Get summary of worker activities in a time range",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "number"},
                        "end_time": {"type": "number"}
                    },
                    "required": ["start_time", "end_time"]
                }
            },
            {
                "name": "get_tool_usage",
                "description": "Get total usage time per tool",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "find_idle_periods",
                "description": "Find idle segments longer than threshold",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_duration_seconds": {"type": "integer"}
                    }
                }
            },
            {
                "name": "get_productivity_score",
                "description": "Get overall productivity score",
                "parameters": {"type": "object", "properties": {}}
            },
           {
                "name": "generate_insights",
                "description": "Generate executive-level productivity insights",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "detect_fatigue",
                "description": "Detect behavioral fatigue within the session",
                "parameters": {"type": "object", "properties": {}}
            }
        ]

    # Tool Execution Router

    def _execute_tool(self, name, args):
        if name == "get_activity_summary":
            return self.tools.get_activity_summary(**args)

        elif name == "get_tool_usage":
            return self.tools.get_tool_usage()

        elif name == "find_idle_periods":
            return self.tools.find_idle_periods(**args)

        elif name == "get_productivity_score":
            return self.tools.get_productivity_score()
        
        elif name == "generate_insights":
            return self.insight_engine.generate_summary()
        
        elif name == "detect_fatigue":
            return self.insight_engine.detect_fatigue()

        else:
            return {"error": "Unknown tool"}

    # Chat Interface

    def chat(self, user_message: str):

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            tools=self.function_definitions,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # If model wants to call a tool
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            tool_result = self._execute_tool(tool_name, tool_args)

            # Send tool result back to LLM for final answer
            second_response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                    message,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    }
                ]
            )

            return second_response.choices[0].message.content

        return message.content