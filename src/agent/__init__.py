"""
CodeAct Agent Package

DSPy-based agent for frame-level productivity analysis.
Uses CodeAct paradigm to generate and execute Python code for queries.
"""

from .agent import CodeActAgent
from .tools import get_tool_registry, get_tool_descriptions
from .sandbox import SafeCodeExecutor, CodeFormatter
from .signatures import (
    FrameQuerySignature,
    CodeGenerationSignature,
    ResultFormatterSignature,
    CodeFixSignature,
    get_data_schema_description,
)

__all__ = [
    # Main agent
    "CodeActAgent",

    # Tools
    "get_tool_registry",
    "get_tool_descriptions",

    # Sandbox
    "SafeCodeExecutor",
    "CodeFormatter",

    # Signatures
    "FrameQuerySignature",
    "CodeGenerationSignature",
    "ResultFormatterSignature",
    "CodeFixSignature",
    "get_data_schema_description",
]
