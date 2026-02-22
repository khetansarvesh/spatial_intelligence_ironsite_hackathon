import dspy

class FrameQuerySignature(dspy.Signature):
    """
    Analyze a user query to determine query type and relevant tools.

    This signature is used for the first step of query processing,
    where we understand what the user is asking and which tools might help.
    """

    query: str = dspy.InputField(
        desc="User's natural language question about productivity data"
    )
    available_tools: str = dspy.InputField(
        desc="List of available tool function names and their descriptions"
    )
    data_schema: str = dspy.InputField(
        desc="Description of the data schema available to query"
    )

    query_type: str = dspy.OutputField(
        desc="Type of query: 'time_based' (idle time, duration), "
             "'spatial' (locations, positions), 'activity' (tools used, patterns), "
             "or 'aggregate' (summaries, counts)"
    )
    relevant_tools: str = dspy.OutputField(
        desc="Comma-separated list of tool function names that would help answer this query"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of how to approach answering this query"
    )


class CodeGenerationSignature(dspy.Signature):
    """
    Generate Python code to answer a frame-level productivity query.

    This signature is used to generate the actual code that will be executed
    in the sandbox to compute the answer.
    """

    query: str = dspy.InputField(
        desc="User's natural language question"
    )
    query_type: str = dspy.InputField(
        desc="Type of query identified (time_based, spatial, activity, aggregate)"
    )
    relevant_tools: str = dspy.InputField(
        desc="Available tool functions that should be used"
    )
    data_schema: str = dspy.InputField(
        desc="Description of data structure and available variables"
    )
    tool_descriptions: str = dspy.InputField(
        desc="Detailed descriptions of how to use each tool function"
    )

    rationale: str = dspy.OutputField(
        desc="Step-by-step reasoning about how to solve this query with code"
    )
    code: str = dspy.OutputField(
        desc="Python code that computes the answer. "
             "MUST assign the final result to a variable named 'answer'. "
             "Use only the provided tool functions. No imports allowed."
    )


class ResultFormatterSignature(dspy.Signature):
    """
    Format a computed result as a natural language response.

    This signature is used to convert the raw result from code execution
    into a human-friendly response.
    """

    query: str = dspy.InputField(
        desc="Original user question"
    )
    result: str = dspy.InputField(
        desc="Computed result from code execution (may be dict, list, or string)"
    )

    response: str = dspy.OutputField(
        desc="Natural language response that directly answers the user's question. "
             "Include specific numbers, times, and percentages. Be concise but complete."
    )


class CodeFixSignature(dspy.Signature):
    """
    Fix code that failed to execute.

    This signature is used when the generated code has errors,
    to attempt to fix them based on the error message.
    """

    query: str = dspy.InputField(
        desc="Original user question"
    )
    original_code: str = dspy.InputField(
        desc="The code that failed to execute"
    )
    error_message: str = dspy.InputField(
        desc="The error message from execution"
    )
    available_tools: str = dspy.InputField(
        desc="Available tool functions"
    )

    analysis: str = dspy.OutputField(
        desc="Analysis of what went wrong and how to fix it"
    )
    fixed_code: str = dspy.OutputField(
        desc="Corrected Python code that should work. "
             "MUST assign result to 'answer' variable."
    )


# Convenience function to get schema descriptions
def get_data_schema_description() -> str:
    """
    Return the description of the frame-level data schema.

    This is passed to the LLM so it knows what data is available.
    """
    return """
The data is available in a variable called 'data' with this structure:

data['total_frames'] - Total number of frames in the video (int)
data['fps'] - Video frame rate (if available)
data['frames'] - List of frame dictionaries, each containing:
  - 'frame_index': int (1-based frame number)
  - 'timestamp_sec': float (time in seconds)
  - 'wearer_productivity_status': "ACTIVE" or "IDLE"
  - 'scene_activity_status': str (scene-level activity status)
  - 'task_name': str (e.g., "align_or_fit_pipe", "cut_material")
  - 'task_family': str (e.g., "positioning_alignment", "cutting")
  - 'task_confidence': float (0-1)
  - 'task_unknown': bool
  - 'scene': str (e.g., "plumbing", "electrical")
  - 'scene_confidence': float (0-1)
  - 'motion': str (e.g., "unknown", "stationary", "walking")
  - 'primary_tool': str (e.g., "hand pipe cutter torch", "drill")
  - 'active_interaction': bool (True if actively using tool)
  - 'hands_count': int (number of hands detected)
  - 'tools_count': int (number of tools detected)
  - 'interactions_count': int (number of interactions)

IMPORTANT NOTES:
- frames is a LIST, not a dict - iterate with: for frame in data['frames']
- Use timestamp_sec for time-based queries
- Use wearer_productivity_status for idle/active analysis
- The 'data' variable is already available - don't try to load or import it
- Time strings like "11:00" mean 11 minutes and 0 seconds (use parse_time_string)
"""
