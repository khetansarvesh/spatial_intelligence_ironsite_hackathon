import os
import json
from typing import Dict, Any, Optional
import dspy
from .tools import (
    get_tool_registry,
    get_tool_names_with_brief_descriptions,
    get_selected_tool_descriptions,
    parse_relevant_tools,
)
from .sandbox import SafeCodeExecutor, CodeFormatter
from .signatures import (
    FrameQuerySignature,
    CodeGenerationSignature,
    ResultFormatterSignature,
    CodeFixSignature,
    get_data_schema_description,
)
from .prompts import CODEACT_SYSTEM_PROMPT
import pickle

class CodeActAgent:
    """
    DSPy CodeAct agent for frame-level productivity analysis.

    Uses Anthropic's Claude model to generate Python code that queries
    frame-level HOI detection data.

    Example:
        >>> agent = CodeActAgent(hoi_results)
        >>> response = agent.query("How much idle time was there?")
        >>> print(response)
    """

    def __init__(
        self,
        hoi_results: Dict[str, Any],
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 3,
        verbose: bool = False,
    ):
        """
        Initialize CodeAct agent.

        Args:
            hoi_results: Frame-level HOI detection results dictionary
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            max_iterations: Max code generation/fix iterations
            verbose: If True, print debug information
        """
        self.hoi_results = hoi_results
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Get API key
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize DSPy with Anthropic
        self.lm = dspy.LM(
            model=f"anthropic/{model}",
            api_key=api_key,
            max_tokens=4096,
        )
        dspy.configure(lm=self.lm)

        # Initialize tools and sandbox
        self.tools = get_tool_registry()
        self.tool_names_brief = get_tool_names_with_brief_descriptions()  # For query analyzer
        self.data_schema = get_data_schema_description()

        self.sandbox = SafeCodeExecutor(
            allowed_functions=self.tools,
            data_context={"data": hoi_results},
        )

        # Initialize DSPy modules
        self.query_analyzer = dspy.Predict(FrameQuerySignature)
        self.code_generator = dspy.ChainOfThought(CodeGenerationSignature)
        self.result_formatter = dspy.Predict(ResultFormatterSignature)
        self.code_fixer = dspy.ChainOfThought(CodeFixSignature)

    def query(self, user_query: str) -> str:
        """
        Answer a user query about frame-level productivity data.

        Args:
            user_query: Natural language question

        Returns:
            Natural language answer
        """
        try:
            # Step 1: Analyze the query (uses brief tool descriptions)
            if self.verbose:
                print(f"[1] Analyzing query: {user_query}")

            analysis = self.query_analyzer(
                query=user_query,
                available_tools=self.tool_names_brief,  # Brief descriptions for selection
                data_schema=self.data_schema,
            )

            if self.verbose:
                print(f"    Query type: {analysis.query_type}")
                print(f"    Relevant tools: {analysis.relevant_tools}")

            # Step 1b: Parse relevant tools and get their full docstrings
            relevant_tool_names = parse_relevant_tools(analysis.relevant_tools)
            selected_tool_descriptions = get_selected_tool_descriptions(relevant_tool_names)

            if self.verbose:
                print(f"    Parsed tools: {relevant_tool_names}")

            # Step 2: Generate code (uses full docstrings for ONLY relevant tools)
            if self.verbose:
                print("[2] Generating code...")

            code_result = self.code_generator(
                query=user_query,
                query_type=analysis.query_type,
                relevant_tools=", ".join(relevant_tool_names),
                data_schema=self.data_schema,
                tool_descriptions=selected_tool_descriptions,  # Full docstrings for selected tools only
            )

            code = CodeFormatter.extract_code_from_response(code_result.code)

            if self.verbose:
                print(f"    Generated code:\n{code}")

            # Step 3: Execute code with retry
            execution_result = self._execute_with_retry(
                code=code,
                user_query=user_query,
                selected_tool_descriptions=selected_tool_descriptions,
            )

            if not execution_result['success']:
                return f"I couldn't compute the answer due to an error: {execution_result.get('error', 'Unknown error')}"

            # Step 4: Format result as natural language
            if self.verbose:
                print(f"[4] Formatting result...")

            formatted = self.result_formatter(
                query=user_query,
                result=str(execution_result['answer']),
            )

            return formatted.response

        except Exception as e:
            if self.verbose:
                print(f"Error in query processing: {e}")
            return f"An error occurred while processing your query: {str(e)}"

    def query_raw(self, user_query: str) -> Dict[str, Any]:
        """
        Answer a query and return raw structured data instead of formatted text.

        Args:
            user_query: Natural language question

        Returns:
            Dictionary with query results and metadata
        """
        try:
            # Step 1: Analyze the query (uses brief tool descriptions)
            analysis = self.query_analyzer(
                query=user_query,
                available_tools=self.tool_names_brief,
                data_schema=self.data_schema,
            )

            # Step 1b: Parse relevant tools and get their full docstrings
            relevant_tool_names = parse_relevant_tools(analysis.relevant_tools)
            selected_tool_descriptions = get_selected_tool_descriptions(relevant_tool_names)

            # Step 2: Generate code (uses full docstrings for ONLY relevant tools)
            code_result = self.code_generator(
                query=user_query,
                query_type=analysis.query_type,
                relevant_tools=", ".join(relevant_tool_names),
                data_schema=self.data_schema,
                tool_descriptions=selected_tool_descriptions,
            )

            code = CodeFormatter.extract_code_from_response(code_result.code)

            # Step 3: Execute code
            execution_result = self._execute_with_retry(
                code=code,
                user_query=user_query,
                selected_tool_descriptions=selected_tool_descriptions,  # Pass for error recovery
            )

            return {
                'success': execution_result['success'],
                'query': user_query,
                'query_type': analysis.query_type,
                'relevant_tools': relevant_tool_names,
                'code': code,
                'result': execution_result.get('answer'),
                'error': execution_result.get('error'),
            }

        except Exception as e:
            return {
                'success': False,
                'query': user_query,
                'error': str(e),
            }

    def _execute_with_retry(
        self,
        code: str,
        user_query: str,
        selected_tool_descriptions: str = "",
    ) -> Dict[str, Any]:
        """
        Execute code with retry on failure.

        Args:
            code: Python code to execute
            user_query: Original query (for context in error recovery)
            selected_tool_descriptions: Full docstrings for relevant tools (for error recovery)

        Returns:
            Execution result dictionary
        """
        current_code = code

        for attempt in range(self.max_iterations):
            if self.verbose:
                print(f"[3] Executing code (attempt {attempt + 1}/{self.max_iterations})...")

            result = self.sandbox.execute(current_code)

            if result['success']:
                if self.verbose:
                    print(f"    Success! Answer: {result['answer_str'][:200]}...")
                return result

            if self.verbose:
                print(f"    Error: {result['error']}")

            # Try to fix the code if we have attempts left
            if attempt < self.max_iterations - 1:
                if self.verbose:
                    print(f"    Attempting to fix code...")

                try:
                    fix_result = self.code_fixer(
                        query=user_query,
                        original_code=current_code,
                        error_message=result['error'],
                        available_tools=selected_tool_descriptions,  # Use selected tools only
                    )

                    fixed_code = CodeFormatter.extract_code_from_response(fix_result.fixed_code)

                    if fixed_code and fixed_code != current_code:
                        if self.verbose:
                            print(f"    Fixed code:\n{fixed_code}")
                        current_code = fixed_code
                        continue
                except Exception as e:
                    if self.verbose:
                        print(f"    Fix attempt failed: {e}")

            break

        return result

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded data.

        Returns:
            Dictionary with data summary
        """
        frames = self.hoi_results.get('frames', [])
        total_frames = self.hoi_results.get('total_frames', len(frames))

        # Count frames by status
        active_frames = 0
        idle_frames = 0
        unique_tasks = set()
        unique_tools = set()
        unique_scenes = set()

        for frame in frames:
            status = frame.get('wearer_productivity_status', '').upper()
            if status == 'ACTIVE':
                active_frames += 1
            else:
                idle_frames += 1

            task = frame.get('task_name')
            if task and not frame.get('task_unknown', False):
                unique_tasks.add(task)

            tool = frame.get('primary_tool')
            if tool:
                unique_tools.add(tool)

            scene = frame.get('scene')
            if scene:
                unique_scenes.add(scene)

        # Calculate duration from timestamps
        if frames:
            max_timestamp = max(f.get('timestamp_sec', 0) for f in frames)
            duration_seconds = max_timestamp
        else:
            duration_seconds = 0

        return {
            'total_frames': total_frames,
            'duration_seconds': duration_seconds,
            'duration_formatted': f"{int(duration_seconds // 60)}m {int(duration_seconds % 60)}s",
            'active_frames': active_frames,
            'idle_frames': idle_frames,
            'productivity_percentage': (active_frames / total_frames * 100) if total_frames > 0 else 0,
            'unique_tasks': list(unique_tasks),
            'unique_tools': list(unique_tools),
            'unique_scenes': list(unique_scenes),
        }


# =============================================================================
# SIMPLE FUNCTION INTERFACE FOR UI
# =============================================================================

DATA_PATH = "outputs/frames_information.json"
_agent: Optional[CodeActAgent] = None


def ask(query: str) -> str:
    """
    Ask a question and get an answer.

    Args:
        query: Natural language question

    Returns:
        Natural language answer
    """
    global _agent

    # Auto-load data on first call
    if _agent is None:
        with open(DATA_PATH, 'r') as f:
            data = json.load(f)
        _agent = CodeActAgent(data, verbose=False)

    return _agent.query(query)
