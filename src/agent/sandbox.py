"""
Safe Code Execution Sandbox for CodeAct Agent.

Provides controlled exec() execution with restricted builtins.
Prevents dangerous operations like imports, file access, and network calls.
"""

from typing import Dict, Any, List, Optional
import copy


class SafeCodeExecutor:
    """
    Executes generated Python code in a controlled sandbox.

    Restrictions:
    - Only allowed functions can be called
    - No imports allowed
    - No file/network operations
    - Limited builtins
    """

    # Safe builtins to allow in the sandbox
    SAFE_BUILTINS = {
        # Type conversions
        'abs': abs,
        'bool': bool,
        'dict': dict,
        'float': float,
        'int': int,
        'list': list,
        'set': set,
        'str': str,
        'tuple': tuple,
        'frozenset': frozenset,

        # Iteration/sequences
        'all': all,
        'any': any,
        'enumerate': enumerate,
        'filter': filter,
        'len': len,
        'map': map,
        'max': max,
        'min': min,
        'range': range,
        'reversed': reversed,
        'sorted': sorted,
        'sum': sum,
        'zip': zip,

        # Math
        'round': round,
        'pow': pow,
        'divmod': divmod,

        # Type checking
        'isinstance': isinstance,
        'type': type,

        # Constants
        'True': True,
        'False': False,
        'None': None,

        # String/repr
        'repr': repr,
        'format': format,

        # Exceptions (for error handling in generated code)
        'Exception': Exception,
        'ValueError': ValueError,
        'KeyError': KeyError,
        'TypeError': TypeError,
        'IndexError': IndexError,
    }

    # Patterns that are forbidden in code
    FORBIDDEN_PATTERNS = [
        'import ',
        '__import__',
        'open(',
        'exec(',
        'eval(',
        'compile(',
        'globals()',
        'locals()',
        '__class__',
        '__bases__',
        '__subclasses__',
        '__mro__',
        '__code__',
        '__globals__',
        '__builtins__',
        'os.',
        'sys.',
        'subprocess',
        'shutil',
        'pathlib',
        'socket',
        'requests',
        'urllib',
        'pickle',
        'marshal',
        'shelve',
        'input(',
        'breakpoint(',
        'help(',
        'dir(',
        'vars(',
        'getattr(',
        'setattr(',
        'delattr(',
        'hasattr(',
    ]

    def __init__(
        self,
        allowed_functions: Dict[str, callable],
        data_context: Dict[str, Any],
        max_output_size: int = 100000,
        max_iterations: int = 1000000,
    ):
        """
        Initialize sandbox.

        Args:
            allowed_functions: Dictionary of allowed function names to functions
            data_context: Data variables available to the code
            max_output_size: Maximum size of result string (truncate if larger)
            max_iterations: Maximum iterations allowed (not enforced, for documentation)
        """
        self.allowed_functions = allowed_functions
        self.data_context = data_context
        self.max_output_size = max_output_size
        self.max_iterations = max_iterations

    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate code without executing it.

        Args:
            code: Python code to validate

        Returns:
            Dictionary with 'valid' and optionally 'errors'
        """
        errors = []

        # Check for forbidden patterns
        code_lower = code.lower()
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern.lower() in code_lower:
                errors.append(f"Forbidden pattern: {pattern}")

        if errors:
            return {'valid': False, 'errors': errors}

        # Try to compile (syntax check)
        try:
            compile(code, '<sandbox>', 'exec')
        except SyntaxError as e:
            return {'valid': False, 'errors': [f"Syntax error: {e}"]}

        return {'valid': True}

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute code in sandbox.

        Args:
            code: Python code to execute

        Returns:
            Dictionary with 'success', 'answer', and optionally 'error'
        """
        # Validate first
        validation = self.validate_code(code)
        if not validation['valid']:
            return {
                'success': False,
                'error': f"Code validation failed: {'; '.join(validation['errors'])}",
                'code': code,
            }

        # Build restricted globals
        restricted_globals = {
            '__builtins__': self.SAFE_BUILTINS,
        }

        # Add allowed functions
        restricted_globals.update(self.allowed_functions)

        # Add data context (make a deep copy to prevent modification of original)
        try:
            data_copy = copy.deepcopy(self.data_context)
        except Exception:
            # If deep copy fails, use shallow copy
            data_copy = self.data_context.copy()

        restricted_globals.update(data_copy)

        # Create locals dict to capture 'answer' variable
        restricted_locals = {}

        try:
            # Execute the code
            exec(code, restricted_globals, restricted_locals)

            # Get the answer
            if 'answer' not in restricted_locals:
                return {
                    'success': False,
                    'error': "Code did not assign a value to 'answer' variable. "
                             "Make sure your code ends with: answer = <result>",
                    'code': code,
                }

            answer = restricted_locals['answer']

            # Convert answer to string and truncate if too large
            try:
                answer_str = str(answer)
            except Exception:
                answer_str = repr(answer)

            if len(answer_str) > self.max_output_size:
                answer_str = answer_str[:self.max_output_size] + "... [truncated]"

            return {
                'success': True,
                'answer': answer,
                'answer_str': answer_str,
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}",
                'code': code,
            }

    def execute_with_retry(
        self,
        code: str,
        max_retries: int = 0,
        error_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute code with optional retry on failure.

        Args:
            code: Python code to execute
            max_retries: Maximum number of retries (0 = no retry)
            error_callback: Optional callback(error, attempt) -> new_code

        Returns:
            Dictionary with execution result
        """
        current_code = code
        last_result = None

        for attempt in range(max_retries + 1):
            result = self.execute(current_code)

            if result['success']:
                return result

            last_result = result

            # If we have retries left and a callback, try to fix the code
            if attempt < max_retries and error_callback:
                try:
                    new_code = error_callback(result['error'], attempt)
                    if new_code and new_code != current_code:
                        current_code = new_code
                        continue
                except Exception:
                    pass

            break

        return last_result


class CodeFormatter:
    """
    Utilities for formatting and cleaning generated code.
    """

    @staticmethod
    def extract_code_from_response(response: str) -> str:
        """
        Extract Python code from an LLM response that may include markdown.

        Args:
            response: Raw response that may contain ```python blocks

        Returns:
            Clean Python code
        """
        # Check for markdown code blocks
        if '```python' in response:
            # Extract code between ```python and ```
            start = response.find('```python') + len('```python')
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()

        if '```' in response:
            # Extract code between ``` and ```
            start = response.find('```') + len('```')
            end = response.find('```', start)
            if end > start:
                return response[start:end].strip()

        # Return as-is if no code blocks found
        return response.strip()

    @staticmethod
    def ensure_answer_assignment(code: str) -> str:
        """
        Ensure code ends with an answer assignment if it doesn't have one.

        Args:
            code: Python code

        Returns:
            Code with answer assignment
        """
        if 'answer' not in code:
            # Try to find the last expression and assign it to answer
            lines = code.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                # If last line is an expression (not an assignment or control flow)
                if (not last_line.startswith(('if ', 'for ', 'while ', 'def ', 'class ', '#'))
                    and '=' not in last_line
                    and last_line):
                    lines[-1] = f"answer = {last_line}"
                    return '\n'.join(lines)

        return code

    @staticmethod
    def add_default_answer(code: str, default: str = "None") -> str:
        """
        Add a default answer assignment at the end if not present.

        Args:
            code: Python code
            default: Default value for answer

        Returns:
            Code with guaranteed answer assignment
        """
        if 'answer =' not in code and 'answer=' not in code:
            return code + f"\n\nif 'answer' not in dir():\n    answer = {default}"
        return code
