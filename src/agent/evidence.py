import json
from typing import List, Dict, Any, Optional

from anthropic import Anthropic


# =============================================================================
# SYSTEM PROMPT - THE KEY PART
# =============================================================================

EVIDENCE_SYSTEM_PROMPT = """You are a video evidence selector for construction worker productivity analysis.

Given a QUERY and its ANSWER, select video timestamp(s) that would serve as visual proof.

## YOUR TASK
1. Analyze the query type and answer content
2. Decide: 0, 1, or 2 evidence timestamps needed
3. Extract specific timestamps from the answer
4. Return valid JSON

## DECISION RULES

### Return 0 evidence when:
- Query asks for "summary", "overview", "total", "all", "entire"
- Answer describes aggregate statistics without specific times
- Example: "Give me overall summary" → no specific clip needed

### Return 1 evidence when:
- Query asks about specific event: "peak", "when", "what time", "at frame"
- Query asks about specific activity: "tools used", "tasks performed"
- Answer mentions a specific time range → pick MIDDLE of that range
- Example: "Peak was 3-5 seconds" → evidence at 4.0 seconds

### Return 2 evidence when:
- Query explicitly compares TWO things: "vs", "compare", "difference between"
- Query mentions "first half AND second half", "before AND after"
- Example: "Compare first vs second half" → one timestamp from each

## TIMESTAMP EXTRACTION
- Look for time patterns in the ANSWER: "3:00-4:00", "at 5 seconds", "0s-6s"
- For short videos (<60s), times like "3:00" likely mean 3.0 seconds, not 3 minutes
- Pick MIDDLE of ranges: "3-5 seconds" → 4.0 seconds
- Stay within video bounds: 0 to video_duration seconds

## OUTPUT FORMAT
Return ONLY valid JSON, no other text:
{
  "reasoning": "Brief explanation of your decision",
  "evidence": [
    {"start_time": <float>, "description": "<5-10 words>"}
  ]
}

## EXAMPLES

INPUT:
Query: "Give me an overall summary"
Answer: "Worker was 99% productive across 13 seconds with 400 frames..."
Video duration: 13.3 seconds

OUTPUT:
{"reasoning": "Summary query about entire video - no specific clip needed", "evidence": []}

---

INPUT:
Query: "When was peak productivity?"
Answer: "Peak productivity occurred from 3-5 seconds with 100% active time"
Video duration: 13.3 seconds

OUTPUT:
{"reasoning": "Answer mentions 3-5 second range, selecting middle at 4.0s", "evidence": [{"start_time": 4.0, "description": "Peak productivity period"}]}

---

INPUT:
Query: "Compare productivity in first half vs second half"
Answer: "First half (0-6s) had 100% active time. Second half (6-13s) had 98% active time."
Video duration: 13.3 seconds

OUTPUT:
{"reasoning": "Comparison query needs evidence from both halves - 3s for first, 9.5s for second", "evidence": [{"start_time": 3.0, "description": "First half productivity sample"}, {"start_time": 9.5, "description": "Second half productivity sample"}]}

---

INPUT:
Query: "What tools were used?"
Answer: "109 unique tools detected including grout bag (16x), trowel (13x), pipe wrench (12x)..."
Video duration: 13.3 seconds

OUTPUT:
{"reasoning": "Tools query - show representative moment of tool usage at middle of video", "evidence": [{"start_time": 6.5, "description": "Worker using tools"}]}
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

DATA_PATH = "outputs/frames_information.json"
_video_duration: Optional[float] = None


def _get_video_duration() -> float:
    """Get video duration from cached data or load from file."""
    global _video_duration

    if _video_duration is None:
        with open(DATA_PATH, 'r') as f:
            data = json.load(f)
        frames = data.get('frames', [])
        if frames:
            _video_duration = max(f.get('timestamp_sec', 0) for f in frames)
        else:
            _video_duration = 0.0

    return _video_duration


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def get_evidence(
    query: str,
    answer: str,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Get video evidence timestamps using LLM reasoning.

    Args:
        query: Original user question
        answer: Agent's answer to the question
        verbose: If True, print raw LLM output

    Returns:
        List of 0, 1, or 2 evidence dicts:
        [{"start_time": float, "description": str}, ...]

    Examples:
        >>> get_evidence("Give me summary", "Worker was 99% active...", verbose=True)
        []

        >>> get_evidence("When was peak?", "Peak was 3-5 seconds", verbose=True)
        [{"start_time": 4.0, "description": "Peak productivity"}]

        >>> get_evidence("Compare halves", "First 0-6s, second 6-13s", verbose=True)
        [{"start_time": 3.0, ...}, {"start_time": 9.5, ...}]
    """
    # Get video duration
    video_duration = _get_video_duration()

    # Build user message
    user_message = f"""Query: {query}

Answer: {answer}

Video duration: {video_duration:.1f} seconds

Return the evidence JSON:"""

    # Call Anthropic API
    client = Anthropic()

    if verbose:
        print("=" * 60)
        print("CALLING LLM FOR EVIDENCE EXTRACTION")
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Answer: {answer[:100]}..." if len(answer) > 100 else f"Answer: {answer}")
        print(f"Video duration: {video_duration:.1f}s")
        print("-" * 60)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system=EVIDENCE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw_output = response.content[0].text

    if verbose:
        print("LLM RAW OUTPUT:")
        print(raw_output)
        print("=" * 60)

    # Parse JSON response (handle markdown code blocks)
    try:
        # Strip markdown code blocks if present
        clean_output = raw_output.strip()
        if clean_output.startswith("```"):
            # Remove opening ```json or ```
            lines = clean_output.split("\n")
            lines = lines[1:]  # Remove first line with ```
            # Remove closing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            clean_output = "\n".join(lines)

        result = json.loads(clean_output)
        evidence = result.get("evidence", [])

        if verbose:
            print(f"PARSED EVIDENCE: {evidence}")
            print("=" * 60)

        return evidence

    except json.JSONDecodeError as e:
        if verbose:
            print(f"JSON PARSE ERROR: {e}")
        return []
