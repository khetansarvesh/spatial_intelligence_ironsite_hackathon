"""
Report Summary Module.

Uses LLM reasoning to convert raw JSON productivity report
into a beautifully formatted markdown summary for users.
"""

import json
from typing import Optional

from anthropic import Anthropic


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SUMMARY_SYSTEM_PROMPT = """You are a construction productivity report summarizer.

Convert the raw JSON productivity data into a clear, visually appealing markdown summary.

## YOUR TASK
1. Extract key metrics from the JSON data
2. Highlight important findings with appropriate formatting
3. Make it easy to scan and understand at a glance
4. Use emoji sparingly but effectively to draw attention

## FORMATTING GUIDELINES

### Header
- Start with a main title and a brief one-line summary
- Include session duration and overall productivity score prominently

### Key Metrics Section
- Use a clean table or bold metrics for quick scanning
- Include: duration, frames analyzed, productivity %, active vs idle time

### Activity Breakdown
- Show what the worker was doing and for how long
- Highlight the dominant activity
- Use percentages and time values

### Insights Section
- Convert the insights array into bullet points
- Group related insights together
- Highlight the most important ones (peak productivity, dominant activity)

### Recommendations
- If there are recommendations, list them clearly
- Frame them constructively

### Scene & Task Context
- Briefly mention the work context (masonry, electrical, etc.)
- Mention the primary task being performed

## VISUAL ELEMENTS TO USE
- **Bold** for key numbers and metrics
- `code` for technical values
- Tables for structured data
- Bullet points for lists
- Horizontal rules (---) to separate sections
- Emojis: Use sparingly - one per section header max
  - Productivity: use a gauge or chart emoji
  - Time: use clock emoji
  - Insights: use lightbulb emoji
  - Recommendations: use target emoji

## OUTPUT FORMAT
Return ONLY the markdown content, no code blocks wrapping it.

## EXAMPLE OUTPUT STYLE

# Worker Productivity Report

**Session Summary:** 13.3 seconds of masonry work analyzed with **95.6%** productivity.

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Duration | 13.3 seconds |
| Frames Analyzed | 400 |
| Productivity Score | 95.6% |
| Active Time | 12.7 seconds |

---

## Activity Breakdown

The worker spent most of their time on **precision work**:

- **PRECISION_WORK**: 95.5% (12.7s) - High focus task
- **SEARCHING**: 0.5% (0.07s) - Looking for materials/tools

---

## Key Insights

- High productivity session maintained throughout
- Peak performance: 0-1.8 seconds (100% score)
- Primary task: Block alignment and placement
- Work context: Masonry

---

## Recommendations

- Reduce interruptions to improve flow

---

*Report generated from 400 frames of video analysis*
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

DATA_PATH = "outputs/final_report.json"


def _load_report() -> dict:
    """Load the final report from JSON file."""
    with open(DATA_PATH, 'r') as f:
        return json.load(f)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def summarize_report(
    report: Optional[dict] = None,
    verbose: bool = False
) -> str:
    """
    Generate a markdown summary of the productivity report using LLM.

    Args:
        report: Report dictionary (if None, loads from default path)
        verbose: If True, print LLM interaction details

    Returns:
        Markdown formatted summary string

    Example:
        >>> summary = summarize_report(verbose=True)
        >>> print(summary)
    """
    # Load report if not provided
    if report is None:
        report = _load_report()

    # Build user message with the JSON data
    user_message = f"""Here is the raw productivity report JSON:

```json
{json.dumps(report, indent=2)}
```

Generate a beautiful markdown summary that highlights the key findings."""

    # Call Anthropic API
    client = Anthropic()

    if verbose:
        print("=" * 60)
        print("CALLING LLM FOR REPORT SUMMARIZATION")
        print("=" * 60)
        print(f"Report keys: {list(report.keys())}")
        print(f"Productivity score: {report.get('productivity_score', 'N/A')}")
        print(f"Session duration: {report.get('session_duration', 'N/A')}s")
        print("-" * 60)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    markdown_output = response.content[0].text

    if verbose:
        print("LLM OUTPUT LENGTH:", len(markdown_output), "characters")
        print("=" * 60)

    return markdown_output


def save_summary(
    output_path: str = "outputs/productivity_summary.md",
    verbose: bool = False
) -> str:
    """
    Generate and save the markdown summary to a file.

    Args:
        output_path: Path to save the markdown file
        verbose: If True, print details

    Returns:
        The generated markdown content
    """
    summary = summarize_report(verbose=verbose)

    with open(output_path, 'w') as f:
        f.write(summary)

    if verbose:
        print(f"Summary saved to: {output_path}")

    return summary
