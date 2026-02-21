"""
Query Agent CLI
Command-line interface for querying productivity reports using natural language.

Usage:
    python query_agent.py --report report.json "What tools were used most?"
    python query_agent.py --report report.json --interactive
    python query_agent.py --report report.json --provider anthropic "How can productivity improve?"
"""

import argparse
import json
import sys
from pathlib import Path

from src.agent import ProductivityAgent
from src.temporal import SessionReport, ActivityBreakdown, ToolUsage, ActivityState, ActivitySegment


def load_report_from_json(json_path: str) -> SessionReport:
    """Load SessionReport from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Reconstruct ActivityBreakdown objects
    activity_breakdown = {}
    for name, breakdown_data in data.get("activity_breakdown", {}).items():
        activity_breakdown[name] = ActivityBreakdown(
            activity=ActivityState[breakdown_data["activity"]],
            total_time=breakdown_data["total_time"],
            percentage=breakdown_data["percentage"],
            segment_count=breakdown_data["segment_count"],
            average_duration=breakdown_data["average_duration"],
            productivity_score=breakdown_data["productivity_score"],
        )

    # Reconstruct ToolUsage objects
    tool_usage = {}
    for name, usage_data in data.get("tool_usage", {}).items():
        tool_usage[name] = ToolUsage(
            tool_name=usage_data["tool_name"],
            total_time=usage_data["total_time"],
            usage_count=usage_data["usage_count"],
            activities=usage_data["activities"],
        )

    # Create SessionReport
    report = SessionReport(
        session_duration=data["session_duration"],
        start_time=data["start_time"],
        end_time=data["end_time"],
        total_frames=data["total_frames"],
        productivity_score=data["productivity_score"],
        productive_time=data["productive_time"],
        idle_time=data["idle_time"],
        idle_percentage=data["idle_percentage"],
        activity_breakdown=activity_breakdown,
        tool_usage=tool_usage,
        most_used_tool=data.get("most_used_tool"),
        tool_switches=data["tool_switches"],
        segments=[],  # Not stored in JSON
        insights=data.get("insights", []),
        recommendations=data.get("recommendations", []),
    )

    return report


def interactive_mode(agent: ProductivityAgent):
    """Run interactive query session."""
    print("\n" + "=" * 60)
    print("SiteIQ - Interactive Productivity Query")
    print("=" * 60)
    print("\nAsk questions about worker productivity.")
    print("Type 'quit' or 'exit' to end the session.\n")
    print("Example questions:")
    print("  - What tools were used the most?")
    print("  - How much time was spent idle?")
    print("  - What was the overall productivity score?")
    print("  - How can productivity be improved?")
    print("  - Compare the first 5 minutes to the last 5 minutes")
    print("\n" + "=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            # Special command to show summary
            if question.lower() in ['summary', 'report']:
                print("\n" + agent.get_quick_summary())
                continue

            # Query the agent
            print("\nSiteIQ: ", end="", flush=True)
            response = agent.chat(question)
            print(response + "\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Query productivity reports using natural language"
    )
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to JSON report file",
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Question to ask (optional, use --interactive for conversation)",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Start interactive query session",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic"],
        default="openai",
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (default: gpt-4o for OpenAI, claude-3-5-sonnet for Anthropic)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show text summary without using LLM",
    )

    args = parser.parse_args()

    # Load report
    print(f"Loading report from {args.report}...")
    try:
        report = load_report_from_json(args.report)
        print(f"✓ Loaded session report ({report.session_duration:.1f}s duration)")
    except Exception as e:
        print(f"Error loading report: {e}")
        sys.exit(1)

    # If summary only, print and exit
    if args.summary:
        print("\n" + report.get_summary())
        sys.exit(0)

    # Initialize agent
    print(f"Initializing {args.provider.upper()} agent...")
    try:
        agent = ProductivityAgent(
            report=report,
            provider=args.provider,
            model=args.model,
        )
        print("✓ Agent ready\n")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        print("\nMake sure you have set the appropriate API key:")
        print("  - For OpenAI: export OPENAI_API_KEY=your_key")
        print("  - For Anthropic: export ANTHROPIC_API_KEY=your_key")
        sys.exit(1)

    # Interactive or single query mode
    if args.interactive:
        interactive_mode(agent)
    elif args.query:
        print(f"Q: {args.query}\n")
        response = agent.chat(args.query)
        print(f"A: {response}\n")
    else:
        print("Error: Provide either a query or use --interactive mode")
        print("\nExamples:")
        print('  python query_agent.py --report report.json "What tools were used?"')
        print('  python query_agent.py --report report.json --interactive')
        sys.exit(1)


if __name__ == "__main__":
    main()
