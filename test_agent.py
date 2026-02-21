"""
Test script for the LLM agent (tools only, no API key required).
Tests the agent tools functionality without calling the LLM.
"""

import json
from src.agent import AgentTools
from src.temporal import (
    SessionReport,
    ActivityBreakdown,
    ToolUsage,
    ActivityState,
    ActivitySegment,
)


def create_mock_report() -> SessionReport:
    """Create a mock session report for testing."""
    # Create sample segments
    segments = [
        ActivitySegment(0.0, 120.0, ActivityState.ACTIVE_TOOL_USE, "drill", 1.0, 0.95, 3600),
        ActivitySegment(120.0, 140.0, ActivityState.IDLE, None, 0.0, 0.8, 600),
        ActivitySegment(140.0, 280.0, ActivityState.ACTIVE_TOOL_USE, "hammer", 1.0, 0.9, 4200),
        ActivitySegment(280.0, 310.0, ActivityState.SEARCHING, None, 0.3, 0.75, 900),
        ActivitySegment(310.0, 450.0, ActivityState.PRECISION_WORK, "level", 1.0, 0.92, 4200),
        ActivitySegment(450.0, 480.0, ActivityState.TRAVELING, None, 0.2, 0.85, 900),
        ActivitySegment(480.0, 600.0, ActivityState.ACTIVE_TOOL_USE, "drill", 1.0, 0.88, 3600),
    ]

    # Create activity breakdown
    activity_breakdown = {
        "ACTIVE_TOOL_USE": ActivityBreakdown(
            ActivityState.ACTIVE_TOOL_USE,
            340.0,
            56.7,
            3,
            113.3,
            1.0,
        ),
        "PRECISION_WORK": ActivityBreakdown(
            ActivityState.PRECISION_WORK,
            140.0,
            23.3,
            1,
            140.0,
            1.0,
        ),
        "SEARCHING": ActivityBreakdown(
            ActivityState.SEARCHING,
            30.0,
            5.0,
            1,
            30.0,
            0.3,
        ),
        "TRAVELING": ActivityBreakdown(
            ActivityState.TRAVELING,
            30.0,
            5.0,
            1,
            30.0,
            0.2,
        ),
        "IDLE": ActivityBreakdown(
            ActivityState.IDLE,
            20.0,
            3.3,
            1,
            20.0,
            0.0,
        ),
    }

    # Create tool usage
    tool_usage = {
        "drill": ToolUsage("drill", 240.0, 2, ["ACTIVE_TOOL_USE"]),
        "hammer": ToolUsage("hammer", 140.0, 1, ["ACTIVE_TOOL_USE"]),
        "level": ToolUsage("level", 140.0, 1, ["PRECISION_WORK"]),
    }

    # Create report
    report = SessionReport(
        session_duration=600.0,
        start_time=0.0,
        end_time=600.0,
        total_frames=18000,
        productivity_score=0.82,
        productive_time=480.0,
        idle_time=20.0,
        idle_percentage=3.3,
        activity_breakdown=activity_breakdown,
        tool_usage=tool_usage,
        most_used_tool="drill",
        tool_switches=3,
        segments=segments,
        insights=[
            "High productivity session (82%)",
            "Low idle time (3.3%) - excellent",
            "Tool usage well distributed",
        ],
        recommendations=[
            "Maintain current efficient workflow",
            "Continue minimizing idle and search time",
        ],
    )

    return report


def test_agent_tools():
    """Test all agent tools."""
    print("=" * 60)
    print("SiteIQ Agent Tools Test")
    print("=" * 60)

    # Create mock report
    report = create_mock_report()
    tools = AgentTools(report)

    print("\n✓ Created mock session report")
    print(f"  Duration: {report.session_duration}s ({report.session_duration/60:.1f} minutes)")
    print(f"  Productivity: {report.productivity_score:.1%}")
    print(f"  Segments: {len(report.segments)}")

    # Test 1: get_activity_summary
    print("\n" + "-" * 60)
    print("Test 1: get_activity_summary()")
    print("-" * 60)
    result = tools.get_activity_summary()
    print(f"Total time: {result['total_time_formatted']}")
    print(f"Activities ({len(result['activities'])}):")
    for activity in result['activities'][:3]:
        print(f"  - {activity['name']}: {activity['time_formatted']} ({activity['percentage']:.1f}%)")

    # Test 2: get_tool_usage
    print("\n" + "-" * 60)
    print("Test 2: get_tool_usage()")
    print("-" * 60)
    result = tools.get_tool_usage()
    print(f"Total tool time: {result['total_tool_time_formatted']}")
    print(f"Most used tool: {result['most_used_tool']}")
    print(f"Tool switches: {result['tool_switches']}")
    print(f"Tools ({result['tool_count']}):")
    for tool in result['tools']:
        print(f"  - {tool['name']}: {tool['total_time_formatted']} ({tool['percentage']:.1f}%), {tool['usage_count']} uses")

    # Test 3: find_idle_periods
    print("\n" + "-" * 60)
    print("Test 3: find_idle_periods(min_duration_seconds=10)")
    print("-" * 60)
    result = tools.find_idle_periods(min_duration_seconds=10)
    print(f"Total idle time: {result['total_idle_time_formatted']} ({result['idle_percentage']:.1f}%)")
    print(f"Idle periods: {result['idle_period_count']}")
    for period in result['idle_periods']:
        print(f"  - {period['start_time_formatted']} to {period['end_time_formatted']} ({period['duration_formatted']})")

    # Test 4: get_productivity_score
    print("\n" + "-" * 60)
    print("Test 4: get_productivity_score()")
    print("-" * 60)
    result = tools.get_productivity_score()
    print(f"Overall score: {result['overall_score_percentage']:.1f}% ({result['rating']})")
    print(f"Productive time: {result['productive_time_formatted']} ({result['productive_percentage']:.1f}%)")
    print(f"Idle time: {result['idle_time_formatted']} ({result['idle_percentage']:.1f}%)")

    # Test 5: compare_periods
    print("\n" + "-" * 60)
    print("Test 5: compare_periods('0-300', '300-600')")
    print("-" * 60)
    result = tools.compare_periods("0-300", "300-600")
    print(f"Period 1 productivity: {result['period1']['productivity_score']:.1%}")
    print(f"Period 2 productivity: {result['period2']['productivity_score']:.1%}")
    print(f"Better period: {result['comparison']['better_period']}")
    print(f"Difference: {result['comparison']['score_difference']:.2%}")

    # Test 6: get_insights_and_recommendations
    print("\n" + "-" * 60)
    print("Test 6: get_insights_and_recommendations()")
    print("-" * 60)
    result = tools.get_insights_and_recommendations()
    print("Insights:")
    for insight in result['insights']:
        print(f"  • {insight}")
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  → {rec}")

    # Test 7: get_time_breakdown
    print("\n" + "-" * 60)
    print("Test 7: get_time_breakdown()")
    print("-" * 60)
    result = tools.get_time_breakdown()
    print(f"Session duration: {result['session_duration_formatted']}")
    print("\nActivity breakdown:")
    for name, activity in result['activities'].items():
        print(f"  {name}:")
        print(f"    Time: {activity['time_formatted']} ({activity['percentage']:.1f}%)")
        print(f"    Productivity: {activity['productivity_score']:.1%}")

    # Summary
    print("\n" + "=" * 60)
    print("✓ All agent tools tested successfully!")
    print("=" * 60)

    print("\nNext steps:")
    print("  1. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
    print("  2. Process a video: python main.py --video your_video.mp4")
    print("  3. Query the agent: python query_agent.py --report report.json --interactive")

    return True


if __name__ == "__main__":
    import sys
    success = test_agent_tools()
    sys.exit(0 if success else 1)
