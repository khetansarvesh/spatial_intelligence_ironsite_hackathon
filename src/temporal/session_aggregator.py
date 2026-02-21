"""
Session Aggregator Module (P3)
Aggregates activity segments into comprehensive session reports with metrics.

Generates insights like:
- Overall productivity score
- Time breakdown by activity
- Tool usage statistics
- Idle periods
- Recommendations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import timedelta
import numpy as np

from .activity_classifier import ActivitySegment, ActivityState


@dataclass
class ToolUsage:
    """Statistics for a specific tool."""
    tool_name: str
    total_time: float  # seconds
    usage_count: int  # number of segments
    activities: List[str]  # which activities used this tool

    @property
    def average_duration(self) -> float:
        """Average time per usage."""
        return self.total_time / self.usage_count if self.usage_count > 0 else 0.0


@dataclass
class ActivityBreakdown:
    """Time breakdown for a specific activity."""
    activity: ActivityState
    total_time: float  # seconds
    percentage: float  # 0-100
    segment_count: int
    average_duration: float  # seconds
    productivity_score: float  # 0-1


@dataclass
class SessionReport:
    """Comprehensive productivity report for a work session."""

    # Session metadata
    session_duration: float  # total seconds
    start_time: float
    end_time: float
    total_frames: int

    # Overall metrics
    productivity_score: float  # weighted average 0-1
    productive_time: float  # seconds with productivity > 0.5
    idle_time: float  # seconds
    idle_percentage: float  # 0-100

    # Activity breakdown
    activity_breakdown: Dict[str, ActivityBreakdown]

    # Tool usage
    tool_usage: Dict[str, ToolUsage]
    most_used_tool: Optional[str]
    tool_switches: int  # number of times worker changed tools

    # Segments
    segments: List[ActivitySegment]

    # Insights
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        td = timedelta(seconds=seconds)
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def get_summary(self) -> str:
        """Get text summary of the session."""
        lines = []
        lines.append("=" * 60)
        lines.append("PRODUCTIVITY SESSION REPORT")
        lines.append("=" * 60)
        lines.append(f"\nSession Duration: {self.format_time(self.session_duration)}")
        lines.append(f"Overall Productivity Score: {self.productivity_score:.1%}")
        lines.append(f"Productive Time: {self.format_time(self.productive_time)} ({self.productive_time/self.session_duration:.1%})")
        lines.append(f"Idle Time: {self.format_time(self.idle_time)} ({self.idle_percentage:.1f}%)")

        lines.append("\n" + "-" * 60)
        lines.append("ACTIVITY BREAKDOWN")
        lines.append("-" * 60)

        # Sort by time
        sorted_activities = sorted(
            self.activity_breakdown.values(),
            key=lambda x: x.total_time,
            reverse=True
        )

        for activity in sorted_activities:
            lines.append(f"\n{activity.activity.value}:")
            lines.append(f"  Time: {self.format_time(activity.total_time)} ({activity.percentage:.1f}%)")
            lines.append(f"  Segments: {activity.segment_count}")
            lines.append(f"  Avg Duration: {self.format_time(activity.average_duration)}")
            lines.append(f"  Productivity: {activity.productivity_score:.1%}")

        if self.tool_usage:
            lines.append("\n" + "-" * 60)
            lines.append("TOOL USAGE")
            lines.append("-" * 60)

            sorted_tools = sorted(
                self.tool_usage.values(),
                key=lambda x: x.total_time,
                reverse=True
            )

            for tool in sorted_tools:
                lines.append(f"\n{tool.tool_name}:")
                lines.append(f"  Total Time: {self.format_time(tool.total_time)}")
                lines.append(f"  Uses: {tool.usage_count}")
                lines.append(f"  Avg Duration: {self.format_time(tool.average_duration)}")

        if self.insights:
            lines.append("\n" + "-" * 60)
            lines.append("INSIGHTS")
            lines.append("-" * 60)
            for insight in self.insights:
                lines.append(f"• {insight}")

        if self.recommendations:
            lines.append("\n" + "-" * 60)
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 60)
            for rec in self.recommendations:
                lines.append(f"→ {rec}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


class SessionAggregator:
    """
    Aggregates activity segments into comprehensive session reports.

    Calculates metrics, generates insights, and provides recommendations.
    """

    # Thresholds for insights
    HIGH_IDLE_THRESHOLD = 0.25  # 25% idle is concerning
    LOW_PRODUCTIVITY_THRESHOLD = 0.6  # Below 60% needs improvement
    HIGH_TOOL_SWITCHES = 15  # Too many tool changes
    SHORT_SEGMENT_THRESHOLD = 5.0  # seconds - segments this short indicate interruptions

    def __init__(self):
        """Initialize session aggregator."""
        pass

    def aggregate(
        self,
        segments: List[ActivitySegment],
        session_start: Optional[float] = None,
        session_end: Optional[float] = None,
    ) -> SessionReport:
        """
        Aggregate activity segments into a session report.

        Args:
            segments: List of ActivitySegment objects
            session_start: Optional session start time (uses first segment if None)
            session_end: Optional session end time (uses last segment if None)

        Returns:
            SessionReport with all metrics
        """
        if not segments:
            # Return empty report
            return self._create_empty_report()

        # Determine session bounds
        start_time = session_start if session_start is not None else segments[0].start_time
        end_time = session_end if session_end is not None else segments[-1].end_time
        session_duration = end_time - start_time

        # Calculate activity breakdown
        activity_breakdown = self._calculate_activity_breakdown(segments, session_duration)

        # Calculate tool usage
        tool_usage, most_used_tool, tool_switches = self._calculate_tool_usage(segments)

        # Calculate overall metrics
        productivity_score = self._calculate_productivity_score(segments, session_duration)
        productive_time = self._calculate_productive_time(segments)
        idle_time = activity_breakdown.get("IDLE", ActivityBreakdown(
            ActivityState.IDLE, 0.0, 0.0, 0, 0.0, 0.0
        )).total_time
        idle_percentage = (idle_time / session_duration * 100) if session_duration > 0 else 0.0

        # Count total frames
        total_frames = sum(seg.frame_count for seg in segments)

        # Generate insights
        insights = self._generate_insights(
            segments=segments,
            productivity_score=productivity_score,
            idle_percentage=idle_percentage,
            tool_switches=tool_switches,
            activity_breakdown=activity_breakdown,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            insights=insights,
            productivity_score=productivity_score,
            idle_percentage=idle_percentage,
        )

        # Create report
        report = SessionReport(
            session_duration=session_duration,
            start_time=start_time,
            end_time=end_time,
            total_frames=total_frames,
            productivity_score=productivity_score,
            productive_time=productive_time,
            idle_time=idle_time,
            idle_percentage=idle_percentage,
            activity_breakdown=activity_breakdown,
            tool_usage=tool_usage,
            most_used_tool=most_used_tool,
            tool_switches=tool_switches,
            segments=segments,
            insights=insights,
            recommendations=recommendations,
        )

        return report

    def _calculate_activity_breakdown(
        self,
        segments: List[ActivitySegment],
        session_duration: float,
    ) -> Dict[str, ActivityBreakdown]:
        """Calculate time breakdown by activity."""
        activity_times = {}
        activity_counts = {}

        # Aggregate by activity
        for segment in segments:
            activity = segment.activity.value
            duration = segment.duration

            if activity not in activity_times:
                activity_times[activity] = 0.0
                activity_counts[activity] = 0

            activity_times[activity] += duration
            activity_counts[activity] += 1

        # Create breakdown
        breakdown = {}
        for activity, total_time in activity_times.items():
            percentage = (total_time / session_duration * 100) if session_duration > 0 else 0.0
            count = activity_counts[activity]
            avg_duration = total_time / count if count > 0 else 0.0

            # Get productivity score for this activity
            from .activity_classifier import ActivityClassifier
            productivity_score = ActivityClassifier.STATES[activity]["productivity"]

            breakdown[activity] = ActivityBreakdown(
                activity=ActivityState[activity],
                total_time=total_time,
                percentage=percentage,
                segment_count=count,
                average_duration=avg_duration,
                productivity_score=productivity_score,
            )

        return breakdown

    def _calculate_tool_usage(
        self,
        segments: List[ActivitySegment],
    ) -> Tuple[Dict[str, ToolUsage], Optional[str], int]:
        """
        Calculate tool usage statistics.

        Returns:
            (tool_usage_dict, most_used_tool, tool_switches)
        """
        tool_stats = {}

        # Track tool switches
        prev_tool = None
        tool_switches = 0

        for segment in segments:
            tool = segment.tool_used

            # Count tool switches
            if tool is not None and prev_tool is not None and tool != prev_tool:
                tool_switches += 1
            prev_tool = tool

            # Skip if no tool
            if tool is None:
                continue

            # Aggregate tool stats
            if tool not in tool_stats:
                tool_stats[tool] = {
                    "total_time": 0.0,
                    "count": 0,
                    "activities": set(),
                }

            tool_stats[tool]["total_time"] += segment.duration
            tool_stats[tool]["count"] += 1
            tool_stats[tool]["activities"].add(segment.activity.value)

        # Create ToolUsage objects
        tool_usage = {}
        for tool, stats in tool_stats.items():
            tool_usage[tool] = ToolUsage(
                tool_name=tool,
                total_time=stats["total_time"],
                usage_count=stats["count"],
                activities=list(stats["activities"]),
            )

        # Find most used tool
        most_used_tool = None
        if tool_usage:
            most_used_tool = max(tool_usage.keys(), key=lambda t: tool_usage[t].total_time)

        return tool_usage, most_used_tool, tool_switches

    def _calculate_productivity_score(
        self,
        segments: List[ActivitySegment],
        session_duration: float,
    ) -> float:
        """
        Calculate overall productivity score (weighted average).

        Returns:
            Score between 0.0 and 1.0
        """
        if session_duration == 0:
            return 0.0

        # Weighted sum of productivity scores
        weighted_sum = sum(
            seg.productivity_score * seg.duration
            for seg in segments
        )

        return weighted_sum / session_duration

    def _calculate_productive_time(self, segments: List[ActivitySegment]) -> float:
        """Calculate total time in productive activities (productivity > 0.5)."""
        productive_time = sum(
            seg.duration
            for seg in segments
            if seg.productivity_score > 0.5
        )
        return productive_time

    def _generate_insights(
        self,
        segments: List[ActivitySegment],
        productivity_score: float,
        idle_percentage: float,
        tool_switches: int,
        activity_breakdown: Dict[str, ActivityBreakdown],
    ) -> List[str]:
        """Generate insights from the session data."""
        insights = []

        # Overall productivity
        if productivity_score >= 0.8:
            insights.append(f"High productivity session ({productivity_score:.1%})")
        elif productivity_score >= 0.6:
            insights.append(f"Moderate productivity session ({productivity_score:.1%})")
        else:
            insights.append(f"Low productivity session ({productivity_score:.1%}) - opportunities for improvement")

        # Idle time
        if idle_percentage > self.HIGH_IDLE_THRESHOLD * 100:
            insights.append(f"High idle time detected ({idle_percentage:.1f}%) - investigate causes")

        # Tool switches
        if tool_switches > self.HIGH_TOOL_SWITCHES:
            insights.append(f"Frequent tool switching ({tool_switches} times) - may indicate inefficient workflow")

        # Short segments (interruptions)
        short_segments = [s for s in segments if s.duration < self.SHORT_SEGMENT_THRESHOLD]
        if len(short_segments) > len(segments) * 0.3:
            insights.append(f"Many short activity segments ({len(short_segments)}) - possible interruptions")

        # Activity-specific insights
        if "SEARCHING" in activity_breakdown:
            search_pct = activity_breakdown["SEARCHING"].percentage
            if search_pct > 20:
                insights.append(f"Significant time spent searching ({search_pct:.1f}%) - improve material organization")

        if "TRAVELING" in activity_breakdown:
            travel_pct = activity_breakdown["TRAVELING"].percentage
            if travel_pct > 15:
                insights.append(f"High travel time ({travel_pct:.1f}%) - optimize workspace layout")

        # Peak productivity period
        peak_period = self._find_peak_productivity_period(segments)
        if peak_period:
            start, end, score = peak_period
            insights.append(f"Peak productivity: {start:.1f}s - {end:.1f}s (score: {score:.1%})")

        return insights

    def _generate_recommendations(
        self,
        insights: List[str],
        productivity_score: float,
        idle_percentage: float,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Low productivity
        if productivity_score < self.LOW_PRODUCTIVITY_THRESHOLD:
            recommendations.append("Review workflow for bottlenecks and inefficiencies")

        # High idle time
        if idle_percentage > self.HIGH_IDLE_THRESHOLD * 100:
            recommendations.append("Minimize idle periods through better task scheduling")
            recommendations.append("Ensure all materials and tools are prepared before starting")

        # Check for searching/traveling issues
        if any("searching" in i.lower() for i in insights):
            recommendations.append("Organize tools in a consistent, easily accessible location")

        if any("traveling" in i.lower() for i in insights):
            recommendations.append("Arrange workspace to minimize movement between tasks")

        if any("tool switching" in i.lower() for i in insights):
            recommendations.append("Batch similar tasks together to reduce tool changes")

        if any("interruptions" in i.lower() for i in insights):
            recommendations.append("Identify and eliminate sources of interruption")

        # If no issues, provide positive feedback
        if not recommendations:
            recommendations.append("Maintain current efficient workflow")

        return recommendations

    def _find_peak_productivity_period(
        self,
        segments: List[ActivitySegment],
        window_duration: float = 300.0,  # 5 minutes
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find the most productive continuous period.

        Returns:
            (start_time, end_time, avg_productivity) or None
        """
        if not segments:
            return None

        # Simple approach: find segment with highest productivity
        best_segment = max(segments, key=lambda s: s.productivity_score)

        return (
            best_segment.start_time,
            best_segment.end_time,
            best_segment.productivity_score,
        )

    def _create_empty_report(self) -> SessionReport:
        """Create an empty report for when there are no segments."""
        return SessionReport(
            session_duration=0.0,
            start_time=0.0,
            end_time=0.0,
            total_frames=0,
            productivity_score=0.0,
            productive_time=0.0,
            idle_time=0.0,
            idle_percentage=0.0,
            activity_breakdown={},
            tool_usage={},
            most_used_tool=None,
            tool_switches=0,
            segments=[],
            insights=["No activity data available"],
            recommendations=["Process video to generate productivity report"],
        )


# Quick test
if __name__ == "__main__":
    """
    Test session aggregator with synthetic data.
    """
    from .activity_classifier import ActivitySegment, ActivityState

    print("Session Aggregator Test")
    print("=" * 60)

    # Create synthetic segments
    segments = [
        ActivitySegment(0.0, 10.0, ActivityState.ACTIVE_TOOL_USE, "drill", 1.0, 0.9, 300),
        ActivitySegment(10.0, 15.0, ActivityState.IDLE, None, 0.0, 0.8, 150),
        ActivitySegment(15.0, 30.0, ActivityState.ACTIVE_TOOL_USE, "drill", 1.0, 0.95, 450),
        ActivitySegment(30.0, 35.0, ActivityState.TRAVELING, None, 0.2, 0.85, 150),
        ActivitySegment(35.0, 50.0, ActivityState.PRECISION_WORK, "level", 1.0, 0.9, 450),
        ActivitySegment(50.0, 55.0, ActivityState.SEARCHING, None, 0.3, 0.7, 150),
        ActivitySegment(55.0, 70.0, ActivityState.ACTIVE_TOOL_USE, "hammer", 1.0, 0.9, 450),
    ]

    aggregator = SessionAggregator()
    report = aggregator.aggregate(segments)

    print(report.get_summary())
