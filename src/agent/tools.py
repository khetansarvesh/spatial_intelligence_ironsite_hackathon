"""
Agent Tools
Functions that the LLM agent can call to query productivity data.
"""

from typing import List, Dict, Any, Optional
from datetime import timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from temporal.session_aggregator import SessionReport, ActivitySegment
from temporal.activity_classifier import ActivityState


class AgentTools:
    """
    Tools that the LLM agent can use to query productivity data.

    Each method returns data in a format that can be easily converted
    to natural language by the LLM.
    """

    def __init__(self, report: SessionReport):
        """
        Initialize agent tools with a session report.

        Args:
            report: SessionReport containing all productivity data
        """
        self.report = report

    def get_activity_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get summary of worker activities over a time period.

        Args:
            start_time: Start time in seconds (None = session start)
            end_time: End time in seconds (None = session end)

        Returns:
            Dictionary with activity summary
        """
        # Filter segments by time range
        segments = self.report.segments

        if start_time is not None or end_time is not None:
            start = start_time if start_time is not None else self.report.start_time
            end = end_time if end_time is not None else self.report.end_time

            segments = [
                s for s in segments
                if s.start_time >= start and s.end_time <= end
            ]

        # Calculate totals
        total_time = sum(s.duration for s in segments)

        # Group by activity
        activity_times = {}
        for segment in segments:
            activity = segment.activity.value
            if activity not in activity_times:
                activity_times[activity] = 0.0
            activity_times[activity] += segment.duration

        # Sort by time
        sorted_activities = sorted(
            activity_times.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "total_time": total_time,
            "total_time_formatted": self._format_time(total_time),
            "activities": [
                {
                    "name": activity,
                    "time": time,
                    "time_formatted": self._format_time(time),
                    "percentage": (time / total_time * 100) if total_time > 0 else 0,
                }
                for activity, time in sorted_activities
            ],
            "segment_count": len(segments),
        }

    def get_tool_usage(self, time_period: Optional[str] = None) -> Dict[str, Any]:
        """
        Get breakdown of which tools were used and for how long.

        Args:
            time_period: Time period to analyze (e.g., "first_half", "last_hour")
                        None = entire session

        Returns:
            Dictionary with tool usage statistics
        """
        tool_usage = self.report.tool_usage

        if not tool_usage:
            return {
                "message": "No tools were used during this session",
                "tools": []
            }

        # Sort by total time
        sorted_tools = sorted(
            tool_usage.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )

        total_tool_time = sum(t.total_time for _, t in sorted_tools)

        return {
            "total_tool_time": total_tool_time,
            "total_tool_time_formatted": self._format_time(total_tool_time),
            "tool_count": len(sorted_tools),
            "most_used_tool": self.report.most_used_tool,
            "tool_switches": self.report.tool_switches,
            "tools": [
                {
                    "name": name,
                    "total_time": usage.total_time,
                    "total_time_formatted": self._format_time(usage.total_time),
                    "usage_count": usage.usage_count,
                    "average_duration": usage.average_duration,
                    "average_duration_formatted": self._format_time(usage.average_duration),
                    "percentage": (usage.total_time / total_tool_time * 100) if total_tool_time > 0 else 0,
                    "activities": usage.activities,
                }
                for name, usage in sorted_tools
            ]
        }

    def find_idle_periods(self, min_duration_seconds: int = 10) -> Dict[str, Any]:
        """
        Find periods where worker was idle/unproductive.

        Args:
            min_duration_seconds: Minimum duration to report (seconds)

        Returns:
            Dictionary with idle period information
        """
        idle_segments = [
            s for s in self.report.segments
            if s.activity == ActivityState.IDLE and s.duration >= min_duration_seconds
        ]

        total_idle_time = sum(s.duration for s in idle_segments)

        return {
            "total_idle_time": total_idle_time,
            "total_idle_time_formatted": self._format_time(total_idle_time),
            "idle_percentage": self.report.idle_percentage,
            "idle_period_count": len(idle_segments),
            "idle_periods": [
                {
                    "start_time": seg.start_time,
                    "start_time_formatted": self._format_time(seg.start_time),
                    "end_time": seg.end_time,
                    "end_time_formatted": self._format_time(seg.end_time),
                    "duration": seg.duration,
                    "duration_formatted": self._format_time(seg.duration),
                }
                for seg in idle_segments
            ],
            "longest_idle_period": max(idle_segments, key=lambda s: s.duration).duration if idle_segments else 0,
        }

    def get_productivity_score(self) -> Dict[str, Any]:
        """
        Calculate overall productivity score and breakdown.

        Returns:
            Dictionary with productivity metrics
        """
        return {
            "overall_score": self.report.productivity_score,
            "overall_score_percentage": self.report.productivity_score * 100,
            "productive_time": self.report.productive_time,
            "productive_time_formatted": self._format_time(self.report.productive_time),
            "session_duration": self.report.session_duration,
            "session_duration_formatted": self._format_time(self.report.session_duration),
            "productive_percentage": (self.report.productive_time / self.report.session_duration * 100) if self.report.session_duration > 0 else 0,
            "idle_time": self.report.idle_time,
            "idle_time_formatted": self._format_time(self.report.idle_time),
            "idle_percentage": self.report.idle_percentage,
            "rating": self._get_productivity_rating(self.report.productivity_score),
        }

    def compare_periods(
        self,
        period1: str,
        period2: str,
    ) -> Dict[str, Any]:
        """
        Compare productivity between two time periods.

        Args:
            period1: First period (e.g., "0-300" for first 5 minutes)
            period2: Second period (e.g., "300-600" for second 5 minutes)

        Returns:
            Dictionary comparing the two periods
        """
        # Parse period strings (format: "start-end")
        try:
            start1, end1 = map(float, period1.split('-'))
            start2, end2 = map(float, period2.split('-'))
        except:
            return {"error": "Invalid period format. Use 'start-end' (e.g., '0-300')"}

        # Get summaries for each period
        summary1 = self.get_activity_summary(start1, end1)
        summary2 = self.get_activity_summary(start2, end2)

        # Calculate productivity scores for each period
        segments1 = [s for s in self.report.segments if s.start_time >= start1 and s.end_time <= end1]
        segments2 = [s for s in self.report.segments if s.start_time >= start2 and s.end_time <= end2]

        score1 = self._calculate_period_productivity(segments1)
        score2 = self._calculate_period_productivity(segments2)

        return {
            "period1": {
                "range": f"{start1}-{end1}",
                "duration": end1 - start1,
                "productivity_score": score1,
                "summary": summary1,
            },
            "period2": {
                "range": f"{start2}-{end2}",
                "duration": end2 - start2,
                "productivity_score": score2,
                "summary": summary2,
            },
            "comparison": {
                "score_difference": score2 - score1,
                "better_period": "period2" if score2 > score1 else "period1",
                "improvement_percentage": ((score2 - score1) / score1 * 100) if score1 > 0 else 0,
            }
        }

    def get_insights_and_recommendations(self) -> Dict[str, Any]:
        """
        Get insights and recommendations for improving productivity.

        Returns:
            Dictionary with insights and recommendations
        """
        return {
            "insights": self.report.insights,
            "recommendations": self.report.recommendations,
            "key_metrics": {
                "productivity_score": self.report.productivity_score,
                "idle_percentage": self.report.idle_percentage,
                "most_used_tool": self.report.most_used_tool,
                "tool_switches": self.report.tool_switches,
            }
        }

    def get_time_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed time breakdown by activity category.

        Returns:
            Dictionary with time breakdown
        """
        return {
            "session_duration": self.report.session_duration,
            "session_duration_formatted": self._format_time(self.report.session_duration),
            "activities": {
                name: {
                    "time": breakdown.total_time,
                    "time_formatted": self._format_time(breakdown.total_time),
                    "percentage": breakdown.percentage,
                    "productivity_score": breakdown.productivity_score,
                    "segment_count": breakdown.segment_count,
                }
                for name, breakdown in self.report.activity_breakdown.items()
            }
        }

    # Helper methods

    def _format_time(self, seconds: float) -> str:
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

    def _get_productivity_rating(self, score: float) -> str:
        """Get productivity rating from score."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.5:
            return "Below Average"
        else:
            return "Poor"

    def _calculate_period_productivity(self, segments: List[ActivitySegment]) -> float:
        """Calculate productivity score for a period."""
        if not segments:
            return 0.0

        total_time = sum(s.duration for s in segments)
        if total_time == 0:
            return 0.0

        weighted_sum = sum(s.productivity_score * s.duration for s in segments)
        return weighted_sum / total_time


# Define tool schemas for LLM function calling
AGENT_TOOL_SCHEMAS = [
    {
        "name": "get_activity_summary",
        "description": "Get summary of worker activities over a time period. Shows what activities were performed and for how long.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "number",
                    "description": "Start time in seconds (optional, defaults to session start)"
                },
                "end_time": {
                    "type": "number",
                    "description": "End time in seconds (optional, defaults to session end)"
                }
            }
        }
    },
    {
        "name": "get_tool_usage",
        "description": "Get breakdown of which tools were used and for how long. Shows most used tools, usage counts, and time spent.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_period": {
                    "type": "string",
                    "description": "Time period to analyze (optional)"
                }
            }
        }
    },
    {
        "name": "find_idle_periods",
        "description": "Find periods where worker was idle or unproductive. Useful for identifying wasted time.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_duration_seconds": {
                    "type": "integer",
                    "description": "Minimum duration to report in seconds (default: 10)"
                }
            }
        }
    },
    {
        "name": "get_productivity_score",
        "description": "Get overall productivity score and detailed metrics. Shows productive time, idle time, and ratings.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "compare_periods",
        "description": "Compare productivity between two time periods. Format: 'start-end' in seconds (e.g., '0-300' for first 5 minutes).",
        "parameters": {
            "type": "object",
            "properties": {
                "period1": {
                    "type": "string",
                    "description": "First period in format 'start-end' (e.g., '0-300')"
                },
                "period2": {
                    "type": "string",
                    "description": "Second period in format 'start-end' (e.g., '300-600')"
                }
            },
            "required": ["period1", "period2"]
        }
    },
    {
        "name": "get_insights_and_recommendations",
        "description": "Get AI-generated insights and recommendations for improving productivity.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_time_breakdown",
        "description": "Get detailed time breakdown by activity category with percentages and productivity scores.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]
