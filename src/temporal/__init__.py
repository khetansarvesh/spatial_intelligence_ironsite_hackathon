"""
Temporal Analysis Module
Analyzes activities and motion over time for productivity insights.
"""

from .motion_analyzer import MotionAnalyzer, MotionType, MotionResult
from .activity_classifier import ActivityClassifier, ActivityState, ActivitySegment
from .session_aggregator import SessionAggregator, SessionReport, ToolUsage, ActivityBreakdown

__all__ = [
    "MotionAnalyzer",
    "MotionType",
    "MotionResult",
    "ActivityClassifier",
    "ActivityState",
    "ActivitySegment",
    "SessionAggregator",
    "SessionReport",
    "ToolUsage",
    "ActivityBreakdown",
]
