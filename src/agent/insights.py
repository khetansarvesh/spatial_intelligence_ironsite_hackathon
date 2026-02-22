"""
Insight Engine
Advanced analytics for productivity insights including trend analysis and fatigue detection.
"""

from typing import List, Dict, Any
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from temporal.session_aggregator import SessionReport
from temporal.activity_classifier import ActivityState, ActivitySegment


class InsightEngine:
    """
    Advanced insight generation for productivity data.
    Provides trend analysis, peak performance detection, and fatigue detection.
    """

    def __init__(self, report: SessionReport):
        """
        Initialize insight engine with a session report.

        Args:
            report: SessionReport containing all productivity data
        """
        self.report = report
        self.segments = report.segments

    def _duration(self, seg: ActivitySegment) -> float:
        """Get duration of a segment."""
        return seg.end_time - seg.start_time

    def _total_time(self) -> float:
        """Get total time of all segments."""
        return sum(self._duration(s) for s in self.segments)

    def analyze_idle_time(self) -> Dict[str, Any]:
        """
        Analyze idle time and flag if excessive.

        Returns:
            Dictionary with idle analysis including flag if > 20%
        """
        total_time = self._total_time()
        idle_time = sum(
            self._duration(s)
            for s in self.segments
            if s.activity == ActivityState.IDLE
        )

        idle_percentage = (idle_time / total_time * 100) if total_time > 0 else 0

        return {
            "idle_seconds": idle_time,
            "idle_percentage": round(idle_percentage, 2),
            "excessive_idle": idle_percentage > 20,
            "recommendation": "Consider reducing idle time" if idle_percentage > 20 else None
        }

    def productivity_trend(self) -> Dict[str, Any]:
        """
        Analyze productivity trend over the session using linear regression.

        Returns:
            Dictionary with trend direction and slope
        """
        if not self.segments:
            return {"trend": "unknown", "slope": 0.0}

        scores = [seg.productivity_score for seg in self.segments]

        if len(scores) < 2:
            return {"trend": "insufficient_data", "slope": 0.0}

        slope = np.polyfit(range(len(scores)), scores, 1)[0]

        if slope > 0.02:
            trend = "improving"
            description = "Productivity increased over the session"
        elif slope < -0.02:
            trend = "declining"
            description = "Productivity decreased over the session"
        else:
            trend = "stable"
            description = "Productivity remained consistent"

        return {
            "trend": trend,
            "slope": round(float(slope), 4),
            "description": description,
            "start_score": round(scores[0], 3) if scores else 0,
            "end_score": round(scores[-1], 3) if scores else 0,
        }

    def tool_switch_analysis(self) -> Dict[str, Any]:
        """
        Analyze tool switching behavior.

        Returns:
            Dictionary with switch count and flag if excessive
        """
        switches = 0
        prev_tool = None

        for seg in self.segments:
            if seg.primary_tool != prev_tool:
                if prev_tool is not None:
                    switches += 1
                prev_tool = seg.primary_tool

        excessive = switches > 15

        return {
            "tool_switches": switches,
            "excessive_switching": excessive,
            "recommendation": "Consider batching similar tasks to reduce tool switches" if excessive else None
        }

    def peak_productivity_period(self) -> Dict[str, Any]:
        """
        Find the peak productivity period during the session.

        Returns:
            Dictionary with peak period details
        """
        if not self.segments:
            return {"found": False}

        best_segment = max(self.segments, key=lambda s: s.productivity_score)

        return {
            "found": True,
            "start_time": best_segment.start_time,
            "end_time": best_segment.end_time,
            "duration": self._duration(best_segment),
            "score": round(best_segment.productivity_score, 3),
            "activity": best_segment.activity.value,
            "tool": best_segment.primary_tool,
        }

    def detect_fatigue(self) -> Dict[str, Any]:
        """
        Detect behavioral fatigue by comparing first and second half of session.

        Fatigue indicators:
        - Productivity drop > 8%
        - Idle time increase > 4%
        - Increased tool switching

        Returns:
            Dictionary with fatigue analysis
        """
        if not self.segments:
            return {"fatigue_detected": False, "reason": "No segments to analyze"}

        total_time = self._total_time()
        mid_time = self.report.start_time + (total_time / 2)

        first_half = []
        second_half = []

        for seg in self.segments:
            if seg.end_time <= mid_time:
                first_half.append(seg)
            else:
                second_half.append(seg)

        if not first_half or not second_half:
            return {"fatigue_detected": False, "reason": "Insufficient data for comparison"}

        def weighted_avg_productivity(segs):
            if not segs:
                return 0
            total_weighted = sum(
                self._duration(s) * s.productivity_score
                for s in segs
            )
            total = sum(self._duration(s) for s in segs)
            return total_weighted / total if total > 0 else 0

        def idle_percentage(segs):
            total = sum(self._duration(s) for s in segs)
            idle = sum(
                self._duration(s)
                for s in segs
                if s.activity == ActivityState.IDLE
            )
            return (idle / total * 100) if total > 0 else 0

        def tool_switches(segs):
            switches = 0
            prev = None
            for s in segs:
                if s.primary_tool != prev:
                    if prev is not None:
                        switches += 1
                    prev = s.primary_tool
            return switches

        first_prod = weighted_avg_productivity(first_half)
        second_prod = weighted_avg_productivity(second_half)

        first_idle = idle_percentage(first_half)
        second_idle = idle_percentage(second_half)

        first_switch = tool_switches(first_half)
        second_switch = tool_switches(second_half)

        productivity_drop = first_prod - second_prod
        idle_increase = second_idle - first_idle
        switch_increase = second_switch - first_switch

        fatigue_flag = (
            productivity_drop > 0.08 and
            idle_increase > 4
        )

        fatigue_score = (
            max(0, productivity_drop) * 0.5 +
            max(0, idle_increase / 100) * 0.3 +
            max(0, switch_increase / 10) * 0.2
        )

        recommendations = []
        if fatigue_flag:
            if productivity_drop > 0.1:
                recommendations.append("Consider shorter work sessions or more frequent breaks")
            if idle_increase > 10:
                recommendations.append("Increased idle time in second half suggests fatigue")
            if switch_increase > 5:
                recommendations.append("Increased tool switching may indicate difficulty focusing")

        return {
            "fatigue_detected": fatigue_flag,
            "fatigue_score": round(min(fatigue_score, 1.0), 3),
            "productivity_drop": round(productivity_drop * 100, 2),
            "idle_increase": round(idle_increase, 2),
            "tool_switch_change": switch_increase,
            "first_half": {
                "productivity_score": round(first_prod, 3),
                "idle_percentage": round(first_idle, 2),
                "tool_switches": first_switch,
            },
            "second_half": {
                "productivity_score": round(second_prod, 3),
                "idle_percentage": round(second_idle, 2),
                "tool_switches": second_switch,
            },
            "recommendations": recommendations,
        }

    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive executive summary of the session.

        Returns:
            Dictionary with all key insights
        """
        idle = self.analyze_idle_time()
        trend = self.productivity_trend()
        switches = self.tool_switch_analysis()
        peak = self.peak_productivity_period()
        fatigue = self.detect_fatigue()

        all_recommendations = []
        if idle.get("recommendation"):
            all_recommendations.append(idle["recommendation"])
        if switches.get("recommendation"):
            all_recommendations.append(switches["recommendation"])
        all_recommendations.extend(fatigue.get("recommendations", []))
        all_recommendations.extend(self.report.recommendations)

        return {
            "overall_productivity": round(self.report.productivity_score * 100, 1),
            "session_duration_seconds": self.report.session_duration,
            "idle_analysis": idle,
            "productivity_trend": trend,
            "tool_switching": switches,
            "peak_period": peak,
            "fatigue_analysis": fatigue,
            "key_insights": self.report.insights,
            "recommendations": list(set(all_recommendations)),
        }
