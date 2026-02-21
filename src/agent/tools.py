from typing import List, Dict
from collections import defaultdict
from collections import defaultdict


class ProductivityTools:
    def __init__(self, activity_segments: List):
        """
        activity_segments: List[ActivitySegment]
        """
        self.segments = activity_segments

    # Helper Utilities

    def _filter_by_time(self, start: float, end: float):
        return [
            seg for seg in self.segments
            if seg.start_time >= start and seg.end_time <= end
        ]

    def _duration(self, seg):
        return seg.end_time - seg.start_time

    # TOOL 1: Activity Summary

    def get_activity_summary(self, start_time: float, end_time: float) -> Dict:
        segments = self._filter_by_time(start_time, end_time)

        breakdown = defaultdict(float)
        total_time = 0

        for seg in segments:
            duration = self._duration(seg)
            breakdown[seg.activity] += duration
            total_time += duration

        return {
            "time_range": f"{start_time:.2f}s - {end_time:.2f}s",
            "total_time_seconds": total_time,
            "activity_breakdown_seconds": dict(breakdown)
        }

    # TOOL 2: Tool Usage

    def get_tool_usage(self) -> Dict[str, float]:
        tool_time = defaultdict(float)

        for seg in self.segments:
            if seg.tool_used:
                tool_time[seg.tool_used] += self._duration(seg)

        return dict(tool_time)

    # TOOL 3: Idle Periods

    def find_idle_periods(self, min_duration_seconds: int = 10):
        idle_segments = []

        for seg in self.segments:
            if seg.activity == "IDLE":
                duration = self._duration(seg)
                if duration >= min_duration_seconds:
                    idle_segments.append({
                        "start": seg.start_time,
                        "end": seg.end_time,
                        "duration": duration
                    })

        return idle_segments

    # TOOL 4: Productivity Score

    def get_productivity_score(self) -> Dict:
        total_weighted = 0
        total_time = 0

        for seg in self.segments:
            duration = self._duration(seg)
            total_weighted += duration * seg.productivity_score
            total_time += duration

        if total_time == 0:
            return {"score": 0.0}

        return {
            "score": round(total_weighted / total_time, 3),
            "total_time_seconds": total_time
        }

    # TOOL 5: Compare Periods

    def compare_periods(self, p1_start: float, p1_end: float,
                        p2_start: float, p2_end: float):

        p1 = self.get_activity_summary(p1_start, p1_end)
        p2 = self.get_activity_summary(p2_start, p2_end)

        return {
            "period_1": p1,
            "period_2": p2
        }