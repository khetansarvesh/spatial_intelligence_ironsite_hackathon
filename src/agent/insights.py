from typing import List, Dict
import numpy as np

class InsightEngine:
    def __init__(self, activity_segments: List):
        self.segments = activity_segments

    # Utility

    def _duration(self, seg):
        return seg.end_time - seg.start_time

    def _total_time(self):
        return sum(self._duration(s) for s in self.segments)

    # Idle Analysis

    def analyze_idle_time(self) -> Dict:
        total_time = self._total_time()
        idle_time = sum(
            self._duration(s)
            for s in self.segments
            if s.activity == "IDLE"
        )

        idle_percentage = idle_time / total_time if total_time > 0 else 0

        return {
            "idle_seconds": idle_time,
            "idle_percentage": round(idle_percentage, 3),
            "flag": idle_percentage > 0.2
        }

    # Productivity Trend Over Time

    def productivity_trend(self) -> Dict:
        timeline = []
        for seg in self.segments:
            timeline.append({
                "start": seg.start_time,
                "end": seg.end_time,
                "score": seg.productivity_score
            })

        scores = [s["score"] for s in timeline]

        if not scores:
            return {"trend": "unknown"}

        slope = np.polyfit(range(len(scores)), scores, 1)[0]

        if slope > 0.02:
            trend = "improving"
        elif slope < -0.02:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(float(slope), 4)
        }

    # Tool Switching Behavior

    def tool_switch_analysis(self) -> Dict:
        switches = 0
        prev_tool = None

        for seg in self.segments:
            if seg.tool_used != prev_tool:
                if prev_tool is not None:
                    switches += 1
                prev_tool = seg.tool_used

        return {
            "tool_switches": switches,
            "flag": switches > 15
        }
    
    # Peak Performance Window

    def peak_productivity_period(self) -> Dict:
        best_segment = None
        best_score = -1

        for seg in self.segments:
            if seg.productivity_score > best_score:
                best_score = seg.productivity_score
                best_segment = seg

        if not best_segment:
            return {}

        return {
            "start": best_segment.start_time,
            "end": best_segment.end_time,
            "score": best_score
        }
    
    # Executive Summary Generator

    def generate_summary(self) -> Dict:
        idle = self.analyze_idle_time()
        trend = self.productivity_trend()
        switches = self.tool_switch_analysis()
        peak = self.peak_productivity_period()

        return {
            "idle_analysis": idle,
            "productivity_trend": trend,
            "tool_switching": switches,
            "peak_period": peak
        }
    
    # Fatigue Detection
        
    def detect_fatigue(self) -> Dict:
        if not self.segments:
            return {"fatigue_detected": False}

        total_time = self._total_time()
        mid_time = total_time / 2

        first_half = []
        second_half = []

        for seg in self.segments:
            if seg.end_time <= mid_time:
                first_half.append(seg)
            else:
                second_half.append(seg)

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
                if s.activity == "IDLE"
            )
            return idle / total if total > 0 else 0

        def tool_switches(segs):
            switches = 0
            prev = None
            for s in segs:
                if s.tool_used != prev:
                    if prev is not None:
                        switches += 1
                    prev = s.tool_used
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

        # fatigue threshold logic
        fatigue_flag = (
            productivity_drop > 0.08 and
            idle_increase > 0.04
        )

        fatigue_score = (
            max(0, productivity_drop) * 0.5 +
            max(0, idle_increase) * 0.3 +
            max(0, switch_increase / 10) * 0.2
        )

        return {
            "fatigue_detected": fatigue_flag,
            "fatigue_score": round(fatigue_score, 3),
            "productivity_drop": round(productivity_drop, 3),
            "idle_increase": round(idle_increase, 3),
            "tool_switch_change": switch_increase,
            "first_half_score": round(first_prod, 3),
            "second_half_score": round(second_prod, 3)
        }