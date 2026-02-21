"""
Activity Classification Module (P3) - CRITICAL PATH
Classifies worker activities over time using HOI detection + motion analysis.

This is the core innovation: combining hand-object interactions with temporal
motion patterns to determine what the worker is actually doing.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import numpy as np

# Import from perception module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from perception.hoi_detector import FrameAnalysis, InteractionStatus
from .motion_analyzer import MotionType, MotionResult


class ActivityState(Enum):
    """Worker activity states with productivity scores."""
    ACTIVE_TOOL_USE = "ACTIVE_TOOL_USE"  # Actively using tool
    PRECISION_WORK = "PRECISION_WORK"  # Careful positioning/measurement
    MATERIAL_HANDLING = "MATERIAL_HANDLING"  # Moving materials
    SETUP_CLEANUP = "SETUP_CLEANUP"  # Preparing workspace
    SEARCHING = "SEARCHING"  # Looking for tools/materials
    TRAVELING = "TRAVELING"  # Moving to different location
    IDLE = "IDLE"  # No productive activity


@dataclass
class ActivitySegment:
    """A continuous segment of a single activity."""
    start_time: float
    end_time: float
    activity: ActivityState
    tool_used: Optional[str]
    productivity_score: float
    confidence: float
    frame_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time


class ActivityClassifier:
    """
    Classifies worker activities by combining:
    1. Hand-Object Interaction detection (HOI)
    2. Camera motion analysis
    3. Temporal state machine logic

    Key Innovation: Multi-modal fusion for activity recognition.
    """

    # Activity states with productivity scores
    STATES = {
        "ACTIVE_TOOL_USE": {"productivity": 1.0, "value": 100},
        "PRECISION_WORK": {"productivity": 1.0, "value": 100},
        "MATERIAL_HANDLING": {"productivity": 0.7, "value": 70},
        "SETUP_CLEANUP": {"productivity": 0.5, "value": 50},
        "SEARCHING": {"productivity": 0.3, "value": 30},
        "TRAVELING": {"productivity": 0.2, "value": 20},
        "IDLE": {"productivity": 0.0, "value": 0},
    }

    # Smoothing parameters
    MIN_SEGMENT_DURATION = 1.0  # seconds - merge shorter segments
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to classify

    def __init__(self, smoothing_window: int = 3):
        """
        Initialize activity classifier.

        Args:
            smoothing_window: Number of frames for smoothing state transitions
        """
        self.smoothing_window = smoothing_window
        self.state_history = []  # Recent activity states for smoothing

    def classify_frame(
        self,
        analysis: FrameAnalysis,
        motion_result: MotionResult,
    ) -> tuple[ActivityState, float]:
        """
        Classify activity for a single frame.

        Args:
            analysis: FrameAnalysis from HOIDetector
            motion_result: MotionResult from MotionAnalyzer

        Returns:
            (ActivityState, confidence)
        """
        # Extract features from HOI analysis
        has_hands = len(analysis.hands) > 0
        has_active_interaction = analysis.has_active_interaction()
        held_tools = analysis.get_held_tools()
        has_tool = len(held_tools) > 0
        primary_tool = analysis.primary_tool

        # Extract motion features
        motion_type = motion_result.motion_type
        motion_magnitude = motion_result.magnitude

        # Decision tree for activity classification
        activity, confidence = self._classify_activity(
            has_hands=has_hands,
            has_tool=has_tool,
            has_active_interaction=has_active_interaction,
            motion_type=motion_type,
            motion_magnitude=motion_magnitude,
            primary_tool=primary_tool,
        )

        # Apply smoothing based on recent history
        if len(self.state_history) >= self.smoothing_window:
            activity, confidence = self._smooth_classification(
                activity, confidence
            )

        # Update history
        self.state_history.append(activity)
        if len(self.state_history) > self.smoothing_window * 2:
            self.state_history = self.state_history[-self.smoothing_window * 2:]

        return activity, confidence

    def _classify_activity(
        self,
        has_hands: bool,
        has_tool: bool,
        has_active_interaction: bool,
        motion_type: MotionType,
        motion_magnitude: float,
        primary_tool: Optional[str],
    ) -> tuple[ActivityState, float]:
        """
        Core decision logic for activity classification.

        Priority order (highest to lowest):
        1. ACTIVE_TOOL_USE - Tool in hand + motion
        2. PRECISION_WORK - Tool in hand + stable/minimal motion
        3. MATERIAL_HANDLING - No tool + hands visible + some motion
        4. SEARCHING - No tool + panning motion
        5. TRAVELING - Walking motion + no tool
        6. SETUP_CLEANUP - Hands visible but ambiguous activity
        7. IDLE - No hands, no activity

        Returns:
            (ActivityState, confidence)
        """

        # Rule 1: ACTIVE_TOOL_USE
        # Tool in hand + stable or rhythmic motion
        if has_tool and has_active_interaction:
            if motion_type == MotionType.RHYTHMIC:
                # Clear rhythmic tool use (drilling, hammering)
                return ActivityState.ACTIVE_TOOL_USE, 0.95
            elif motion_type == MotionType.STABLE and motion_magnitude > 1.0:
                # Tool use with some motion
                return ActivityState.ACTIVE_TOOL_USE, 0.85
            elif motion_type == MotionType.STABLE and motion_magnitude <= 1.0:
                # Precision work - very minimal motion
                return ActivityState.PRECISION_WORK, 0.9
            else:
                # Tool in hand but unclear activity
                return ActivityState.ACTIVE_TOOL_USE, 0.7

        # Rule 2: PRECISION_WORK
        # Tool in hand but minimal motion (measuring, positioning)
        if has_tool and not has_active_interaction:
            if motion_type == MotionType.STABLE:
                return ActivityState.PRECISION_WORK, 0.75
            else:
                return ActivityState.SETUP_CLEANUP, 0.6

        # Rule 3: TRAVELING
        # Walking motion detected
        if motion_type == MotionType.WALKING:
            return ActivityState.TRAVELING, 0.85

        # Rule 4: SEARCHING
        # No tool + panning motion (looking around)
        if not has_tool and motion_type == MotionType.PANNING:
            return ActivityState.SEARCHING, 0.8

        # Rule 5: MATERIAL_HANDLING
        # No tool but hands visible with some motion
        if has_hands and not has_tool:
            if motion_type == MotionType.STABLE and motion_magnitude > 2.0:
                # Hands moving but no tool
                return ActivityState.MATERIAL_HANDLING, 0.7
            elif motion_type == MotionType.RHYTHMIC:
                # Could be manual work without tools
                return ActivityState.MATERIAL_HANDLING, 0.75
            else:
                # Hands visible but unclear
                return ActivityState.SETUP_CLEANUP, 0.6

        # Rule 6: SETUP_CLEANUP
        # Hands visible but low activity
        if has_hands and motion_type == MotionType.STABLE:
            return ActivityState.SETUP_CLEANUP, 0.5

        # Rule 7: IDLE
        # No hands visible or no motion
        if not has_hands or motion_type == MotionType.STABLE:
            return ActivityState.IDLE, 0.7

        # Default: Unknown activity - classify as SETUP_CLEANUP
        return ActivityState.SETUP_CLEANUP, 0.4

    def _smooth_classification(
        self,
        current_state: ActivityState,
        current_confidence: float,
    ) -> tuple[ActivityState, float]:
        """
        Smooth activity classification using recent history.
        Avoids rapid flickering between states.

        Returns:
            (smoothed_state, smoothed_confidence)
        """
        if not self.state_history:
            return current_state, current_confidence

        # Get recent history
        recent = self.state_history[-self.smoothing_window:]

        # Count occurrences of each state
        state_counts = {}
        for state in recent:
            state_counts[state] = state_counts.get(state, 0) + 1

        # Most common state in recent history
        most_common = max(state_counts.items(), key=lambda x: x[1])
        common_state, common_count = most_common

        # If current state matches most common, boost confidence
        if current_state == common_state:
            boosted_confidence = min(1.0, current_confidence * 1.1)
            return current_state, boosted_confidence

        # If current state is different but has low confidence, use common state
        if current_confidence < self.CONFIDENCE_THRESHOLD:
            # Use the common state with reduced confidence
            return common_state, 0.6

        # Otherwise use current state
        return current_state, current_confidence

    def segment_activities(
        self,
        frame_analyses: List[FrameAnalysis],
        motion_results: List[MotionResult],
        fps: float = 30.0,
    ) -> List[ActivitySegment]:
        """
        Segment activities over time from frame-by-frame analyses.

        Args:
            frame_analyses: List of FrameAnalysis from HOI detection
            motion_results: List of MotionResult from motion analysis
            fps: Video frame rate for time calculation

        Returns:
            List of ActivitySegment objects
        """
        if not frame_analyses or len(frame_analyses) != len(motion_results):
            return []

        segments = []
        current_segment = None

        for i, (frame_analysis, motion_result) in enumerate(zip(frame_analyses, motion_results)):
            # Get timestamp
            timestamp = frame_analysis.timestamp

            # Classify this frame
            activity, confidence = self.classify_frame(frame_analysis, motion_result)

            # Get productivity score
            productivity = self.STATES[activity.value]["productivity"]

            # Get primary tool used
            tool_used = frame_analysis.primary_tool

            # Start new segment or continue existing
            if current_segment is None:
                # Start first segment
                current_segment = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "activity": activity,
                    "tool_used": tool_used,
                    "productivity": productivity,
                    "confidence": confidence,
                    "frame_count": 1,
                }
            elif (current_segment["activity"] == activity and
                  current_segment["tool_used"] == tool_used):
                # Continue current segment
                current_segment["end_time"] = timestamp
                current_segment["frame_count"] += 1
                # Update average confidence
                n = current_segment["frame_count"]
                current_segment["confidence"] = (
                    (current_segment["confidence"] * (n - 1) + confidence) / n
                )
            else:
                # Activity changed - save current segment and start new one
                segments.append(ActivitySegment(
                    start_time=current_segment["start_time"],
                    end_time=current_segment["end_time"],
                    activity=current_segment["activity"],
                    tool_used=current_segment["tool_used"],
                    productivity_score=current_segment["productivity"],
                    confidence=current_segment["confidence"],
                    frame_count=current_segment["frame_count"],
                ))

                # Start new segment
                current_segment = {
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "activity": activity,
                    "tool_used": tool_used,
                    "productivity": productivity,
                    "confidence": confidence,
                    "frame_count": 1,
                }

        # Add final segment
        if current_segment is not None:
            segments.append(ActivitySegment(
                start_time=current_segment["start_time"],
                end_time=current_segment["end_time"],
                activity=current_segment["activity"],
                tool_used=current_segment["tool_used"],
                productivity_score=current_segment["productivity"],
                confidence=current_segment["confidence"],
                frame_count=current_segment["frame_count"],
            ))

        # Post-process: merge very short segments
        segments = self._merge_short_segments(segments)

        return segments

    def _merge_short_segments(
        self,
        segments: List[ActivitySegment],
    ) -> List[ActivitySegment]:
        """
        Merge segments shorter than MIN_SEGMENT_DURATION with neighbors.

        Returns:
            Merged segments
        """
        if len(segments) <= 1:
            return segments

        merged = []
        i = 0

        while i < len(segments):
            current = segments[i]

            # If segment is long enough, keep it
            if current.duration >= self.MIN_SEGMENT_DURATION:
                merged.append(current)
                i += 1
                continue

            # Segment is too short - try to merge with neighbors
            if i > 0 and i < len(segments) - 1:
                # Merge with most similar neighbor
                prev = segments[i - 1]
                next_seg = segments[i + 1]

                # Prefer merging with same activity
                if prev.activity == current.activity:
                    # Merge with previous
                    merged[-1] = ActivitySegment(
                        start_time=prev.start_time,
                        end_time=current.end_time,
                        activity=prev.activity,
                        tool_used=prev.tool_used,
                        productivity_score=prev.productivity_score,
                        confidence=(prev.confidence + current.confidence) / 2,
                        frame_count=prev.frame_count + current.frame_count,
                    )
                else:
                    # Keep as is, will be merged in next iteration
                    merged.append(current)

            elif i == 0 and len(segments) > 1:
                # First segment - merge with next
                next_seg = segments[i + 1]
                current = ActivitySegment(
                    start_time=current.start_time,
                    end_time=next_seg.end_time,
                    activity=next_seg.activity,
                    tool_used=next_seg.tool_used,
                    productivity_score=next_seg.productivity_score,
                    confidence=(current.confidence + next_seg.confidence) / 2,
                    frame_count=current.frame_count + next_seg.frame_count,
                )
                merged.append(current)
                i += 1  # Skip next since we merged it

            else:
                # Last segment - keep as is
                merged.append(current)

            i += 1

        return merged

    def reset(self):
        """Reset internal state."""
        self.state_history = []


# Quick test
if __name__ == "__main__":
    """
    Test activity classifier with synthetic data.
    """
    from perception.hoi_detector import FrameAnalysis, HandResult, Detection, Interaction, InteractionStatus

    classifier = ActivityClassifier(smoothing_window=3)

    print("Activity Classifier Test")
    print("=" * 50)

    # Test Case 1: Active tool use
    print("\nTest 1: Active tool use (drill + rhythmic motion)")
    frame_analysis = FrameAnalysis(
        timestamp=0.0,
        frame_index=0,
        hands=[HandResult(0, "right", [], (0, 0, 0, 0), {}, 0.9)],
        tools=[Detection("drill", (100, 100, 200, 200), 0.9, (150, 150))],
        workpieces=[],
        interactions=[Interaction("right", 0, "drill", (100, 100, 200, 200), InteractionStatus.HOLDING, 0.9, 10.0)],
        primary_tool="drill",
    )
    motion_result = MotionResult(MotionType.RHYTHMIC, 0.9, 15.0, (0, 0), 2.5, {})
    activity, conf = classifier.classify_frame(frame_analysis, motion_result)
    print(f"  Activity: {activity.value}, Confidence: {conf:.2f}")

    # Test Case 2: Searching
    print("\nTest 2: Searching (no tool + panning)")
    frame_analysis = FrameAnalysis(
        timestamp=1.0,
        frame_index=1,
        hands=[HandResult(0, "right", [], (0, 0, 0, 0), {}, 0.9)],
        tools=[],
        workpieces=[],
        interactions=[],
        primary_tool=None,
    )
    motion_result = MotionResult(MotionType.PANNING, 0.85, 8.0, (5, 0), None, {})
    activity, conf = classifier.classify_frame(frame_analysis, motion_result)
    print(f"  Activity: {activity.value}, Confidence: {conf:.2f}")

    # Test Case 3: Idle
    print("\nTest 3: Idle (no hands + stable)")
    frame_analysis = FrameAnalysis(
        timestamp=2.0,
        frame_index=2,
        hands=[],
        tools=[],
        workpieces=[],
        interactions=[],
        primary_tool=None,
    )
    motion_result = MotionResult(MotionType.STABLE, 0.95, 0.5, (0, 0), None, {})
    activity, conf = classifier.classify_frame(frame_analysis, motion_result)
    print(f"  Activity: {activity.value}, Confidence: {conf:.2f}")

    print("\n" + "=" * 50)
    print("Activity Classifier tests complete!")
    print("\nProductivity Scores:")
    for state, values in ActivityClassifier.STATES.items():
        print(f"  {state}: {values['productivity']}")
