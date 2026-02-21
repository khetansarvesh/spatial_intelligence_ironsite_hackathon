"""
Hand-Object Interaction (HOI) Detection Module
Combines hand detection and tool detection to determine interactions.

Updated to work with Grounding DINO-based detectors.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import math
import cv2

from .hand_detector import HandDetector, HandResult, filter_gloves_by_fingers, compute_iou
from .tool_detector import ToolDetector, Detection, DetectionResult, DetectorBackend


class InteractionStatus(Enum):
    WORKING = "working"  # Hand is interacting with tool (IOU > 0 or close distance)
    IDLE = "idle"        # No interaction


@dataclass
class Interaction:
    """Represents an interaction between a hand and an object."""
    hand_id: int
    object_label: str
    object_bbox: Tuple[int, int, int, int]
    status: InteractionStatus
    confidence: float
    distance_pixels: float
    iou: float  # Overlap between hand and tool


@dataclass
class FrameAnalysis:
    """Complete analysis of a single frame."""
    timestamp: float
    frame_index: int
    hands: List[HandResult]
    tools: List[Detection]
    interactions: List[Interaction]
    primary_tool: Optional[str] = None  # Main tool being used
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_working(self) -> bool:
        """Check if worker is actively working with any tool."""
        return any(
            i.status == InteractionStatus.WORKING
            for i in self.interactions
        )

    def get_active_tools(self) -> List[str]:
        """Get list of tools currently being worked with."""
        return [
            i.object_label
            for i in self.interactions
            if i.status == InteractionStatus.WORKING
        ]

    def has_active_interaction(self) -> bool:
        """Check if there's an active interaction (alias for is_working)."""
        return self.is_working()

    def get_held_tools(self) -> List[str]:
        """Get list of tools being held/used (alias for get_active_tools)."""
        return self.get_active_tools()


class HOIDetector:
    """
    Detects Hand-Object Interactions by combining hand and tool detection.

    Uses Grounding DINO for both hand (glove) and tool detection.
    Hands are filtered using finger detection for accuracy.
    """

    # Distance threshold (in pixels, will be scaled by frame size)
    WORKING_DISTANCE = 100  # If IOU=0, tool center within this distance = working

    def __init__(
        self,
        hand_confidence: float = 0.25,
        tool_confidence: float = 0.25,
        working_distance: float = None,
    ):
        """
        Initialize HOI detector.

        Args:
            hand_confidence: Confidence threshold for hand detection
            tool_confidence: Confidence threshold for tool detection
            working_distance: Distance threshold for WORKING status when IOU=0
        """
        self.hand_detector = HandDetector(confidence_threshold=hand_confidence)
        self.tool_detector = ToolDetector(
            backend=DetectorBackend.GROUNDING_DINO,
            confidence_threshold=tool_confidence,
        )

        if working_distance:
            self.WORKING_DISTANCE = working_distance

        self.frame_count = 0

    def analyze_frame(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> FrameAnalysis:
        """
        Analyze a single frame for hand-object interactions.

        Args:
            frame: BGR image from OpenCV
            timestamp: Timestamp in seconds

        Returns:
            FrameAnalysis with all detections and interactions
        """
        height, width = frame.shape[:2]

        # Scale distance threshold based on frame size
        scale = min(width, height) / 640  # Normalize to 640px base
        working_dist = self.WORKING_DISTANCE * scale

        # Detect hands (gloves) with finger filtering
        gloves = self.hand_detector.detect(frame)
        fingers = self.hand_detector.detect_fingers(frame)
        hands, _ = filter_gloves_by_fingers(gloves, fingers, min_avg_iou=0.05)

        # Detect tools (skip workpieces for speed)
        detections = self.tool_detector.detect(frame, tools_only=True)
        tools = detections.tools

        # Find interactions
        interactions = []

        for hand in hands:
            hand_center = hand.center

            for tool in tools:
                # Calculate IOU (overlap)
                iou = compute_iou(hand.bbox, tool.bbox)

                # Calculate distance between centers
                distance = self._distance(hand_center, tool.center)

                # Determine interaction status:
                # - IOU > 0 means WORKING (bounding boxes overlap)
                # - IOU = 0 but distance < threshold means WORKING
                # - Otherwise IDLE (no interaction recorded)
                if iou > 0:
                    status = InteractionStatus.WORKING
                    confidence = min(1.0, iou * 5)  # Scale IOU to confidence
                elif distance < working_dist:
                    status = InteractionStatus.WORKING
                    confidence = 1 - (distance / working_dist)
                else:
                    continue  # IDLE - no interaction with this tool

                interactions.append(Interaction(
                    hand_id=hand.hand_id,
                    object_label=tool.label,
                    object_bbox=tool.bbox,
                    status=status,
                    confidence=max(0, min(1, confidence)),
                    distance_pixels=distance,
                    iou=iou,
                ))

        # Determine primary tool (highest confidence WORKING interaction)
        working_interactions = [
            i for i in interactions
            if i.status == InteractionStatus.WORKING
        ]
        primary_tool = None
        if working_interactions:
            best = max(working_interactions, key=lambda x: x.confidence)
            primary_tool = best.object_label

        self.frame_count += 1

        return FrameAnalysis(
            timestamp=timestamp,
            frame_index=self.frame_count,
            hands=hands,
            tools=tools,
            interactions=interactions,
            primary_tool=primary_tool,
        )

    def _distance(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def draw_analysis(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
    ) -> np.ndarray:
        """
        Visualize frame analysis with all detections and interactions.

        Args:
            frame: BGR image
            analysis: FrameAnalysis result

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        HAND_COLOR = (0, 255, 0)      # Green
        TOOL_COLOR = (0, 165, 255)    # Orange
        WORKING_COLOR = (0, 0, 255)   # Red - tool being worked with

        # Draw hands
        for hand in analysis.hands:
            x1, y1, x2, y2 = hand.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), HAND_COLOR, 2)
            cv2.putText(
                annotated,
                f"hand ({hand.confidence:.2f})",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                HAND_COLOR,
                2,
            )
            cv2.circle(annotated, hand.center, 4, HAND_COLOR, -1)

        # Draw tools
        for tool in analysis.tools:
            x1, y1, x2, y2 = tool.bbox
            # Check if this tool is being worked with
            is_working = any(
                i.object_bbox == tool.bbox and i.status == InteractionStatus.WORKING
                for i in analysis.interactions
            )
            color = WORKING_COLOR if is_working else TOOL_COLOR

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"{tool.label} ({tool.confidence:.2f})",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Draw interaction lines for WORKING status
        for interaction in analysis.interactions:
            if interaction.status == InteractionStatus.WORKING:
                # Find the hand
                hand = next(
                    (h for h in analysis.hands if h.hand_id == interaction.hand_id),
                    None
                )
                if hand:
                    obj_center = (
                        (interaction.object_bbox[0] + interaction.object_bbox[2]) // 2,
                        (interaction.object_bbox[1] + interaction.object_bbox[3]) // 2,
                    )
                    cv2.line(annotated, hand.center, obj_center, WORKING_COLOR, 2)

        # Draw status overlay
        status_lines = []
        if analysis.is_working():
            active_tools = analysis.get_active_tools()
            status_lines.append(f"WORKING: {', '.join(active_tools)}")
        else:
            status_lines.append("IDLE")

        status_lines.append(f"Hands: {len(analysis.hands)} | Tools: {len(analysis.tools)}")

        y_offset = 30
        for text in status_lines:
            cv2.putText(
                annotated,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y_offset += 30

        return annotated

    def close(self):
        """Release resources."""
        self.hand_detector.close()


# Quick test
if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    video_path = project_root / "data" / "videos" / "07_production_mp.mp4"

    print(f"Testing HOI detector on: {video_path}")

    detector = HOIDetector()
    cap = cv2.VideoCapture(str(video_path))

    frame_idx = 0
    while cap.isOpened() and frame_idx < 10:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 5 == 0:  # Every 5th frame
            analysis = detector.analyze_frame(frame, timestamp=frame_idx / 5.0)
            status = "WORKING" if analysis.is_working() else "IDLE"
            print(f"Frame {frame_idx}: {len(analysis.hands)} hands, {len(analysis.tools)} tools, "
                  f"{status}: {analysis.get_active_tools()}")

        frame_idx += 1

    cap.release()
    detector.close()
    print("Done!")
