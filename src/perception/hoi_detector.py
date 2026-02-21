"""
Hand-Object Interaction (HOI) Detection Module (P1 + P2)
Combines hand detection and tool detection to determine interactions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import numpy as np
import math

from .hand_detector import HandDetector, HandResult
from .tool_detector import ToolDetector, Detection, DetectionResult, DetectorBackend


class InteractionStatus(Enum):
    HOLDING = "holding"
    REACHING = "reaching"
    NEAR = "near"
    NONE = "none"


@dataclass
class Interaction:
    """Represents an interaction between a hand and an object."""
    hand_side: str  # "left" | "right"
    hand_id: int
    object_label: str
    object_bbox: Tuple[int, int, int, int]
    status: InteractionStatus
    confidence: float
    distance_pixels: float


@dataclass
class FrameAnalysis:
    """Complete analysis of a single frame."""
    timestamp: float
    frame_index: int
    hands: List[HandResult]
    tools: List[Detection]
    workpieces: List[Detection]
    interactions: List[Interaction]
    primary_tool: Optional[str] = None  # Main tool being used
    primary_workpiece: Optional[str] = None  # Main workpiece being worked on
    camera_motion: Optional[str] = None  # Will be filled by motion analyzer
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_active_interaction(self) -> bool:
        """Check if there's any active tool interaction."""
        return any(
            i.status == InteractionStatus.HOLDING
            for i in self.interactions
        )

    def get_held_tools(self) -> List[str]:
        """Get list of tools currently being held."""
        return [
            i.object_label
            for i in self.interactions
            if i.status == InteractionStatus.HOLDING
        ]


class HOIDetector:
    """
    Detects Hand-Object Interactions by combining hand and tool detection.

    The key insight: In egocentric video, if a tool's bounding box overlaps
    significantly with the hand region (especially fingertips), the worker
    is likely holding that tool.
    """

    # Distance thresholds (in pixels, will be scaled by frame size)
    HOLDING_THRESHOLD = 50  # Tool center within this distance of fingertips
    REACHING_THRESHOLD = 150  # Hand moving toward tool
    NEAR_THRESHOLD = 300  # Tool in general vicinity

    def __init__(
        self,
        hand_detector: Optional[HandDetector] = None,
        tool_detector: Optional[ToolDetector] = None,
        holding_threshold: float = None,
    ):
        """
        Initialize HOI detector.

        Args:
            hand_detector: HandDetector instance (created if None)
            tool_detector: ToolDetector instance (created if None)
            holding_threshold: Custom threshold for "holding" detection
        """
        self.hand_detector = hand_detector or HandDetector()
        self.tool_detector = tool_detector or ToolDetector(
            backend=DetectorBackend.YOLO
        )

        if holding_threshold:
            self.HOLDING_THRESHOLD = holding_threshold

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

        # Scale thresholds based on frame size
        scale = min(width, height) / 640  # Normalize to 640px base
        holding_thresh = self.HOLDING_THRESHOLD * scale
        reaching_thresh = self.REACHING_THRESHOLD * scale
        near_thresh = self.NEAR_THRESHOLD * scale

        # Detect hands
        hands = self.hand_detector.detect(frame)

        # Detect tools and workpieces
        detections = self.tool_detector.detect(frame)

        # Find interactions
        interactions = []

        for hand in hands:
            # Get fingertip positions
            fingertips = list(hand.fingertip_positions.values())
            wrist = self.hand_detector.get_wrist_position(hand)

            # Calculate hand center (average of fingertips)
            hand_center = (
                sum(f[0] for f in fingertips) / len(fingertips),
                sum(f[1] for f in fingertips) / len(fingertips),
            )

            # Check interaction with each tool
            for tool in detections.tools:
                # Calculate distances
                distance_to_center = self._distance(hand_center, tool.center)

                # Check if any fingertip is inside or very close to tool bbox
                fingertips_near = sum(
                    1 for f in fingertips
                    if self._point_near_bbox(f, tool.bbox, holding_thresh)
                )

                # Determine interaction status
                if fingertips_near >= 2 or distance_to_center < holding_thresh:
                    status = InteractionStatus.HOLDING
                    confidence = min(1.0, (fingertips_near / 3) + (1 - distance_to_center / holding_thresh) * 0.5)
                elif distance_to_center < reaching_thresh:
                    status = InteractionStatus.REACHING
                    confidence = 1 - (distance_to_center - holding_thresh) / (reaching_thresh - holding_thresh)
                elif distance_to_center < near_thresh:
                    status = InteractionStatus.NEAR
                    confidence = 1 - (distance_to_center - reaching_thresh) / (near_thresh - reaching_thresh)
                else:
                    continue  # Too far, no interaction

                interactions.append(Interaction(
                    hand_side=hand.side,
                    hand_id=hand.hand_id,
                    object_label=tool.label,
                    object_bbox=tool.bbox,
                    status=status,
                    confidence=max(0, min(1, confidence)),
                    distance_pixels=distance_to_center,
                ))

        # Determine primary tool (highest confidence HOLDING interaction)
        holding_interactions = [
            i for i in interactions
            if i.status == InteractionStatus.HOLDING
        ]
        primary_tool = None
        if holding_interactions:
            best = max(holding_interactions, key=lambda x: x.confidence)
            primary_tool = best.object_label

        # Determine primary workpiece (closest to hand center)
        primary_workpiece = None
        if hands and detections.workpieces:
            hand_center = (
                sum(h.bbox[0] + h.bbox[2] for h in hands) / (2 * len(hands)),
                sum(h.bbox[1] + h.bbox[3] for h in hands) / (2 * len(hands)),
            )
            closest_wp = min(
                detections.workpieces,
                key=lambda wp: self._distance(hand_center, wp.center)
            )
            primary_workpiece = closest_wp.label

        self.frame_count += 1

        return FrameAnalysis(
            timestamp=timestamp,
            frame_index=self.frame_count,
            hands=hands,
            tools=detections.tools,
            workpieces=detections.workpieces,
            interactions=interactions,
            primary_tool=primary_tool,
            primary_workpiece=primary_workpiece,
        )

    def _distance(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _point_near_bbox(
        self,
        point: Tuple[float, float],
        bbox: Tuple[int, int, int, int],
        threshold: float,
    ) -> bool:
        """Check if point is inside or near bounding box."""
        x, y = point
        x1, y1, x2, y2 = bbox

        # Check if inside
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True

        # Check distance to nearest edge
        nearest_x = max(x1, min(x, x2))
        nearest_y = max(y1, min(y, y2))
        distance = self._distance((x, y), (nearest_x, nearest_y))

        return distance < threshold

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

        # Draw hands
        annotated = self.hand_detector.draw_landmarks(
            annotated, analysis.hands, draw_bbox=True
        )

        # Draw tools
        TOOL_COLOR = (0, 165, 255)  # Orange
        for tool in analysis.tools:
            x1, y1, x2, y2 = tool.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), TOOL_COLOR, 2)
            cv2.putText(
                annotated,
                f"{tool.label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TOOL_COLOR,
                2,
            )

        # Draw workpieces
        WORKPIECE_COLOR = (255, 165, 0)  # Blue
        for wp in analysis.workpieces:
            x1, y1, x2, y2 = wp.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), WORKPIECE_COLOR, 1)

        # Draw interaction lines
        for interaction in analysis.interactions:
            if interaction.status == InteractionStatus.HOLDING:
                color = (0, 255, 0)  # Green
                thickness = 3
            elif interaction.status == InteractionStatus.REACHING:
                color = (0, 255, 255)  # Yellow
                thickness = 2
            else:
                color = (128, 128, 128)  # Gray
                thickness = 1

            # Find the hand
            hand = next(
                (h for h in analysis.hands if h.hand_id == interaction.hand_id),
                None
            )
            if hand:
                # Draw line from hand center to object center
                hand_center = (
                    (hand.bbox[0] + hand.bbox[2]) // 2,
                    (hand.bbox[1] + hand.bbox[3]) // 2,
                )
                obj_center = (
                    (interaction.object_bbox[0] + interaction.object_bbox[2]) // 2,
                    (interaction.object_bbox[1] + interaction.object_bbox[3]) // 2,
                )
                cv2.line(annotated, hand_center, obj_center, color, thickness)

        # Draw status overlay
        status_text = []
        if analysis.primary_tool:
            status_text.append(f"Tool: {analysis.primary_tool}")
        if analysis.primary_workpiece:
            status_text.append(f"Working on: {analysis.primary_workpiece}")
        if analysis.has_active_interaction():
            status_text.append("Status: ACTIVE")
        else:
            status_text.append("Status: IDLE")

        y_offset = 30
        for text in status_text:
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


# Need cv2 for drawing
import cv2


# Quick test
if __name__ == "__main__":
    detector = HOIDetector()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        analysis = detector.analyze_frame(frame, timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        annotated = detector.draw_analysis(frame, analysis)

        cv2.imshow("HOI Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
