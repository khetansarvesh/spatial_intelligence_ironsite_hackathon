"""
Hand Detection Module (P1)
Uses MediaPipe Hands for egocentric hand detection.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    import mediapipe as mp
    import cv2
except ImportError:
    raise ImportError("Please install mediapipe and opencv-python")


@dataclass
class HandResult:
    """Result from hand detection."""
    hand_id: int
    side: str  # "left" | "right"
    landmarks: List[Tuple[float, float, float]]  # 21 points (x, y, z)
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    fingertip_positions: dict  # {"thumb": (x,y), "index": (x,y), ...}
    confidence: float


class HandDetector:
    """
    Detects hands in egocentric video frames using MediaPipe.

    Optimized for construction hardhat camera footage where:
    - Hands are often close to camera
    - Workers may wear gloves
    - Lighting can vary significantly
    """

    # MediaPipe hand landmark indices
    FINGERTIP_INDICES = {
        "thumb": 4,
        "index": 8,
        "middle": 12,
        "ring": 16,
        "pinky": 20,
    }

    WRIST_INDEX = 0

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize hand detector.

        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect(self, frame: np.ndarray) -> List[HandResult]:
        """
        Detect hands in a frame.

        Args:
            frame: BGR image from OpenCV (np.ndarray)

        Returns:
            List of HandResult objects
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]

        # Process frame
        results = self.hands.process(rgb_frame)

        hand_results = []

        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Get hand side (Note: MediaPipe returns mirrored, so we flip)
                side = handedness.classification[0].label.lower()
                confidence = handedness.classification[0].score

                # Extract landmarks as list of (x, y, z) tuples
                landmarks = [
                    (lm.x * width, lm.y * height, lm.z)
                    for lm in hand_landmarks.landmark
                ]

                # Calculate bounding box
                bbox = self._calculate_bbox(landmarks, width, height)

                # Extract fingertip positions
                fingertips = self._get_fingertip_positions(landmarks)

                hand_results.append(HandResult(
                    hand_id=idx,
                    side=side,
                    landmarks=landmarks,
                    bbox=bbox,
                    fingertip_positions=fingertips,
                    confidence=confidence,
                ))

        return hand_results

    def _calculate_bbox(
        self,
        landmarks: List[Tuple[float, float, float]],
        width: int,
        height: int,
        padding: float = 0.1,
    ) -> Tuple[int, int, int, int]:
        """Calculate bounding box from landmarks with padding."""
        x_coords = [lm[0] for lm in landmarks]
        y_coords = [lm[1] for lm in landmarks]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add padding
        pad_x = (x_max - x_min) * padding
        pad_y = (y_max - y_min) * padding

        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(width, int(x_max + pad_x))
        y2 = min(height, int(y_max + pad_y))

        return (x1, y1, x2, y2)

    def _get_fingertip_positions(
        self,
        landmarks: List[Tuple[float, float, float]],
    ) -> dict:
        """Extract fingertip positions from landmarks."""
        return {
            name: (landmarks[idx][0], landmarks[idx][1])
            for name, idx in self.FINGERTIP_INDICES.items()
        }

    def get_wrist_position(self, hand_result: HandResult) -> Tuple[float, float]:
        """Get wrist position from hand result."""
        return (
            hand_result.landmarks[self.WRIST_INDEX][0],
            hand_result.landmarks[self.WRIST_INDEX][1],
        )

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_results: List[HandResult],
        draw_bbox: bool = True,
    ) -> np.ndarray:
        """
        Draw hand landmarks on frame for visualization.

        Args:
            frame: BGR image
            hand_results: List of detected hands
            draw_bbox: Whether to draw bounding box

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for hand in hand_results:
            # Draw landmarks
            for i, (x, y, z) in enumerate(hand.landmarks):
                color = (0, 255, 0) if hand.side == "right" else (255, 0, 0)
                cv2.circle(annotated, (int(x), int(y)), 4, color, -1)

            # Draw connections (simplified)
            for finger in self.FINGERTIP_INDICES.values():
                # Draw line from wrist to fingertip (simplified skeleton)
                wrist = hand.landmarks[self.WRIST_INDEX]
                tip = hand.landmarks[finger]
                cv2.line(
                    annotated,
                    (int(wrist[0]), int(wrist[1])),
                    (int(tip[0]), int(tip[1])),
                    (200, 200, 200),
                    1,
                )

            # Draw bounding box
            if draw_bbox:
                x1, y1, x2, y2 = hand.bbox
                color = (0, 255, 0) if hand.side == "right" else (255, 0, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    f"{hand.side} ({hand.confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        return annotated

    def close(self):
        """Release resources."""
        self.hands.close()


# Quick test
if __name__ == "__main__":
    import cv2

    detector = HandDetector()

    # Test with webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hands = detector.detect(frame)
        annotated = detector.draw_landmarks(frame, hands)

        cv2.imshow("Hand Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
