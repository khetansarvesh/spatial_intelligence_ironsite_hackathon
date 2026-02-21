"""
Glove/Hand Detection Module (P1)
Uses Grounding DINO for zero-shot glove detection in egocentric construction footage.

Optimized for workers wearing gloves (MediaPipe fails on gloved hands).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# Detection prompts for construction hands/gloves
DETECTION_PROMPTS = [
    "work glove",
    "safety glove",
    "gloved hand",
    "hand",
    "construction glove",
]

# Detection prompts for fingers
FINGER_PROMPTS = [
    "finger",
    "thumb",
    "index finger",
    "fingertip",
    "hand finger",
]

# Label normalization map - all variations map to canonical labels
LABEL_NORMALIZATION = {
    "glove": "glove",
    "hand": "glove",
    "gloved hand": "glove",
    "work glove": "glove",
    "safety glove": "glove",
    "construction glove": "glove",
    "work": "glove",
    "safety": "glove",
    "construction": "glove",
}


def normalize_label(label: str) -> str:
    """Normalize detection label to canonical form."""
    label_lower = label.lower().strip()

    # Check direct match
    if label_lower in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[label_lower]

    # Check if any key is contained in the label
    for key, normalized in LABEL_NORMALIZATION.items():
        if key in label_lower:
            return normalized

    # Default: return original
    return label_lower


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def apply_nms(detections: list, iou_threshold: float = 0.5) -> list:
    """
    Apply Non-Maximum Suppression to remove overlapping detections.

    Args:
        detections: List of detection dicts with 'bbox', 'confidence', 'label'
        iou_threshold: IoU threshold for considering boxes as overlapping

    Returns:
        Filtered list of detections
    """
    if not detections:
        return []

    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []

    while sorted_dets:
        # Take the highest confidence detection
        best = sorted_dets.pop(0)
        keep.append(best)

        # Remove all detections that overlap significantly with this one
        remaining = []
        for det in sorted_dets:
            iou = compute_iou(best['bbox'], det['bbox'])
            if iou < iou_threshold:
                remaining.append(det)

        sorted_dets = remaining

    return keep


def filter_gloves_by_fingers(
    gloves: list,
    fingers: list,
    min_avg_iou: float = 0.05,
) -> Tuple[list, List[dict]]:
    """
    Filter glove detections to keep only those with sufficient finger overlap.

    A valid glove should have fingers detected inside it. This filters out
    false positive glove detections (e.g., other objects detected as gloves).

    Args:
        gloves: List of HandResult objects (glove detections)
        fingers: List of HandResult objects (finger detections)
        min_avg_iou: Minimum average IOU with fingers to keep a glove

    Returns:
        Tuple of:
            - filtered_gloves: List of gloves that passed the filter
            - glove_finger_info: List of dicts with IOU info for each glove
    """
    if not gloves or not fingers:
        # If no fingers detected, return all gloves (can't filter)
        return gloves, [{"glove_id": g.hand_id, "avg_iou": 0.0, "finger_count": 0} for g in gloves]

    filtered_gloves = []
    glove_finger_info = []

    for glove in gloves:
        # Calculate IOU with each finger
        ious = []
        for finger in fingers:
            iou = compute_iou(glove.bbox, finger.bbox)
            ious.append(iou)

        # Calculate average IOU
        avg_iou = sum(ious) / len(ious) if ious else 0.0

        # Count fingers with non-zero IOU (fingers inside this glove)
        fingers_inside = sum(1 for iou in ious if iou > 0)

        info = {
            "glove_id": glove.hand_id,
            "avg_iou": avg_iou,
            "max_iou": max(ious) if ious else 0.0,
            "finger_count": fingers_inside,
            "all_ious": ious,
        }
        glove_finger_info.append(info)

        # Keep glove if average IOU is above threshold
        if avg_iou >= min_avg_iou:
            filtered_gloves.append(glove)

    # Fallback: if all gloves were filtered out, keep the one with highest avg IOU
    if not filtered_gloves and gloves and glove_finger_info:
        # Find the glove with highest avg_iou
        best_idx = max(range(len(glove_finger_info)), key=lambda i: glove_finger_info[i]["avg_iou"])
        filtered_gloves.append(gloves[best_idx])

    return filtered_gloves, glove_finger_info


@dataclass
class HandResult:
    """Result from glove/hand detection."""
    hand_id: int
    label: str  # Normalized label (usually "glove")
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # Center point of bounding box
    confidence: float
    original_label: str = ""  # Original label from model


class HandDetector:
    """
    Detects gloves/hands in egocentric video frames using Grounding DINO.

    Optimized for construction hardhat camera footage where:
    - Workers wear work gloves (MediaPipe fails)
    - Hands are often close to camera
    - Lighting can vary significantly

    Uses zero-shot object detection for robust glove detection.
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        device: str = None,
        prompts: List[str] = None,
    ):
        """
        Initialize glove detector.

        Args:
            model_id: HuggingFace model ID for Grounding DINO
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for NMS
            device: Device to run on (auto-detected if None)
            prompts: Custom detection prompts (uses defaults if None)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.prompts = prompts or DETECTION_PROMPTS

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        print(f"HandDetector using device: {device}")

        # Load model
        print(f"Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded!")

    def detect(self, frame: np.ndarray) -> List[HandResult]:
        """
        Detect gloves/hands in a frame.

        Args:
            frame: BGR image from OpenCV (np.ndarray)

        Returns:
            List of HandResult objects
        """
        height, width = frame.shape[:2]

        # Convert BGR to RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Create text prompt (Grounding DINO format)
        text = ". ".join(self.prompts) + "."

        # Process inputs
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold,
            target_sizes=[(height, width)],
        )[0]

        # Format results
        detections = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Normalize label
            normalized_label = normalize_label(label)

            detections.append({
                "label": normalized_label,
                "original_label": label,
                "bbox": (x1, y1, x2, y2),
                "confidence": float(score),
                "center": center,
            })

        # Apply NMS to remove overlapping detections
        detections = apply_nms(detections, iou_threshold=self.nms_threshold)

        # Convert to HandResult objects
        hand_results = []
        for idx, det in enumerate(detections):
            hand_results.append(HandResult(
                hand_id=idx,
                label=det["label"],
                bbox=det["bbox"],
                center=det["center"],
                confidence=det["confidence"],
                original_label=det["original_label"],
            ))

        return hand_results

    def draw_detections(
        self,
        frame: np.ndarray,
        hand_results: List[HandResult],
        draw_center: bool = True,
    ) -> np.ndarray:
        """
        Draw detection boxes on frame for visualization.

        Args:
            frame: BGR image
            hand_results: List of detected hands/gloves
            draw_center: Whether to draw center point

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for hand in hand_results:
            x1, y1, x2, y2 = hand.bbox
            label = hand.label
            conf = hand.confidence

            # Color based on label
            if "glove" in label.lower():
                color = (0, 255, 0)  # Green for gloves
            elif "hand" in label.lower():
                color = (255, 165, 0)  # Orange for hands
            else:
                color = (255, 255, 255)  # White for other

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_text = f"{label} ({conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                annotated, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

            # Draw center point
            if draw_center:
                cv2.circle(annotated, hand.center, 4, color, -1)

        return annotated

    # Alias for backward compatibility
    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_results: List[HandResult],
        draw_bbox: bool = True,
    ) -> np.ndarray:
        """Alias for draw_detections (backward compatibility)."""
        return self.draw_detections(frame, hand_results, draw_center=draw_bbox)

    def detect_fingers(self, frame: np.ndarray) -> List[HandResult]:
        """
        Detect fingers in a frame.

        Args:
            frame: BGR image from OpenCV (np.ndarray)

        Returns:
            List of HandResult objects for fingers
        """
        height, width = frame.shape[:2]

        # Convert BGR to RGB PIL Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Create text prompt for fingers
        text = ". ".join(FINGER_PROMPTS) + "."

        # Process inputs
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold,
            target_sizes=[(height, width)],
        )[0]

        # Format results
        finger_results = []
        for idx, (box, score, label) in enumerate(zip(
            results["boxes"], results["scores"], results["labels"]
        )):
            x1, y1, x2, y2 = map(int, box.tolist())
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            finger_results.append(HandResult(
                hand_id=idx,
                label="finger",
                bbox=(x1, y1, x2, y2),
                center=center,
                confidence=float(score),
                original_label=label,
            ))

        return finger_results

    def _detect_batch_internal(
        self,
        frames: List[np.ndarray],
        prompts: List[str],
        normalize_labels: bool = True,
    ) -> List[List[HandResult]]:
        """
        Internal batch detection method.

        Args:
            frames: List of BGR images from OpenCV
            prompts: Detection prompts to use
            normalize_labels: Whether to normalize labels

        Returns:
            List of List[HandResult], one list per frame
        """
        if not frames:
            return []

        # Convert all frames to PIL images
        images = []
        sizes = []
        for frame in frames:
            h, w = frame.shape[:2]
            sizes.append((h, w))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))

        # Create text prompt
        text = ". ".join(prompts) + "."

        # Process inputs as batch
        inputs = self.processor(
            images=images,
            text=[text] * len(images),
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference on batch
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process for each frame
        all_results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold,
            target_sizes=sizes,
        )

        # Format results for each frame
        batch_results = []
        for frame_idx, results in enumerate(all_results):
            detections = []
            for box, score, label in zip(
                results["boxes"], results["scores"], results["labels"]
            ):
                x1, y1, x2, y2 = map(int, box.tolist())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                if normalize_labels:
                    normalized_label = normalize_label(label)
                else:
                    normalized_label = "finger"

                detections.append({
                    "label": normalized_label,
                    "original_label": label,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(score),
                    "center": center,
                })

            # Apply NMS if normalizing (gloves)
            if normalize_labels:
                detections = apply_nms(detections, iou_threshold=self.nms_threshold)

            # Convert to HandResult objects
            hand_results = []
            for idx, det in enumerate(detections):
                hand_results.append(HandResult(
                    hand_id=idx,
                    label=det["label"],
                    bbox=det["bbox"],
                    center=det["center"],
                    confidence=det["confidence"],
                    original_label=det["original_label"],
                ))

            batch_results.append(hand_results)

        return batch_results

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[HandResult]]:
        """
        Detect gloves/hands in multiple frames (batch inference).

        Args:
            frames: List of BGR images from OpenCV

        Returns:
            List of List[HandResult], one list per frame
        """
        return self._detect_batch_internal(frames, self.prompts, normalize_labels=True)

    def detect_fingers_batch(self, frames: List[np.ndarray]) -> List[List[HandResult]]:
        """
        Detect fingers in multiple frames (batch inference).

        Args:
            frames: List of BGR images from OpenCV

        Returns:
            List of List[HandResult], one list per frame
        """
        return self._detect_batch_internal(frames, FINGER_PROMPTS, normalize_labels=False)

    def detect_with_fingers_batch(
        self,
        frames: List[np.ndarray],
        min_avg_iou: float = 0.05,
    ) -> List[List[HandResult]]:
        """
        Detect and filter gloves using fingers for multiple frames.

        This is the most efficient method for processing multiple frames:
        - Batch inference for gloves
        - Batch inference for fingers
        - Filter each frame's gloves using its fingers

        Args:
            frames: List of BGR images from OpenCV
            min_avg_iou: Minimum average IOU with fingers to keep a glove

        Returns:
            List of filtered hand results, one list per frame
        """
        # Batch detect gloves
        all_gloves = self.detect_batch(frames)

        # Batch detect fingers
        all_fingers = self.detect_fingers_batch(frames)

        # Filter each frame
        filtered_results = []
        for gloves, fingers in zip(all_gloves, all_fingers):
            filtered, _ = filter_gloves_by_fingers(gloves, fingers, min_avg_iou)
            filtered_results.append(filtered)

        return filtered_results

    def close(self):
        """Release resources."""
        # Clear model from memory if needed
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Quick test
if __name__ == "__main__":
    from pathlib import Path

    # Test on sample video
    project_root = Path(__file__).parent.parent.parent
    video_path = project_root / "data" / "videos" / "07_production_mp.mp4"

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        print("Testing with webcam instead...")
        cap = cv2.VideoCapture(0)
    else:
        print(f"Testing on: {video_path}")
        cap = cv2.VideoCapture(str(video_path))

    detector = HandDetector()

    frame_count = 0
    while cap.isOpened() and frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            break

        hands = detector.detect(frame)
        annotated = detector.draw_detections(frame, hands)

        print(f"Frame {frame_count}: {len(hands)} detections")
        for h in hands:
            print(f"  - {h.label} ({h.confidence:.2f}) at {h.bbox}")

        # Save sample frame
        if frame_count == 0:
            output_path = project_root / "outputs" / "hand_detector_test.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)
            print(f"Saved sample to: {output_path}")

        frame_count += 1

    cap.release()
    detector.close()
    print("Done!")
