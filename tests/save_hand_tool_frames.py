#!/usr/bin/env python3
"""
Save 20 frames with hand and tool detections.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from perception.hand_detector import HandDetector, filter_gloves_by_fingers
from perception.tool_detector import ToolDetector, DetectorBackend


def draw_detections(frame, hands, tools):
    """Draw hands and tools on frame (no filtering)."""
    annotated = frame.copy()

    HAND_COLOR = (0, 255, 0)      # Green
    TOOL_COLOR = (0, 165, 255)    # Orange

    # Draw hands
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), HAND_COLOR, 2)
        cv2.putText(annotated, f"hand ({hand.confidence:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HAND_COLOR, 2)

    # Draw all tools
    for tool in tools:
        x1, y1, x2, y2 = tool.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), TOOL_COLOR, 2)
        label = f"{tool.label} ({tool.confidence:.2f})"
        cv2.putText(annotated, label, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TOOL_COLOR, 2)

    # Status
    status = f"Hands: {len(hands)} | Tools: {len(tools)}"
    cv2.putText(annotated, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


def main():
    project_root = Path(__file__).parent.parent
    video_path = project_root / "data" / "videos" / "07_production_mp.mp4"
    output_dir = project_root / "outputs" / "hand_tool_20_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Select 20 evenly spaced frames
    num_frames = 20
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    print(f"Selected frame indices: {frame_indices}")

    # Load frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append((idx, frame))
    cap.release()
    print(f"Loaded {len(frames)} frames")

    # Initialize detectors
    print("\nInitializing detectors...")
    hand_detector = HandDetector(confidence_threshold=0.25)
    tool_detector = ToolDetector(backend=DetectorBackend.GROUNDING_DINO, confidence_threshold=0.25)

    # Process and save each frame
    print("\nProcessing frames...")
    for i, (frame_idx, frame) in enumerate(frames):
        print(f"  Frame {i+1}/{len(frames)} (index {frame_idx})...", end=" ")

        # Detect hands with finger filtering
        gloves = hand_detector.detect(frame)
        fingers = hand_detector.detect_fingers(frame)
        hands, _ = filter_gloves_by_fingers(gloves, fingers, min_avg_iou=0.05)

        # Detect all tools (no filtering)
        tool_result = tool_detector.detect(frame)
        tools = tool_result.tools

        # Draw and save (no filtering)
        annotated = draw_detections(frame, hands, tools)
        output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), annotated)

        print(f"hands={len(hands)}, tools={len(tools)}")

    hand_detector.close()

    print(f"\nSaved {len(frames)} frames to: {output_dir}")


if __name__ == "__main__":
    main()
