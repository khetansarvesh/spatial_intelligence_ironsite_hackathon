#!/usr/bin/env python3
"""
Final Hand Detection Test
Only shows filtered glove detections (no fingers, no removed gloves).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from tqdm import tqdm

from perception.hand_detector import HandDetector, filter_gloves_by_fingers


def test_final_hand_detection(
    video_path: str,
    output_video_path: str = None,
    output_frames_dir: str = None,
    frame_skip: int = 5,
    max_frames: int = None,
    save_sample_frames: int = 10,
    min_avg_iou: float = 0.05,
):
    """Final hand detection test - only shows kept gloves."""
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    frames_to_process = total_frames // frame_skip
    if max_frames:
        frames_to_process = min(frames_to_process, max_frames)
    print(f"Processing {frames_to_process} frames (every {frame_skip}th)")

    print("\nInitializing detector...")
    detector = HandDetector(confidence_threshold=0.25)

    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps / frame_skip, (width, height))
        print(f"Output video: {output_video_path}")

    if output_frames_dir:
        Path(output_frames_dir).mkdir(parents=True, exist_ok=True)

    stats = {"frames": 0, "with_hands": 0, "total_hands": 0}
    processing_times = []

    sample_indices = set()
    if save_sample_frames > 0 and output_frames_dir:
        step = max(1, frames_to_process // save_sample_frames)
        sample_indices = set(range(0, frames_to_process, step)[:save_sample_frames])

    print("\nProcessing...")
    frame_idx = 0
    processed_idx = 0

    pbar = tqdm(total=frames_to_process, desc="Final Hand Detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        if max_frames and processed_idx >= max_frames:
            break

        start = time.time()

        # Detect gloves and fingers
        gloves = detector.detect(frame)
        fingers = detector.detect_fingers(frame)

        # Filter gloves using fingers
        filtered_gloves, _ = filter_gloves_by_fingers(gloves, fingers, min_avg_iou=min_avg_iou)

        processing_times.append(time.time() - start)

        # Update stats
        stats["frames"] += 1
        if filtered_gloves:
            stats["with_hands"] += 1
            stats["total_hands"] += len(filtered_gloves)

        # Draw only kept gloves
        annotated = draw_final_detections(frame, filtered_gloves)

        if writer:
            writer.write(annotated)

        if processed_idx in sample_indices and output_frames_dir:
            cv2.imwrite(
                str(Path(output_frames_dir) / f"frame_{frame_idx:06d}.jpg"),
                annotated
            )

        frame_idx += 1
        processed_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer:
        writer.release()
    detector.close()

    # Print results
    print("\n" + "="*50)
    print("FINAL HAND DETECTION RESULTS")
    print("="*50)
    n = stats["frames"]
    print(f"Frames processed:      {n}")
    print(f"Frames with hands:     {stats['with_hands']} ({100*stats['with_hands']/n:.1f}%)")
    print(f"Total hand detections: {stats['total_hands']}")
    print(f"Avg hands per frame:   {stats['total_hands']/n:.2f}")

    avg_time = np.mean(processing_times) * 1000
    print(f"\nProcessing: {avg_time:.0f}ms/frame ({1000/avg_time:.2f} FPS)")


def draw_final_detections(frame: np.ndarray, hands) -> np.ndarray:
    """Draw only the kept hand/glove detections."""
    annotated = frame.copy()

    HAND_COLOR = (0, 255, 0)  # Green

    for hand in hands:
        x1, y1, x2, y2 = hand.bbox

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), HAND_COLOR, 2)

        # Draw label
        label = f"hand ({hand.confidence:.2f})"
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HAND_COLOR, 2)

        # Draw center point
        cv2.circle(annotated, hand.center, 4, HAND_COLOR, -1)

    # Status overlay
    status = f"Hands: {len(hands)}"
    cv2.putText(annotated, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    video_path = project_root / "data" / "videos" / "07_production_mp.mp4"
    output_video = project_root / "outputs" / "final_hand_detection.mp4"
    output_frames = project_root / "outputs" / "final_hand_detection_frames"

    test_final_hand_detection(
        video_path=str(video_path),
        output_video_path=str(output_video),
        output_frames_dir=str(output_frames),
        frame_skip=5,
        max_frames=100,
        save_sample_frames=10,
        min_avg_iou=0.05,
    )
