#!/usr/bin/env python3
"""
Test Batch Hand Detection
Compares sequential vs batch inference performance.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from tqdm import tqdm

from perception.hand_detector import HandDetector, filter_gloves_by_fingers


def load_frames(video_path: str, frame_skip: int = 5, max_frames: int = None):
    """Load frames from video into memory."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


def test_sequential(detector, frames):
    """Test sequential (one-by-one) detection."""
    print("\n--- Sequential Detection ---")
    start = time.time()

    all_results = []
    for frame in tqdm(frames, desc="Sequential"):
        gloves = detector.detect(frame)
        fingers = detector.detect_fingers(frame)
        filtered, _ = filter_gloves_by_fingers(gloves, fingers, min_avg_iou=0.05)
        all_results.append(filtered)

    elapsed = time.time() - start
    total_detections = sum(len(r) for r in all_results)

    print(f"Time: {elapsed:.1f}s")
    print(f"Frames: {len(frames)}")
    print(f"Time per frame: {elapsed/len(frames)*1000:.0f}ms")
    print(f"FPS: {len(frames)/elapsed:.2f}")
    print(f"Total detections: {total_detections}")

    return all_results, elapsed


def test_batch(detector, frames, batch_size: int = 4):
    """Test batch detection."""
    print(f"\n--- Batch Detection (batch_size={batch_size}) ---")
    start = time.time()

    all_results = []

    # Process in batches
    for i in tqdm(range(0, len(frames), batch_size), desc=f"Batch({batch_size})"):
        batch = frames[i:i + batch_size]
        batch_results = detector.detect_with_fingers_batch(batch, min_avg_iou=0.05)
        all_results.extend(batch_results)

    elapsed = time.time() - start
    total_detections = sum(len(r) for r in all_results)

    print(f"Time: {elapsed:.1f}s")
    print(f"Frames: {len(frames)}")
    print(f"Time per frame: {elapsed/len(frames)*1000:.0f}ms")
    print(f"FPS: {len(frames)/elapsed:.2f}")
    print(f"Total detections: {total_detections}")

    return all_results, elapsed


def main():
    project_root = Path(__file__).parent.parent
    video_path = project_root / "data" / "videos" / "07_production_mp.mp4"

    print(f"Loading video: {video_path}")

    # Load frames into memory first (to isolate detection time)
    print("Loading frames into memory...")
    frames = load_frames(str(video_path), frame_skip=5, max_frames=20)
    print(f"Loaded {len(frames)} frames")

    # Initialize detector
    print("\nInitializing detector...")
    detector = HandDetector(confidence_threshold=0.25)

    # Test sequential
    seq_results, seq_time = test_sequential(detector, frames)

    # Test different batch sizes
    batch_results = {}
    for batch_size in [2, 4, 8]:
        try:
            results, elapsed = test_batch(detector, frames, batch_size=batch_size)
            batch_results[batch_size] = (results, elapsed)
        except Exception as e:
            print(f"Batch size {batch_size} failed: {e}")

    # Summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    print(f"{'Method':<20} {'Time (s)':<12} {'ms/frame':<12} {'Speedup':<10}")
    print("-"*50)
    print(f"{'Sequential':<20} {seq_time:<12.1f} {seq_time/len(frames)*1000:<12.0f} {'1.0x':<10}")

    for batch_size, (results, elapsed) in batch_results.items():
        speedup = seq_time / elapsed
        print(f"{'Batch(' + str(batch_size) + ')':<20} {elapsed:<12.1f} {elapsed/len(frames)*1000:<12.0f} {speedup:<10.1f}x")

    detector.close()


if __name__ == "__main__":
    main()
