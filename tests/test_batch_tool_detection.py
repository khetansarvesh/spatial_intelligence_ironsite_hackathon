#!/usr/bin/env python3
"""
Test batch vs sequential tool detection performance.
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from perception.tool_detector import ToolDetector, DetectorBackend


def main():
    project_root = Path(__file__).parent.parent
    video_path = project_root / "data" / "videos" / "07_production_mp.mp4"

    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Select 10 evenly spaced frames for testing
    num_frames = 10
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    print(f"Selected frame indices: {frame_indices}")

    # Load frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames\n")

    # Initialize detector
    print("Initializing tool detector...")
    detector = ToolDetector(backend=DetectorBackend.GROUNDING_DINO, confidence_threshold=0.25)

    # Test sequential detection
    print("\n=== Sequential Detection ===")
    start = time.time()
    seq_results = []
    for frame in frames:
        result = detector.detect(frame, tools_only=True)
        seq_results.append(result)
    seq_time = time.time() - start
    seq_tools = [len(r.tools) for r in seq_results]
    print(f"Time: {seq_time:.2f}s")
    print(f"Tools per frame: {seq_tools}")
    print(f"Avg time per frame: {seq_time / len(frames):.3f}s")

    # Test batch detection
    print("\n=== Batch Detection ===")
    start = time.time()
    batch_results = detector.detect_batch(frames, tools_only=True)
    batch_time = time.time() - start
    batch_tools = [len(r.tools) for r in batch_results]
    print(f"Time: {batch_time:.2f}s")
    print(f"Tools per frame: {batch_tools}")
    print(f"Avg time per frame: {batch_time / len(frames):.3f}s")

    # Compare
    print("\n=== Comparison ===")
    print(f"Sequential: {seq_time:.2f}s")
    print(f"Batch: {batch_time:.2f}s")
    speedup = seq_time / batch_time if batch_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")

    # Verify results match
    print("\n=== Result Verification ===")
    match = seq_tools == batch_tools
    print(f"Tool counts match: {match}")
    if not match:
        print(f"Sequential: {seq_tools}")
        print(f"Batch: {batch_tools}")

    print("\nDone!")


if __name__ == "__main__":
    main()
