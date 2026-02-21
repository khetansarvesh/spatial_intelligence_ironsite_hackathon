#!/usr/bin/env python3
"""
Test Tool Detection on Sample Video
Validates Grounding DINO tool detection on egocentric construction footage.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from tqdm import tqdm
from perception.tool_detector import ToolDetector, DetectorBackend


def test_tool_detection(
    video_path: str,
    output_video_path: str = None,
    output_frames_dir: str = None,
    frame_skip: int = 5,
    max_frames: int = None,
    save_sample_frames: int = 10,
    confidence_threshold: float = 0.25,
):
    """
    Test tool detection on a video file.

    Args:
        video_path: Path to input video
        output_video_path: Path to save annotated video (optional)
        output_frames_dir: Directory to save sample frames (optional)
        frame_skip: Process every Nth frame (default: 5)
        max_frames: Maximum frames to process (None = all)
        save_sample_frames: Number of sample frames to save
        confidence_threshold: Detection confidence threshold
    """
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"  Processing every {frame_skip} frame(s)")

    # Limit frames if specified
    frames_to_process = total_frames // frame_skip
    if max_frames:
        frames_to_process = min(frames_to_process, max_frames)
    print(f"  Frames to process: {frames_to_process}")

    # Initialize tool detector with Grounding DINO
    print("\nInitializing Grounding DINO tool detector...")
    detector = ToolDetector(
        backend=DetectorBackend.GROUNDING_DINO,
        confidence_threshold=confidence_threshold,
    )

    print(f"\nTool prompts: {detector.TOOLS}")
    print(f"Workpiece prompts: {detector.WORKPIECES}")

    # Setup video writer if output path specified
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = fps / frame_skip
        writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, (width, height))
        print(f"  Output video: {output_video_path}")

    # Setup frame output directory
    if output_frames_dir:
        Path(output_frames_dir).mkdir(parents=True, exist_ok=True)
        print(f"  Output frames: {output_frames_dir}")

    # Statistics
    frames_processed = 0
    frames_with_tools = 0
    frames_with_workpieces = 0
    total_tools = 0
    total_workpieces = 0
    total_all = 0
    processing_times = []
    tool_counts = {}  # Count by label
    workpiece_counts = {}
    all_counts = {}

    # Frames to save samples from (evenly distributed)
    sample_frame_indices = set()
    if save_sample_frames > 0 and output_frames_dir:
        step = max(1, frames_to_process // save_sample_frames)
        sample_frame_indices = set(range(0, frames_to_process, step)[:save_sample_frames])

    print("\nProcessing video...")
    frame_idx = 0
    processed_idx = 0

    pbar = tqdm(total=frames_to_process, desc="Detecting tools")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Check max frames
        if max_frames and processed_idx >= max_frames:
            break

        # Process frame
        start_time = time.time()
        result = detector.detect(frame)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

        # Update statistics
        frames_processed += 1

        if result.tools:
            frames_with_tools += 1
            total_tools += len(result.tools)
            for det in result.tools:
                tool_counts[det.label] = tool_counts.get(det.label, 0) + 1

        if result.workpieces:
            frames_with_workpieces += 1
            total_workpieces += len(result.workpieces)
            for det in result.workpieces:
                workpiece_counts[det.label] = workpiece_counts.get(det.label, 0) + 1

        total_all += len(result.all_detections)
        for det in result.all_detections:
            all_counts[det.label] = all_counts.get(det.label, 0) + 1

        # Draw annotations (show all detections)
        annotated = detector.draw_detections(frame, result, show_all=True)

        # Add info overlay
        info_text = f"Frame: {frame_idx} | Tools: {len(result.tools)} | Workpieces: {len(result.workpieces)} | All: {len(result.all_detections)}"
        cv2.putText(
            annotated, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Save to video
        if writer:
            writer.write(annotated)

        # Save sample frame
        if processed_idx in sample_frame_indices and output_frames_dir:
            frame_path = Path(output_frames_dir) / f"tool_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), annotated)

        frame_idx += 1
        processed_idx += 1
        pbar.update(1)

    pbar.close()

    # Cleanup
    cap.release()
    if writer:
        writer.release()

    # Print statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Frames processed:       {frames_processed}")
    print(f"Frames with tools:      {frames_with_tools} ({100*frames_with_tools/frames_processed:.1f}%)")
    print(f"Frames with workpieces: {frames_with_workpieces} ({100*frames_with_workpieces/frames_processed:.1f}%)")
    print(f"Total tool detections:      {total_tools}")
    print(f"Total workpiece detections: {total_workpieces}")
    print(f"Total all detections:       {total_all}")

    if tool_counts:
        print(f"\nTools detected:")
        for label, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count}")

    if workpiece_counts:
        print(f"\nWorkpieces detected:")
        for label, count in sorted(workpiece_counts.items(), key=lambda x: -x[1]):
            print(f"  {label}: {count}")

    print(f"\nAll detections by label:")
    for label, count in sorted(all_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {label}: {count}")

    avg_time = np.mean(processing_times) * 1000
    std_time = np.std(processing_times) * 1000
    fps_achieved = 1.0 / np.mean(processing_times)

    print(f"\nProcessing speed:")
    print(f"  Avg time per frame: {avg_time:.0f}ms (+/- {std_time:.0f}ms)")
    print(f"  Effective FPS:      {fps_achieved:.2f}")

    if output_video_path:
        print(f"\nOutput video saved: {output_video_path}")
    if output_frames_dir:
        print(f"Sample frames saved: {output_frames_dir}")


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent
    video_path = project_root / "data" / "videos" / "07_production_mp.mp4"
    output_video = project_root / "outputs" / "tool_detection_test.mp4"
    output_frames = project_root / "outputs" / "tool_frames"

    # Run test
    test_tool_detection(
        video_path=str(video_path),
        output_video_path=str(output_video),
        output_frames_dir=str(output_frames),
        frame_skip=5,
        max_frames=100,
        save_sample_frames=10,
        confidence_threshold=0.25,
    )
