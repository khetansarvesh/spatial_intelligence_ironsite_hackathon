#!/usr/bin/env python3
"""
Save 20 frames with tool detections only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from perception.tool_detector import ToolDetector, DetectorBackend


def main():
    project_root = Path(__file__).parent.parent
    video_path = project_root / "data" / "videos" / "07_production_mp.mp4"
    output_dir = project_root / "outputs" / "tool_20_frames"
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

    # Initialize tool detector
    print("\nInitializing tool detector...")
    tool_detector = ToolDetector(backend=DetectorBackend.GROUNDING_DINO, confidence_threshold=0.25)

    # Process and save each frame
    print("\nProcessing frames...")
    for i, (frame_idx, frame) in enumerate(frames):
        print(f"  Frame {i+1}/{len(frames)} (index {frame_idx})...", end=" ")

        # Detect tools only (skip workpieces)
        result = tool_detector.detect(frame, tools_only=True)
        tools = result.tools

        # Draw detections
        annotated = tool_detector.draw_detections(frame, result, show_all=True)

        # Save
        output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(output_path), annotated)

        print(f"tools={len(tools)}")

    print(f"\nSaved {len(frames)} frames to: {output_dir}")


if __name__ == "__main__":
    main()
