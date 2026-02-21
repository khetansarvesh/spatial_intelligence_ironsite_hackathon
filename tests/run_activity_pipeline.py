"""
Run full pipeline: HOI Detection → Motion Analysis → Activity Classification
Outputs annotated video with activity labels.

OPTIMIZED: Parallel hand + tool detection with batch processing.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.perception.hand_detector import HandDetector, filter_gloves_by_fingers, compute_iou
from src.perception.tool_detector import ToolDetector, DetectorBackend
from src.perception.hoi_detector import InteractionStatus, Interaction, FrameAnalysis
from src.temporal.motion_analyzer import MotionAnalyzer, MotionType, MotionResult
from src.temporal.activity_classifier import ActivityClassifier, ActivityState

import math

# Configuration
BATCH_SIZE = 8  # Number of frames to process in parallel
MOTION_WINDOW = 5  # Frames needed for motion analysis


def compute_hoi(hands, tools, frame_height, frame_width, working_distance=100):
    """Compute HOI interactions between hands and tools."""
    scale = min(frame_width, frame_height) / 640
    working_dist = working_distance * scale

    interactions = []

    for hand in hands:
        hand_center = hand.center

        for tool in tools:
            iou = compute_iou(hand.bbox, tool.bbox)
            dx = hand_center[0] - tool.center[0]
            dy = hand_center[1] - tool.center[1]
            distance = math.sqrt(dx*dx + dy*dy)

            if iou > 0:
                status = InteractionStatus.WORKING
                confidence = min(1.0, iou * 5)
            elif distance < working_dist:
                status = InteractionStatus.WORKING
                confidence = 1 - (distance / working_dist)
            else:
                continue

            interactions.append(Interaction(
                hand_id=hand.hand_id,
                object_label=tool.label,
                object_bbox=tool.bbox,
                status=status,
                confidence=max(0, min(1, confidence)),
                distance_pixels=distance,
                iou=iou,
            ))

    working_interactions = [i for i in interactions if i.status == InteractionStatus.WORKING]
    primary_tool = None
    if working_interactions:
        best = max(working_interactions, key=lambda x: x.confidence)
        primary_tool = best.object_label

    return interactions, primary_tool


def draw_activity_frame(
    frame,
    hands,
    tools,
    interactions,
    activity_state,
    activity_confidence,
    motion_type,
    motion_magnitude,
):
    """Draw all annotations on frame."""
    annotated = frame.copy()
    height, width = frame.shape[:2]

    HAND_COLOR = (0, 255, 0)      # Green
    TOOL_COLOR = (0, 165, 255)    # Orange
    WORKING_COLOR = (0, 0, 255)   # Red

    # Activity state colors
    ACTIVITY_COLORS = {
        ActivityState.ACTIVE_TOOL_USE: (0, 255, 0),    # Green
        ActivityState.PRECISION_WORK: (0, 255, 128),   # Light green
        ActivityState.MATERIAL_HANDLING: (255, 255, 0), # Cyan
        ActivityState.SETUP_CLEANUP: (255, 165, 0),    # Blue
        ActivityState.SEARCHING: (0, 165, 255),        # Orange
        ActivityState.TRAVELING: (255, 0, 255),        # Magenta
        ActivityState.IDLE: (128, 128, 128),           # Gray
    }

    # Draw hands
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), HAND_COLOR, 2)
        cv2.circle(annotated, hand.center, 4, HAND_COLOR, -1)

    # Draw tools
    for tool in tools:
        x1, y1, x2, y2 = tool.bbox
        tool_is_working = any(
            i.object_bbox == tool.bbox and i.status == InteractionStatus.WORKING
            for i in interactions
        )
        color = WORKING_COLOR if tool_is_working else TOOL_COLOR

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{tool.label}",
            (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # Draw interaction lines
    for interaction in interactions:
        if interaction.status == InteractionStatus.WORKING:
            hand = next((h for h in hands if h.hand_id == interaction.hand_id), None)
            if hand:
                obj_center = (
                    (interaction.object_bbox[0] + interaction.object_bbox[2]) // 2,
                    (interaction.object_bbox[1] + interaction.object_bbox[3]) // 2,
                )
                cv2.line(annotated, hand.center, obj_center, WORKING_COLOR, 2)

    # Draw activity status bar at top
    activity_color = ACTIVITY_COLORS.get(activity_state, (255, 255, 255))

    # Semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

    # Activity state
    cv2.putText(
        annotated,
        f"Activity: {activity_state.value}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        activity_color,
        2,
    )

    # Confidence bar
    bar_width = int(150 * activity_confidence)
    cv2.rectangle(annotated, (10, 40), (10 + bar_width, 55), activity_color, -1)
    cv2.rectangle(annotated, (10, 40), (160, 55), (255, 255, 255), 1)
    cv2.putText(
        annotated,
        f"{activity_confidence:.0%}",
        (170, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # Motion info
    cv2.putText(
        annotated,
        f"Motion: {motion_type.value} ({motion_magnitude:.1f})",
        (300, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Detection counts
    cv2.putText(
        annotated,
        f"Hands: {len(hands)} | Tools: {len(tools)}",
        (300, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return annotated


def detect_hands_batch(hand_detector, frames):
    """Detect hands in batch with finger filtering."""
    gloves_batch = hand_detector.detect_batch(frames)
    fingers_batch = hand_detector.detect_fingers_batch(frames)

    hands_batch = []
    for gloves, fingers in zip(gloves_batch, fingers_batch):
        hands, _ = filter_gloves_by_fingers(gloves, fingers, min_avg_iou=0.05)
        hands_batch.append(hands)

    return hands_batch


def detect_tools_batch(tool_detector, frames):
    """Detect tools in batch."""
    results = tool_detector.detect_batch(frames, tools_only=True)
    return [r.tools for r in results]


def run_pipeline(
    video_path: str,
    output_video_path: str,
    max_frames: int = None,
):
    """Run full activity detection pipeline on video with parallel processing."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Processing mode: PARALLEL (hand + tool detection)")

    # Initialize detectors
    print("\nInitializing detectors...")
    hand_detector = HandDetector(confidence_threshold=0.25)
    tool_detector = ToolDetector(
        backend=DetectorBackend.GROUNDING_DINO,
        confidence_threshold=0.25,
    )
    motion_analyzer = MotionAnalyzer(window_size=10, sample_rate=fps)
    activity_classifier = ActivityClassifier(smoothing_window=3)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Frame buffer for motion analysis (maintains history across batches)
    motion_frame_buffer = []

    # Activity stats
    activity_counts = {state: 0 for state in ActivityState}

    # Thread pool for parallel detection
    executor = ThreadPoolExecutor(max_workers=2)

    print("\nProcessing with parallel batching...")
    start_time = time.time()
    frame_idx = 0

    with tqdm(total=total_frames, desc="Processing") as pbar:
        while frame_idx < total_frames:
            # Load batch of frames
            batch_frames = []
            batch_indices = []

            for _ in range(BATCH_SIZE):
                if frame_idx >= total_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                frame_idx += 1

            if not batch_frames:
                break

            # === PARALLEL DETECTION ===
            # Submit hand and tool detection to run in parallel
            hands_future = executor.submit(detect_hands_batch, hand_detector, batch_frames)
            tools_future = executor.submit(detect_tools_batch, tool_detector, batch_frames)

            # While waiting, run motion analysis sequentially (preserves history)
            motion_results = []
            for frame in batch_frames:
                motion_frame_buffer.append(frame.copy())
                if len(motion_frame_buffer) > MOTION_WINDOW:
                    motion_frame_buffer.pop(0)

                if len(motion_frame_buffer) >= 2:
                    motion_result = motion_analyzer.analyze(motion_frame_buffer)
                else:
                    motion_result = MotionResult(
                        motion_type=MotionType.UNKNOWN,
                        confidence=0.0,
                        magnitude=0.0,
                        direction=None,
                        frequency=None,
                        metadata={},
                    )
                motion_results.append(motion_result)

            # Wait for parallel detection to complete
            hands_batch = hands_future.result()
            tools_batch = tools_future.result()

            # === PROCESS EACH FRAME IN BATCH ===
            for i, frame in enumerate(batch_frames):
                idx = batch_indices[i]
                timestamp = idx / fps
                hands = hands_batch[i]
                tools = tools_batch[i]
                motion_result = motion_results[i]

                # Compute HOI
                interactions, primary_tool = compute_hoi(hands, tools, height, width)

                # Create FrameAnalysis for activity classifier
                frame_analysis = FrameAnalysis(
                    timestamp=timestamp,
                    frame_index=idx,
                    hands=hands,
                    tools=tools,
                    interactions=interactions,
                    primary_tool=primary_tool,
                )

                # Activity classification
                activity_state, activity_confidence = activity_classifier.classify_frame(
                    frame_analysis, motion_result
                )

                # Track stats
                activity_counts[activity_state] += 1

                # Draw annotations
                annotated = draw_activity_frame(
                    frame=frame,
                    hands=hands,
                    tools=tools,
                    interactions=interactions,
                    activity_state=activity_state,
                    activity_confidence=activity_confidence,
                    motion_type=motion_result.motion_type,
                    motion_magnitude=motion_result.magnitude,
                )

                out.write(annotated)

            pbar.update(len(batch_frames))

    # Cleanup
    executor.shutdown()
    cap.release()
    out.release()
    hand_detector.close()

    elapsed_time = time.time() - start_time
    frames_processed = sum(activity_counts.values())

    # Print summary
    print(f"\n{'='*50}")
    print("PERFORMANCE")
    print(f"{'='*50}")
    print(f"Total time: {elapsed_time:.1f}s")
    print(f"Frames processed: {frames_processed}")
    print(f"Average: {elapsed_time/frames_processed:.2f}s/frame")
    print(f"Throughput: {frames_processed/elapsed_time:.2f} FPS")

    print(f"\n{'='*50}")
    print("ACTIVITY SUMMARY")
    print(f"{'='*50}")
    for state, count in sorted(activity_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = 100 * count / frames_processed
            print(f"  {state.value}: {count} frames ({pct:.1f}%)")

    print(f"\nOutput video: {output_video_path}")


if __name__ == "__main__":
    video_path = str(project_root / "artifacts" / "clip_3_masonry.mp4")
    output_video = str(project_root / "outputs" / "clip3_activity_pipeline.mp4")

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    run_pipeline(
        video_path=video_path,
        output_video_path=output_video,
        max_frames=None,  # Process all frames
    )
