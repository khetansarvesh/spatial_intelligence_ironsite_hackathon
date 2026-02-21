"""
Process video for HOI detection - outputs annotated video and JSON data.

Optimized with:
- Parallel hand and tool detection (using threads)
- Batch inference (8 frames at a time)
"""

import sys
from pathlib import Path
import json
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import math

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.perception.hand_detector import HandDetector, HandResult, filter_gloves_by_fingers, compute_iou
from src.perception.tool_detector import ToolDetector, DetectorBackend
from src.perception.hoi_detector import InteractionStatus, Interaction, FrameAnalysis


BATCH_SIZE = 8
WORKING_DISTANCE = 100  # pixels (will be scaled by frame size)


def compute_hoi(
    hands: list,
    tools: list,
    frame_height: int,
    frame_width: int,
) -> tuple:
    """
    Compute HOI interactions between hands and tools.

    Returns:
        (interactions, primary_tool)
    """
    # Scale distance threshold based on frame size
    scale = min(frame_width, frame_height) / 640
    working_dist = WORKING_DISTANCE * scale

    interactions = []

    for hand in hands:
        hand_center = hand.center

        for tool in tools:
            # Calculate IOU (overlap)
            iou = compute_iou(hand.bbox, tool.bbox)

            # Calculate distance between centers
            dx = hand_center[0] - tool.center[0]
            dy = hand_center[1] - tool.center[1]
            distance = math.sqrt(dx*dx + dy*dy)

            # Determine interaction status
            if iou > 0:
                status = InteractionStatus.WORKING
                confidence = min(1.0, iou * 5)
            elif distance < working_dist:
                status = InteractionStatus.WORKING
                confidence = 1 - (distance / working_dist)
            else:
                continue  # IDLE - no interaction

            interactions.append(Interaction(
                hand_id=hand.hand_id,
                object_label=tool.label,
                object_bbox=tool.bbox,
                status=status,
                confidence=max(0, min(1, confidence)),
                distance_pixels=distance,
                iou=iou,
            ))

    # Determine primary tool
    working_interactions = [i for i in interactions if i.status == InteractionStatus.WORKING]
    primary_tool = None
    if working_interactions:
        best = max(working_interactions, key=lambda x: x.confidence)
        primary_tool = best.object_label

    return interactions, primary_tool


def draw_frame_analysis(
    frame,
    hands,
    tools,
    interactions,
    is_working,
    active_tools,
):
    """Draw annotations on frame."""
    annotated = frame.copy()

    HAND_COLOR = (0, 255, 0)      # Green
    TOOL_COLOR = (0, 165, 255)    # Orange
    WORKING_COLOR = (0, 0, 255)   # Red

    # Draw hands
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), HAND_COLOR, 2)
        cv2.putText(
            annotated,
            f"hand ({hand.confidence:.2f})",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            HAND_COLOR,
            2,
        )
        cv2.circle(annotated, hand.center, 4, HAND_COLOR, -1)

    # Draw tools
    for tool in tools:
        x1, y1, x2, y2 = tool.bbox
        # Check if this tool is being worked with
        tool_is_working = any(
            i.object_bbox == tool.bbox and i.status == InteractionStatus.WORKING
            for i in interactions
        )
        color = WORKING_COLOR if tool_is_working else TOOL_COLOR

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{tool.label} ({tool.confidence:.2f})",
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

    # Draw status overlay
    if is_working:
        status_text = f"WORKING: {', '.join(active_tools)}"
    else:
        status_text = "IDLE"

    cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"Hands: {len(hands)} | Tools: {len(tools)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated


def process_video(
    video_path: str,
    output_video_path: str,
    output_json_path: str,
    duration_seconds: float = 10.0,
):
    """
    Process video for HOI detection with parallel batch processing.
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frames to process
    frames_to_process = int(fps * duration_seconds)
    frames_to_process = min(frames_to_process, total_frames)

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Processing first {duration_seconds}s = {frames_to_process} frames")
    print(f"Batch size: {BATCH_SIZE}")

    # Initialize detectors
    print("\nInitializing detectors...")
    hand_detector = HandDetector(confidence_threshold=0.25)
    tool_detector = ToolDetector(
        backend=DetectorBackend.GROUNDING_DINO,
        confidence_threshold=0.25,
    )

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # JSON data
    json_data = {
        "video_path": video_path,
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "duration_processed": duration_seconds,
        "batch_size": BATCH_SIZE,
        "frames": []
    }

    print("\nProcessing frames in parallel batches...")
    frame_idx = 0

    # Thread pool for parallel detection
    executor = ThreadPoolExecutor(max_workers=2)

    with tqdm(total=frames_to_process, desc="Processing") as pbar:
        while frame_idx < frames_to_process:
            # Load batch of frames
            batch_frames = []
            batch_indices = []

            for _ in range(BATCH_SIZE):
                if frame_idx >= frames_to_process:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                batch_indices.append(frame_idx)
                frame_idx += 1

            if not batch_frames:
                break

            # Run hand and tool detection in parallel
            def detect_hands_batch(frames):
                gloves_batch = hand_detector.detect_batch(frames)
                fingers_batch = hand_detector.detect_fingers_batch(frames)
                hands_batch = []
                for gloves, fingers in zip(gloves_batch, fingers_batch):
                    hands, _ = filter_gloves_by_fingers(gloves, fingers, min_avg_iou=0.05)
                    hands_batch.append(hands)
                return hands_batch

            def detect_tools_batch(frames):
                results = tool_detector.detect_batch(frames, tools_only=True)
                return [r.tools for r in results]

            # Submit both tasks
            hands_future = executor.submit(detect_hands_batch, batch_frames)
            tools_future = executor.submit(detect_tools_batch, batch_frames)

            # Wait for both to complete
            hands_batch = hands_future.result()
            tools_batch = tools_future.result()

            # Process each frame in batch
            for i, (frame, hands, tools) in enumerate(zip(batch_frames, hands_batch, tools_batch)):
                idx = batch_indices[i]
                timestamp = idx / fps

                # Compute HOI
                interactions, primary_tool = compute_hoi(hands, tools, height, width)

                # Check if working
                is_working = any(inter.status == InteractionStatus.WORKING for inter in interactions)
                active_tools = [inter.object_label for inter in interactions if inter.status == InteractionStatus.WORKING]

                # Draw annotations
                annotated = draw_frame_analysis(frame, hands, tools, interactions, is_working, active_tools)
                out.write(annotated)

                # Build frame data for JSON
                frame_data = {
                    "frame_number": idx,
                    "timestamp": round(timestamp, 3),
                    "hands": [],
                    "tools": [],
                    "hoi_status": "working" if is_working else "idle",
                    "active_tools": active_tools,
                    "interactions": []
                }

                for hand in hands:
                    frame_data["hands"].append({
                        "hand_id": hand.hand_id,
                        "bbox": {"x1": hand.bbox[0], "y1": hand.bbox[1], "x2": hand.bbox[2], "y2": hand.bbox[3]},
                        "center": {"x": hand.center[0], "y": hand.center[1]},
                        "confidence": round(hand.confidence, 3)
                    })

                for tool in tools:
                    frame_data["tools"].append({
                        "label": tool.label,
                        "bbox": {"x1": tool.bbox[0], "y1": tool.bbox[1], "x2": tool.bbox[2], "y2": tool.bbox[3]},
                        "center": {"x": tool.center[0], "y": tool.center[1]},
                        "confidence": round(tool.confidence, 3)
                    })

                for interaction in interactions:
                    frame_data["interactions"].append({
                        "hand_id": interaction.hand_id,
                        "tool_label": interaction.object_label,
                        "status": interaction.status.value,
                        "confidence": round(interaction.confidence, 3),
                        "iou": round(interaction.iou, 3),
                        "distance_pixels": round(interaction.distance_pixels, 1)
                    })

                json_data["frames"].append(frame_data)

            pbar.update(len(batch_frames))

    # Cleanup
    executor.shutdown()
    cap.release()
    out.release()
    hand_detector.close()

    # Save JSON
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Print summary
    total_processed = len(json_data["frames"])
    working_frames = sum(1 for f in json_data["frames"] if f["hoi_status"] == "working")
    idle_frames = sum(1 for f in json_data["frames"] if f["hoi_status"] == "idle")
    total_hands = sum(len(f["hands"]) for f in json_data["frames"])
    total_tools = sum(len(f["tools"]) for f in json_data["frames"])

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Frames processed: {total_processed}")
    print(f"WORKING frames: {working_frames} ({100*working_frames/total_processed:.1f}%)")
    print(f"IDLE frames: {idle_frames} ({100*idle_frames/total_processed:.1f}%)")
    print(f"Total hand detections: {total_hands}")
    print(f"Total tool detections: {total_tools}")
    print(f"\nOutput video: {output_video_path}")
    print(f"Output JSON: {output_json_path}")


if __name__ == "__main__":
    video_path = str(project_root / "artifacts" / "07_production_mp.mp4")
    output_video = str(project_root / "outputs" / "hoi_detection_10s.mp4")
    output_json = str(project_root / "outputs" / "hoi_detection_10s.json")

    # Ensure output directory exists
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    process_video(
        video_path=video_path,
        output_video_path=output_video,
        output_json_path=output_json,
        duration_seconds=2.0,
    )
