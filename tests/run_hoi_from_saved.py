"""
Run HOI detection using pre-saved hand and tool detection results.
"""

import sys
from pathlib import Path
import json
import pickle
import math

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.perception.hand_detector import HandResult, compute_iou
from src.perception.tool_detector import Detection
from src.perception.hoi_detector import InteractionStatus, Interaction


def compute_hoi(hands, tools, frame_height, frame_width, working_distance=100):
    """Compute HOI interactions between hands and tools."""
    scale = min(frame_width, frame_height) / 640
    working_dist = working_distance * scale

    interactions = []

    for hand in hands:
        hand_center = hand["center"]

        for tool in tools:
            # Calculate IOU
            hand_bbox = hand["bbox"]
            tool_bbox = tool["bbox"]
            iou = compute_iou(tuple(hand_bbox), tuple(tool_bbox))

            # Calculate distance
            dx = hand_center[0] - tool["center"][0]
            dy = hand_center[1] - tool["center"][1]
            distance = math.sqrt(dx*dx + dy*dy)

            # Determine status
            if iou > 0:
                status = "working"
                confidence = min(1.0, iou * 5)
            elif distance < working_dist:
                status = "working"
                confidence = 1 - (distance / working_dist)
            else:
                status = "idle"
                confidence = 0.0

            interactions.append({
                "hand_id": hand["hand_id"],
                "tool_label": tool["label"],
                "tool_bbox": tool["bbox"],
                "status": status,
                "confidence": confidence,
                "distance": distance,
                "iou": iou,
            })

    return interactions


def run_hoi_detection(
    hand_results_path: str,
    tool_results_path: str,
    output_path: str,
):
    """Combine hand and tool detections to compute HOI."""

    # Load saved results
    print(f"Loading hand detections from: {hand_results_path}")
    with open(hand_results_path, 'rb') as f:
        hand_data = pickle.load(f)

    print(f"Loading tool detections from: {tool_results_path}")
    with open(tool_results_path, 'rb') as f:
        tool_data = pickle.load(f)

    # Verify they're from the same video
    assert hand_data["video_path"] == tool_data["video_path"], "Video mismatch!"
    assert hand_data["total_frames"] == tool_data["total_frames"], "Frame count mismatch!"

    width = hand_data["width"]
    height = hand_data["height"]
    fps = hand_data["fps"]
    total_frames = hand_data["total_frames"]

    print(f"\nVideo: {hand_data['video_path']}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {total_frames}")

    # Compute HOI for each frame
    hoi_results = {
        "video_path": hand_data["video_path"],
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frames": {}
    }

    working_frames = 0
    total_interactions = 0

    print("\nComputing HOI...")
    for frame_idx in range(total_frames):
        hands = hand_data["frames"].get(frame_idx, [])
        tools = tool_data["frames"].get(frame_idx, [])

        # Compute interactions
        interactions = compute_hoi(hands, tools, height, width)

        # Determine frame status
        working_interactions = [i for i in interactions if i["status"] == "working"]
        is_working = len(working_interactions) > 0

        if is_working:
            working_frames += 1

        total_interactions += len(working_interactions)

        # Get primary tool (highest confidence working interaction)
        primary_tool = None
        if working_interactions:
            best = max(working_interactions, key=lambda x: x["confidence"])
            primary_tool = best["tool_label"]

        hoi_results["frames"][frame_idx] = {
            "hands": hands,
            "tools": tools,
            "interactions": interactions,
            "is_working": is_working,
            "primary_tool": primary_tool,
        }

    # Save results
    with open(output_path, 'wb') as f:
        pickle.dump(hoi_results, f)

    # Save JSON summary
    json_path = output_path.replace('.pkl', '_summary.json')
    summary = {
        "video_path": hand_data["video_path"],
        "total_frames": total_frames,
        "working_frames": working_frames,
        "working_percentage": 100 * working_frames / total_frames,
        "idle_frames": total_frames - working_frames,
        "idle_percentage": 100 * (total_frames - working_frames) / total_frames,
        "total_working_interactions": total_interactions,
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("HOI DETECTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total frames: {total_frames}")
    print(f"WORKING frames: {working_frames} ({100*working_frames/total_frames:.1f}%)")
    print(f"IDLE frames: {total_frames - working_frames} ({100*(total_frames-working_frames)/total_frames:.1f}%)")
    print(f"Total working interactions: {total_interactions}")
    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    hand_results = str(project_root / "outputs" / "clip3_hand_detections.pkl")
    tool_results = str(project_root / "outputs" / "clip3_tool_detections.pkl")
    output_path = str(project_root / "outputs" / "clip3_hoi_results.pkl")

    run_hoi_detection(hand_results, tool_results, output_path)
