"""
Create annotated video from HOI detection results.
"""

import sys
from pathlib import Path
import cv2
import pickle
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_hoi_video(
    hoi_results_path: str,
    output_video_path: str,
):
    """Create annotated video showing hands, tools, and interactions."""

    # Load HOI results
    print(f"Loading HOI results from: {hoi_results_path}")
    with open(hoi_results_path, 'rb') as f:
        hoi_data = pickle.load(f)

    video_path = hoi_data["video_path"]
    fps = hoi_data["fps"]
    width = hoi_data["width"]
    height = hoi_data["height"]
    total_frames = hoi_data["total_frames"]

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {total_frames}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Colors
    HAND_COLOR = (0, 255, 0)       # Green for hands
    TOOL_COLOR = (0, 165, 255)     # Orange for tools
    WORKING_COLOR = (0, 255, 255)  # Yellow for working interaction
    IDLE_COLOR = (128, 128, 128)   # Gray for idle

    print("\nCreating annotated video...")
    for frame_idx in tqdm(range(total_frames), desc="Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = hoi_data["frames"].get(frame_idx, {})
        hands = frame_data.get("hands", [])
        tools = frame_data.get("tools", [])
        interactions = frame_data.get("interactions", [])
        is_working = frame_data.get("is_working", False)
        primary_tool = frame_data.get("primary_tool", None)

        # Draw hands
        for hand in hands:
            bbox = hand["bbox"]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), HAND_COLOR, 2)
            cv2.putText(
                frame,
                f"Hand: {hand['hand_id']}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                HAND_COLOR,
                2,
            )
            # Draw center
            cx, cy = hand["center"]
            cv2.circle(frame, (cx, cy), 5, HAND_COLOR, -1)

        # Draw tools
        for tool in tools:
            bbox = tool["bbox"]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), TOOL_COLOR, 2)
            conf = tool.get("confidence", 0)
            cv2.putText(
                frame,
                f"{tool['label']} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TOOL_COLOR,
                2,
            )
            # Draw center
            cx, cy = tool["center"]
            cv2.circle(frame, (cx, cy), 5, TOOL_COLOR, -1)

        # Draw working interactions (lines between hand and tool)
        for interaction in interactions:
            if interaction["status"] == "working":
                # Find hand center
                hand_id = interaction["hand_id"]
                hand_center = None
                for h in hands:
                    if h["hand_id"] == hand_id:
                        hand_center = tuple(h["center"])
                        break

                # Find tool center
                tool_bbox = interaction["tool_bbox"]
                tool_center = (
                    (tool_bbox[0] + tool_bbox[2]) // 2,
                    (tool_bbox[1] + tool_bbox[3]) // 2
                )

                if hand_center:
                    # Draw line connecting hand and tool
                    cv2.line(frame, hand_center, tool_center, WORKING_COLOR, 3)

        # Status overlay
        status_color = WORKING_COLOR if is_working else IDLE_COLOR
        status_text = "WORKING" if is_working else "IDLE"

        # Background rectangle for status
        cv2.rectangle(frame, (10, 10), (200, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (200, 80), status_color, 2)

        cv2.putText(
            frame,
            status_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            status_color,
            2,
        )

        if primary_tool:
            cv2.putText(
                frame,
                f"Tool: {primary_tool}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                1,
            )

        # Frame counter
        cv2.putText(
            frame,
            f"Frame: {frame_idx}/{total_frames}",
            (width - 180, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        out.write(frame)

    cap.release()
    out.release()

    print(f"\nVideo saved to: {output_video_path}")


if __name__ == "__main__":
    hoi_results = str(project_root / "outputs" / "clip3_hoi_results.pkl")
    output_video = str(project_root / "outputs" / "clip3_hoi_annotated.mp4")

    create_hoi_video(hoi_results, output_video)
