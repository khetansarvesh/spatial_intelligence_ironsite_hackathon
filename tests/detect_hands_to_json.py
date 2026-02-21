"""
Detect hands in video frames and save results to JSON.
"""

import sys
from pathlib import Path
import json
import cv2
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.perception.hand_detector import HandDetector, filter_gloves_by_fingers


def detect_hands_to_json(
    video_path: str,
    output_json_path: str,
    num_frames: int = 100,
):
    """
    Detect hands in video frames and save to JSON.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_to_process = min(num_frames, total_frames)

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Processing {frames_to_process} frames")

    # Initialize detector
    print("\nInitializing hand detector...")
    detector = HandDetector(confidence_threshold=0.25)

    # JSON data
    json_data = {
        "video_path": video_path,
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "frames_processed": frames_to_process,
        "frames": []
    }

    print("\nDetecting hands...")
    for frame_idx in tqdm(range(frames_to_process), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # Detect gloves and fingers
        gloves = detector.detect(frame)
        fingers = detector.detect_fingers(frame)

        # Filter gloves by finger overlap
        hands, rejected = filter_gloves_by_fingers(gloves, fingers, min_avg_iou=0.05)

        # Build frame data
        frame_data = {
            "frame_number": frame_idx,
            "timestamp": round(timestamp, 3),
            "hands": [],
            "rejected_gloves": [],  # For debugging
            "fingers": []  # For debugging
        }

        # Add kept hands
        for hand in hands:
            frame_data["hands"].append({
                "hand_id": hand.hand_id,
                "label": "hand",
                "bbox": {
                    "x1": hand.bbox[0],
                    "y1": hand.bbox[1],
                    "x2": hand.bbox[2],
                    "y2": hand.bbox[3]
                },
                "center": {"x": hand.center[0], "y": hand.center[1]},
                "confidence": round(hand.confidence, 3)
            })

        # Add rejected gloves (for debugging/visualization)
        for glove in rejected:
            frame_data["rejected_gloves"].append({
                "hand_id": glove.hand_id,
                "label": "rejected_glove",
                "bbox": {
                    "x1": glove.bbox[0],
                    "y1": glove.bbox[1],
                    "x2": glove.bbox[2],
                    "y2": glove.bbox[3]
                },
                "center": {"x": glove.center[0], "y": glove.center[1]},
                "confidence": round(glove.confidence, 3)
            })

        # Add finger detections (for debugging/visualization)
        for finger in fingers:
            frame_data["fingers"].append({
                "label": finger.label,
                "bbox": {
                    "x1": finger.bbox[0],
                    "y1": finger.bbox[1],
                    "x2": finger.bbox[2],
                    "y2": finger.bbox[3]
                },
                "confidence": round(finger.confidence, 3)
            })

        json_data["frames"].append(frame_data)

    # Cleanup
    cap.release()
    detector.close()

    # Save JSON
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # Summary
    total_hands = sum(len(f["hands"]) for f in json_data["frames"])
    frames_with_hands = sum(1 for f in json_data["frames"] if len(f["hands"]) > 0)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Frames processed: {len(json_data['frames'])}")
    print(f"Frames with hands: {frames_with_hands} ({100*frames_with_hands/len(json_data['frames']):.1f}%)")
    print(f"Total hand detections: {total_hands}")
    print(f"\nOutput JSON: {output_json_path}")


if __name__ == "__main__":
    video_path = str(project_root / "artifacts" / "07_production_mp.mp4")
    output_json = str(project_root / "outputs" / "hand_detections.json")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    detect_hands_to_json(
        video_path=video_path,
        output_json_path=output_json,
        num_frames=100,
    )
