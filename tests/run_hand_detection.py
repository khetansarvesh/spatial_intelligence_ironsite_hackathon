"""
Run hand detection on video and save results for later use by HOI detector.
Sequential processing (no batching - faster on Mac MPS).
"""

import sys
from pathlib import Path
import cv2
import json
import pickle
from tqdm import tqdm
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.perception.hand_detector import HandDetector, filter_gloves_by_fingers


def run_hand_detection(
    video_path: str,
    output_path: str,
    max_frames: int = None,
):
    """Run hand detection sequentially and save results."""

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
    print(f"Processing {total_frames} frames (sequential)")

    # Initialize detector
    print("\nInitializing hand detector...")
    detector = HandDetector(confidence_threshold=0.25)

    # Results storage
    results = {
        "video_path": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frames": {}
    }

    print("\nRunning hand detection...")
    start_time = time.time()

    for frame_idx in tqdm(range(total_frames), desc="Detecting hands"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect gloves and fingers (sequential - one frame at a time)
        gloves = detector.detect(frame)
        fingers = detector.detect_fingers(frame)
        hands, rejected = filter_gloves_by_fingers(gloves, fingers, min_avg_iou=0.05)

        # Convert to serializable format
        frame_hands = []
        for hand in hands:
            frame_hands.append({
                "hand_id": hand.hand_id,
                "bbox": list(hand.bbox),
                "center": list(hand.center),
                "confidence": float(hand.confidence),
            })

        results["frames"][frame_idx] = frame_hands

    cap.release()
    detector.close()

    elapsed = time.time() - start_time

    # Save results
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    # Summary
    json_path = output_path.replace('.pkl', '_summary.json')
    frames_with_hands = sum(1 for f in results["frames"].values() if f)
    total_detections = sum(len(f) for f in results["frames"].values())

    summary = {
        "video_path": video_path,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frames_with_hands": frames_with_hands,
        "total_hand_detections": total_detections,
        "processing_time_seconds": elapsed,
        "seconds_per_frame": elapsed / total_frames if total_frames > 0 else 0,
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"Frames processed: {total_frames}")
    print(f"Frames with hands: {frames_with_hands} ({100*frames_with_hands/total_frames:.1f}%)")
    print(f"Total hand detections: {total_detections}")
    print(f"Processing time: {elapsed:.1f}s ({elapsed/total_frames:.2f}s/frame)")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    video_path = str(project_root / "artifacts" / "clip_3_masonry.mp4")
    output_path = str(project_root / "outputs" / "clip3_hand_detections.pkl")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    run_hand_detection(
        video_path=video_path,
        output_path=output_path,
        max_frames=50,  # Just first 50 frames for quick test
    )
