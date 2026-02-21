"""
SiteIQ Main Pipeline
End-to-end video processing for egocentric productivity analysis.

Usage:
    python main.py --video path/to/video.mp4
    python main.py --video path/to/video.mp4 --output report.json
    python main.py --video path/to/video.mp4 --visualize
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# Import our modules
from src.perception import HOIDetector, HandDetector, ToolDetector, DetectorBackend
from src.temporal import (
    MotionAnalyzer,
    ActivityClassifier,
    SessionAggregator,
    SessionReport,
)


class SiteIQPipeline:
    """
    End-to-end pipeline for egocentric productivity analysis.

    Processes video through:
    1. Hand-Object Interaction detection (frame-by-frame)
    2. Motion analysis (temporal windows)
    3. Activity classification (combines HOI + motion)
    4. Session aggregation (metrics and insights)
    """

    def __init__(
        self,
        detector_backend: str = "yolo",
        motion_window: int = 10,
        fps: float = 30.0,
        frame_skip: int = 1,  # Process every Nth frame
        verbose: bool = True,
    ):
        """
        Initialize the pipeline.

        Args:
            detector_backend: "yolo" or "grounding_dino"
            motion_window: Number of frames for motion analysis
            fps: Video frame rate (will be auto-detected if possible)
            frame_skip: Process every Nth frame (1 = all frames)
            verbose: Print progress information
        """
        self.verbose = verbose
        self.frame_skip = frame_skip
        self.fps = fps

        if self.verbose:
            print("Initializing SiteIQ Pipeline...")

        # Initialize perception modules
        if self.verbose:
            print("  Loading hand detector...")
        hand_detector = HandDetector(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        if self.verbose:
            print(f"  Loading tool detector ({detector_backend})...")
        backend = DetectorBackend.YOLO if detector_backend == "yolo" else DetectorBackend.GROUNDING_DINO
        tool_detector = ToolDetector(
            backend=backend,
            confidence_threshold=0.3,
        )

        self.hoi_detector = HOIDetector(hand_detector, tool_detector)

        # Initialize temporal modules
        if self.verbose:
            print("  Initializing motion analyzer...")
        self.motion_analyzer = MotionAnalyzer(
            window_size=motion_window,
            sample_rate=fps,
        )

        if self.verbose:
            print("  Initializing activity classifier...")
        self.activity_classifier = ActivityClassifier(smoothing_window=3)

        if self.verbose:
            print("  Initializing session aggregator...")
        self.session_aggregator = SessionAggregator()

        if self.verbose:
            print("✓ Pipeline initialized successfully!\n")

    def process_video(
        self,
        video_path: str,
        output_video: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> SessionReport:
        """
        Process a video file and generate productivity report.

        Args:
            video_path: Path to input video file
            output_video: Optional path to save annotated video
            max_frames: Optional limit on number of frames to process

        Returns:
            SessionReport with complete analysis
        """
        if self.verbose:
            print(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps > 0:
            self.fps = fps
            self.motion_analyzer.sample_rate = fps

        if max_frames:
            total_frames = min(total_frames, max_frames)

        if self.verbose:
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {fps}")
            print(f"  Total frames: {total_frames}")
            print(f"  Processing every {self.frame_skip} frame(s)\n")

        # Setup video writer if needed
        video_writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        # Storage for analysis results
        frame_analyses = []
        motion_results = []
        frames_buffer = []  # For motion analysis

        # Process frames
        frame_count = 0
        processed_count = 0

        progress_bar = tqdm(total=total_frames, desc="Processing frames", disable=not self.verbose)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                timestamp = frame_count / fps

                # Skip frames if needed
                if frame_count % self.frame_skip != 0:
                    progress_bar.update(1)
                    continue

                # Add to frames buffer for motion analysis
                frames_buffer.append(frame.copy())
                if len(frames_buffer) > self.motion_analyzer.window_size:
                    frames_buffer.pop(0)

                # HOI detection
                frame_analysis = self.hoi_detector.analyze_frame(frame, timestamp)

                # Motion analysis (if we have enough frames)
                if len(frames_buffer) >= 2:
                    motion_result = self.motion_analyzer.analyze(frames_buffer)
                    frame_analysis.camera_motion = motion_result.motion_type.value
                else:
                    # Not enough frames yet, create dummy result
                    from src.temporal.motion_analyzer import MotionResult, MotionType
                    motion_result = MotionResult(
                        MotionType.UNKNOWN, 0.0, 0.0, None, None, {}
                    )

                # Store results
                frame_analyses.append(frame_analysis)
                motion_results.append(motion_result)

                # Annotate frame if saving video
                if video_writer:
                    annotated = self.hoi_detector.draw_analysis(frame, frame_analysis)
                    # Add motion info
                    cv2.putText(
                        annotated,
                        f"Motion: {motion_result.motion_type.value}",
                        (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    video_writer.write(annotated)

                processed_count += 1
                progress_bar.update(1)

                # Check max frames limit
                if max_frames and processed_count >= max_frames:
                    break

        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            progress_bar.close()

        if self.verbose:
            print(f"\n✓ Processed {processed_count} frames\n")

        # Activity classification
        if self.verbose:
            print("Classifying activities...")

        segments = self.activity_classifier.segment_activities(
            frame_analyses, motion_results, fps=self.fps
        )

        if self.verbose:
            print(f"  Generated {len(segments)} activity segments\n")

        # Session aggregation
        if self.verbose:
            print("Generating session report...")

        report = self.session_aggregator.aggregate(segments)

        if self.verbose:
            print("✓ Analysis complete!\n")

        return report

    def process_video_batch(
        self,
        video_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> List[SessionReport]:
        """
        Process multiple videos.

        Args:
            video_paths: List of video file paths
            output_dir: Optional directory to save reports

        Returns:
            List of SessionReport objects
        """
        reports = []

        for i, video_path in enumerate(video_paths, 1):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Processing video {i}/{len(video_paths)}")
                print(f"{'='*60}\n")

            report = self.process_video(video_path)
            reports.append(report)

            # Save report if output directory specified
            if output_dir:
                output_path = Path(output_dir) / f"{Path(video_path).stem}_report.json"
                self._save_report(report, str(output_path))

        return reports

    def _save_report(self, report: SessionReport, output_path: str):
        """Save report to JSON file."""
        # Convert report to JSON-serializable dict
        report_dict = {
            "session_duration": report.session_duration,
            "start_time": report.start_time,
            "end_time": report.end_time,
            "total_frames": report.total_frames,
            "productivity_score": report.productivity_score,
            "productive_time": report.productive_time,
            "idle_time": report.idle_time,
            "idle_percentage": report.idle_percentage,
            "most_used_tool": report.most_used_tool,
            "tool_switches": report.tool_switches,
            "activity_breakdown": {
                name: {
                    "activity": breakdown.activity.value,
                    "total_time": breakdown.total_time,
                    "percentage": breakdown.percentage,
                    "segment_count": breakdown.segment_count,
                    "average_duration": breakdown.average_duration,
                    "productivity_score": breakdown.productivity_score,
                }
                for name, breakdown in report.activity_breakdown.items()
            },
            "tool_usage": {
                name: {
                    "tool_name": usage.tool_name,
                    "total_time": usage.total_time,
                    "usage_count": usage.usage_count,
                    "average_duration": usage.average_duration,
                    "activities": usage.activities,
                }
                for name, usage in report.tool_usage.items()
            },
            "insights": report.insights,
            "recommendations": report.recommendations,
        }

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        if self.verbose:
            print(f"✓ Report saved to: {output_path}")


def main():
    """Command-line interface for SiteIQ pipeline."""
    parser = argparse.ArgumentParser(
        description="SiteIQ - Egocentric Productivity Intelligence"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save JSON report (default: video_name_report.json)",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        help="Path to save annotated video",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["yolo", "grounding_dino"],
        default="yolo",
        help="Object detection backend (default: yolo)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to process (for testing)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = SiteIQPipeline(
        detector_backend=args.detector,
        frame_skip=args.frame_skip,
        verbose=not args.quiet,
    )

    # Process video
    start_time = time.time()

    report = pipeline.process_video(
        video_path=args.video,
        output_video=args.output_video,
        max_frames=args.max_frames,
    )

    elapsed_time = time.time() - start_time

    # Print report
    print("\n" + report.get_summary())

    # Save report
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.video).stem + "_report.json"

    pipeline._save_report(report, output_path)

    print(f"\nProcessing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
