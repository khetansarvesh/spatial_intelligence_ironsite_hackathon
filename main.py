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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from tqdm import tqdm

# Import our modules
from src.perception import (
    HOIDetector,
    HandDetector,
    ToolDetector,
    DetectorBackend,
    SceneClassifier,
)
from src.temporal import (
    MotionAnalyzer,
    ActivityClassifier,
    SessionAggregator,
    SessionReport,
)
from src.agent.task_classifier import TaskClassifier


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
        # Support both legacy MediaPipe-style and current GroundingDINO-style signatures.
        try:
            hand_detector = HandDetector(confidence_threshold=0.25)

        except TypeError:
            hand_detector = HandDetector(
                confidence_threshold=0.25,
            )

        if self.verbose:
            print(f"  Loading tool detector ({detector_backend})...")
        backend = DetectorBackend.YOLO if detector_backend == "yolo" else DetectorBackend.GROUNDING_DINO
        # Grounding DINO tends to need lower thresholds in egocentric construction footage.
        detector_conf = 0.18 if backend == DetectorBackend.GROUNDING_DINO else 0.25
        tool_detector = ToolDetector(
            backend=backend,
            confidence_threshold=detector_conf,
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
            print("  Initializing scene classifier...")
        self.scene_classifier = SceneClassifier()

        if self.verbose:
            print("  Initializing task classifier...")
        # Uses deterministic fallback unless provider/api key is configured.
        self.task_classifier = TaskClassifier()

        if self.verbose:
            print("✓ Pipeline initialized successfully!\n")

    def process_video(
        self,
        video_path: str,
        output_video: Optional[str] = None,
        frame_output: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> SessionReport:
        """
        Process a video file and generate productivity report.

        Args:
            video_path: Path to input video file
            output_video: Optional path to save annotated video
            frame_output: Optional path to save per-frame JSON data
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

                # Scene context classification
                frame_workpieces = getattr(frame_analysis, "workpieces", [])
                detected_labels = (
                    [d.label for d in frame_analysis.tools]
                    + [d.label for d in frame_workpieces]
                )
                scene_result = self.scene_classifier.classify_scene_with_confidence(detected_labels)
                frame_analysis.metadata["scene"] = scene_result.scene
                frame_analysis.metadata["scene_confidence"] = scene_result.confidence
                frame_analysis.metadata["scene_matches"] = scene_result.matched_objects
                if frame_analysis.primary_tool:
                    frame_analysis.metadata["tool_scene_valid"] = self.scene_classifier.validate_tool_for_scene(
                        frame_analysis.primary_tool, scene_result.scene
                    )

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

                # Frame-level task classification
                interaction_status = "none"
                if any(i.status.value == "holding" for i in frame_analysis.interactions):
                    interaction_status = "holding"
                elif any(i.status.value == "reaching" for i in frame_analysis.interactions):
                    interaction_status = "reaching"
                elif frame_analysis.interactions:
                    interaction_status = "near"

                task_evidence = {
                    "objects": [d.label for d in frame_analysis.tools] + [d.label for d in frame_workpieces],
                    "tools": [d.label for d in frame_analysis.tools],
                    "motion": motion_result.motion_type.value,
                    "interaction": interaction_status,
                    "scene": frame_analysis.metadata.get("scene", "unknown"),
                    "primary_tool": frame_analysis.primary_tool,
                    "primary_workpiece": getattr(frame_analysis, "primary_workpiece", None),
                }
                task_result = self.task_classifier.classify(task_evidence)
                frame_analysis.metadata["task_trade"] = task_result["trade"]
                frame_analysis.metadata["task_family"] = task_result["task_family"]
                frame_analysis.metadata["task_name"] = task_result["task_name"]
                frame_analysis.metadata["task_confidence"] = task_result["confidence"]
                frame_analysis.metadata["task_unknown"] = task_result["unknown_flag"]
                frame_analysis.metadata["task_reason"] = task_result["reason"]
                wearer_status, scene_activity_status = self._derive_statuses(
                    frame_analysis=frame_analysis,
                    motion_type=motion_result.motion_type.value,
                    task_result=task_result,
                )
                frame_analysis.metadata["wearer_productivity_status"] = wearer_status
                frame_analysis.metadata["scene_activity_status"] = scene_activity_status

                # Store results
                frame_analyses.append(frame_analysis)
                motion_results.append(motion_result)

                # Annotate frame if saving video
                if video_writer:
                    annotated = self.hoi_detector.draw_analysis(frame, frame_analysis)
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
        report.metadata = self._build_scene_summary(frame_analyses)
        report.metadata.update(self._build_task_summary(frame_analyses))
        report.metadata.update(self._build_status_summary(frame_analyses))
        activity_insights = self._build_activity_insights(report)
        report.metadata["activity_insights"] = activity_insights
        for insight in activity_insights:
            if insight not in report.insights:
                report.insights.append(insight)

        scene_summary = report.metadata.get("scene_summary", {})
        dominant_scene = scene_summary.get("dominant_scene")
        if dominant_scene and dominant_scene != "unknown":
            report.insights.append(
                f"Dominant scene context detected: {dominant_scene} "
                f"({scene_summary.get('dominant_scene_percentage', 0.0):.1f}% of analyzed frames)"
            )

        if self.verbose:
            print("✓ Analysis complete!\n")

        if frame_output:
            self._save_frame_level_report(frame_analyses, frame_output)

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
            "metadata": report.metadata,
        }

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        if self.verbose:
            print(f"✓ Report saved to: {output_path}")

    def _save_frame_level_report(self, frame_analyses: List, output_path: str):
        """Save frame-level analysis to a separate JSON file."""
        frames = []
        for fa in frame_analyses:
            if hasattr(fa, "has_active_interaction"):
                is_active = bool(fa.has_active_interaction())
            elif hasattr(fa, "is_working"):
                is_active = bool(fa.is_working())
            else:
                is_active = False

            camera_motion = fa.metadata.get("camera_motion")
            if not camera_motion:
                camera_motion = getattr(fa, "camera_motion", "unknown")

            frames.append({
                "frame_index": fa.frame_index,
                "timestamp_sec": fa.timestamp,
                "wearer_productivity_status": fa.metadata.get("wearer_productivity_status", "UNOBSERVABLE"),
                "scene_activity_status": fa.metadata.get("scene_activity_status", "SCENE_UNCLEAR"),
                "task_name": fa.metadata.get("task_name", "unknown_task"),
                "task_family": fa.metadata.get("task_family", "unknown"),
                "task_confidence": fa.metadata.get("task_confidence", 0.0),
                "task_unknown": fa.metadata.get("task_unknown", True),
                "scene": fa.metadata.get("scene", "unknown"),
                "scene_confidence": fa.metadata.get("scene_confidence", 0.0),
                "motion": camera_motion,
                "primary_tool": fa.primary_tool,
                "active_interaction": is_active,
                "hands_count": len(getattr(fa, "hands", [])),
                "tools_count": len(getattr(fa, "tools", [])),
                "interactions_count": len(getattr(fa, "interactions", [])),
            })

        payload = {
            "total_frames": len(frames),
            "frames": frames,
        }

        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

        if self.verbose:
            print(f"✓ Frame-level report saved to: {output_path}")

    def _build_scene_summary(self, frame_analyses: List) -> Dict[str, object]:
        """Build per-session scene classification summary from frame metadata."""
        scenes = [
            fa.metadata.get("scene", "unknown")
            for fa in frame_analyses
        ]
        if not scenes:
            return {
                "scene_summary": {
                    "dominant_scene": "unknown",
                    "dominant_scene_percentage": 0.0,
                    "scene_distribution": {},
                }
            }

        counts = Counter(scenes)
        total = len(scenes)
        dominant_scene, dominant_count = counts.most_common(1)[0]
        scene_distribution = {
            scene: {
                "frames": count,
                "percentage": (count / total) * 100.0,
            }
            for scene, count in counts.items()
        }

        return {
            "scene_summary": {
                "dominant_scene": dominant_scene,
                "dominant_scene_percentage": (dominant_count / total) * 100.0,
                "scene_distribution": scene_distribution,
            }
        }

    def _build_task_summary(self, frame_analyses: List) -> Dict[str, object]:
        """Build per-session task classification summary from frame metadata."""
        task_names = [fa.metadata.get("task_name", "unknown_task") for fa in frame_analyses]
        families = [fa.metadata.get("task_family", "unknown") for fa in frame_analyses]
        trades = [fa.metadata.get("task_trade", "unknown") for fa in frame_analyses]

        if not task_names:
            return {
                "task_summary": {
                    "dominant_task": "unknown_task",
                    "task_distribution": {},
                    "family_distribution": {},
                    "trade_distribution": {},
                }
            }

        def dist(values: List[str]) -> Dict[str, Dict[str, float]]:
            counts = Counter(values)
            total = len(values)
            return {
                k: {"frames": v, "percentage": (v / total) * 100.0}
                for k, v in counts.items()
            }

        task_counts = Counter(task_names)
        dominant_task, dominant_task_count = task_counts.most_common(1)[0]
        total_tasks = len(task_names)

        return {
            "task_summary": {
                "dominant_task": dominant_task,
                "dominant_task_percentage": (dominant_task_count / total_tasks) * 100.0,
                "task_distribution": dist(task_names),
                "family_distribution": dist(families),
                "trade_distribution": dist(trades),
            }
        }

    def _derive_statuses(
        self,
        frame_analysis,
        motion_type: str,
        task_result: Dict[str, object],
    ) -> Tuple[str, str]:
        """
        Build two status metrics:
        - wearer_productivity_status: POV-centric
        - scene_activity_status: any visible activity in frame context
        """
        has_hands = len(getattr(frame_analysis, "hands", [])) > 0
        if hasattr(frame_analysis, "has_active_interaction"):
            has_active_hoi = bool(frame_analysis.has_active_interaction())
        elif hasattr(frame_analysis, "is_working"):
            has_active_hoi = bool(frame_analysis.is_working())
        else:
            has_active_hoi = False
        has_tools = len(frame_analysis.tools) > 0

        task_family = str(task_result.get("task_family", "unknown"))
        task_conf = float(task_result.get("confidence", 0.0))
        strong_task = task_family not in {"unknown", "idle"} and task_conf >= 0.65

        # Wearer-centric status
        if has_active_hoi:
            wearer = "ACTIVE"
        elif not has_hands and not has_tools and motion_type in {"stable", "unknown"}:
            wearer = "UNOBSERVABLE"
        elif motion_type == "walking":
            wearer = "ACTIVE_TRAVEL"
        elif strong_task:
            wearer = "ACTIVE_CONTEXTUAL"
        elif has_hands:
            wearer = "IDLE"
        else:
            wearer = "UNOBSERVABLE"

        # Scene-level activity status (includes visible non-POV activity cues)
        scene = str(frame_analysis.metadata.get("scene", "unknown"))
        scene_conf = float(frame_analysis.metadata.get("scene_confidence", 0.0))
        if strong_task:
            scene_activity = "SCENE_ACTIVE"
        elif scene != "unknown" and scene_conf >= 0.35:
            scene_activity = "SCENE_ACTIVE"
        else:
            scene_activity = "SCENE_UNCLEAR"

        return wearer, scene_activity

    def _build_status_summary(self, frame_analyses: List) -> Dict[str, object]:
        """Build distribution summaries for wearer vs scene activity statuses."""
        wearer_statuses = [fa.metadata.get("wearer_productivity_status", "UNOBSERVABLE") for fa in frame_analyses]
        scene_statuses = [fa.metadata.get("scene_activity_status", "SCENE_UNCLEAR") for fa in frame_analyses]

        def dist(values: List[str]) -> Dict[str, Dict[str, float]]:
            counts = Counter(values)
            total = max(len(values), 1)
            return {
                k: {"frames": v, "percentage": (v / total) * 100.0}
                for k, v in counts.items()
            }

        return {
            "status_summary": {
                "wearer_productivity_status": dist(wearer_statuses),
                "scene_activity_status": dist(scene_statuses),
            }
        }

    def _build_activity_insights(self, report: SessionReport) -> List[str]:
        """Generate additional activity-focused insights for report.json."""
        insights: List[str] = []

        # Dominant activity from activity breakdown
        if report.activity_breakdown:
            dominant = max(
                report.activity_breakdown.values(),
                key=lambda b: b.percentage,
            )
            insights.append(
                f"Dominant activity: {dominant.activity.value} "
                f"({dominant.percentage:.1f}% of analyzed activity time)"
            )

        status_summary = report.metadata.get("status_summary", {})
        wearer_dist = status_summary.get("wearer_productivity_status", {})
        scene_dist = status_summary.get("scene_activity_status", {})

        # POV observability insight
        unobs = wearer_dist.get("UNOBSERVABLE", {}).get("percentage", 0.0)
        if isinstance(unobs, (int, float)):
            observable = max(0.0, 100.0 - float(unobs))
            insights.append(f"POV observability coverage: {observable:.1f}%")
            if float(unobs) > 40.0:
                insights.append(
                    f"High unobservable share ({float(unobs):.1f}%) - "
                    "wearer productivity confidence may be limited"
                )

        # Scene-vs-wearer discrepancy insight
        scene_active = scene_dist.get("SCENE_ACTIVE", {}).get("percentage", 0.0)
        wearer_idle = wearer_dist.get("IDLE", {}).get("percentage", 0.0)
        if isinstance(scene_active, (int, float)) and isinstance(wearer_idle, (int, float)):
            if float(scene_active) > 30.0 and float(wearer_idle) > 40.0:
                insights.append(
                    "Scene activity is visible while POV wearer appears mostly idle/unobservable; "
                    "consider reviewing camera positioning or wearer-centric evidence"
                )

        # Task distribution insight
        task_summary = report.metadata.get("task_summary", {})
        dominant_task = task_summary.get("dominant_task")
        dominant_task_pct = task_summary.get("dominant_task_percentage", 0.0)
        if dominant_task:
            insights.append(
                f"Dominant inferred task: {dominant_task} "
                f"({float(dominant_task_pct):.1f}% of frames)"
            )

        return insights


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
        "--frame-output",
        type=str,
        help="Path to save frame-level JSON (default: <report_name>_frames.json)",
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

    # Save report
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.video).stem + "_report.json"

    if args.frame_output:
        frame_output_path = args.frame_output
    else:
        report_path = Path(output_path)
        frame_output_path = str(report_path.with_name(f"{report_path.stem}_frames.json"))

    # Process video
    start_time = time.time()

    report = pipeline.process_video(
        video_path=args.video,
        output_video=args.output_video,
        frame_output=frame_output_path,
        max_frames=args.max_frames,
    )

    elapsed_time = time.time() - start_time

    # Print report
    print("\n" + report.get_summary())

    pipeline._save_report(report, output_path)

    print(f"\nProcessing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
