"""
Tool & Workpiece Detection Module (P2)
Uses Grounding DINO for zero-shot detection or YOLOv8 as fallback.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from enum import Enum
import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("Please install opencv-python")


class DetectorBackend(Enum):
    GROUNDING_DINO = "grounding_dino"
    YOLO = "yolo"


@dataclass
class Detection:
    """Single object detection result."""
    label: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[int, int]  # center point


@dataclass
class DetectionResult:
    """Combined detection results for a frame."""
    tools: List[Detection]
    workpieces: List[Detection]
    all_detections: List[Detection]


class ToolDetector:
    """
    Detects construction tools and workpieces in egocentric video frames.

    Supports two backends:
    1. Grounding DINO (zero-shot, more accurate, slower)
    2. YOLOv8 (requires training/fine-tuning, faster)
    """

    # Construction tools to detect
    TOOLS = [
        "drill",
        "hammer",
        "screwdriver",
        "wrench",
        "measuring tape",
        "level",
        "saw",
        "pliers",
        "nail gun",
        "screw gun",
        "power tool",
        "hand tool",
        # Masonry-specific tools
        "trowel",
        "masonry trowel",
        "mortar board",
        "mason line",
        "leveling tool",
        "grout bag",
        "float",
        # Plumbing-specific tools
        "pipe wrench",
        "pipe cutter",
        "torch",
        "soldering torch",
        "flux brush",
        # Marking/documentation tools
        "marker",
        "pen",
        "pencil",
        "chalk",
        "clipboard",
    ]

    # Workpieces/materials to detect
    WORKPIECES = [
        "drywall",
        "lumber",
        "wood board",
        "pipe",
        "wire",
        "cable",
        "metal stud",
        "concrete",
        "insulation",
        "panel",
        "screw",
        "nail",
        "bracket",
        # Masonry-specific materials/workpieces
        "brick",
        "block",
        "cinder block",
        "concrete block",
        "cmu",
        "mortar",
        "grout",
        "rebar",
        "masonry wall",
        "masonry unit",
        # Plumbing-specific materials/workpieces
        "copper pipe",
        "pvc pipe",
        "pex pipe",
        "pipe fitting",
        "elbow fitting",
        "tee fitting",
        "coupling",
        "valve",
        "drain",
        "trap",
        "manifold",
        "solder",
        "flux",
        # Packaging/labeling context objects
        "package",
        "box",
        "carton",
        "label",
        "tag",
        "sticker",
        "clipboard",
    ]

    def __init__(
        self,
        backend: DetectorBackend = DetectorBackend.YOLO,
        confidence_threshold: float = 0.2,
        text_threshold: float = 0.15,
        min_bbox_area: int = 300,
        device: str = "auto",  # "auto" | "mps" | "cuda" | "cpu"
    ):
        """
        Initialize tool detector.

        Args:
            backend: Which detection backend to use
            confidence_threshold: Minimum confidence for detections
            text_threshold: Text matching threshold for Grounding DINO
            min_bbox_area: Minimum bounding box area to keep detection
            device: Device to run inference on (auto prefers mps on Mac, then cuda, then cpu)
        """
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.text_threshold = text_threshold
        self.min_bbox_area = min_bbox_area
        self.device = device
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load the detection model based on backend."""
        if self.backend == DetectorBackend.YOLO:
            self._load_yolo()
        elif self.backend == DetectorBackend.GROUNDING_DINO:
            self._load_grounding_dino()

    def _load_yolo(self):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            # Use pretrained COCO model - will detect some tools
            # For better results, fine-tune on construction dataset
            self.model = YOLO("yolov8n.pt")  # nano model for speed
            print("Loaded YOLOv8 model")
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")

    def _load_grounding_dino(self):
        """Load Grounding DINO model."""
        try:
            # Option 1: Using transformers
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            model_id = "IDEA-Research/grounding-dino-tiny"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            resolved_device = self._resolve_device(self.device)
            self.model.to(resolved_device)
            self.device = resolved_device
            print(f"Loaded Grounding DINO model on {resolved_device}")
        except Exception as e:
            print(f"Failed to load Grounding DINO: {e}")
            print("Falling back to YOLO")
            self.backend = DetectorBackend.YOLO
            self._load_yolo()

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _has_mps(self) -> bool:
        """Check if Apple Metal (MPS) is available."""
        try:
            import torch
            return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
        except:
            return False

    def _resolve_device(self, requested_device: str) -> str:
        """Resolve requested device to an available backend."""
        req = (requested_device or "auto").lower()
        if req == "auto":
            if self._has_mps():
                return "mps"
            if self._has_cuda():
                return "cuda"
            return "cpu"
        if req == "mps":
            return "mps" if self._has_mps() else "cpu"
        if req == "cuda":
            return "cuda" if self._has_cuda() else "cpu"
        return "cpu"

    def detect(self, frame: np.ndarray, tools_only: bool = False) -> DetectionResult:
        """
        Detect tools and workpieces in a frame.

        Args:
            frame: BGR image from OpenCV
            tools_only: If True, skip workpiece detection (faster)

        Returns:
            DetectionResult with tools and workpieces
        """
        if self.backend == DetectorBackend.YOLO:
            return self._detect_yolo(frame)
        else:
            return self._detect_grounding_dino(frame, tools_only=tools_only)

    def _detect_yolo(self, frame: np.ndarray) -> DetectionResult:
        """Detect using YOLOv8."""
        results = self.model(frame, verbose=False)[0]

        tools = []
        workpieces = []
        all_detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            cls_id = int(box.cls[0])
            label = results.names[cls_id]

            # Get bbox
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            detection = Detection(
                label=label,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                center=center,
            )

            all_detections.append(detection)

            # Categorize (YOLO COCO classes)
            # Tools in COCO: scissors (76), knife (43)
            # We'll need custom model for construction tools
            tool_keywords = ["scissors", "knife", "tool"]
            if any(kw in label.lower() for kw in tool_keywords):
                tools.append(detection)

        return DetectionResult(
            tools=tools,
            workpieces=workpieces,
            all_detections=all_detections,
        )

    def _detect_grounding_dino(self, frame: np.ndarray, tools_only: bool = False) -> DetectionResult:
        """Detect using Grounding DINO (zero-shot)."""
        from PIL import Image
        import torch

        # Convert to PIL
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        height, width = frame.shape[:2]

        # Combine prompts
        tool_prompt = ". ".join(self.TOOLS) + "."
        if tools_only:
            full_prompt = tool_prompt
        else:
            workpiece_prompt = ". ".join(self.WORKPIECES) + "."
            full_prompt = tool_prompt + " " + workpiece_prompt

        # Process
        inputs = self.processor(images=image, text=full_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        # results = self.processor.post_process_grounded_object_detection(
        #     outputs,
        #     inputs["input_ids"],
        #     threshold=self.confidence_threshold,
        #     text_threshold=self.confidence_threshold,
        #     target_sizes=[(height, width)],
        # )[0]
        # transformers API compatibility:
        # - newer versions use `threshold`
        # - older versions used `box_threshold`
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.confidence_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(height, width)],
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.confidence_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(height, width)],
            )[0]

        tools = []
        workpieces = []
        all_detections = []

        # Max allowed bbox size (half of image dimensions)
        max_bbox_width = width // 2
        max_bbox_height = height // 2

        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x1, y1, x2, y2 = map(int, box.tolist())

            # Filter out bounding boxes that are too large
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width > max_bbox_width or bbox_height > max_bbox_height:
                continue

            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            detection = Detection(
                label=label,
                bbox=(x1, y1, x2, y2),
                confidence=float(score),
                center=center,
            )

            # Basic noise filtering for low-quality / tiny boxes
            area = max(0, (x2 - x1)) * max(0, (y2 - y1))
            if area < self.min_bbox_area:
                continue

            all_detections.append(detection)

            # Categorize (normalized match)
            label_lower = label.lower().strip()
            if self._matches_any(label_lower, self.TOOLS):
                tools.append(detection)
            elif self._matches_any(label_lower, self.WORKPIECES):
                workpieces.append(detection)

        return DetectionResult(
            tools=tools,
            workpieces=workpieces,
            all_detections=all_detections,
        )

    def detect_batch(
        self,
        frames: List[np.ndarray],
        tools_only: bool = False,
    ) -> List[DetectionResult]:
        """
        Detect tools and workpieces in multiple frames (batch inference).

        Args:
            frames: List of BGR images from OpenCV
            tools_only: If True, skip workpiece detection (faster)

        Returns:
            List of DetectionResult, one per frame
        """
        if self.backend == DetectorBackend.YOLO:
            # YOLO batch: process sequentially (YOLO handles its own batching)
            return [self._detect_yolo(frame) for frame in frames]
        else:
            return self._detect_grounding_dino_batch(frames, tools_only=tools_only)

    def _detect_grounding_dino_batch(
        self,
        frames: List[np.ndarray],
        tools_only: bool = False,
    ) -> List[DetectionResult]:
        """Batch detection using Grounding DINO."""
        from PIL import Image
        import torch

        if not frames:
            return []

        # Convert all frames to PIL images and track sizes
        images = []
        sizes = []
        for frame in frames:
            h, w = frame.shape[:2]
            sizes.append((h, w))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))

        # Combine prompts
        tool_prompt = ". ".join(self.TOOLS) + "."
        if tools_only:
            full_prompt = tool_prompt
        else:
            workpiece_prompt = ". ".join(self.WORKPIECES) + "."
            full_prompt = tool_prompt + " " + workpiece_prompt

        # Process inputs as batch
        inputs = self.processor(
            images=images,
            text=[full_prompt] * len(images),
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Run inference on batch
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process for each frame
        all_results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.confidence_threshold,
            text_threshold=self.confidence_threshold,
            target_sizes=sizes,
        )

        # Format results for each frame
        batch_results = []
        for frame_idx, results in enumerate(all_results):
            height, width = sizes[frame_idx]
            max_bbox_width = width // 2
            max_bbox_height = height // 2

            tools = []
            workpieces = []
            all_detections = []

            for box, score, label in zip(
                results["boxes"], results["scores"], results["labels"]
            ):
                x1, y1, x2, y2 = map(int, box.tolist())

                # Filter out bounding boxes that are too large
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                if bbox_width > max_bbox_width or bbox_height > max_bbox_height:
                    continue

                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                detection = Detection(
                    label=label,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(score),
                    center=center,
                )

                all_detections.append(detection)

                # Categorize
                label_lower = label.lower()
                if any(tool.lower() in label_lower for tool in self.TOOLS):
                    tools.append(detection)
                elif any(wp.lower() in label_lower for wp in self.WORKPIECES):
                    workpieces.append(detection)

            batch_results.append(DetectionResult(
                tools=tools,
                workpieces=workpieces,
                all_detections=all_detections,
            ))

        return batch_results

    def detect_with_custom_prompts(
        self,
        frame: np.ndarray,
        prompts: List[str],
    ) -> List[Detection]:
        """
        Detect objects with custom prompts (Grounding DINO only).

        Args:
            frame: BGR image
            prompts: List of object descriptions to detect

        Returns:
            List of detections
        """
        if self.backend != DetectorBackend.GROUNDING_DINO:
            print("Custom prompts only supported with Grounding DINO")
            return []

        from PIL import Image
        import torch

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        height, width = frame.shape[:2]

        prompt = ". ".join(prompts) + "."

        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # results = self.processor.post_process_grounded_object_detection(
        #     outputs,
        #     inputs["input_ids"],
        #     threshold=self.confidence_threshold,
        #     text_threshold=self.confidence_threshold,
        #     target_sizes=[(height, width)],
        # )[0]
        try:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=self.confidence_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(height, width)],
            )[0]
        except TypeError:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.confidence_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(height, width)],
            )[0]

        detections = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append(Detection(
                label=label,
                bbox=(x1, y1, x2, y2),
                confidence=float(score),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
            ))

        return detections

    def _matches_any(self, label: str, candidates: List[str]) -> bool:
        """Robust phrase match for noisy zero-shot labels."""
        normalized = label.replace("-", " ").replace("_", " ")
        for c in candidates:
            cc = c.lower().replace("-", " ").replace("_", " ")
            if cc in normalized or normalized in cc:
                return True
        return False

    def draw_detections(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        show_all: bool = False,
    ) -> np.ndarray:
        """
        Draw detection boxes on frame.

        Args:
            frame: BGR image
            result: Detection results
            show_all: If True, show all detections; else only tools/workpieces

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Colors
        TOOL_COLOR = (0, 165, 255)  # Orange
        WORKPIECE_COLOR = (255, 165, 0)  # Blue
        OTHER_COLOR = (128, 128, 128)  # Gray

        detections_to_draw = []

        if show_all:
            for det in result.all_detections:
                color = OTHER_COLOR
                if det in result.tools:
                    color = TOOL_COLOR
                elif det in result.workpieces:
                    color = WORKPIECE_COLOR
                detections_to_draw.append((det, color))
        else:
            for det in result.tools:
                detections_to_draw.append((det, TOOL_COLOR))
            for det in result.workpieces:
                detections_to_draw.append((det, WORKPIECE_COLOR))

        for det, color in detections_to_draw:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label_text = f"{det.label} ({det.confidence:.2f})"
            cv2.putText(
                annotated,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Draw center point
            cv2.circle(annotated, det.center, 4, color, -1)

        return annotated


# Quick test
if __name__ == "__main__":
    # Test with YOLO (easier setup)
    detector = ToolDetector(backend=DetectorBackend.YOLO)

    # Test with webcam or image
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        annotated = detector.draw_detections(frame, result, show_all=True)

        # Show info
        cv2.putText(
            annotated,
            f"Tools: {len(result.tools)} | Workpieces: {len(result.workpieces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Tool Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
