# SiteIQ - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Process a Video
```bash
# Basic usage
python main.py --video path/to/construction_video.mp4

# Save annotated video
python main.py --video input.mp4 --output-video annotated.mp4

# Specify output report location
python main.py --video input.mp4 --output my_report.json

# Quick test (process first 100 frames)
python main.py --video input.mp4 --max-frames 100
```

## Command-Line Options

```
--video PATH           Input video file (required)
--output PATH          Output JSON report path (default: video_name_report.json)
--output-video PATH    Save annotated video with detections
--detector TYPE        Detection backend: "yolo" or "grounding_dino" (default: yolo)
--max-frames N         Process only first N frames (for testing)
--frame-skip N         Process every Nth frame (default: 1)
--quiet               Suppress progress output
```

## Python API Usage

### End-to-End Pipeline

```python
from main import SiteIQPipeline

# Initialize pipeline
pipeline = SiteIQPipeline(
    detector_backend="yolo",
    frame_skip=1,  # Process every frame
    verbose=True
)

# Process video
report = pipeline.process_video(
    video_path="construction_site.mp4",
    output_video="annotated_output.mp4"
)

# Access results
print(f"Productivity Score: {report.productivity_score:.1%}")
print(f"Idle Time: {report.idle_percentage:.1f}%")
print(f"Most Used Tool: {report.most_used_tool}")

# Print full report
print(report.get_summary())

# Activity breakdown
for activity_name, breakdown in report.activity_breakdown.items():
    print(f"{activity_name}: {breakdown.total_time:.1f}s ({breakdown.percentage:.1f}%)")

# Tool usage
for tool_name, usage in report.tool_usage.items():
    print(f"{tool_name}: {usage.total_time:.1f}s, {usage.usage_count} uses")
```

### Individual Modules

#### Hand Detection
```python
from src.perception import HandDetector
import cv2

detector = HandDetector()
frame = cv2.imread("frame.jpg")

hands = detector.detect(frame)
for hand in hands:
    print(f"{hand.side} hand: confidence {hand.confidence:.2f}")

annotated = detector.draw_landmarks(frame, hands)
cv2.imwrite("output.jpg", annotated)
```

#### Tool Detection
```python
from src.perception import ToolDetector, DetectorBackend

detector = ToolDetector(backend=DetectorBackend.YOLO)
result = detector.detect(frame)

print(f"Found {len(result.tools)} tools, {len(result.workpieces)} workpieces")
for tool in result.tools:
    print(f"  {tool.label}: {tool.confidence:.2f}")
```

#### Hand-Object Interaction
```python
from src.perception import HOIDetector

detector = HOIDetector()
analysis = detector.analyze_frame(frame, timestamp=1.5)

print(f"Active interaction: {analysis.has_active_interaction()}")
print(f"Held tools: {analysis.get_held_tools()}")
print(f"Primary tool: {analysis.primary_tool}")

annotated = detector.draw_analysis(frame, analysis)
```

#### Motion Analysis
```python
from src.temporal import MotionAnalyzer

analyzer = MotionAnalyzer(window_size=10, sample_rate=30.0)

# Analyze sequence of frames
frames = [frame1, frame2, frame3, ...]
result = analyzer.analyze(frames)

print(f"Motion type: {result.motion_type.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Magnitude: {result.magnitude:.2f}")
```

#### Activity Classification
```python
from src.temporal import ActivityClassifier

classifier = ActivityClassifier()

# Classify single frame
activity, confidence = classifier.classify_frame(frame_analysis, motion_result)
print(f"Activity: {activity.value}, Confidence: {confidence:.2f}")

# Segment activities over time
segments = classifier.segment_activities(frame_analyses, motion_results, fps=30.0)
for segment in segments:
    print(f"{segment.activity.value}: {segment.duration:.1f}s (productivity: {segment.productivity_score:.1%})")
```

#### Session Aggregation
```python
from src.temporal import SessionAggregator

aggregator = SessionAggregator()
report = aggregator.aggregate(segments)

print(report.get_summary())
```

## Output Format

### JSON Report Structure
```json
{
  "session_duration": 300.5,
  "productivity_score": 0.75,
  "productive_time": 225.3,
  "idle_time": 45.2,
  "idle_percentage": 15.0,
  "most_used_tool": "drill",
  "tool_switches": 8,
  "activity_breakdown": {
    "ACTIVE_TOOL_USE": {
      "total_time": 180.5,
      "percentage": 60.1,
      "segment_count": 12,
      "productivity_score": 1.0
    }
  },
  "tool_usage": {
    "drill": {
      "total_time": 150.0,
      "usage_count": 8,
      "average_duration": 18.75
    }
  },
  "insights": [
    "High productivity session (75.0%)",
    "Peak productivity: 50.0s - 120.0s (score: 95.0%)"
  ],
  "recommendations": [
    "Maintain current efficient workflow"
  ]
}
```

## Performance Tips

1. **Use frame skipping for faster processing:**
   ```bash
   python main.py --video input.mp4 --frame-skip 2  # Process every 2nd frame
   ```

2. **Use YOLO for speed, Grounding DINO for accuracy:**
   ```bash
   python main.py --video input.mp4 --detector yolo  # Faster
   python main.py --video input.mp4 --detector grounding_dino  # More accurate
   ```

3. **Process subset for quick testing:**
   ```bash
   python main.py --video input.mp4 --max-frames 300  # First 10 seconds at 30fps
   ```

## Troubleshooting

### Common Issues

1. **Out of memory:**
   - Use `--frame-skip 2` or higher
   - Process shorter video clips
   - Use YOLO instead of Grounding DINO

2. **Slow processing:**
   - Ensure GPU is available for PyTorch
   - Use frame skipping
   - Use YOLO backend

3. **Import errors:**
   - Ensure you're running from the project root directory
   - Check all dependencies are installed: `pip install -r requirements.txt`

4. **No hands detected:**
   - Check video quality and lighting
   - Lower `min_detection_confidence` in HandDetector
   - Verify workers' hands are visible

5. **No tools detected:**
   - Lower `confidence_threshold` in ToolDetector
   - Try switching between YOLO and Grounding DINO
   - Check if tools are in the predefined list

## Next Steps

- See `HACKATHON_PLAN.md` for the full system architecture
- Check individual module files for detailed documentation
- Explore the demo dashboard (coming soon)
- Try the LLM agent for natural language queries (coming soon)
