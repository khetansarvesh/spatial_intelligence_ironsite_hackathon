# SiteIQ - Egocentric Productivity Intelligence

## Hackathon Plan (36 Hours)

**Team Size:** 5 people
**Event:** UMD x Ironsite Spatial Intelligence Hackathon
**Dates:** February 20-22, 2025

---

## Project Overview

### The Problem
Current VLMs looking at egocentric (hardhat camera) video can only say:
> "I see hands, a drill, and a wall"

They **cannot** determine:
> "The worker has been actively drilling for 3 minutes, completed approximately 8 screw insertions, and is 70% through this panel"

### Our Solution
**SiteIQ** combines Hand-Object Interaction (HOI) detection with temporal activity analysis and an LLM agent to provide actionable productivity insights from egocentric construction footage.

### Key Innovation
- **Egocentric HOI Detection:** Hand tracking + tool detection + proximity-based interaction logic
- **Temporal State Machine:** Classify activities over time (active work, idle, traveling, etc.)
- **Agentic Interface:** Natural language queries about worker productivity

---

## Team Roles

| Person | Role | Primary Skills Needed |
|--------|------|----------------------|
| **P1** | Perception Lead | CV, object detection, MediaPipe |
| **P2** | Perception Support | CV, model integration |
| **P3** | Temporal/Logic Lead | Python, state machines, data processing |
| **P4** | Agent/Backend Lead | LLMs, APIs, prompt engineering |
| **P5** | Integration/Demo Lead | Full-stack, visualization, demo |

---

## Project Structure

```
spatial-productivity/
├── data/
│   └── videos/               # Sample videos from Ironsite
├── src/
│   ├── perception/           # P1, P2
│   │   ├── hand_detector.py
│   │   ├── tool_detector.py
│   │   └── hoi_detector.py
│   ├── temporal/             # P3
│   │   ├── activity_classifier.py
│   │   ├── motion_analyzer.py
│   │   └── session_aggregator.py
│   ├── agent/                # P4
│   │   ├── tools.py
│   │   ├── prompts.py
│   │   └── agent.py
│   └── demo/                 # P5
│       ├── visualizer.py
│       └── dashboard.py
├── outputs/                  # Results
├── notebooks/                # Exploration
├── requirements.txt
└── main.py
```

---

## Phase 0: Setup & Alignment (Hours 0-2)

**Goal:** Everyone on same page, environment ready, video data accessible

### All Team Members (Together)

- [ ] Clone repo, set up virtual environment
- [ ] Install core dependencies (requirements.txt)
- [ ] Verify GPU access (if available)
- [ ] Download/access sample videos from Ironsite
- [ ] Review video characteristics (resolution, FPS, lighting)
- [ ] Quick team sync: confirm understanding of the plan
- [ ] Create shared folder structure

### Deliverable
Everyone can run `python test_setup.py` successfully

---

## Phase 1: Core Perception Pipeline (Hours 2-10)

**Goal:** Given a video frame → Output hand positions, detected tools, and HOI status

**Dependencies:** Phase 0 complete

### Architecture

```
Video Frame
     │
     ├──────────────────┬──────────────────┐
     ▼                  ▼                  ▼
┌─────────┐       ┌──────────┐      ┌──────────┐
│ Hand    │       │ Tool     │      │ Workpiece│
│ Detector│       │ Detector │      │ Detector │
│ (P1)    │       │ (P2)     │      │ (P2)     │
└────┬────┘       └────┬─────┘      └────┬─────┘
     │                 │                 │
     └────────────┬────┴─────────────────┘
                  ▼
           ┌─────────────┐
           │ HOI Merger  │
           │ (P1 + P2)   │
           └─────────────┘
                  │
                  ▼
Output: {"hands": [...], "tools": [...], "holding": {...}}
```

---

### P1: Hand Detection (Hours 2-8)

**File:** `src/perception/hand_detector.py`

```python
class HandDetector:
    def detect(self, frame) -> List[HandResult]:
        """
        Returns:
        [
            {
                "hand_id": 0,
                "side": "left" | "right",
                "landmarks": [(x,y,z), ...],  # 21 points
                "bbox": [x1, y1, x2, y2],
                "fingertip_positions": {...},
                "confidence": 0.95
            }
        ]
        """
```

**Hour-by-hour:**
| Hours | Task |
|-------|------|
| 2-3 | MediaPipe setup, basic detection working |
| 3-5 | Landmark extraction, confidence filtering |
| 5-7 | Handle gloves, partial visibility, low light |
| 7-8 | Clean API, unit tests, documentation |

---

### P2: Tool & Workpiece Detection (Hours 2-8)

**File:** `src/perception/tool_detector.py`

```python
class ToolDetector:
    TOOLS = ["drill", "hammer", "screwdriver", "wrench",
             "measuring tape", "level", "saw", "pliers", "nail gun"]

    WORKPIECES = ["drywall", "lumber", "pipe", "wire",
                  "metal stud", "concrete", "insulation", "panel"]

    def detect(self, frame) -> DetectionResult:
        """
        Returns:
        {
            "tools": [
                {"label": "drill", "bbox": [...], "confidence": 0.89}
            ],
            "workpieces": [
                {"label": "drywall", "bbox": [...], "confidence": 0.76}
            ]
        }
        """
```

**Hour-by-hour:**
| Hours | Task |
|-------|------|
| 2-3 | Grounding DINO setup (or YOLOv8 fallback) |
| 3-5 | Tool detection with prompt tuning |
| 5-7 | Workpiece detection, threshold optimization |
| 7-8 | Unified API, handle no-detection cases |

---

### P1 + P2: HOI Integration (Hours 8-10)

**File:** `src/perception/hoi_detector.py`

```python
class HOIDetector:
    def __init__(self, hand_detector, tool_detector):
        self.hand_detector = hand_detector
        self.tool_detector = tool_detector

    def analyze_frame(self, frame) -> FrameAnalysis:
        """
        Returns:
        {
            "timestamp": 1234.56,
            "hands": [...],
            "tools": [...],
            "workpieces": [...],
            "interactions": [
                {
                    "hand": "right",
                    "tool": "drill",
                    "confidence": 0.92,
                    "status": "holding"  # holding | reaching | none
                }
            ],
            "target_workpiece": "drywall"
        }
        """
```

**Hour-by-hour:**
| Hours | Task |
|-------|------|
| 8-9 | Proximity calculation, threshold tuning |
| 9-10 | Edge cases, confidence scoring, testing |

---

### P3: Data Schema & Motion Analysis (Hours 2-10) - Parallel

**Files:** `src/temporal/motion_analyzer.py`

```python
@dataclass
class FrameAnalysis:
    timestamp: float
    hands: List[HandResult]
    tools: List[ToolDetection]
    interactions: List[HOI]
    camera_motion: str  # "stable" | "rhythmic" | "moving"

@dataclass
class ActivitySegment:
    start_time: float
    end_time: float
    activity: str
    tool_used: Optional[str]
    productivity_score: float

class MotionAnalyzer:
    def analyze(self, frames: List[np.ndarray]) -> str:
        """
        Classify camera motion:
        - "stable": focused work or idle
        - "rhythmic": repetitive task (hammering)
        - "panning": looking around
        - "walking": traveling
        """
```

**Hour-by-hour:**
| Hours | Task |
|-------|------|
| 2-5 | Define data schemas |
| 5-10 | Camera motion analysis using optical flow |

---

### P4: Agent Architecture (Hours 2-10) - Parallel

**Files:** `src/agent/tools.py`, `src/agent/prompts.py`

```python
AGENT_TOOLS = [
    {
        "name": "get_activity_summary",
        "description": "Get summary of worker activities over a time period",
        "parameters": {"start_time": "float", "end_time": "float"}
    },
    {
        "name": "get_tool_usage",
        "description": "Get breakdown of which tools were used and for how long",
        "parameters": {"time_period": "str"}
    },
    {
        "name": "find_idle_periods",
        "description": "Find periods where worker was idle/unproductive",
        "parameters": {"min_duration_seconds": "int"}
    },
    {
        "name": "get_productivity_score",
        "description": "Calculate overall productivity score",
        "parameters": {}
    },
    {
        "name": "compare_periods",
        "description": "Compare productivity between two time periods",
        "parameters": {"period1": "str", "period2": "str"}
    }
]

SYSTEM_PROMPT = """
You are SiteIQ, an AI assistant that analyzes construction worker
productivity from egocentric video footage. You have access to
detailed activity data including tool usage, idle periods, and
productivity metrics.

Always be specific with numbers and time ranges. If asked about
anomalies or issues, provide actionable insights.
"""
```

---

### P5: Visualization Foundation (Hours 2-10) - Parallel

**Files:** `src/demo/visualizer.py`, `src/demo/dashboard.py`

```python
class FrameVisualizer:
    def annotate_frame(self, frame, analysis: FrameAnalysis) -> np.ndarray:
        """
        Draw on frame:
        - Hand landmarks (skeleton)
        - Tool bounding boxes with labels
        - HOI status indicator
        - Current activity label
        """

class DemoPipeline:
    def __init__(self):
        self.hoi_detector = None
        self.activity_classifier = None
        self.agent = None

    def process_video(self, video_path):
        """End-to-end pipeline"""
        pass
```

---

### Phase 1 Checkpoint (Hour 10)

**Team Sync Meeting (30 min)**

| Person | Deliverable | Status |
|--------|-------------|--------|
| P1 | `HandDetector` works on sample video | ☐ |
| P2 | `ToolDetector` works on sample video | ☐ |
| P1+P2 | `HOIDetector` merges both correctly | ☐ |
| P3 | Data schemas + motion analyzer ready | ☐ |
| P4 | Agent tools defined, prompts written | ☐ |
| P5 | Frame visualizer + dashboard skeleton | ☐ |

**Exit Criteria:**
```python
hoi = HOIDetector(HandDetector(), ToolDetector())
result = hoi.analyze_frame(sample_frame)
print(result)  # Shows hands, tools, interactions
```

---

## Phase 2: Temporal Intelligence (Hours 10-18)

**Goal:** Process video over time → Activity segments with productivity scores

**Dependencies:** Phase 1 `HOIDetector` working

### Architecture

```
Video Stream
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Frame-by-frame HOI Analysis (from Phase 1)             │
│  → List[FrameAnalysis]                                  │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ State       │     │ Activity    │     │ Productivity│
│ Machine     │────▶│ Segmenter   │────▶│ Calculator  │
│ (P3)        │     │ (P3)        │     │ (P3)        │
└─────────────┘     └─────────────┘     └─────────────┘
     │
     ▼
Output: List[ActivitySegment] + ProductivityReport
```

---

### P3: Activity Classification (Hours 10-18) - CRITICAL PATH

**File:** `src/temporal/activity_classifier.py`

```python
class ActivityClassifier:

    STATES = {
        "ACTIVE_TOOL_USE": {"productivity": 1.0},
        "PRECISION_WORK": {"productivity": 1.0},
        "MATERIAL_HANDLING": {"productivity": 0.7},
        "SETUP_CLEANUP": {"productivity": 0.5},
        "SEARCHING": {"productivity": 0.3},
        "TRAVELING": {"productivity": 0.2},
        "IDLE": {"productivity": 0.0},
    }

    def classify_frame(self, analysis: FrameAnalysis, motion: str) -> str:
        """
        Decision logic:
        - Tool in hand + stable/rhythmic motion → ACTIVE_TOOL_USE
        - Tool in hand + minimal motion → PRECISION_WORK
        - No tool + object in hand + movement → MATERIAL_HANDLING
        - No tool + scanning motion → SEARCHING
        - Walking detected + no tool → TRAVELING
        - Stationary + no tool + no activity → IDLE
        """

    def segment_activities(self, frame_analyses: List[FrameAnalysis]) -> List[ActivitySegment]:
        """
        Merge consecutive frames with same activity into segments.
        Apply smoothing to avoid rapid state changes.
        """
```

**Hour-by-hour:**
| Hours | Task |
|-------|------|
| 10-12 | State machine logic implementation |
| 12-14 | Segmentation algorithm |
| 14-16 | Productivity calculation and reporting |
| 16-18 | Edge cases, smoothing, testing |

---

### P1: Perception Refinement (Hours 10-14)

- [ ] Tune thresholds based on actual Ironsite footage
- [ ] Handle specific edge cases discovered in testing
- [ ] Add depth estimation (optional)
- [ ] Optimize for speed (target: 5+ FPS)

---

### P2: Scene Context (Hours 10-14)

**File:** `src/perception/scene_classifier.py`

```python
class SceneClassifier:
    SCENE_TYPES = {
        "framing": ["metal stud", "lumber", "drywall", "screw"],
        "electrical": ["wire", "conduit", "junction box", "panel"],
        "plumbing": ["pipe", "fitting", "valve", "wrench"],
        "finishing": ["paint", "trim", "tape", "sander"],
    }

    def classify_scene(self, detected_objects: List[str]) -> str:
        """Based on objects visible, infer scene type"""

    def validate_tool_for_scene(self, tool: str, scene: str) -> bool:
        """Is this tool appropriate for this scene type?"""
```

---

### P4: Agent Implementation (Hours 10-18)

**File:** `src/agent/agent.py`

```python
class ProductivityAgent:
    def __init__(self, activity_data: List[ActivitySegment]):
        self.data = activity_data
        self.client = OpenAI()  # or Anthropic

    def get_activity_summary(self, start: float, end: float) -> str:
        segments = self._filter_by_time(start, end)
        # Aggregate and format

    def get_tool_usage(self) -> Dict[str, float]:
        # Calculate time per tool

    def find_idle_periods(self, min_duration: int) -> List[Dict]:
        # Find gaps in productivity

    def chat(self, user_message: str) -> str:
        # Route to appropriate tool, format response
```

**Hour-by-hour:**
| Hours | Task |
|-------|------|
| 10-12 | Implement tool functions with real data queries |
| 12-14 | Connect to LLM with function calling |
| 14-16 | Test queries, refine prompts |
| 16-18 | Error handling, edge cases |

---

### P5: Integration Pipeline (Hours 10-18)

**File:** `main.py`

```python
class SiteIQPipeline:
    def __init__(self):
        self.hoi_detector = HOIDetector(HandDetector(), ToolDetector())
        self.motion_analyzer = MotionAnalyzer()
        self.activity_classifier = ActivityClassifier()
        self.visualizer = FrameVisualizer()

    def process_video(self, video_path: str) -> SessionReport:
        frames = self._extract_frames(video_path)

        analyses = []
        for i, frame in enumerate(frames):
            analysis = self.hoi_detector.analyze_frame(frame)

            if i % 5 == 0:
                motion = self.motion_analyzer.analyze(frames[max(0,i-10):i+1])
                analysis.camera_motion = motion

            analyses.append(analysis)

        segments = self.activity_classifier.segment_activities(analyses)
        report = self._generate_report(segments)

        return report
```

---

### Phase 2 Checkpoint (Hour 18)

**Team Sync Meeting (30 min)**

| Person | Deliverable | Status |
|--------|-------------|--------|
| P3 | Activity classifier + segmenter working | ☐ |
| P1 | Perception refined, optional depth added | ☐ |
| P2 | Scene classifier working | ☐ |
| P4 | Agent answers queries correctly | ☐ |
| P5 | End-to-end pipeline processes video | ☐ |

**Exit Criteria:**
```python
pipeline = SiteIQPipeline()
report = pipeline.process_video("sample.mp4")
print(report.productivity_score)  # e.g., 0.73
print(report.activity_breakdown)  # {"ACTIVE_TOOL_USE": "2h 15m", ...}

agent = ProductivityAgent(report.segments)
response = agent.chat("What tools were used most?")
print(response)  # Natural language answer
```

---

## Phase 3: Demo & Polish (Hours 18-30)

**Goal:** Impressive, reliable demo that tells a compelling story

**Dependencies:** Phase 2 pipeline working end-to-end

### Hours 18-22: Integration & Bug Fixing (All Team)

- [ ] Connect all components
- [ ] Fix integration bugs
- [ ] Test on multiple videos
- [ ] Handle edge cases
- [ ] Ensure stability for demo

---

### Hours 22-28: Demo Polish

#### P5 + P4: Demo Interface

**File:** `src/demo/dashboard.py`

```python
import streamlit as st

st.title("SiteIQ - Egocentric Productivity Intelligence")

# Sidebar: Upload or select video
video = st.file_uploader("Upload hardhat video")

# Main area: Tabs
tab1, tab2, tab3 = st.tabs(["Video Analysis", "Metrics", "Ask SiteIQ"])

with tab1:
    # Video player with real-time annotations

with tab2:
    # Productivity dashboard
    # - Overall score (big number)
    # - Activity breakdown (pie chart)
    # - Timeline view
    # - Tool usage stats

with tab3:
    # Chat interface with agent
```

#### P1 + P2: Robustness & Speed

- [ ] Optimize for real-time processing
- [ ] Add fallbacks for detection failures
- [ ] Test on varied lighting conditions
- [ ] Create confidence indicators

#### P3: Metrics & Insights

```python
class InsightGenerator:
    def generate_insights(self, report: SessionReport) -> List[str]:
        insights = []

        if report.idle_percentage > 0.2:
            insights.append(f"High idle time ({report.idle_percentage:.0%})")

        if report.tool_switches > 10:
            insights.append(f"Frequent tool switching ({report.tool_switches} times)")

        peak = report.get_peak_productivity_period()
        insights.append(f"Peak productivity: {peak.start}-{peak.end}")

        return insights
```

---

### Phase 3 Checkpoint (Hours 28-30)

**Full Demo Rehearsal**

1. Show video with detections
2. Show productivity dashboard
3. Ask agent 3-4 questions
4. Show insights/recommendations

Fix any issues found during rehearsal.

---

## Phase 4: Final Prep (Hours 30-36)

### Hours 30-34: Presentation & Backup

| Task | Owner | Hours |
|------|-------|-------|
| Slide deck (5-7 slides) | P5 | 30-32 |
| Demo script (exact flow) | P4 | 30-32 |
| Backup video (pre-recorded) | P1 | 30-32 |
| Technical talking points | P2, P3 | 30-32 |
| Practice run #1 | All | 32-33 |
| Practice run #2 | All | 33-34 |

---

### Demo Script

```
[SLIDE 1] Problem (30 sec)
"Supervisors need to know: Is this worker productive?
Current AI can't answer this from egocentric video."

[SLIDE 2] Our Solution (30 sec)
"SiteIQ combines hand-object interaction detection
with temporal activity analysis."

[LIVE DEMO] (2-3 min)
1. Show raw video → "Current AI sees nothing useful"
2. Show our annotated video → "We detect hands, tools, interactions"
3. Show dashboard → "Automatic productivity metrics"
4. Chat with agent → "Ask any question about the shift"

[SLIDE 3] Technical Innovation (30 sec)
"Key insight: Egocentric HOI + temporal state machine + LLM agent"

[SLIDE 4] Results (30 sec)
"X% accuracy on activity classification, real-time processing"

[Q&A]
```

---

### Hours 34-36: Buffer & Rest

- [ ] Final testing
- [ ] Ensure demo machine is ready
- [ ] Charge laptops
- [ ] Get some rest before presentation
- [ ] Have backup plan ready

---

## Summary Timeline

```
HOUR   PHASE             P1          P2          P3          P4          P5
─────────────────────────────────────────────────────────────────────────────
0-2    Setup             ◆─────────────────── ALL TOGETHER ───────────────◆
─────────────────────────────────────────────────────────────────────────────
2-8    Perception        Hand Det.   Tool Det.   Schemas     Agent Arch  Visualizer
8-10   Integration       HOI ◄───────┤           Motion      Prompts     Dashboard
─────────────────────────────────────────────────────────────────────────────
10     ★ CHECKPOINT 1    ◆─────────────────── TEAM SYNC ──────────────────◆
─────────────────────────────────────────────────────────────────────────────
10-14  Refinement        Tune/Depth  Scene Class State Mach. Agent Tools Pipeline
14-18  Temporal          Speed Opt   Validation  Segmenter   LLM Connect Integration
─────────────────────────────────────────────────────────────────────────────
18     ★ CHECKPOINT 2    ◆─────────────────── TEAM SYNC ──────────────────◆
─────────────────────────────────────────────────────────────────────────────
18-22  Bug Fixing        ◆─────────────────── ALL TOGETHER ───────────────◆
22-28  Demo Polish       Robustness  Robustness  Insights    Demo UI     Demo UI
─────────────────────────────────────────────────────────────────────────────
28-30  ★ CHECKPOINT 3    ◆────────────────── DEMO REHEARSAL ──────────────◆
─────────────────────────────────────────────────────────────────────────────
30-34  Final Prep        Backup Vid  Tech Points Tech Points Demo Script Slides
34-36  Buffer            ◆─────────────────── REST & READY ───────────────◆
```

---

## Dependencies

### Python Packages (requirements.txt)

```
# Core
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0

# Hand Detection
mediapipe>=0.10.0

# Object Detection
torch>=2.0.0
transformers>=4.35.0
groundingdino  # or ultralytics for YOLO

# Depth Estimation (optional)
# depth-anything-v2

# Agent
openai>=1.0.0
# anthropic>=0.18.0

# Demo
streamlit>=1.28.0
plotly>=5.18.0

# Utilities
tqdm>=4.66.0
python-dotenv>=1.0.0
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Grounding DINO too slow | Fall back to YOLOv8 |
| Hand detection fails with gloves | Tune confidence, add fallback heuristics |
| Video quality issues | Pre-process frames, adjust thresholds |
| Demo crashes | Pre-recorded backup video ready |
| Agent gives wrong answers | Constrain responses, add validation |

---

## Success Criteria

1. **Working Demo:** End-to-end pipeline processes video and shows results
2. **Accurate Detection:** Hands and tools detected in >80% of relevant frames
3. **Meaningful Metrics:** Productivity scores correlate with visible activity
4. **Interactive Agent:** Can answer 5+ different query types accurately
5. **Compelling Story:** Clear before/after showing improvement over baseline VLMs
