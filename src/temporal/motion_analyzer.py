"""
Motion Analysis Module (P3)
Analyzes camera motion patterns to help classify worker activities.

In egocentric video, camera motion reveals worker behavior:
- Stable: Focused work or idle
- Rhythmic: Repetitive tasks (hammering, drilling)
- Panning: Looking around, searching
- Walking: Traveling to different location
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np
import cv2


class MotionType(Enum):
    """Types of camera motion patterns."""
    STABLE = "stable"  # Minimal movement - focused work or idle
    RHYTHMIC = "rhythmic"  # Periodic motion - repetitive tasks
    PANNING = "panning"  # Smooth scanning - looking around
    WALKING = "walking"  # Consistent forward motion - traveling
    UNKNOWN = "unknown"  # Unable to classify


@dataclass
class MotionResult:
    """Result of motion analysis."""
    motion_type: MotionType
    confidence: float  # 0.0 to 1.0
    magnitude: float  # Overall motion intensity
    direction: Optional[Tuple[float, float]]  # Average motion vector (dx, dy)
    frequency: Optional[float]  # For rhythmic motion, cycles per second
    metadata: dict  # Additional metrics


class MotionAnalyzer:
    """
    Analyzes camera motion in egocentric video using optical flow.

    Key insight: In hardhat camera footage, motion patterns reveal activity:
    - Stable camera + hands visible = active work
    - Rhythmic motion = repetitive task (hammering, drilling)
    - Smooth panning = searching for tools/materials
    - Forward walking motion = traveling between locations
    """

    # Thresholds for motion classification
    STABLE_THRESHOLD = 2.0  # Below this = stable
    WALKING_MAGNITUDE_MIN = 5.0  # Walking has consistent motion
    WALKING_MAGNITUDE_MAX = 25.0
    RHYTHMIC_FREQUENCY_MIN = 0.5  # Hz - minimum for rhythmic
    RHYTHMIC_FREQUENCY_MAX = 5.0  # Hz - maximum for rhythmic
    PANNING_SMOOTHNESS_THRESHOLD = 0.7  # How smooth the motion is

    def __init__(
        self,
        window_size: int = 10,  # Number of frames to analyze
        sample_rate: float = 30.0,  # Assumed FPS
    ):
        """
        Initialize motion analyzer.

        Args:
            window_size: Number of frames to analyze together
            sample_rate: Video frame rate (FPS)
        """
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.prev_gray = None
        self.motion_history = []  # Store recent motion magnitudes

    def analyze(
        self,
        frames: List[np.ndarray],
        timestamps: Optional[List[float]] = None,
    ) -> MotionResult:
        """
        Analyze motion across a sequence of frames.

        Args:
            frames: List of BGR frames (most recent last)
            timestamps: Optional timestamps for each frame

        Returns:
            MotionResult with classified motion type
        """
        if len(frames) < 2:
            return MotionResult(
                motion_type=MotionType.UNKNOWN,
                confidence=0.0,
                magnitude=0.0,
                direction=None,
                frequency=None,
                metadata={"error": "Need at least 2 frames"}
            )

        # Calculate optical flow between consecutive frames
        flow_data = self._calculate_optical_flow(frames)

        if not flow_data:
            return MotionResult(
                motion_type=MotionType.UNKNOWN,
                confidence=0.0,
                magnitude=0.0,
                direction=None,
                frequency=None,
                metadata={"error": "Failed to calculate optical flow"}
            )

        # Extract motion features
        magnitudes = [f["magnitude"] for f in flow_data]
        directions = [f["direction"] for f in flow_data]

        avg_magnitude = np.mean(magnitudes)
        std_magnitude = np.std(magnitudes)

        # Update motion history
        self.motion_history.extend(magnitudes)
        self.motion_history = self.motion_history[-self.window_size:]

        # Classify motion type
        motion_type, confidence, metadata = self._classify_motion(
            magnitudes=magnitudes,
            directions=directions,
            avg_magnitude=avg_magnitude,
            std_magnitude=std_magnitude,
            timestamps=timestamps,
        )

        # Calculate average direction
        avg_direction = self._calculate_average_direction(directions)

        # Detect frequency for rhythmic motion
        frequency = None
        if motion_type == MotionType.RHYTHMIC:
            frequency = self._detect_frequency(magnitudes, self.sample_rate)
            metadata["frequency_hz"] = frequency

        return MotionResult(
            motion_type=motion_type,
            confidence=confidence,
            magnitude=avg_magnitude,
            direction=avg_direction,
            frequency=frequency,
            metadata=metadata,
        )

    def _calculate_optical_flow(
        self,
        frames: List[np.ndarray],
    ) -> List[dict]:
        """
        Calculate dense optical flow between consecutive frames.

        Returns:
            List of flow data dictionaries
        """
        flow_data = []

        for i in range(len(frames) - 1):
            prev_frame = frames[i]
            next_frame = frames[i + 1]

            # Convert to grayscale
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                next_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            # Calculate magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Average flow magnitude (overall motion intensity)
            avg_mag = np.mean(magnitude)

            # Dominant flow direction (median angle)
            median_angle = np.median(angle)

            # Convert to direction vector
            direction = (
                np.cos(median_angle) * avg_mag,
                np.sin(median_angle) * avg_mag,
            )

            flow_data.append({
                "magnitude": avg_mag,
                "direction": direction,
                "angle": median_angle,
                "flow_field": flow,
            })

        return flow_data

    def _classify_motion(
        self,
        magnitudes: List[float],
        directions: List[Tuple[float, float]],
        avg_magnitude: float,
        std_magnitude: float,
        timestamps: Optional[List[float]],
    ) -> Tuple[MotionType, float, dict]:
        """
        Classify motion type based on features.

        Returns:
            (motion_type, confidence, metadata)
        """
        metadata = {
            "avg_magnitude": avg_magnitude,
            "std_magnitude": std_magnitude,
        }

        # Check for stable (minimal motion)
        if avg_magnitude < self.STABLE_THRESHOLD:
            confidence = 1.0 - (avg_magnitude / self.STABLE_THRESHOLD)
            return MotionType.STABLE, confidence, metadata

        # Calculate smoothness of motion
        smoothness = self._calculate_smoothness(directions)
        metadata["smoothness"] = smoothness

        # Check for rhythmic motion (periodic pattern)
        is_rhythmic, rhythm_confidence = self._detect_rhythmic_pattern(magnitudes)
        metadata["rhythmic_score"] = rhythm_confidence

        if is_rhythmic and rhythm_confidence > 0.6:
            return MotionType.RHYTHMIC, rhythm_confidence, metadata

        # Check for walking (consistent forward motion with typical magnitude)
        is_walking, walk_confidence = self._detect_walking(
            avg_magnitude, std_magnitude, smoothness
        )
        metadata["walking_score"] = walk_confidence

        if is_walking and walk_confidence > 0.6:
            return MotionType.WALKING, walk_confidence, metadata

        # Check for panning (smooth, continuous motion)
        if smoothness > self.PANNING_SMOOTHNESS_THRESHOLD and avg_magnitude > self.STABLE_THRESHOLD:
            confidence = smoothness * (avg_magnitude / 10.0)
            confidence = min(1.0, confidence)
            return MotionType.PANNING, confidence, metadata

        # Default to unknown
        return MotionType.UNKNOWN, 0.5, metadata

    def _calculate_smoothness(self, directions: List[Tuple[float, float]]) -> float:
        """
        Calculate smoothness of motion (0=jerky, 1=smooth).

        Smooth motion has consistent direction.
        """
        if len(directions) < 2:
            return 0.0

        # Calculate angle differences between consecutive directions
        angles = [np.arctan2(d[1], d[0]) for d in directions]
        angle_diffs = []

        for i in range(len(angles) - 1):
            diff = abs(angles[i+1] - angles[i])
            # Normalize to [0, pi]
            diff = min(diff, 2 * np.pi - diff)
            angle_diffs.append(diff)

        # Low variance = smooth
        if len(angle_diffs) == 0:
            return 0.0

        avg_diff = np.mean(angle_diffs)
        # Map to 0-1 scale (pi radians = 0, 0 radians = 1)
        smoothness = 1.0 - (avg_diff / np.pi)
        return max(0.0, min(1.0, smoothness))

    def _detect_rhythmic_pattern(
        self,
        magnitudes: List[float],
    ) -> Tuple[bool, float]:
        """
        Detect rhythmic/periodic pattern in motion.

        Returns:
            (is_rhythmic, confidence)
        """
        if len(magnitudes) < 4:
            return False, 0.0

        # Use autocorrelation to detect periodicity
        mags = np.array(magnitudes)
        mags = mags - np.mean(mags)  # Remove DC component

        # Autocorrelation
        correlation = np.correlate(mags, mags, mode='full')
        correlation = correlation[len(correlation)//2:]

        # Normalize
        if correlation[0] != 0:
            correlation = correlation / correlation[0]
        else:
            return False, 0.0

        # Find peaks (excluding the first one at lag=0)
        if len(correlation) < 3:
            return False, 0.0

        # Look for strong peak indicating periodicity
        max_peak = np.max(correlation[1:])

        # If peak is strong enough, it's rhythmic
        if max_peak > 0.5:
            return True, max_peak

        return False, max_peak

    def _detect_frequency(
        self,
        magnitudes: List[float],
        sample_rate: float,
    ) -> Optional[float]:
        """
        Detect the frequency of rhythmic motion using FFT.

        Returns:
            Frequency in Hz, or None if not detected
        """
        if len(magnitudes) < 4:
            return None

        # Perform FFT
        fft = np.fft.rfft(magnitudes - np.mean(magnitudes))
        freqs = np.fft.rfftfreq(len(magnitudes), d=1.0/sample_rate)

        # Find dominant frequency
        magnitudes_fft = np.abs(fft)

        # Ignore DC component and very high frequencies
        valid_range = (freqs >= self.RHYTHMIC_FREQUENCY_MIN) & (freqs <= self.RHYTHMIC_FREQUENCY_MAX)

        if not np.any(valid_range):
            return None

        valid_mags = magnitudes_fft[valid_range]
        valid_freqs = freqs[valid_range]

        if len(valid_mags) == 0:
            return None

        dominant_idx = np.argmax(valid_mags)
        dominant_freq = valid_freqs[dominant_idx]

        return float(dominant_freq)

    def _detect_walking(
        self,
        avg_magnitude: float,
        std_magnitude: float,
        smoothness: float,
    ) -> Tuple[bool, float]:
        """
        Detect walking motion pattern.

        Walking has:
        - Moderate consistent magnitude
        - Some variance (bobbing motion)
        - Moderate smoothness

        Returns:
            (is_walking, confidence)
        """
        # Check magnitude is in walking range
        if not (self.WALKING_MAGNITUDE_MIN <= avg_magnitude <= self.WALKING_MAGNITUDE_MAX):
            return False, 0.0

        # Walking should have some variance (not perfectly smooth)
        # but not too jerky
        if std_magnitude < 1.0 or std_magnitude > 10.0:
            return False, 0.0

        # Moderate smoothness (not perfectly smooth like panning)
        if smoothness < 0.3 or smoothness > 0.8:
            return False, 0.0

        # Calculate confidence based on how well it fits walking profile
        mag_score = 1.0 - abs(avg_magnitude - 15.0) / 15.0  # Optimal around 15
        var_score = 1.0 - abs(std_magnitude - 5.0) / 5.0  # Optimal around 5
        smooth_score = smoothness

        confidence = (mag_score + var_score + smooth_score) / 3.0
        confidence = max(0.0, min(1.0, confidence))

        is_walking = confidence > 0.6

        return is_walking, confidence

    def _calculate_average_direction(
        self,
        directions: List[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """Calculate average direction vector."""
        if not directions:
            return (0.0, 0.0)

        avg_x = np.mean([d[0] for d in directions])
        avg_y = np.mean([d[1] for d in directions])

        return (float(avg_x), float(avg_y))

    def reset(self):
        """Reset internal state."""
        self.prev_gray = None
        self.motion_history = []


# Quick test
if __name__ == "__main__":
    """
    Test motion analyzer with synthetic data.
    """
    analyzer = MotionAnalyzer(window_size=10, sample_rate=30.0)

    # Create synthetic frames with different motion patterns

    # Test 1: Stable (no motion)
    print("Test 1: Stable motion")
    stable_frames = [np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
    result = analyzer.analyze(stable_frames)
    print(f"  Type: {result.motion_type.value}, Confidence: {result.confidence:.2f}, Magnitude: {result.magnitude:.2f}")

    # Test 2: Panning (smooth horizontal motion)
    print("\nTest 2: Panning motion")
    panning_frames = []
    base_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    for i in range(10):
        # Shift frame horizontally
        shifted = np.roll(base_frame, shift=i*20, axis=1)
        panning_frames.append(shifted)
    result = analyzer.analyze(panning_frames)
    print(f"  Type: {result.motion_type.value}, Confidence: {result.confidence:.2f}, Magnitude: {result.magnitude:.2f}")

    print("\nMotion analyzer tests complete!")
