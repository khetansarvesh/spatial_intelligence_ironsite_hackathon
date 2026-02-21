"""
Scene Context Classification Module (P2)
Infers construction scene type from detected objects and validates tool-scene fit.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class SceneClassificationResult:
    """Result of scene classification."""
    scene: str
    confidence: float
    matched_objects: List[str]
    scene_scores: Dict[str, float]


class SceneClassifier:
    """
    Classifies construction scene context from detected object labels.

    Example:
    - Objects: ["wire", "panel", "screwdriver"]
    - Scene: "electrical"
    """

    SCENE_TYPES: Dict[str, List[str]] = {
        "framing": ["metal stud", "lumber", "drywall", "screw", "drill", "level"],
        "electrical": ["wire", "conduit", "junction box", "panel", "cable", "screwdriver"],
        "plumbing": ["pipe", "fitting", "valve", "wrench", "tube", "pvc"],
        "finishing": ["paint", "trim", "tape", "sander", "brush", "roller"],
        "masonry": [
            "brick",
            "block",
            "cinder block",
            "concrete block",
            "cmu",
            "mortar",
            "grout",
            "trowel",
            "masonry wall",
            "rebar",
        ],
    }

    # Tool compatibility map used by validate_tool_for_scene()
    SCENE_TOOL_HINTS: Dict[str, Set[str]] = {
        "framing": {"drill", "screw gun", "nail gun", "hammer", "saw", "level"},
        "electrical": {"screwdriver", "pliers", "drill", "wire stripper", "multimeter"},
        "plumbing": {"wrench", "pliers", "pipe cutter", "drill", "saw"},
        "finishing": {"sander", "paint sprayer", "brush", "roller", "tape"},
        "masonry": {"trowel", "masonry trowel", "float", "hammer", "level", "mason line"},
    }

    def __init__(self, min_confidence: float = 0.25):
        """
        Initialize scene classifier.

        Args:
            min_confidence: Minimum confidence required to return a specific scene.
                            Otherwise returns "unknown".
        """
        self.min_confidence = min_confidence

    def classify_scene(self, detected_objects: List[str]) -> str:
        """
        Infer scene type from visible objects.

        Args:
            detected_objects: list of object labels from detector output

        Returns:
            Scene label (framing/electrical/plumbing/finishing/unknown)
        """
        result = self.classify_scene_with_confidence(detected_objects)
        return result.scene

    def classify_scene_with_confidence(
        self,
        detected_objects: List[str],
    ) -> SceneClassificationResult:
        """
        Infer scene type and return confidence details.

        Confidence is based on keyword coverage against each scene vocabulary.
        """
        normalized = [o.strip().lower() for o in detected_objects if o and o.strip()]

        if not normalized:
            return SceneClassificationResult(
                scene="unknown",
                confidence=0.0,
                matched_objects=[],
                scene_scores={name: 0.0 for name in self.SCENE_TYPES},
            )

        scene_scores: Dict[str, float] = {}
        scene_matches: Dict[str, List[str]] = {}

        for scene, keywords in self.SCENE_TYPES.items():
            matched = self._find_keyword_matches(normalized, keywords)
            score = len(matched) / max(len(keywords), 1)
            scene_scores[scene] = score
            scene_matches[scene] = matched

        best_scene, best_score = max(scene_scores.items(), key=lambda x: x[1])
        if best_score < self.min_confidence:
            return SceneClassificationResult(
                scene="unknown",
                confidence=best_score,
                matched_objects=[],
                scene_scores=scene_scores,
            )

        return SceneClassificationResult(
            scene=best_scene,
            confidence=best_score,
            matched_objects=scene_matches[best_scene],
            scene_scores=scene_scores,
        )

    def validate_tool_for_scene(self, tool: str, scene: str) -> bool:
        """
        Check whether a tool is appropriate for a given scene.

        Args:
            tool: tool label (e.g., "drill")
            scene: scene label (e.g., "framing")

        Returns:
            True if tool is compatible with the scene, else False.
        """
        if not tool or not scene:
            return False

        tool_norm = tool.strip().lower()
        scene_norm = scene.strip().lower()

        allowed_tools = self.SCENE_TOOL_HINTS.get(scene_norm)
        if not allowed_tools:
            return False

        return any(self._contains_token(tool_norm, allowed) for allowed in allowed_tools)

    def get_scene_relevance(self, tool: str, detected_objects: List[str]) -> Dict[str, bool]:
        """
        Return compatibility of a tool across all scene types.
        """
        classification = self.classify_scene_with_confidence(detected_objects)
        relevance = {
            scene: self.validate_tool_for_scene(tool, scene)
            for scene in self.SCENE_TYPES.keys()
        }
        relevance["predicted_scene"] = classification.scene  # type: ignore[assignment]
        return relevance

    def _find_keyword_matches(self, objects: List[str], keywords: List[str]) -> List[str]:
        """Find scene keywords present in object labels using relaxed token matching."""
        matches: List[str] = []
        for keyword in keywords:
            if any(self._contains_token(obj, keyword) for obj in objects):
                matches.append(keyword)
        return matches

    def _contains_token(self, text: str, token: str) -> bool:
        """Case-insensitive containment check in both directions."""
        return token in text or text in token


if __name__ == "__main__":
    classifier = SceneClassifier()
    sample_objects = ["wire", "junction box", "screwdriver", "panel"]
    result = classifier.classify_scene_with_confidence(sample_objects)

    print("Scene classification test")
    print("=" * 40)
    print(f"Objects: {sample_objects}")
    print(f"Predicted scene: {result.scene}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Matched objects: {result.matched_objects}")
    print(f"Tool valid (screwdriver): {classifier.validate_tool_for_scene('screwdriver', result.scene)}")
