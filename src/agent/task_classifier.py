"""
Task Classification Module
Constrained task labeling with fixed schema and fallback rule-based inference.
"""

from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Dict, List, Optional


TASK_CLASSIFICATION_JSON_SCHEMA: Dict[str, Any] = {
    "name": "task_classification",
    "schema": {
        "type": "object",
        "properties": {
            "trade": {"type": "string"},
            "task_family": {"type": "string"},
            "task_name": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
            "unknown_flag": {"type": "boolean"},
        },
        "required": [
            "trade",
            "task_family",
            "task_name",
            "confidence",
            "reason",
            "unknown_flag",
        ],
        "additionalProperties": False,
    },
}


TASK_CLASSIFICATION_PROMPT = """You are a construction task classifier.

Given structured evidence from an egocentric construction video segment, output ONLY JSON
with this exact schema:
- trade (string)
- task_family (string)
- task_name (string)
- confidence (float 0.0-1.0)
- reason (short string grounded in evidence)
- unknown_flag (boolean)

Rules:
1) Ground your answer strictly in evidence. Do not invent objects/tools.
2) If evidence is weak/ambiguous, set unknown_flag=true and low confidence.
3) task_family should be one of:
   tool_operation, material_handling, transport, positioning_alignment,
   inspection_verification, setup_cleanup, coordination_waiting, travel, idle, unknown
4) Use concise canonical task_name values like:
   apply_mortar, place_block, align_block, pull_wire, push_cart, walk_between_zones,
   inspect_joint, cleanup_area, unknown_task
"""


@dataclass
class TaskClassification:
    trade: str
    task_family: str
    task_name: str
    confidence: float
    reason: str
    unknown_flag: bool


class TaskClassifier:
    """
    Classifies tasks from structured evidence with:
    1) LLM constrained JSON output (when configured), or
    2) Deterministic fallback rules.
    """

    ALLOWED_FAMILIES = {
        "tool_operation",
        "material_handling",
        "transport",
        "positioning_alignment",
        "inspection_verification",
        "setup_cleanup",
        "coordination_waiting",
        "travel",
        "idle",
        "unknown",
    }

    def __init__(
        self,
        provider: Optional[str] = None,  # "openai" | "anthropic" | None
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider.lower() if provider else None
        self.model = model
        self.client = None

        if self.provider == "openai":
            from openai import OpenAI

            key = api_key or os.environ.get("OPENAI_API_KEY")
            if key:
                self.client = OpenAI(api_key=key)
            if not self.model:
                self.model = "gpt-4o-mini"

        elif self.provider == "anthropic":
            from anthropic import Anthropic

            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if key:
                self.client = Anthropic(api_key=key)
            if not self.model:
                self.model = "claude-3-5-sonnet-20241022"

    def classify(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Return constrained task classification result."""
        llm_result: Optional[Dict[str, Any]] = None

        if self.client and self.provider == "openai":
            llm_result = self._classify_openai(evidence)
        elif self.client and self.provider == "anthropic":
            llm_result = self._classify_anthropic(evidence)

        if llm_result is None:
            llm_result = self._classify_rules(evidence)

        return self._sanitize(llm_result)

    def classify_batch(self, evidences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.classify(e) for e in evidences]

    def _classify_openai(self, evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Constrained JSON classification using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TASK_CLASSIFICATION_PROMPT},
                    {
                        "role": "user",
                        "content": "Evidence JSON:\n" + json.dumps(evidence, ensure_ascii=True),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": TASK_CLASSIFICATION_JSON_SCHEMA,
                },
                temperature=0.1,
            )
            content = response.choices[0].message.content
            if not content:
                return None
            return json.loads(content)
        except Exception:
            return None

    def _classify_anthropic(self, evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """JSON-only classification using Anthropic with post-parse validation."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=TASK_CLASSIFICATION_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": "Evidence JSON:\n" + json.dumps(evidence, ensure_ascii=True),
                    }
                ],
                temperature=0.1,
            )
            text_blocks = [b.text for b in response.content if getattr(b, "type", "") == "text"]
            if not text_blocks:
                return None
            return json.loads(text_blocks[0])
        except Exception:
            return None

    def _classify_rules(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Deterministic fallback classification when LLM is unavailable."""
        objects = [str(x).lower() for x in evidence.get("objects", [])]
        tools = [str(x).lower() for x in evidence.get("tools", [])]
        motion = str(evidence.get("motion", "unknown")).lower()
        interactions = str(evidence.get("interaction", "none")).lower()

        has = lambda terms: any(any(t in o for o in objects + tools) for t in terms)

        # Travel / transport
        if motion in {"walking", "moving"} and has(["wheelbarrow", "cart"]):
            return asdict(
                TaskClassification(
                    trade="general_construction",
                    task_family="transport",
                    task_name="push_cart",
                    confidence=0.8,
                    reason="Forward movement with cart-like object evidence.",
                    unknown_flag=False,
                )
            )
        if motion in {"walking", "moving"}:
            return asdict(
                TaskClassification(
                    trade="general_construction",
                    task_family="travel",
                    task_name="walk_between_zones",
                    confidence=0.72,
                    reason="Consistent walking motion without strong tool-operation evidence.",
                    unknown_flag=False,
                )
            )

        # Plumbing tasks
        if has(["pipe", "copper pipe", "pvc pipe", "pex", "fitting", "elbow", "tee", "coupling", "valve"]):
            if has(["torch", "solder", "flux"]) and interactions in {"holding", "reaching"}:
                return asdict(
                    TaskClassification(
                        trade="plumbing",
                        task_family="tool_operation",
                        task_name="join_or_solder_pipe",
                        confidence=0.84,
                        reason="Pipe/fitting evidence with soldering-related cues and active interaction.",
                        unknown_flag=False,
                    )
                )
            if interactions in {"holding", "reaching", "near"}:
                return asdict(
                    TaskClassification(
                        trade="plumbing",
                        task_family="positioning_alignment",
                        task_name="align_or_fit_pipe",
                        confidence=0.78,
                        reason="Pipe/fitting evidence with active hand interaction indicates fit-up/alignment.",
                        unknown_flag=False,
                    )
                )
            return asdict(
                TaskClassification(
                    trade="plumbing",
                    task_family="material_handling",
                    task_name="handle_pipe_or_fittings",
                    confidence=0.66,
                    reason="Plumbing components present without strong operation cue.",
                    unknown_flag=False,
                )
            )

        # Masonry tasks
        if has(["mortar", "trowel"]) and interactions in {"holding", "reaching"}:
            return asdict(
                TaskClassification(
                    trade="masonry",
                    task_family="tool_operation",
                    task_name="apply_mortar",
                    confidence=0.83,
                    reason="Masonry tool/material evidence with active interaction.",
                    unknown_flag=False,
                )
            )
        if has(["cinder block", "concrete block", "cmu", "block", "brick"]):
            return asdict(
                TaskClassification(
                    trade="masonry",
                    task_family="positioning_alignment",
                    task_name="align_or_place_block",
                    confidence=0.68,
                    reason="Block-unit evidence with stationary work posture.",
                    unknown_flag=False,
                )
            )

        # Electrical tasks
        if has(["wire", "conduit", "junction box", "panel"]):
            return asdict(
                TaskClassification(
                    trade="electrical",
                    task_family="tool_operation",
                    task_name="wire_or_panel_work",
                    confidence=0.64,
                    reason="Electrical object evidence present in segment.",
                    unknown_flag=False,
                )
            )

        # Documentation / package marking tasks
        if has(["marker", "pen", "pencil", "chalk"]) and has(["package", "box", "carton", "label", "tag", "sticker"]):
            return asdict(
                TaskClassification(
                    trade="general_construction",
                    task_family="inspection_verification",
                    task_name="mark_or_label_package",
                    confidence=0.74,
                    reason="Marking tool plus package/label evidence suggests writing or labeling task.",
                    unknown_flag=False,
                )
            )
        if has(["clipboard", "label", "tag", "sticker"]) and interactions in {"holding", "reaching", "near"}:
            return asdict(
                TaskClassification(
                    trade="general_construction",
                    task_family="inspection_verification",
                    task_name="record_or_verify_item",
                    confidence=0.66,
                    reason="Checklist/label context with hand interaction suggests recording or verification.",
                    unknown_flag=False,
                )
            )

        # Idle fallback
        if motion in {"stable", "unknown"} and interactions in {"none", "near"} and not tools:
            return asdict(
                TaskClassification(
                    trade="general_construction",
                    task_family="idle",
                    task_name="idle_or_waiting",
                    confidence=0.62,
                    reason="Low interaction and no strong tool/object cues.",
                    unknown_flag=False,
                )
            )

        # Unknown fallback
        return asdict(
            TaskClassification(
                trade="unknown",
                task_family="unknown",
                task_name="unknown_task",
                confidence=0.2,
                reason="Insufficient evidence to map to a reliable task.",
                unknown_flag=True,
            )
        )

    def _sanitize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and enforce constrained schema."""
        trade = str(payload.get("trade", "unknown")).strip() or "unknown"
        family = str(payload.get("task_family", "unknown")).strip() or "unknown"
        task = str(payload.get("task_name", "unknown_task")).strip() or "unknown_task"
        reason = str(payload.get("reason", "")).strip()
        unknown_flag = bool(payload.get("unknown_flag", False))

        try:
            confidence = float(payload.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        if family not in self.ALLOWED_FAMILIES:
            family = "unknown"
            unknown_flag = True

        if unknown_flag and confidence > 0.5:
            confidence = 0.5
        if not reason:
            reason = "Classification derived from available evidence."

        return {
            "trade": trade,
            "task_family": family,
            "task_name": task,
            "confidence": confidence,
            "reason": reason,
            "unknown_flag": unknown_flag,
        }
