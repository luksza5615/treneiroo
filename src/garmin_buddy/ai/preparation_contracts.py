from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _validate_non_empty_string(field_name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    return cleaned


def _validate_optional_string(field_name: str, value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null")
    cleaned = value.strip()
    return cleaned or None


def _validate_string_list(
    field_name: str,
    value: Any,
    *,
    min_length: int = 0,
    max_length: int = 50,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings")
    if len(value) < min_length or len(value) > max_length:
        raise ValueError(
            f"{field_name} must contain between {min_length} and {max_length} items"
        )

    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
        cleaned.append(item.strip())
    return cleaned


def _validate_mapping(field_name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _validate_optional_float(field_name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a float or null")
    return float(value)


@dataclass(frozen=True)
class RunnerProfileArtifact:
    profile_context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_context": self.profile_context,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RunnerProfileArtifact:
        return cls(
            profile_context=_validate_non_empty_string(
                "profile_context", payload.get("profile_context")
            ),
        )


@dataclass(frozen=True)
class NormalizedPreparationContext:
    profile: RunnerProfileArtifact
    lab_summary: str
    lab_markers: dict[str, Any]
    planned_training_summary: dict[str, Any]
    executed_training_summary: dict[str, Any]
    source_provenance: list[str]
    missing_data: list[str]
    input_hash: str
    lab_fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "lab_summary": self.lab_summary,
            "lab_markers": self.lab_markers,
            "planned_training_summary": self.planned_training_summary,
            "executed_training_summary": self.executed_training_summary,
            "source_provenance": self.source_provenance,
            "missing_data": self.missing_data,
            "input_hash": self.input_hash,
            "lab_fingerprint": self.lab_fingerprint,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> NormalizedPreparationContext:
        return cls(
            profile=RunnerProfileArtifact.from_payload(
                _validate_mapping("profile", payload.get("profile"))
            ),
            lab_summary=_validate_non_empty_string(
                "lab_summary", payload.get("lab_summary", "No lab summary available.")
            ),
            lab_markers=_validate_mapping("lab_markers", payload.get("lab_markers", {})),
            planned_training_summary=_validate_mapping(
                "planned_training_summary", payload.get("planned_training_summary", {})
            ),
            executed_training_summary=_validate_mapping(
                "executed_training_summary", payload.get("executed_training_summary", {})
            ),
            source_provenance=_validate_string_list(
                "source_provenance", payload.get("source_provenance", []), min_length=1
            ),
            missing_data=_validate_string_list("missing_data", payload.get("missing_data", [])),
            input_hash=_validate_non_empty_string("input_hash", payload.get("input_hash")),
            lab_fingerprint=_validate_optional_string(
                "lab_fingerprint", payload.get("lab_fingerprint")
            ),
        )


@dataclass(frozen=True)
class StageArtifactBase:
    missing_data: list[str] = field(default_factory=list)
    confidence: float | None = None

    def _common_dict(self) -> dict[str, Any]:
        return {
            "missing_data": self.missing_data,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class LabAnalysisArtifact(StageArtifactBase):
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    training_implications: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self._common_dict()
        payload.update(
            {
                "summary": self.summary,
                "findings": self.findings,
                "training_implications": self.training_implications,
                "risk_flags": self.risk_flags,
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LabAnalysisArtifact:
        return cls(
            summary=_validate_non_empty_string("summary", payload.get("summary")),
            findings=_validate_string_list("findings", payload.get("findings", []), min_length=1),
            training_implications=_validate_string_list(
                "training_implications",
                payload.get("training_implications", []),
                min_length=1,
            ),
            risk_flags=_validate_string_list("risk_flags", payload.get("risk_flags", [])),
            missing_data=_validate_string_list("missing_data", payload.get("missing_data", [])),
            confidence=_validate_optional_float("confidence", payload.get("confidence")),
        )


@dataclass(frozen=True)
class PastPhaseReviewArtifact(StageArtifactBase):
    summary: str = ""
    adherence_summary: str = ""
    positive_patterns: list[str] = field(default_factory=list)
    execution_issues: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self._common_dict()
        payload.update(
            {
                "summary": self.summary,
                "adherence_summary": self.adherence_summary,
                "positive_patterns": self.positive_patterns,
                "execution_issues": self.execution_issues,
                "risk_flags": self.risk_flags,
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PastPhaseReviewArtifact:
        return cls(
            summary=_validate_non_empty_string("summary", payload.get("summary")),
            adherence_summary=_validate_non_empty_string(
                "adherence_summary", payload.get("adherence_summary")
            ),
            positive_patterns=_validate_string_list(
                "positive_patterns", payload.get("positive_patterns", []), min_length=1
            ),
            execution_issues=_validate_string_list(
                "execution_issues", payload.get("execution_issues", []), min_length=1
            ),
            risk_flags=_validate_string_list("risk_flags", payload.get("risk_flags", [])),
            missing_data=_validate_string_list("missing_data", payload.get("missing_data", [])),
            confidence=_validate_optional_float("confidence", payload.get("confidence")),
        )


@dataclass(frozen=True)
class PreparationSynthesisArtifact(StageArtifactBase):
    summary: str = ""
    key_constraints: list[str] = field(default_factory=list)
    key_opportunities: list[str] = field(default_factory=list)
    planning_priorities: list[str] = field(default_factory=list)
    risk_controls: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self._common_dict()
        payload.update(
            {
                "summary": self.summary,
                "key_constraints": self.key_constraints,
                "key_opportunities": self.key_opportunities,
                "planning_priorities": self.planning_priorities,
                "risk_controls": self.risk_controls,
                "assumptions": self.assumptions,
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PreparationSynthesisArtifact:
        return cls(
            summary=_validate_non_empty_string("summary", payload.get("summary")),
            key_constraints=_validate_string_list(
                "key_constraints", payload.get("key_constraints", []), min_length=1
            ),
            key_opportunities=_validate_string_list(
                "key_opportunities", payload.get("key_opportunities", []), min_length=1
            ),
            planning_priorities=_validate_string_list(
                "planning_priorities", payload.get("planning_priorities", []), min_length=1
            ),
            risk_controls=_validate_string_list(
                "risk_controls", payload.get("risk_controls", [])
            ),
            assumptions=_validate_string_list("assumptions", payload.get("assumptions", [])),
            missing_data=_validate_string_list("missing_data", payload.get("missing_data", [])),
            confidence=_validate_optional_float("confidence", payload.get("confidence")),
        )


@dataclass(frozen=True)
class StrengthPlanArtifact(StageArtifactBase):
    objectives: list[str] = field(default_factory=list)
    weekly_frequency: str = ""
    session_focuses: list[str] = field(default_factory=list)
    integration_notes: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self._common_dict()
        payload.update(
            {
                "objectives": self.objectives,
                "weekly_frequency": self.weekly_frequency,
                "session_focuses": self.session_focuses,
                "integration_notes": self.integration_notes,
                "contraindications": self.contraindications,
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> StrengthPlanArtifact:
        return cls(
            objectives=_validate_string_list(
                "objectives", payload.get("objectives", []), min_length=1
            ),
            weekly_frequency=_validate_non_empty_string(
                "weekly_frequency", payload.get("weekly_frequency")
            ),
            session_focuses=_validate_string_list(
                "session_focuses", payload.get("session_focuses", []), min_length=1
            ),
            integration_notes=_validate_string_list(
                "integration_notes", payload.get("integration_notes", [])
            ),
            contraindications=_validate_string_list(
                "contraindications", payload.get("contraindications", [])
            ),
            missing_data=_validate_string_list("missing_data", payload.get("missing_data", [])),
            confidence=_validate_optional_float("confidence", payload.get("confidence")),
        )


@dataclass(frozen=True)
class MacroStrategyArtifact(StageArtifactBase):
    strategy_id: str = ""
    planning_horizon: str = ""
    strategic_goal: str = ""
    mesocycles: list[str] = field(default_factory=list)
    progression_logic: list[str] = field(default_factory=list)
    recovery_logic: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    approval_status: str = "pending"
    input_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self._common_dict()
        payload.update(
            {
                "strategy_id": self.strategy_id,
                "planning_horizon": self.planning_horizon,
                "strategic_goal": self.strategic_goal,
                "mesocycles": self.mesocycles,
                "progression_logic": self.progression_logic,
                "recovery_logic": self.recovery_logic,
                "risks": self.risks,
                "assumptions": self.assumptions,
                "approval_status": self.approval_status,
                "input_hash": self.input_hash,
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MacroStrategyArtifact:
        approval_status = _validate_non_empty_string(
            "approval_status", payload.get("approval_status")
        )
        if approval_status not in {"pending", "approved", "stale"}:
            raise ValueError("approval_status must be one of: pending, approved, stale")
        return cls(
            strategy_id=_validate_non_empty_string("strategy_id", payload.get("strategy_id")),
            planning_horizon=_validate_non_empty_string(
                "planning_horizon", payload.get("planning_horizon")
            ),
            strategic_goal=_validate_non_empty_string(
                "strategic_goal", payload.get("strategic_goal")
            ),
            mesocycles=_validate_string_list("mesocycles", payload.get("mesocycles", []), min_length=1),
            progression_logic=_validate_string_list(
                "progression_logic", payload.get("progression_logic", []), min_length=1
            ),
            recovery_logic=_validate_string_list(
                "recovery_logic", payload.get("recovery_logic", []), min_length=1
            ),
            risks=_validate_string_list("risks", payload.get("risks", [])),
            assumptions=_validate_string_list("assumptions", payload.get("assumptions", [])),
            approval_status=approval_status,
            input_hash=_validate_non_empty_string("input_hash", payload.get("input_hash")),
            missing_data=_validate_string_list("missing_data", payload.get("missing_data", [])),
            confidence=_validate_optional_float("confidence", payload.get("confidence")),
        )


@dataclass(frozen=True)
class PhasePlanArtifact(StageArtifactBase):
    phase_length_weeks: int = 0
    weekly_goals: list[str] = field(default_factory=list)
    session_plan: list[str] = field(default_factory=list)
    strength_integration: list[str] = field(default_factory=list)
    rationale_links: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self._common_dict()
        payload.update(
            {
                "phase_length_weeks": self.phase_length_weeks,
                "weekly_goals": self.weekly_goals,
                "session_plan": self.session_plan,
                "strength_integration": self.strength_integration,
                "rationale_links": self.rationale_links,
                "risks": self.risks,
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PhasePlanArtifact:
        weeks = payload.get("phase_length_weeks")
        if isinstance(weeks, bool) or not isinstance(weeks, int) or weeks <= 0:
            raise ValueError("phase_length_weeks must be a positive integer")
        return cls(
            phase_length_weeks=weeks,
            weekly_goals=_validate_string_list(
                "weekly_goals", payload.get("weekly_goals", []), min_length=1
            ),
            session_plan=_validate_string_list(
                "session_plan", payload.get("session_plan", []), min_length=1
            ),
            strength_integration=_validate_string_list(
                "strength_integration", payload.get("strength_integration", [])
            ),
            rationale_links=_validate_string_list(
                "rationale_links", payload.get("rationale_links", []), min_length=1
            ),
            risks=_validate_string_list("risks", payload.get("risks", [])),
            missing_data=_validate_string_list("missing_data", payload.get("missing_data", [])),
            confidence=_validate_optional_float("confidence", payload.get("confidence")),
        )


@dataclass(frozen=True)
class CritiqueArtifact(StageArtifactBase):
    decision: str = "revise"
    blocking_issues: list[str] = field(default_factory=list)
    non_blocking_improvements: list[str] = field(default_factory=list)
    required_adjustments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = self._common_dict()
        payload.update(
            {
                "decision": self.decision,
                "blocking_issues": self.blocking_issues,
                "non_blocking_improvements": self.non_blocking_improvements,
                "required_adjustments": self.required_adjustments,
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CritiqueArtifact:
        decision = _validate_non_empty_string("decision", payload.get("decision"))
        if decision not in {"accept", "revise"}:
            raise ValueError("decision must be one of: accept, revise")
        return cls(
            decision=decision,
            blocking_issues=_validate_string_list(
                "blocking_issues", payload.get("blocking_issues", [])
            ),
            non_blocking_improvements=_validate_string_list(
                "non_blocking_improvements",
                payload.get("non_blocking_improvements", []),
            ),
            required_adjustments=_validate_string_list(
                "required_adjustments", payload.get("required_adjustments", [])
            ),
            missing_data=_validate_string_list("missing_data", payload.get("missing_data", [])),
            confidence=_validate_optional_float("confidence", payload.get("confidence")),
        )


@dataclass(frozen=True)
class PreparationResult:
    context: NormalizedPreparationContext
    lab_analysis: LabAnalysisArtifact
    past_phase_review: PastPhaseReviewArtifact
    synthesis: PreparationSynthesisArtifact
    strength_plan: StrengthPlanArtifact
    strategy: MacroStrategyArtifact
    phase_plan: PhasePlanArtifact | None
    critique: CritiqueArtifact | None
    parse_ok: bool
    retry_count: int
    strategy_stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "lab_analysis": self.lab_analysis.to_dict(),
            "past_phase_review": self.past_phase_review.to_dict(),
            "synthesis": self.synthesis.to_dict(),
            "strength_plan": self.strength_plan.to_dict(),
            "strategy": self.strategy.to_dict(),
            "phase_plan": self.phase_plan.to_dict() if self.phase_plan else None,
            "critique": self.critique.to_dict() if self.critique else None,
            "parse_ok": self.parse_ok,
            "retry_count": self.retry_count,
            "strategy_stale": self.strategy_stale,
        }


def build_fallback_lab_analysis(error_reason: str | None = None) -> LabAnalysisArtifact:
    missing_data = ["lab_analysis_unavailable"]
    if error_reason:
        missing_data.append(error_reason)
    return LabAnalysisArtifact(
        summary="Lab analysis could not be generated reliably.",
        findings=["Lab input was unavailable or could not be parsed reliably."],
        training_implications=["Use conservative training progression until lab context is clarified."],
        risk_flags=["lab_context_missing"],
        missing_data=missing_data,
        confidence=0.0,
    )


def build_fallback_past_phase_review(
    error_reason: str | None = None,
) -> PastPhaseReviewArtifact:
    missing_data = ["past_phase_review_unavailable"]
    if error_reason:
        missing_data.append(error_reason)
    return PastPhaseReviewArtifact(
        summary="Past phase review could not be generated reliably.",
        adherence_summary="Planned versus executed comparison is unavailable.",
        positive_patterns=["Executed data was collected for the selected period."],
        execution_issues=["Review planned versus executed adherence manually."],
        risk_flags=["phase_review_missing"],
        missing_data=missing_data,
        confidence=0.0,
    )


def build_fallback_synthesis(error_reason: str | None = None) -> PreparationSynthesisArtifact:
    missing_data = ["preparation_synthesis_unavailable"]
    if error_reason:
        missing_data.append(error_reason)
    return PreparationSynthesisArtifact(
        summary="Preparation synthesis could not be generated reliably.",
        key_constraints=["Use a conservative first phase until synthesis is regenerated."],
        key_opportunities=["Reuse simple aerobic progression and stable recovery patterns."],
        planning_priorities=["Avoid aggressive load jumps."],
        risk_controls=["Insert explicit recovery checks each week."],
        assumptions=["Some upstream conclusions were unavailable."],
        missing_data=missing_data,
        confidence=0.0,
    )


def build_fallback_strength_plan(error_reason: str | None = None) -> StrengthPlanArtifact:
    missing_data = ["strength_plan_unavailable"]
    if error_reason:
        missing_data.append(error_reason)
    return StrengthPlanArtifact(
        objectives=["Maintain general durability with low-complexity accessory work."],
        weekly_frequency="1-2 sessions per week",
        session_focuses=["Core stability", "Single-leg strength", "Hip/glute support"],
        integration_notes=["Keep strength away from the hardest run when possible."],
        contraindications=["Reduce or pause strength if fatigue spikes or pain emerges."],
        missing_data=missing_data,
        confidence=0.0,
    )


def build_fallback_macro_strategy(
    *,
    strategy_id: str,
    input_hash: str,
    error_reason: str | None = None,
) -> MacroStrategyArtifact:
    missing_data = ["macro_strategy_unavailable"]
    if error_reason:
        missing_data.append(error_reason)
    return MacroStrategyArtifact(
        strategy_id=strategy_id,
        planning_horizon="Unknown horizon",
        strategic_goal="Use a conservative progression until a valid strategy is regenerated.",
        mesocycles=["Stabilize volume", "Reassess constraints before progression"],
        progression_logic=["Increase only when recovery markers remain stable."],
        recovery_logic=["Keep a lighter week after each block of harder work."],
        risks=["Planning context is incomplete."],
        assumptions=["This is a fallback strategy artifact."],
        approval_status="pending",
        input_hash=input_hash,
        missing_data=missing_data,
        confidence=0.0,
    )


def build_fallback_phase_plan(
    *,
    weeks: int,
    error_reason: str | None = None,
) -> PhasePlanArtifact:
    missing_data = ["phase_plan_unavailable"]
    if error_reason:
        missing_data.append(error_reason)
    return PhasePlanArtifact(
        phase_length_weeks=weeks,
        weekly_goals=["Use a conservative week while the detailed plan is regenerated."],
        session_plan=["1 easy aerobic run", "1 rest day", "1 light strength session"],
        strength_integration=["Keep strength low volume and technique-focused."],
        rationale_links=["Fallback phase plan generated because the detailed artifact failed validation."],
        risks=["Detailed progression is unavailable."],
        missing_data=missing_data,
        confidence=0.0,
    )


def build_fallback_critique(error_reason: str | None = None) -> CritiqueArtifact:
    missing_data = ["phase_critique_unavailable"]
    if error_reason:
        missing_data.append(error_reason)
    return CritiqueArtifact(
        decision="revise",
        blocking_issues=["Critique output could not be generated reliably."],
        non_blocking_improvements=[],
        required_adjustments=["Review the first phase manually before execution."],
        missing_data=missing_data,
        confidence=0.0,
    )
