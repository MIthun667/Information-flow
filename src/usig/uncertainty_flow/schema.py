"""Validated schemas for source-specific uncertainty experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence


class UncertaintySource(StrEnum):
    """Operational uncertainty-source labels."""

    LOW_UNCERTAINTY = "low_uncertainty"
    KNOWLEDGE = "knowledge"
    AMBIGUITY = "ambiguity"
    REASONING = "reasoning"
    MIXED = "mixed"


class InterventionAction(StrEnum):
    """Candidate actions available to an uncertainty-aware system."""

    ANSWER = "answer"
    RETRIEVE = "retrieve"
    CLARIFY = "clarify"
    REASON_MORE = "reason_more"
    ABSTAIN = "abstain"


class UncertaintyVariant(StrEnum):
    """Role of one record within a controlled counterfactual group."""

    ORIGINAL = "original"
    RESOLVED = "resolved"
    IRRELEVANT_CONTROL = "irrelevant_control"
    ADVERSARIAL_CONTROL = "adversarial_control"
    ALTERNATIVE_INTERPRETATION = "alternative_interpretation"


_DEFAULT_ACTION_BY_SOURCE: dict[UncertaintySource, InterventionAction] = {
    UncertaintySource.LOW_UNCERTAINTY: InterventionAction.ANSWER,
    UncertaintySource.KNOWLEDGE: InterventionAction.RETRIEVE,
    UncertaintySource.AMBIGUITY: InterventionAction.CLARIFY,
    UncertaintySource.REASONING: InterventionAction.REASON_MORE,
}


@dataclass(frozen=True, slots=True)
class UncertaintyFlowRecord:
    """One immutable record in an uncertainty-flow benchmark.

    Counterfactual variants derived from the same base question must share the
    same ``group_id``. Splitting must therefore happen at the group level.
    """

    record_id: str
    group_id: str
    base_id: str
    prompt: str
    source: UncertaintySource
    variant: UncertaintyVariant
    optimal_action: InterventionAction

    dataset_name: str
    gold_answers: tuple[str, ...] = field(default_factory=tuple)

    is_resolved_variant: bool = False
    answerable: bool = True
    intervention_cost: float = 0.0

    evidence: str | None = None
    clarification: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_nonempty("record_id", self.record_id)
        self._validate_nonempty("group_id", self.group_id)
        self._validate_nonempty("base_id", self.base_id)
        self._validate_nonempty("prompt", self.prompt)
        self._validate_nonempty("dataset_name", self.dataset_name)

        if self.intervention_cost < 0:
            raise ValueError("intervention_cost must be non-negative")

        if any(not answer.strip() for answer in self.gold_answers):
            raise ValueError("gold_answers must not contain blank answers")

        if self.variant is UncertaintyVariant.ORIGINAL and self.is_resolved_variant:
            raise ValueError(
                "an original variant cannot be marked as already resolved"
            )

        if self.variant is UncertaintyVariant.RESOLVED and not self.is_resolved_variant:
            raise ValueError(
                "a resolved variant must set is_resolved_variant=True"
            )

        if self.source is UncertaintySource.MIXED:
            mixed_sources = self.metadata.get("mixed_sources")
            if not isinstance(mixed_sources, Sequence) or isinstance(
                mixed_sources, (str, bytes)
            ):
                raise ValueError(
                    "mixed-source records must provide metadata['mixed_sources']"
                )

            normalized_sources = {
                UncertaintySource(value) for value in mixed_sources
            }

            if len(normalized_sources) < 2:
                raise ValueError(
                    "mixed-source records must contain at least two sources"
                )

            if UncertaintySource.MIXED in normalized_sources:
                raise ValueError(
                    "metadata['mixed_sources'] cannot contain 'mixed'"
                )

    @staticmethod
    def _validate_nonempty(name: str, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    @property
    def expected_default_action(self) -> InterventionAction | None:
        """Return the canonical first-pilot action for a single source."""

        return _DEFAULT_ACTION_BY_SOURCE.get(self.source)

    @property
    def is_control(self) -> bool:
        """Return whether this record is a non-resolving or adversarial control."""

        return self.variant in {
            UncertaintyVariant.IRRELEVANT_CONTROL,
            UncertaintyVariant.ADVERSARIAL_CONTROL,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record to a JSON-compatible dictionary."""

        payload = asdict(self)
        payload["source"] = self.source.value
        payload["variant"] = self.variant.value
        payload["optimal_action"] = self.optimal_action.value
        payload["gold_answers"] = list(self.gold_answers)
        payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UncertaintyFlowRecord":
        """Construct and validate a record from a JSON-compatible mapping."""

        required_fields = {
            "record_id",
            "group_id",
            "base_id",
            "prompt",
            "source",
            "variant",
            "optimal_action",
            "dataset_name",
        }

        missing = sorted(required_fields.difference(payload))
        if missing:
            raise ValueError(
                f"missing required record fields: {', '.join(missing)}"
            )

        return cls(
            record_id=str(payload["record_id"]),
            group_id=str(payload["group_id"]),
            base_id=str(payload["base_id"]),
            prompt=str(payload["prompt"]),
            source=UncertaintySource(payload["source"]),
            variant=UncertaintyVariant(payload["variant"]),
            optimal_action=InterventionAction(payload["optimal_action"]),
            dataset_name=str(payload["dataset_name"]),
            gold_answers=tuple(
                str(answer) for answer in payload.get("gold_answers", ())
            ),
            is_resolved_variant=bool(
                payload.get("is_resolved_variant", False)
            ),
            answerable=bool(payload.get("answerable", True)),
            intervention_cost=float(payload.get("intervention_cost", 0.0)),
            evidence=(
                None
                if payload.get("evidence") is None
                else str(payload["evidence"])
            ),
            clarification=(
                None
                if payload.get("clarification") is None
                else str(payload["clarification"])
            ),
            metadata=dict(payload.get("metadata", {})),
        )


def canonical_action_for_source(
    source: UncertaintySource,
) -> InterventionAction | None:
    """Return the canonical action used by the first controlled pilot."""

    return _DEFAULT_ACTION_BY_SOURCE.get(source)
