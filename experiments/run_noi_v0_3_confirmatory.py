"""Per-seed evaluator for locked NOI v0.3 confirmatory execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.evidence_conflict import (
    EvidenceConflictDetector,
)
from src.evaluation.multisensory_conditions import (
    ConditionGenerationResult,
)
from src.evaluation.multisensory_records import (
    MultisensorySplit,
)
from src.evaluation.noi_v0_3_retrieval import (
    NOIV03System,
    NOIV03SystemPolicy,
)
from src.evaluation.reliability_fusion import (
    LockedFusionConfig,
    fuse_locked_evidence,
)
from src.evaluation.support_gate import (
    SupportGate,
    SupportMethod,
    apply_locked_support_threshold,
)


class ConfirmatoryExecutionError(ValueError):
    """Raised when confirmatory evaluation inputs are invalid."""


@dataclass(frozen=True, slots=True)
class ConfirmatorySeedReport:
    """Deterministic results for one independent generator seed."""

    seed: int
    final_latent_event_count: int
    condition_view_count: int
    system_count: int
    system_evaluation_count: int
    locked_values: dict[str, float]
    view_results: tuple[dict[str, Any], ...]
    system_summaries: dict[str, dict[str, float | int]]
    condition_summaries: dict[
        str,
        dict[str, dict[str, float | int]],
    ]
    support_regime_summaries: dict[
        str,
        dict[str, dict[str, float | int]],
    ]
    integrity: dict[str, bool | int]


_REQUIRED_LOCKED_VALUES = (
    "support_threshold",
    "support_uncertainty_lower",
    "support_uncertainty_upper",
    "reliability_threshold",
    "conflict_threshold",
)


def score_to_confidence(
    *,
    top_score: float | None,
    abstained: bool,
) -> float:
    """Apply the preconfirmatory clipped cosine confidence rule."""

    if not isinstance(abstained, bool):
        raise ConfirmatoryExecutionError(
            "abstained must be a Boolean."
        )

    if abstained:
        return 0.0

    if (
        isinstance(top_score, bool)
        or not isinstance(top_score, Real)
        or not math.isfinite(float(top_score))
    ):
        raise ConfirmatoryExecutionError(
            "A non-abstaining top_score must be finite."
        )

    return min(
        1.0,
        max(
            0.0,
            (float(top_score) + 1.0) / 2.0,
        ),
    )


def _locked_values(
    values: Mapping[str, float],
) -> dict[str, float]:
    """Validate and copy the five seedwise frozen values."""

    if not isinstance(values, Mapping):
        raise ConfirmatoryExecutionError(
            "locked_values must be a mapping."
        )

    if set(values) != set(_REQUIRED_LOCKED_VALUES):
        raise ConfirmatoryExecutionError(
            "locked_values must contain exactly the five "
            "registered seedwise values."
        )

    converted: dict[str, float] = {}

    for name in _REQUIRED_LOCKED_VALUES:
        value = values[name]

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ConfirmatoryExecutionError(
                f"locked value {name} must be finite."
            )

        converted[name] = float(value)

    if not (
        converted["support_uncertainty_lower"]
        <= converted["support_threshold"]
        <= converted["support_uncertainty_upper"]
    ):
        raise ConfirmatoryExecutionError(
            "locked support values must satisfy lower <= threshold <= upper."
        )

    for name in (
        "reliability_threshold",
        "conflict_threshold",
    ):
        if not 0.0 <= converted[name] <= 1.0:
            raise ConfirmatoryExecutionError(
                f"locked value {name} must be between 0 and 1."
            )

    return converted


def _reciprocal_rank(
    ranking: tuple[str, ...],
    target_item_id: str,
) -> float:
    """Return exact-item reciprocal rank for posthoc scoring."""

    for rank, item_id in enumerate(ranking, start=1):
        if item_id == target_item_id:
            return 1.0 / rank

    return 0.0


def _summary(
    records: tuple[dict[str, Any], ...],
    *,
    false_confident_threshold: float,
) -> dict[str, float | int]:
    """Summarize one deterministic record stratum."""

    if not records:
        raise ConfirmatoryExecutionError(
            "Cannot summarize an empty record stratum."
        )

    count = len(records)
    nonabstained = sum(
        not bool(record["abstained"])
        for record in records
    )
    correct = sum(
        bool(record["correct"])
        for record in records
    )
    false_confident = sum(
        (not bool(record["correct"]))
        and (not bool(record["abstained"]))
        and float(record["confidence"])
        >= false_confident_threshold
        for record in records
    )

    return {
        "evaluation_count": count,
        "nonabstained_count": nonabstained,
        "abstained_count": count - nonabstained,
        "coverage": nonabstained / count,
        "accuracy": correct / count,
        "mean_reciprocal_rank": (
            sum(
                float(record["reciprocal_rank"])
                for record in records
            )
            / count
        ),
        "mean_confidence": (
            sum(
                float(record["confidence"])
                for record in records
            )
            / count
        ),
        "false_confident_count": false_confident,
        "false_confident_rate": false_confident / count,
        "touch_request_count": sum(
            bool(record["touch_requested"])
            for record in records
        ),
        "touch_request_rate": (
            sum(
                bool(record["touch_requested"])
                for record in records
            )
            / count
        ),
    }


def _group_summaries(
    records: tuple[dict[str, Any], ...],
    *,
    group_field: str,
    expected_groups: tuple[str, ...],
    false_confident_threshold: float,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Build group-by-system summaries with fixed expected strata."""

    output: dict[
        str,
        dict[str, dict[str, float | int]],
    ] = {}

    for group in expected_groups:
        output[group] = {}

        for system in NOIV03System:
            selected = tuple(
                record
                for record in records
                if record[group_field] == group
                and record["system"] == system.value
            )

            output[group][system.value] = _summary(
                selected,
                false_confident_threshold=(
                    false_confident_threshold
                ),
            )

    return output


def evaluate_confirmatory_seed(
    *,
    generated: Any,
    condition_result: ConditionGenerationResult,
    locked_values: Mapping[str, float],
    top_k: int,
    false_confident_threshold: float,
) -> ConfirmatorySeedReport:
    """Evaluate one generated seed under frozen metadata-blind rules."""

    values = _locked_values(locked_values)

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
    ):
        raise ConfirmatoryExecutionError(
            "top_k must be a positive integer."
        )

    if (
        isinstance(false_confident_threshold, bool)
        or not isinstance(false_confident_threshold, Real)
        or not math.isfinite(
            float(false_confident_threshold)
        )
        or not 0.0
        <= float(false_confident_threshold)
        <= 1.0
    ):
        raise ConfirmatoryExecutionError(
            "false_confident_threshold must be finite "
            "and between 0 and 1."
        )

    if not isinstance(
        condition_result,
        ConditionGenerationResult,
    ):
        raise ConfirmatoryExecutionError(
            "condition_result must be a ConditionGenerationResult."
        )

    try:
        latent_events = tuple(generated.latent_events)
        targets = tuple(generated.targets)
        generator_seeds = {
            int(event.generator_seed)
            for event in latent_events
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ConfirmatoryExecutionError(
            "generated must be a validated NOI v0.3 generation result."
        ) from error

    if len(generator_seeds) != 1:
        raise ConfirmatoryExecutionError(
            "All generated events must share one generator seed."
        )

    seed = next(iter(generator_seeds))

    training_events = tuple(
        event
        for event in latent_events
        if event.split is MultisensorySplit.TRAIN
    )
    final_events = tuple(
        event
        for event in latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )

    if not training_events or not final_events:
        raise ConfirmatoryExecutionError(
            "Both training and final-test events are required."
        )

    final_by_id = {
        event.latent_event_id: event
        for event in final_events
    }

    if len(final_by_id) != len(final_events):
        raise ConfirmatoryExecutionError(
            "Final latent event identifiers must be unique."
        )

    if any(
        view.latent_event_id not in final_by_id
        for view in condition_result.views
    ):
        raise ConfirmatoryExecutionError(
            "Every condition view must reference a final latent event."
        )

    policy = NOIV03SystemPolicy.fit(
        training_events=training_events,
        targets=targets,
        ridge_alpha=1.0,
    )

    support_gate = SupportGate(
        method=SupportMethod.MAHALANOBIS,
    )
    support_gate.fit(training_events)

    evidence_detector = EvidenceConflictDetector()
    evidence_detector.fit(training_events)

    fusion_config = LockedFusionConfig(
        reliability_threshold=values[
            "reliability_threshold"
        ],
        conflict_threshold=values[
            "conflict_threshold"
        ],
        generator_version=(
            condition_result.provenance.generator_version
        ),
    )

    view_results: list[dict[str, Any]] = []

    for view in sorted(
        condition_result.views,
        key=lambda item: item.view_id,
    ):
        latent = final_by_id[view.latent_event_id]

        if view.olfactory_vector is None:
            support_score = values["support_threshold"]
        else:
            support_score = support_gate.score(
                view.olfactory_vector
            )

        support_decision = apply_locked_support_threshold(
            event_id=view.view_id,
            method=SupportMethod.MAHALANOBIS,
            support_score=support_score,
            threshold=values["support_threshold"],
            uncertainty_lower=values[
                "support_uncertainty_lower"
            ],
            uncertainty_upper=values[
                "support_uncertainty_upper"
            ],
        )

        evidence = evidence_detector.assess(
            olfactory_vector=view.olfactory_vector,
            tactile_vector=view.tactile_vector,
        )

        fusion_decision = fuse_locked_evidence(
            event_id=view.view_id,
            olfactory_vector=view.olfactory_vector,
            tactile_vector=view.tactile_vector,
            temporal_offset_steps=(
                view.temporal_offset_steps
            ),
            evidence=evidence,
            config=fusion_config,
        )

        for system in NOIV03System:
            system_result = policy.evaluate(
                system=system,
                event_id=view.view_id,
                olfactory_vector=view.olfactory_vector,
                tactile_vector=view.tactile_vector,
                support_decision=support_decision,
                fusion_decision=fusion_decision,
                top_k=top_k,
            )
            retrieval = system_result.retrieval
            reciprocal_rank = _reciprocal_rank(
                retrieval.ranking,
                latent.target_item_id,
            )
            correct = (
                not retrieval.abstained
                and bool(retrieval.ranking)
                and retrieval.ranking[0]
                == latent.target_item_id
            )
            top_score = (
                retrieval.scores[0]
                if retrieval.scores
                else None
            )
            confidence = score_to_confidence(
                top_score=top_score,
                abstained=retrieval.abstained,
            )

            touch_requested = (
                system
                is NOIV03System
                .SUPPORT_GATE_RELIABILITY_FUSION_WITH_ABSTENTION
                and (
                    support_decision.request_touch
                    or (
                        view.tactile_vector is not None
                        and fusion_decision.odor_reliability
                        < values["reliability_threshold"]
                    )
                )
            )

            view_results.append(
                {
                    "view_id": view.view_id,
                    "latent_event_id": view.latent_event_id,
                    "statistical_unit": "latent_event_id",
                    "condition": view.condition.value,
                    "support_regime": (
                        latent.support_regime.value
                    ),
                    "system": system.value,
                    "target_item_id": latent.target_item_id,
                    "target_family_id": latent.target_family_id,
                    "ranking": list(retrieval.ranking),
                    "scores": list(retrieval.scores),
                    "abstained": retrieval.abstained,
                    "correct": correct,
                    "reciprocal_rank": reciprocal_rank,
                    "confidence": confidence,
                    "odor_weight": retrieval.odor_weight,
                    "touch_weight": retrieval.touch_weight,
                    "touch_requested": touch_requested,
                    "support_score": support_decision.support_score,
                    "predicted_supported": (
                        support_decision.is_supported
                    ),
                    "true_supported": (
                        latent.support_regime.value
                        == "seen_item"
                    ),
                    "support_uncertainty_status": (
                        support_decision
                        .uncertainty_status.value
                    ),
                    "conflict_score": (
                        fusion_decision.conflict_score
                    ),
                    "conflict_detected": (
                        fusion_decision.conflict_detected
                    ),
                    "temporal_conflict_detected": (
                        fusion_decision
                        .temporal_conflict_detected
                    ),
                }
            )

    records = tuple(view_results)
    expected_count = (
        len(condition_result.views)
        * len(tuple(NOIV03System))
    )

    if len(records) != expected_count:
        raise ConfirmatoryExecutionError(
            "Every condition view must be evaluated by all systems."
        )

    threshold = float(false_confident_threshold)

    system_summaries = {
        system.value: _summary(
            tuple(
                record
                for record in records
                if record["system"] == system.value
            ),
            false_confident_threshold=threshold,
        )
        for system in NOIV03System
    }

    condition_summaries = _group_summaries(
        records,
        group_field="condition",
        expected_groups=(
            "clean",
            "degraded_odor",
            "degraded_touch",
            "missing_touch",
            "missing_odor",
            "contradictory_modalities",
            "temporal_misalignment",
        ),
        false_confident_threshold=threshold,
    )

    support_summaries = _group_summaries(
        records,
        group_field="support_regime",
        expected_groups=(
            "seen_item",
            "known_family_unseen_item",
            "unseen_family",
        ),
        false_confident_threshold=threshold,
    )

    return ConfirmatorySeedReport(
        seed=seed,
        final_latent_event_count=len(final_events),
        condition_view_count=len(condition_result.views),
        system_count=len(tuple(NOIV03System)),
        system_evaluation_count=len(records),
        locked_values=values,
        view_results=records,
        system_summaries=system_summaries,
        condition_summaries=condition_summaries,
        support_regime_summaries=support_summaries,
        integrity={
            "training_events_used": len(training_events),
            "validation_events_used_for_fitting": 0,
            "final_test_labels_used_for_training": False,
            "final_test_labels_used_for_calibration": False,
            "final_test_labels_used_for_scoring_only": True,
            "condition_metadata_used_as_model_input": False,
            "target_labels_used_as_inference_input": False,
            "quality_metadata_used_as_model_input": False,
            "paired_views_treated_as_independent": False,
            "thresholds_changed_from_final_test": False,
        },
    )


def build_confirmatory_seed_payload(
    report: ConfirmatorySeedReport,
) -> dict[str, Any]:
    """Return the canonical deterministic per-seed payload."""

    if not isinstance(report, ConfirmatorySeedReport):
        raise ConfirmatoryExecutionError(
            "report must be a ConfirmatorySeedReport."
        )

    payload = {
        "schema_version": "noi-v0.3-confirmatory-seed-v1",
        "study_phase": "confirmatory",
        "confirmatory_execution": True,
        **asdict(report),
    }

    return json.loads(
        json.dumps(
            payload,
            sort_keys=True,
            allow_nan=False,
        )
    )


def _validate_payload(
    payload: Mapping[str, Any],
) -> None:
    """Validate the top-level export schema."""

    if not isinstance(payload, Mapping):
        raise ConfirmatoryExecutionError(
            "payload must be a mapping."
        )

    if payload.get("schema_version") != (
        "noi-v0.3-confirmatory-seed-v1"
    ):
        raise ConfirmatoryExecutionError(
            "payload has an invalid confirmatory schema."
        )

    if payload.get("study_phase") != "confirmatory":
        raise ConfirmatoryExecutionError(
            "payload must describe confirmatory execution."
        )

    if payload.get("confirmatory_execution") is not True:
        raise ConfirmatoryExecutionError(
            "payload must mark confirmatory execution."
        )


def export_confirmatory_seed_payload(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Write canonical JSON while preventing silent reruns."""

    _validate_payload(payload)

    if not isinstance(output_path, Path):
        raise ConfirmatoryExecutionError(
            "output_path must be a Path."
        )

    if not isinstance(overwrite, bool):
        raise ConfirmatoryExecutionError(
            "overwrite must be a Boolean."
        )

    if output_path.exists() and not overwrite:
        raise ConfirmatoryExecutionError(
            f"Output already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized = json.dumps(
        dict(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"

    output_path.write_text(
        serialized,
        encoding="utf-8",
    )
