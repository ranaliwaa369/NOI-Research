"""Per-seed evaluator for locked NOI v0.3 confirmatory execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.evaluation.evidence_conflict import (
    EvidenceConflictDetector,
)
from src.evaluation.multisensory_conditions import (
    ConditionGenerationConfig,
    ConditionGenerationResult,
    generate_multisensory_condition_views,
)
from src.evaluation.multisensory_records import (
    MultisensorySplit,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationConfig,
    generate_noi_v0_3_events,
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
                        not retrieval.abstained
                    ),
                    "support_gate_predicted_supported": (
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

REGISTERED_SEEDS = tuple(range(1301, 1311))

_VALIDATION_LOCK_PATH = Path(
    "configs/noi_v0.3_validation_lock.yaml"
)


def _registered_seed(seed: int) -> int:
    """Return one exact preregistered generator seed."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in REGISTERED_SEEDS
    ):
        raise ConfirmatoryExecutionError(
            "seed must be one of the registered seeds 1301 through 1310."
        )

    return seed


def generation_config_for_seed(
    seed: int,
) -> NOIV03GenerationConfig:
    """Return the full locked confirmatory allocation."""

    registered = _registered_seed(seed)

    return NOIV03GenerationConfig(
        seed=registered,
        train_event_count=7000,
        validation_event_count=1000,
        final_test_event_count=2000,
        validation_seen_item_count=400,
        validation_known_family_unseen_item_count=300,
        validation_unseen_family_count=300,
        final_seen_item_count=800,
        final_known_family_unseen_item_count=600,
        final_unseen_family_count=600,
        known_family_count=4,
        training_items_per_family=4,
        withheld_items_per_known_family=2,
        validation_unknown_family_count=2,
        final_unknown_family_count=2,
        items_per_unknown_family=3,
        generator_version="0.3.1-confirmatory",
        feasibility_only=False,
    )


def condition_config_for_seed(
    seed: int,
) -> ConditionGenerationConfig:
    """Return the locked seven-condition controls."""

    registered = _registered_seed(seed)

    return ConditionGenerationConfig(
        seed=registered,
        odor_noise_scale=0.10,
        tactile_noise_scale=0.10,
        degraded_quality=0.40,
        locked_temporal_offset_steps=3,
        generator_version="0.3.1-confirmatory",
    )


def load_seed_locked_values(
    seed: int,
) -> dict[str, float]:
    """Load exactly one seed's five frozen validation values."""

    registered = _registered_seed(seed)

    try:
        payload = yaml.safe_load(
            _VALIDATION_LOCK_PATH.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, yaml.YAMLError) as error:
        raise ConfirmatoryExecutionError(
            "Unable to load the validation-lock artifact."
        ) from error

    try:
        lock = payload["validation_lock"]
        integrity = payload["integrity"]
        raw_values = payload["values_by_seed"][
            str(registered)
        ]
    except (KeyError, TypeError) as error:
        raise ConfirmatoryExecutionError(
            "The validation-lock artifact has an invalid schema."
        ) from error

    if lock.get("status") != "validation_locked":
        raise ConfirmatoryExecutionError(
            "The protocol must remain validation_locked."
        )

    if lock.get("confirmatory_evaluation_executed") is not False:
        raise ConfirmatoryExecutionError(
            "The validation lock already records confirmatory execution."
        )

    if (
        integrity.get("final_test_events_used") != 0
        or integrity.get("final_test_labels_used") is not False
    ):
        raise ConfirmatoryExecutionError(
            "The validation lock fails final-test integrity."
        )

    return _locked_values(raw_values)


def run_registered_confirmatory_seed(
    *,
    seed: int,
    output_path: Path,
) -> dict[str, Any]:
    """Execute one registered seed exactly once and hash its result."""

    registered = _registered_seed(seed)

    if not isinstance(output_path, Path):
        raise ConfirmatoryExecutionError(
            "output_path must be a Path."
        )

    hash_path = Path(f"{output_path}.sha256")

    if output_path.exists():
        raise ConfirmatoryExecutionError(
            f"Output already exists: {output_path}"
        )

    if hash_path.exists():
        raise ConfirmatoryExecutionError(
            f"Hash output already exists: {hash_path}"
        )

    locked_values = load_seed_locked_values(
        registered
    )
    generation_config = generation_config_for_seed(
        registered
    )
    condition_config = condition_config_for_seed(
        registered
    )

    generated = generate_noi_v0_3_events(
        generation_config
    )

    final_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.FINAL_TEST
    )

    condition_result = generate_multisensory_condition_views(
        latent_events=final_events,
        targets=generated.targets,
        config=condition_config,
    )

    report = evaluate_confirmatory_seed(
        generated=generated,
        condition_result=condition_result,
        locked_values=locked_values,
        top_k=10,
        false_confident_threshold=0.80,
    )

    payload = build_confirmatory_seed_payload(
        report
    )

    export_confirmatory_seed_payload(
        payload,
        output_path,
        overwrite=False,
    )

    digest = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    hash_path.write_text(
        f"{digest}  {output_path.name}\n",
        encoding="utf-8",
    )

    return payload


def main() -> None:
    """Run one explicitly selected registered seed."""

    parser = argparse.ArgumentParser(
        description=(
            "Execute one locked NOI v0.3 confirmatory seed "
            "exactly once."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        choices=REGISTERED_SEEDS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    arguments = parser.parse_args()

    output = (
        arguments.output
        if arguments.output is not None
        else Path(
            "results/v0.3-confirmatory/"
            f"seed-{arguments.seed}.json"
        )
    )

    payload = run_registered_confirmatory_seed(
        seed=arguments.seed,
        output_path=output,
    )

    print("CONFIRMATORY SEED: COMPLETE")
    print("SEED:", payload["seed"])
    print(
        "FINAL LATENT EVENTS:",
        payload["final_latent_event_count"],
    )
    print(
        "CONDITION VIEWS:",
        payload["condition_view_count"],
    )
    print(
        "SYSTEM EVALUATIONS:",
        payload["system_evaluation_count"],
    )
    print("WROTE:", output)
    print("SHA256:", Path(f"{output}.sha256"))


if __name__ == "__main__":
    main()
