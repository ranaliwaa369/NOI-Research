"""Run the non-confirmatory NOI v0.3 feasibility pilot."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from src.evaluation.multisensory_conditions import (
    ConditionGenerationConfig,
    generate_multisensory_condition_views,
)
from src.evaluation.multisensory_records import (
    ConditionLabel,
    MultisensorySplit,
    SupportRegime,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationConfig,
    generate_noi_v0_3_events,
)
from src.evaluation.noi_v0_3_metrics import (
    calibration_metrics,
    open_set_metrics,
    risk_coverage_summary,
)
from src.evaluation.reliability_fusion import (
    FusionAction,
    FusionConfig,
    FusionMethod,
    fuse_multisensory_view,
)
from src.evaluation.support_gate import (
    SupportGate,
    SupportMethod,
)


class FeasibilityPilotError(ValueError):
    """Raised when the feasibility artifact violates its schema."""


@dataclass(frozen=True, slots=True)
class FeasibilityPilotReport:
    """Immutable wrapper around deterministic feasibility content."""

    payload: dict[str, Any]


def _generation_config() -> NOIV03GenerationConfig:
    """Return the documented reduced-size pilot allocation."""

    return NOIV03GenerationConfig(
        seed=1301,
        train_event_count=70,
        validation_event_count=10,
        final_test_event_count=20,
        validation_seen_item_count=4,
        validation_known_family_unseen_item_count=3,
        validation_unseen_family_count=3,
        final_seen_item_count=8,
        final_known_family_unseen_item_count=6,
        final_unseen_family_count=6,
        known_family_count=4,
        training_items_per_family=4,
        withheld_items_per_known_family=2,
        validation_unknown_family_count=2,
        final_unknown_family_count=2,
        items_per_unknown_family=3,
        generator_version="0.3.0-feasibility",
        feasibility_only=True,
    )


def _split_events(
    events,
    split: MultisensorySplit,
):
    """Return events from one declared split."""

    return tuple(
        event
        for event in events
        if event.split is split
    )


def _confidence_from_margin(
    score: float,
    threshold: float,
) -> float:
    """Map threshold distance to a bounded confidence for mechanics checks."""

    distance = abs(float(score) - float(threshold))

    return 0.5 + 0.5 * (
        distance / (1.0 + distance)
    )


def _leakage_audit(generated) -> dict[str, bool]:
    """Audit every prespecified split and reachability boundary."""

    training = _split_events(
        generated.latent_events,
        MultisensorySplit.TRAIN,
    )
    validation = _split_events(
        generated.latent_events,
        MultisensorySplit.VALIDATION,
    )
    final_test = _split_events(
        generated.latent_events,
        MultisensorySplit.FINAL_TEST,
    )

    training_items = {
        event.target_item_id
        for event in training
    }
    training_families = {
        event.target_family_id
        for event in training
    }

    seen_events = tuple(
        event
        for event in validation + final_test
        if event.support_regime is SupportRegime.SEEN_ITEM
    )
    known_unseen_events = tuple(
        event
        for event in validation + final_test
        if (
            event.support_regime
            is SupportRegime.KNOWN_FAMILY_UNSEEN_ITEM
        )
    )
    unseen_events = tuple(
        event
        for event in validation + final_test
        if event.support_regime is SupportRegime.UNSEEN_FAMILY
    )

    checks = {
        "seen_items_reachable": all(
            event.target_item_id in training_items
            for event in seen_events
        ),
        "known_family_items_absent_from_training": all(
            event.target_item_id not in training_items
            for event in known_unseen_events
        ),
        "known_families_reachable": all(
            event.target_family_id in training_families
            for event in known_unseen_events
        ),
        "unseen_families_absent_from_training": all(
            event.target_family_id not in training_families
            for event in unseen_events
        ),
        "validation_unknown_families_disjoint_from_final": (
            set(
                generated.reachability.validation_unknown_family_ids,
            ).isdisjoint(
                generated.reachability.final_test_unknown_family_ids,
            )
        ),
    }
    checks["passed"] = all(checks.values())

    return checks


def _condition_audit(
    *,
    final_events,
    condition_result,
) -> dict[str, Any]:
    """Verify seven paired views and invariant ground truth."""

    event_lookup = {
        event.latent_event_id: event
        for event in final_events
    }
    condition_sets: dict[str, set[ConditionLabel]] = {
        event.latent_event_id: set()
        for event in final_events
    }

    ground_truth_preserved = True

    for view in condition_result.views:
        event = event_lookup[view.latent_event_id]
        condition_sets[view.latent_event_id].add(
            view.condition,
        )

        if (
            view.target_item_id != event.target_item_id
            or view.target_family_id != event.target_family_id
        ):
            ground_truth_preserved = False

    expected_support = tuple(
        (
            event.latent_event_id,
            event.support_regime,
        )
        for event in final_events
    )

    return {
        "latent_event_count": len(final_events),
        "view_count": len(condition_result.views),
        "views_per_event": len(ConditionLabel),
        "all_conditions_present": all(
            conditions == set(ConditionLabel)
            for conditions in condition_sets.values()
        ),
        "ground_truth_preserved": ground_truth_preserved,
        "support_regimes_preserved": (
            condition_result.support_regimes
            == expected_support
        ),
    }


def _support_gate_audit(
    *,
    training_events,
    validation_events,
    final_events,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Fit, validation-calibrate, and mechanically evaluate all gates."""

    methods: dict[str, Any] = {}
    aurocs: list[float] = []
    balanced_accuracies: list[float] = []
    touch_request_rates: list[float] = []
    calibration_errors: list[float] = []
    risk_areas: list[float] = []

    true_supported = tuple(
        event.support_regime
        in {
            SupportRegime.SEEN_ITEM,
            SupportRegime.KNOWN_FAMILY_UNSEEN_ITEM,
        }
        for event in final_events
    )

    for method in SupportMethod:
        gate = SupportGate(method=method)
        gate.fit(training_events)
        calibration = gate.calibrate(
            validation_events,
            uncertainty_width=0.05,
        )

        decisions = tuple(
            gate.decide(
                event_id=event.latent_event_id,
                query_vector=event.olfactory_vector,
            )
            for event in final_events
        )
        predicted_supported = tuple(
            decision.is_supported
            for decision in decisions
        )
        scores = tuple(
            decision.support_score
            for decision in decisions
        )

        open_metrics = open_set_metrics(
            true_supported=true_supported,
            predicted_supported=predicted_supported,
            support_scores=scores,
        )
        correctness = tuple(
            predicted == actual
            for predicted, actual in zip(
                predicted_supported,
                true_supported,
                strict=True,
            )
        )
        confidences = tuple(
            _confidence_from_margin(
                decision.support_score,
                calibration.threshold,
            )
            for decision in decisions
        )
        calibration_summary = calibration_metrics(
            confidences=confidences,
            correctness=correctness,
            bin_count=5,
        )
        risk_summary = risk_coverage_summary(
            confidences=confidences,
            correctness=correctness,
        )
        touch_request_rate = (
            sum(
                decision.request_touch
                for decision in decisions
            )
            / len(decisions)
        )

        methods[method.value] = {
            "fitted_on_split": "train",
            "calibrated_on_split": "validation",
            "training_event_count": gate.training_event_count,
            "validation_event_count": (
                calibration.validation_event_count
            ),
            "final_test_labels_used_for_calibration": (
                calibration.final_test_labels_used
            ),
            "threshold": calibration.threshold,
            "balanced_accuracy": (
                open_metrics.balanced_accuracy
            ),
            "final_auroc": open_metrics.auroc,
            "final_false_known_rate": (
                open_metrics.false_known_rate
            ),
            "touch_request_rate": touch_request_rate,
        }

        aurocs.append(open_metrics.auroc)
        balanced_accuracies.append(
            open_metrics.balanced_accuracy,
        )
        touch_request_rates.append(touch_request_rate)
        calibration_errors.append(
            calibration_summary.expected_calibration_error,
        )
        risk_areas.append(
            risk_summary.area_under_risk_coverage,
        )

    metric_behavior = {
        "mean_final_auroc": sum(aurocs) / len(aurocs),
        "mean_final_balanced_accuracy": (
            sum(balanced_accuracies)
            / len(balanced_accuracies)
        ),
        "mean_touch_request_rate": (
            sum(touch_request_rates)
            / len(touch_request_rates)
        ),
        "mean_expected_calibration_error": (
            sum(calibration_errors)
            / len(calibration_errors)
        ),
        "mean_area_under_risk_coverage": (
            sum(risk_areas)
            / len(risk_areas)
        ),
    }

    return methods, metric_behavior


def _fusion_audit(condition_result) -> dict[str, Any]:
    """Exercise proposed and baseline fusion mechanics."""

    output: dict[str, Any] = {}
    config = FusionConfig(
        minimum_reliability=0.30,
        generator_version="0.3.0-feasibility",
    )

    for method in FusionMethod:
        decisions = tuple(
            fuse_multisensory_view(
                view=view,
                config=config,
                method=method,
            )
            for view in condition_result.views
        )
        counts = Counter(
            decision.action.value
            for decision in decisions
        )

        unavailable_received_weight = any(
            (
                not decision.trace.odor_available
                and decision.odor_weight != 0.0
            )
            or (
                not decision.trace.touch_available
                and decision.touch_weight != 0.0
            )
            for decision in decisions
        )

        output[method.value] = {
            "decision_count": len(decisions),
            "action_counts": {
                action.value: counts.get(
                    action.value,
                    0,
                )
                for action in FusionAction
            },
            "unavailable_modalities_received_weight": (
                unavailable_received_weight
            ),
        }

    return output


def run_feasibility_pilot() -> FeasibilityPilotReport:
    """Run deterministic integration checks without testing hypotheses."""

    config = _generation_config()
    generated = generate_noi_v0_3_events(config)

    training_events = _split_events(
        generated.latent_events,
        MultisensorySplit.TRAIN,
    )
    validation_events = _split_events(
        generated.latent_events,
        MultisensorySplit.VALIDATION,
    )
    final_events = _split_events(
        generated.latent_events,
        MultisensorySplit.FINAL_TEST,
    )

    condition_result = generate_multisensory_condition_views(
        latent_events=final_events,
        targets=generated.targets,
        config=ConditionGenerationConfig(
            seed=1301,
            odor_noise_scale=0.25,
            tactile_noise_scale=0.20,
            degraded_quality=0.40,
            locked_temporal_offset_steps=2,
            generator_version="0.3.0-feasibility",
        ),
    )

    gate_methods, metric_behavior = _support_gate_audit(
        training_events=training_events,
        validation_events=validation_events,
        final_events=final_events,
    )

    payload: dict[str, Any] = {
        "schema_version": "noi-v0.3-feasibility-v1",
        "study_phase": "feasibility",
        "confirmatory": False,
        "seed": 1301,
        "generator_version": "0.3.0-feasibility",
        "allocation": {
            "train": len(training_events),
            "validation": len(validation_events),
            "final_test": len(final_events),
        },
        "final_support_allocation": {
            "seen_item": sum(
                event.support_regime
                is SupportRegime.SEEN_ITEM
                for event in final_events
            ),
            "known_family_unseen_item": sum(
                event.support_regime
                is SupportRegime.KNOWN_FAMILY_UNSEEN_ITEM
                for event in final_events
            ),
            "unseen_family": sum(
                event.support_regime
                is SupportRegime.UNSEEN_FAMILY
                for event in final_events
            ),
        },
        "condition_audit": _condition_audit(
            final_events=final_events,
            condition_result=condition_result,
        ),
        "leakage_audit": _leakage_audit(generated),
        "support_gate_methods": gate_methods,
        "metric_behavior": metric_behavior,
        "fusion_audit": _fusion_audit(
            condition_result,
        ),
        "hypotheses": {
            "H6": "not_tested",
            "H7": "not_tested",
            "H8": "not_tested",
        },
        "supports_h6": False,
        "supports_h7": False,
        "supports_h8": False,
        "integrity": {
            "test_labels_for_training_used": False,
            "test_labels_for_calibration_used": False,
            "thresholds_changed_from_final_test": False,
            "confirmatory_claims_allowed": False,
        },
    }

    return FeasibilityPilotReport(
        payload=payload,
    )


def _validate_payload(payload: object) -> dict[str, Any]:
    """Require the exact feasibility schema identity."""

    if not isinstance(payload, dict):
        raise FeasibilityPilotError(
            "payload must be a dictionary."
        )

    if payload.get("schema_version") != "noi-v0.3-feasibility-v1":
        raise FeasibilityPilotError(
            "payload has an invalid feasibility schema."
        )

    if payload.get("study_phase") != "feasibility":
        raise FeasibilityPilotError(
            "payload must identify the feasibility phase."
        )

    if payload.get("confirmatory") is not False:
        raise FeasibilityPilotError(
            "feasibility payload cannot be confirmatory."
        )

    return payload


def build_feasibility_payload(
    report: FeasibilityPilotReport,
) -> dict[str, Any]:
    """Return detached deterministic JSON-ready pilot content."""

    if not isinstance(report, FeasibilityPilotReport):
        raise FeasibilityPilotError(
            "report must be a FeasibilityPilotReport."
        )

    payload = deepcopy(report.payload)
    _validate_payload(payload)

    return payload


def export_feasibility_payload(
    payload: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write canonical JSON while refusing silent replacement."""

    checked = _validate_payload(payload)

    if not isinstance(output_path, Path):
        raise FeasibilityPilotError(
            "output_path must be a pathlib.Path."
        )

    if output_path.exists() and not overwrite:
        raise FeasibilityPilotError(
            f"Output already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    serialized = json.dumps(
        checked,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    output_path.write_text(
        serialized + "\n",
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    """Run and optionally export the feasibility artifact."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the non-confirmatory NOI v0.3 feasibility pilot."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    arguments = parser.parse_args()

    report = run_feasibility_pilot()
    payload = build_feasibility_payload(report)
    written = export_feasibility_payload(
        payload,
        arguments.output,
        overwrite=arguments.overwrite,
    )

    print(f"WROTE: {written}")
    print("STUDY PHASE: feasibility")
    print("H6/H7/H8: not tested")


if __name__ == "__main__":
    main()
