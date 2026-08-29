"""Locked aggregate analysis for NOI v0.3 confirmatory results."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.noi_v0_3_analysis import (
    HypothesisTest,
    PairedObservation,
    holm_correction,
    paired_bootstrap,
)


REGISTERED_SEEDS = tuple(range(1301, 1311))

H6_COMPARATORS = (
    "odor_only_ridge",
    "odor_only_cosine",
    "naive_concatenation",
    "fixed_weight_fusion",
)

H7_COMPARATORS = (
    "odor_only_ridge",
    "odor_only_cosine",
    "touch_only_ridge",
    "touch_only_cosine",
    "fixed_weight_fusion",
)

H7_CONDITIONS = (
    "degraded_odor",
    "missing_odor",
)

H8_CONDITIONS = (
    "degraded_odor",
    "degraded_touch",
    "missing_touch",
    "missing_odor",
    "contradictory_modalities",
    "temporal_misalignment",
)

H8_COMPARATORS = (
    "naive_concatenation",
    "fixed_weight_fusion",
)


class ConfirmatoryAggregateError(ValueError):
    """Raised when confirmatory aggregation inputs are invalid."""


def _validate_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Validate all ten per-seed payloads and integrity controls."""

    if (
        not isinstance(payloads, Sequence)
        or isinstance(payloads, (str, bytes))
    ):
        raise ConfirmatoryAggregateError(
            "payloads must be a sequence."
        )

    items = tuple(payloads)

    if len(items) != len(REGISTERED_SEEDS):
        raise ConfirmatoryAggregateError(
            "All registered seeds must be supplied exactly once."
        )

    observed_seeds: list[int] = []

    for payload in items:
        if not isinstance(payload, Mapping):
            raise ConfirmatoryAggregateError(
                "Every seed payload must be a mapping."
            )

        if payload.get("schema_version") != (
            "noi-v0.3-confirmatory-seed-v1"
        ):
            raise ConfirmatoryAggregateError(
                "Every payload must use the registered seed schema."
            )

        if (
            payload.get("study_phase") != "confirmatory"
            or payload.get("confirmatory_execution") is not True
        ):
            raise ConfirmatoryAggregateError(
                "Every payload must be a confirmatory result."
            )

        seed = payload.get("seed")

        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ConfirmatoryAggregateError(
                "Every payload seed must be an integer."
            )

        observed_seeds.append(seed)

        integrity = payload.get("integrity")

        if not isinstance(integrity, Mapping):
            raise ConfirmatoryAggregateError(
                "Every payload requires an integrity audit."
            )

        required_false = (
            "final_test_labels_used_for_training",
            "final_test_labels_used_for_calibration",
            "condition_metadata_used_as_model_input",
            "target_labels_used_as_inference_input",
            "paired_views_treated_as_independent",
            "thresholds_changed_from_final_test",
        )

        if any(
            integrity.get(field) is not False
            for field in required_false
        ):
            raise ConfirmatoryAggregateError(
                "A seed payload fails confirmatory integrity."
            )

        if integrity.get(
            "final_test_labels_used_for_scoring_only"
        ) is not True:
            raise ConfirmatoryAggregateError(
                "Final labels must be isolated to posthoc scoring."
            )

        records = payload.get("view_results")

        if (
            not isinstance(records, list)
            or not records
        ):
            raise ConfirmatoryAggregateError(
                "Every seed payload requires view_results."
            )

    if tuple(sorted(observed_seeds)) != REGISTERED_SEEDS:
        raise ConfirmatoryAggregateError(
            "All registered seeds must be supplied exactly once."
        )

    return tuple(
        sorted(
            items,
            key=lambda payload: int(payload["seed"]),
        )
    )


def _all_records(
    payloads: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Attach seed identity to every posthoc result record."""

    records: list[dict[str, Any]] = []

    for payload in payloads:
        seed = int(payload["seed"])

        for raw in payload["view_results"]:
            if not isinstance(raw, Mapping):
                raise ConfirmatoryAggregateError(
                    "Every view result must be a mapping."
                )

            record = dict(raw)
            record["_seed"] = seed
            records.append(record)

    return tuple(records)


def _mean(
    values: tuple[float, ...],
    *,
    label: str,
) -> float:
    """Return one finite nonempty arithmetic mean."""

    if not values or not all(
        math.isfinite(value)
        for value in values
    ):
        raise ConfirmatoryAggregateError(
            f"{label} must contain finite observations."
        )

    return sum(values) / len(values)


def _selected_comparator(
    records: tuple[dict[str, Any], ...],
    *,
    candidates: tuple[str, ...],
    conditions: tuple[str, ...],
    support_regime: str | None,
) -> tuple[str, dict[str, float]]:
    """Select maximum MRR with ascending-name deterministic ties."""

    scores: dict[str, float] = {}

    for candidate in candidates:
        values = tuple(
            float(record["reciprocal_rank"])
            for record in records
            if record["system"] == candidate
            and record["condition"] in conditions
            and (
                support_regime is None
                or record["support_regime"]
                == support_regime
            )
        )

        scores[candidate] = _mean(
            values,
            label=f"{candidate} comparator values",
        )

    selected = sorted(
        candidates,
        key=lambda name: (
            -scores[name],
            name,
        ),
    )[0]

    return selected, scores


def _paired_records(
    records: tuple[dict[str, Any], ...],
    *,
    baseline_system: str,
    proposed_system: str,
    conditions: tuple[str, ...],
    support_regime: str | None,
    value_field: str,
    reduction: bool,
) -> tuple[PairedObservation, ...]:
    """Build exact seed/event/condition paired observations."""

    def selected(system: str) -> dict[
        tuple[int, str, str],
        dict[str, Any],
    ]:
        output = {}

        for record in records:
            if (
                record["system"] != system
                or record["condition"] not in conditions
                or (
                    support_regime is not None
                    and record["support_regime"]
                    != support_regime
                )
            ):
                continue

            key = (
                int(record["_seed"]),
                str(record["latent_event_id"]),
                str(record["condition"]),
            )

            if key in output:
                raise ConfirmatoryAggregateError(
                    "System records must be unique per paired view."
                )

            output[key] = record

        return output

    baseline = selected(baseline_system)
    proposed = selected(proposed_system)

    if not baseline or set(baseline) != set(proposed):
        raise ConfirmatoryAggregateError(
            "Baseline and proposed paired views must align exactly."
        )

    observations = []

    for key in sorted(baseline):
        baseline_value = float(
            baseline[key][value_field]
        )
        proposed_value = float(
            proposed[key][value_field]
        )

        if reduction:
            first = proposed_value
            second = baseline_value
        else:
            first = baseline_value
            second = proposed_value

        observations.append(
            PairedObservation(
                latent_event_id=(
                    f"{key[0]}:{key[1]}"
                ),
                baseline_value=first,
                proposed_value=second,
            )
        )

    return tuple(observations)


def _bootstrap(
    observations: tuple[PairedObservation, ...],
) -> dict[str, Any]:
    """Run the locked paired bootstrap and serialize its record."""

    return asdict(
        paired_bootstrap(
            observations,
            bootstrap_seed=4242,
            bootstrap_resamples=10_000,
            confidence_level=0.95,
        )
    )


def _false_confident_records(
    records: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    """Add the locked false-confident indicator to every record."""

    output = []

    for record in records:
        enriched = dict(record)
        enriched["_false_confident"] = float(
            (not bool(record["correct"]))
            and (not bool(record["abstained"]))
            and float(record["confidence"]) >= 0.80
        )
        enriched["_false_known"] = float(
            bool(record["predicted_supported"])
        )
        output.append(enriched)

    return tuple(output)


def analyze_confirmatory_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate H6, H7, and H8 under the pre-result rules."""

    validated = _validate_payloads(payloads)
    records = _false_confident_records(
        _all_records(validated)
    )

    h6_comparator, h6_candidate_scores = (
        _selected_comparator(
            records,
            candidates=H6_COMPARATORS,
            conditions=("clean",),
            support_regime="seen_item",
        )
    )

    h6_pairs = _paired_records(
        records,
        baseline_system=h6_comparator,
        proposed_system="support_gate_odor_only",
        conditions=(
            "clean",
            "degraded_odor",
            "degraded_touch",
            "missing_touch",
            "missing_odor",
            "contradictory_modalities",
            "temporal_misalignment",
        ),
        support_regime="unseen_family",
        value_field="_false_known",
        reduction=True,
    )
    h6_bootstrap = _bootstrap(h6_pairs)

    h6_proposed_clean = _mean(
        tuple(
            float(record["reciprocal_rank"])
            for record in records
            if record["system"] == "support_gate_odor_only"
            and record["condition"] == "clean"
            and record["support_regime"] == "seen_item"
        ),
        label="H6 proposed clean seen MRR",
    )
    h6_comparator_clean = h6_candidate_scores[
        h6_comparator
    ]
    h6_loss = (
        h6_comparator_clean
        - h6_proposed_clean
    )
    h6_reduction = float(
        h6_bootstrap["mean_difference"]
    )
    h6_supported = (
        h6_reduction >= 0.05
        and float(
            h6_bootstrap[
                "confidence_interval_lower"
            ]
        ) > 0.0
        and h6_loss <= 0.02
    )

    h7_comparator, h7_candidate_scores = (
        _selected_comparator(
            records,
            candidates=H7_COMPARATORS,
            conditions=H7_CONDITIONS,
            support_regime=None,
        )
    )
    h7_pairs = _paired_records(
        records,
        baseline_system=h7_comparator,
        proposed_system=(
            "reliability_gated_olfactory_tactile_fusion"
        ),
        conditions=H7_CONDITIONS,
        support_regime=None,
        value_field="reciprocal_rank",
        reduction=False,
    )
    h7_bootstrap = _bootstrap(h7_pairs)
    h7_absolute = float(
        h7_bootstrap["mean_difference"]
    )
    h7_baseline_mean = h7_candidate_scores[
        h7_comparator
    ]
    h7_relative = (
        h7_absolute / h7_baseline_mean
        if h7_baseline_mean > 0.0
        else 0.0
    )
    h7_effect_pass = (
        h7_absolute >= 0.05
        or h7_relative >= 0.10
    )
    h7_ci_pass = float(
        h7_bootstrap["confidence_interval_lower"]
    ) > 0.0

    h8_comparisons: dict[str, dict[str, Any]] = {}
    secondary_tests = [
        HypothesisTest(
            name="H7",
            p_value=float(
                h7_bootstrap["two_sided_p_value"]
            ),
        )
    ]

    for comparator in H8_COMPARATORS:
        pairs = _paired_records(
            records,
            baseline_system=comparator,
            proposed_system=(
                "support_gate_reliability_fusion_with_abstention"
            ),
            conditions=H8_CONDITIONS,
            support_regime=None,
            value_field="_false_confident",
            reduction=True,
        )
        bootstrap = _bootstrap(pairs)

        comparator_clean = _mean(
            tuple(
                float(record["reciprocal_rank"])
                for record in records
                if record["system"] == comparator
                and record["condition"] == "clean"
            ),
            label=f"{comparator} clean MRR",
        )
        proposed_clean = _mean(
            tuple(
                float(record["reciprocal_rank"])
                for record in records
                if record["system"] == (
                    "support_gate_reliability_fusion_with_abstention"
                )
                and record["condition"] == "clean"
            ),
            label="H8 proposed clean MRR",
        )
        clean_loss = comparator_clean - proposed_clean
        reduction_value = float(
            bootstrap["mean_difference"]
        )
        local_supported = (
            reduction_value >= 0.05
            and float(
                bootstrap[
                    "confidence_interval_lower"
                ]
            ) > 0.0
            and clean_loss <= 0.02
        )

        h8_comparisons[comparator] = {
            "status": (
                "supported"
                if local_supported
                else "not_supported"
            ),
            "false_confident_reduction": (
                reduction_value
            ),
            "clean_mrr_loss": clean_loss,
            "paired_bootstrap": bootstrap,
        }

        secondary_tests.append(
            HypothesisTest(
                name=f"H8_vs_{comparator}",
                p_value=float(
                    bootstrap["two_sided_p_value"]
                ),
            )
        )

    holm = holm_correction(
        tuple(secondary_tests),
        alpha=0.05,
    )
    holm_payload = asdict(holm)
    rejected = {
        comparison.name: comparison.rejected
        for comparison in holm.comparisons
    }

    h7_supported = (
        h7_effect_pass
        and h7_ci_pass
        and rejected["H7"]
    )

    for comparator in H8_COMPARATORS:
        name = f"H8_vs_{comparator}"

        if not rejected[name]:
            h8_comparisons[comparator][
                "status"
            ] = "not_supported"

    h8_supported = all(
        comparison["status"] == "supported"
        for comparison in h8_comparisons.values()
    )

    payload = {
        "schema_version": (
            "noi-v0.3-confirmatory-aggregate-v1"
        ),
        "study_phase": "confirmatory",
        "seed_count": len(validated),
        "completed_seeds": [
            int(payload["seed"])
            for payload in validated
        ],
        "hypotheses": {
            "H6": {
                "role": "primary",
                "status": (
                    "supported"
                    if h6_supported
                    else "not_supported"
                ),
                "selected_comparator": h6_comparator,
                "candidate_clean_seen_mrr": (
                    h6_candidate_scores
                ),
                "false_known_reduction": h6_reduction,
                "seen_item_clean_mrr_loss": h6_loss,
                "paired_bootstrap": h6_bootstrap,
            },
            "H7": {
                "role": "secondary",
                "status": (
                    "supported"
                    if h7_supported
                    else "not_supported"
                ),
                "eligible_conditions": list(
                    H7_CONDITIONS
                ),
                "selected_comparator": h7_comparator,
                "candidate_pooled_mrr": (
                    h7_candidate_scores
                ),
                "absolute_mrr_improvement": h7_absolute,
                "relative_mrr_improvement": h7_relative,
                "paired_bootstrap": h7_bootstrap,
            },
            "H8": {
                "role": "secondary",
                "status": (
                    "supported"
                    if h8_supported
                    else "not_supported"
                ),
                "eligible_conditions": list(
                    H8_CONDITIONS
                ),
                "comparisons": h8_comparisons,
            },
        },
        "holm_correction": holm_payload,
        "integrity": {
            "all_registered_seeds_retained": True,
            "silent_seed_removal_used": False,
            "paired_resampling_unit": "latent_event_id",
            "bootstrap_seed": 4242,
            "bootstrap_resamples": 10_000,
            "confidence_level": 0.95,
            "final_test_tuning_used": False,
            "negative_results_reported": True,
            "null_results_reported": True,
        },
    }

    return json.loads(
        json.dumps(
            payload,
            sort_keys=True,
            allow_nan=False,
        )
    )


def export_confirmatory_aggregate(
    analysis: Mapping[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Write canonical aggregate JSON without silent replacement."""

    if not isinstance(analysis, Mapping):
        raise ConfirmatoryAggregateError(
            "analysis must be a mapping."
        )

    if analysis.get("schema_version") != (
        "noi-v0.3-confirmatory-aggregate-v1"
    ):
        raise ConfirmatoryAggregateError(
            "analysis has an invalid aggregate schema."
        )

    if not isinstance(output_path, Path):
        raise ConfirmatoryAggregateError(
            "output_path must be a Path."
        )

    if output_path.exists() and not overwrite:
        raise ConfirmatoryAggregateError(
            f"Output already exists: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            dict(analysis),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
