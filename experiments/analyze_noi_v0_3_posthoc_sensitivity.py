"""Post-hoc sensitivity analysis for locked NOI v0.3 results.

This module reads retained confirmatory artifacts without changing them.
Every output is explicitly classified as exploratory and post hoc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.analyze_noi_v0_3_confirmatory import (
    H7_CONDITIONS,
    H8_COMPARATORS,
    H8_CONDITIONS,
    REGISTERED_SEEDS,
    _false_confident_records,
    _paired_records,
)


ALL_CONDITIONS = (
    "clean",
    "degraded_odor",
    "degraded_touch",
    "missing_touch",
    "missing_odor",
    "contradictory_modalities",
    "temporal_misalignment",
)

SUPPORT_REGIMES = (
    "seen_item",
    "known_family_unseen_item",
    "unseen_family",
)

POSTHOC_BOOTSTRAP_SEED = 20260830
POSTHOC_BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95


class PosthocSensitivityError(ValueError):
    """Raised when exploratory sensitivity inputs are invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _verify_sidecar(path: Path) -> str:
    sidecar = Path(f"{path}.sha256")

    if not path.is_file() or not sidecar.is_file():
        raise PosthocSensitivityError(
            f"Artifact or SHA-256 sidecar is missing: {path}"
        )

    recorded = sidecar.read_text(
        encoding="utf-8"
    ).split()[0]
    observed = _sha256(path)

    if recorded != observed:
        raise PosthocSensitivityError(
            f"SHA-256 mismatch: {path}"
        )

    return observed


def _mean(values: Sequence[float], *, label: str) -> float:
    if not values:
        raise PosthocSensitivityError(
            f"{label} must not be empty."
        )

    result = float(np.mean(
        np.asarray(values, dtype=np.float64)
    ))

    if not math.isfinite(result):
        raise PosthocSensitivityError(
            f"{label} must be finite."
        )

    return result


def _event_differences(
    pairs: Sequence[Any],
) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)

    for pair in pairs:
        difference = (
            float(pair.proposed_value)
            - float(pair.baseline_value)
        )
        grouped[str(pair.latent_event_id)].append(
            difference
        )

    if not grouped:
        raise PosthocSensitivityError(
            "Paired observations must not be empty."
        )

    values = np.asarray(
        [
            float(np.mean(grouped[event_id]))
            for event_id in sorted(grouped)
        ],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(values)):
        raise PosthocSensitivityError(
            "Event differences must be finite."
        )

    return values


def _distribution(
    values: Sequence[float],
) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)

    if array.size == 0 or not np.all(np.isfinite(array)):
        raise PosthocSensitivityError(
            "Distribution values must be nonempty and finite."
        )

    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q01": float(np.quantile(array, 0.01)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(
            np.std(array, ddof=0)
        ),
    }


def hierarchical_bootstrap(
    arrays_by_seed: Sequence[np.ndarray],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Resample seed clusters, then latent events within each seed."""

    if not arrays_by_seed:
        raise PosthocSensitivityError(
            "arrays_by_seed must not be empty."
        )

    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in arrays_by_seed
    )

    if any(
        values.size == 0
        or not np.all(np.isfinite(values))
        for values in arrays
    ):
        raise PosthocSensitivityError(
            "Every seed array must be nonempty and finite."
        )

    if (
        isinstance(bootstrap_resamples, bool)
        or bootstrap_resamples < 1
    ):
        raise PosthocSensitivityError(
            "bootstrap_resamples must be positive."
        )

    if not 0.0 < float(confidence_level) < 1.0:
        raise PosthocSensitivityError(
            "confidence_level must be between zero and one."
        )

    generator = np.random.default_rng(
        int(bootstrap_seed)
    )
    seed_count = len(arrays)
    bootstrap_means = np.empty(
        int(bootstrap_resamples),
        dtype=np.float64,
    )

    for index in range(int(bootstrap_resamples)):
        selected_seeds = generator.integers(
            low=0,
            high=seed_count,
            size=seed_count,
        )
        selected_means = np.empty(
            seed_count,
            dtype=np.float64,
        )

        for position, seed_index in enumerate(
            selected_seeds
        ):
            values = arrays[int(seed_index)]
            selected_events = generator.integers(
                low=0,
                high=values.size,
                size=values.size,
            )
            selected_means[position] = float(
                np.mean(values[selected_events])
            )

        bootstrap_means[index] = float(
            np.mean(selected_means)
        )

    observed = float(np.mean(
        np.asarray(
            [float(np.mean(values)) for values in arrays],
            dtype=np.float64,
        )
    ))
    tail = (1.0 - float(confidence_level)) / 2.0
    lower = float(np.quantile(
        bootstrap_means,
        tail,
        method="linear",
    ))
    upper = float(np.quantile(
        bootstrap_means,
        1.0 - tail,
        method="linear",
    ))
    nonpositive = int(np.count_nonzero(
        bootstrap_means <= 0.0
    ))
    nonnegative = int(np.count_nonzero(
        bootstrap_means >= 0.0
    ))
    p_value = min(
        1.0,
        2.0 * min(
            (nonpositive + 1)
            / (int(bootstrap_resamples) + 1),
            (nonnegative + 1)
            / (int(bootstrap_resamples) + 1),
        ),
    )

    return {
        "seed_count": seed_count,
        "observed_mean_difference": observed,
        "confidence_interval_lower": lower,
        "confidence_interval_upper": upper,
        "two_sided_p_value": float(p_value),
        "confidence_level": float(confidence_level),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_resamples": int(
            bootstrap_resamples
        ),
        "resampling_hierarchy": [
            "seed",
            "latent_event_id",
        ],
        "classification": "post_hoc_exploratory",
    }


def _system_mean(
    records: Sequence[Mapping[str, Any]],
    *,
    system: str,
    condition: str,
    support_regime: str | None,
) -> float:
    values = [
        float(record["reciprocal_rank"])
        for record in records
        if record["system"] == system
        and record["condition"] == condition
        and (
            support_regime is None
            or record["support_regime"]
            == support_regime
        )
    ]

    return _mean(
        values,
        label=f"{system} {condition} MRR",
    )


def _support_diagnostic(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, list[float]]],
]:
    condition_payload: dict[str, Any] = {}
    pooled: dict[str, dict[str, list[float]]] = {
        condition: {
            regime: []
            for regime in SUPPORT_REGIMES
        }
        for condition in ALL_CONDITIONS
    }

    for condition in ALL_CONDITIONS:
        selected = [
            record
            for record in records
            if record["system"] == "support_gate_odor_only"
            and record["condition"] == condition
        ]
        regimes: dict[str, Any] = {}

        for regime in SUPPORT_REGIMES:
            regime_records = [
                record
                for record in selected
                if record["support_regime"] == regime
            ]
            scores = [
                float(record["support_score"])
                for record in regime_records
            ]
            margins = [
                score - float(threshold)
                for score in scores
            ]
            predicted = [
                float(bool(record["predicted_supported"]))
                for record in regime_records
            ]
            gate_predicted = [
                float(bool(
                    record[
                        "support_gate_predicted_supported"
                    ]
                ))
                for record in regime_records
            ]

            pooled[condition][regime].extend(scores)
            regimes[regime] = {
                "support_score": _distribution(scores),
                "threshold_margin": _distribution(margins),
                "predicted_supported_rate": _mean(
                    predicted,
                    label=(
                        f"{condition} {regime} "
                        "predicted-supported rate"
                    ),
                ),
                "support_gate_predicted_supported_rate": (
                    _mean(
                        gate_predicted,
                        label=(
                            f"{condition} {regime} "
                            "gate-supported rate"
                        ),
                    )
                ),
            }

        seen = [
            float(record["support_score"])
            for record in selected
            if record["support_regime"] == "seen_item"
        ]
        unseen = [
            float(record["support_score"])
            for record in selected
            if record["support_regime"] == "unseen_family"
        ]

        condition_payload[condition] = {
            "threshold": float(threshold),
            "regimes": regimes,
            "seen_minimum_minus_unseen_maximum": (
                float(min(seen) - max(unseen))
            ),
            "score_ranges_disjoint": bool(
                min(seen) > max(unseen)
            ),
            "threshold_minus_unseen_maximum": (
                float(threshold - max(unseen))
            ),
            "seen_minimum_minus_threshold": (
                float(min(seen) - threshold)
            ),
        }

    return condition_payload, pooled


def _analyze_seed(
    payload: Mapping[str, Any],
    *,
    h6_comparator: str,
    h7_comparator: str,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, dict[str, list[float]]],
]:
    seed = int(payload["seed"])
    records = tuple(
        {
            **dict(record),
            "_seed": seed,
        }
        for record in payload["view_results"]
    )
    enriched = _false_confident_records(records)

    h6_pairs = _paired_records(
        enriched,
        baseline_system=h6_comparator,
        proposed_system="support_gate_odor_only",
        conditions=ALL_CONDITIONS,
        support_regime="unseen_family",
        value_field="_false_known",
        reduction=True,
    )
    h7_pairs = _paired_records(
        enriched,
        baseline_system=h7_comparator,
        proposed_system=(
            "reliability_gated_olfactory_tactile_fusion"
        ),
        conditions=H7_CONDITIONS,
        support_regime=None,
        value_field="reciprocal_rank",
        reduction=False,
    )

    arrays = {
        "H6_false_known_reduction": (
            _event_differences(h6_pairs)
        ),
        "H7_absolute_mrr_difference": (
            _event_differences(h7_pairs)
        ),
    }

    for comparator in H8_COMPARATORS:
        pairs = _paired_records(
            enriched,
            baseline_system=comparator,
            proposed_system=(
                "support_gate_reliability_fusion_with_abstention"
            ),
            conditions=H8_CONDITIONS,
            support_regime=None,
            value_field="_false_confident",
            reduction=True,
        )
        arrays[
            f"H8_false_confident_reduction_vs_{comparator}"
        ] = _event_differences(pairs)

    h6_clean_loss = (
        _system_mean(
            enriched,
            system=h6_comparator,
            condition="clean",
            support_regime="seen_item",
        )
        - _system_mean(
            enriched,
            system="support_gate_odor_only",
            condition="clean",
            support_regime="seen_item",
        )
    )

    h8_clean_losses = {}

    for comparator in H8_COMPARATORS:
        h8_clean_losses[comparator] = (
            _system_mean(
                enriched,
                system=comparator,
                condition="clean",
                support_regime=None,
            )
            - _system_mean(
                enriched,
                system=(
                    "support_gate_reliability_fusion_with_abstention"
                ),
                condition="clean",
                support_regime=None,
            )
        )

    threshold = float(
        payload["locked_values"]["support_threshold"]
    )
    diagnostic, pooled = _support_diagnostic(
        enriched,
        threshold=threshold,
    )

    summary = {
        "seed": seed,
        "effects": {
            name: float(np.mean(values))
            for name, values in arrays.items()
        },
        "event_counts": {
            name: int(values.size)
            for name, values in arrays.items()
        },
        "H6_clean_seen_mrr_loss": float(
            h6_clean_loss
        ),
        "H8_clean_mrr_loss": {
            name: float(value)
            for name, value in h8_clean_losses.items()
        },
        "support_threshold": threshold,
        "support_separability_by_condition": diagnostic,
    }

    return summary, arrays, pooled


def analyze_posthoc_sensitivity(
    results_directory: Path,
    *,
    bootstrap_seed: int = POSTHOC_BOOTSTRAP_SEED,
    bootstrap_resamples: int = POSTHOC_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Read verified raw artifacts and compute exploratory checks."""

    aggregate_path = results_directory / "aggregate.json"
    aggregate_hash = _verify_sidecar(aggregate_path)
    aggregate = json.loads(
        aggregate_path.read_text(encoding="utf-8")
    )

    if aggregate.get("schema_version") != (
        "noi-v0.3-confirmatory-aggregate-v1"
    ):
        raise PosthocSensitivityError(
            "Aggregate schema is invalid."
        )

    if tuple(aggregate["completed_seeds"]) != (
        REGISTERED_SEEDS
    ):
        raise PosthocSensitivityError(
            "Aggregate seed registry is invalid."
        )

    h6_comparator = str(
        aggregate["hypotheses"]["H6"][
            "selected_comparator"
        ]
    )
    h7_comparator = str(
        aggregate["hypotheses"]["H7"][
            "selected_comparator"
        ]
    )

    per_seed = []
    arrays_by_metric: dict[str, list[np.ndarray]] = (
        defaultdict(list)
    )
    pooled_support: dict[
        str,
        dict[str, list[float]],
    ] = {
        condition: {
            regime: []
            for regime in SUPPORT_REGIMES
        }
        for condition in ALL_CONDITIONS
    }
    source_hashes = {}

    for seed in REGISTERED_SEEDS:
        path = results_directory / f"seed-{seed}.json"
        source_hashes[str(seed)] = _verify_sidecar(path)
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        if int(payload.get("seed", -1)) != seed:
            raise PosthocSensitivityError(
                f"Seed identity mismatch: {path}"
            )

        summary, arrays, support = _analyze_seed(
            payload,
            h6_comparator=h6_comparator,
            h7_comparator=h7_comparator,
        )
        per_seed.append(summary)

        for name, values in arrays.items():
            arrays_by_metric[name].append(values)

        for condition in ALL_CONDITIONS:
            for regime in SUPPORT_REGIMES:
                pooled_support[condition][regime].extend(
                    support[condition][regime]
                )

        print(f"READ AND ANALYZED SEED {seed}: PASS")

    hierarchical = {}

    for index, name in enumerate(sorted(arrays_by_metric)):
        hierarchical[name] = hierarchical_bootstrap(
            arrays_by_metric[name],
            bootstrap_seed=int(bootstrap_seed) + index,
            bootstrap_resamples=int(
                bootstrap_resamples
            ),
            confidence_level=CONFIDENCE_LEVEL,
        )
        print(f"HIERARCHICAL BOOTSTRAP {name}: PASS")

    pooled_diagnostic = {}

    for condition in ALL_CONDITIONS:
        regime_payload = {
            regime: {
                "support_score": _distribution(
                    pooled_support[condition][regime]
                ),
            }
            for regime in SUPPORT_REGIMES
        }
        seen = pooled_support[condition]["seen_item"]
        unseen = pooled_support[condition]["unseen_family"]

        pooled_diagnostic[condition] = {
            "regimes": regime_payload,
            "seen_minimum_minus_unseen_maximum": (
                float(min(seen) - max(unseen))
            ),
            "score_ranges_disjoint": bool(
                min(seen) > max(unseen)
            ),
        }

    expected = {
        "H6_false_known_reduction": float(
            aggregate["hypotheses"]["H6"][
                "false_known_reduction"
            ]
        ),
        "H7_absolute_mrr_difference": float(
            aggregate["hypotheses"]["H7"][
                "absolute_mrr_improvement"
            ]
        ),
        (
            "H8_false_confident_reduction_vs_"
            "fixed_weight_fusion"
        ): float(
            aggregate["hypotheses"]["H8"][
                "comparisons"
            ]["fixed_weight_fusion"][
                "false_confident_reduction"
            ]
        ),
        (
            "H8_false_confident_reduction_vs_"
            "naive_concatenation"
        ): float(
            aggregate["hypotheses"]["H8"][
                "comparisons"
            ]["naive_concatenation"][
                "false_confident_reduction"
            ]
        ),
    }

    reproduction = {}

    for name, expected_value in expected.items():
        observed = float(
            hierarchical[name][
                "observed_mean_difference"
            ]
        )
        matched = math.isclose(
            observed,
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

        if not matched:
            raise PosthocSensitivityError(
                f"Point estimate mismatch: {name}"
            )

        reproduction[name] = {
            "confirmatory_value": expected_value,
            "posthoc_recomputed_value": observed,
            "absolute_tolerance": 1e-12,
            "matched": True,
        }

    return {
        "schema_version": (
            "noi-v0.3-posthoc-sensitivity-v1"
        ),
        "analysis_classification": (
            "post_hoc_exploratory"
        ),
        "confirmatory_results_modified": False,
        "confirmatory_hypothesis_statuses_modified": (
            False
        ),
        "study_phase": "post_confirmatory_audit",
        "registered_seeds": list(REGISTERED_SEEDS),
        "fixed_confirmatory_comparators": {
            "H6": h6_comparator,
            "H7": h7_comparator,
            "H8": list(H8_COMPARATORS),
        },
        "point_estimate_reproduction": reproduction,
        "per_seed_results": per_seed,
        "hierarchical_seed_event_bootstrap": hierarchical,
        "pooled_support_score_diagnostic": (
            pooled_diagnostic
        ),
        "interpretation_limits": {
            "H6_scope": (
                "unseen_family only; it does not establish "
                "rejection of known-family unseen items"
            ),
            "perfect_H6_warning": (
                "Perfect confirmatory separation may reflect "
                "the registered synthetic family geometry"
            ),
            "new_confirmatory_claim_created": False,
        },
        "integrity": {
            "aggregate_sha256": aggregate_hash,
            "source_seed_sha256": source_hashes,
            "raw_artifacts_modified": False,
            "final_test_tuning_used": False,
            "thresholds_changed": False,
            "seed_removal_used": False,
        },
    }


def export_posthoc_sensitivity(
    analysis: Mapping[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    """Write deterministic JSON and SHA-256 sidecar."""

    if output_path.exists() and not overwrite:
        raise PosthocSensitivityError(
            f"Output already exists: {output_path}"
        )

    hash_path = Path(f"{output_path}.sha256")

    if hash_path.exists() and not overwrite:
        raise PosthocSensitivityError(
            f"Output already exists: {hash_path}"
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
    digest = _sha256(output_path)
    hash_path.write_text(
        f"{digest}  {output_path.name}\n",
        encoding="utf-8",
    )


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute post-hoc exploratory NOI v0.3 "
            "seed-level and hierarchical sensitivity checks."
        )
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path("results/v0.3-confirmatory"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/noi_v0.3_confirmatory/"
            "posthoc_sensitivity.json"
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=POSTHOC_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=POSTHOC_BOOTSTRAP_RESAMPLES,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parsed = parser.parse_args(arguments)

    analysis = analyze_posthoc_sensitivity(
        parsed.results_directory,
        bootstrap_seed=parsed.bootstrap_seed,
        bootstrap_resamples=(
            parsed.bootstrap_resamples
        ),
    )
    export_posthoc_sensitivity(
        analysis,
        parsed.output,
        overwrite=parsed.overwrite,
    )

    print("POST-HOC SENSITIVITY ANALYSIS: PASS")
    print("OUTPUT:", parsed.output)
    print("SHA256:", f"{parsed.output}.sha256")


if __name__ == "__main__":
    main()
