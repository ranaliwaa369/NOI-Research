"""Read-only trace audit for the completed NOI v0.3 evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from experiments.run_noi_v0_3_confirmatory import (
    score_to_confidence,
)
from src.evaluation.support_gate import (
    SupportMethod,
    apply_locked_support_threshold,
)


REGISTERED_SEEDS = tuple(range(1301, 1311))

CONDITIONS = (
    "clean",
    "degraded_odor",
    "degraded_touch",
    "missing_touch",
    "missing_odor",
    "contradictory_modalities",
    "temporal_misalignment",
)

SYSTEMS = (
    "odor_only_ridge",
    "odor_only_cosine",
    "touch_only_ridge",
    "touch_only_cosine",
    "naive_concatenation",
    "fixed_weight_fusion",
    "support_gate_odor_only",
    "reliability_gated_olfactory_tactile_fusion",
    "support_gate_reliability_fusion_with_abstention",
)

SUPPORT_REGIMES = (
    "seen_item",
    "known_family_unseen_item",
    "unseen_family",
)

EXPECTED_REGIME_COUNTS = {
    "seen_item": 800,
    "known_family_unseen_item": 600,
    "unseen_family": 600,
}

EXPECTED_INTEGRITY = {
    "condition_metadata_used_as_model_input": False,
    "final_test_labels_used_for_calibration": False,
    "final_test_labels_used_for_scoring_only": True,
    "final_test_labels_used_for_training": False,
    "paired_views_treated_as_independent": False,
    "quality_metadata_used_as_model_input": False,
    "target_labels_used_as_inference_input": False,
    "thresholds_changed_from_final_test": False,
    "training_events_used": 7000,
    "validation_events_used_for_fitting": 0,
}

LOCK_FIELDS = (
    "support_threshold",
    "support_uncertainty_lower",
    "support_uncertainty_upper",
    "reliability_threshold",
    "conflict_threshold",
)


class TraceAuditError(ValueError):
    """Raised when a trace or integrity invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceAuditError(message)


def _close(first: float, second: float) -> bool:
    return math.isclose(
        float(first),
        float(second),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _verify_sidecar(path: Path) -> str:
    appended_sidecar = Path(
        f"{path}.sha256"
    )
    replaced_sidecar = path.with_suffix(
        ".sha256"
    )

    if appended_sidecar.is_file():
        sidecar = appended_sidecar
    else:
        sidecar = replaced_sidecar

    _require(
        path.is_file(),
        f"Missing artifact: {path}",
    )
    _require(
        sidecar.is_file(),
        "Missing SHA-256 sidecar for "
        f"{path}: checked {appended_sidecar} "
        f"and {replaced_sidecar}",
    )

    expected = sidecar.read_text(
        encoding="utf-8"
    ).split()[0]
    observed = _sha256(path)

    _require(
        expected == observed,
        f"SHA-256 mismatch: {path}",
    )

    return observed


def _load_verified_json(
    path: Path,
) -> tuple[dict[str, Any], str]:
    digest = _verify_sidecar(path)
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    _require(
        isinstance(payload, dict),
        f"JSON root must be a mapping: {path}",
    )

    return payload, digest


def _seed_lock(
    lock: Mapping[str, Any],
    seed: int,
) -> dict[str, float]:
    values_by_seed = lock.get("values_by_seed")

    _require(
        isinstance(values_by_seed, Mapping),
        "Validation lock must contain values_by_seed.",
    )

    values = values_by_seed.get(str(seed))

    _require(
        isinstance(values, Mapping),
        f"Missing validation lock for seed {seed}.",
    )

    output: dict[str, float] = {}

    for field in LOCK_FIELDS:
        value = values.get(field)

        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            f"Invalid {field} for seed {seed}.",
        )

        output[field] = float(value)

    return output


def _reciprocal_rank(
    ranking: Sequence[str],
    target_item_id: str,
) -> float:
    try:
        index = ranking.index(target_item_id)
    except ValueError:
        return 0.0

    return 1.0 / float(index + 1)


def audit_seed_payload(
    *,
    payload: Mapping[str, Any],
    raw_sha256: str,
    expected_seed: int,
    locked_values: Mapping[str, float],
) -> dict[str, Any]:
    _require(
        payload.get("schema_version")
        == "noi-v0.3-confirmatory-seed-v1",
        f"Unexpected schema for seed {expected_seed}.",
    )
    _require(
        payload.get("study_phase") == "confirmatory",
        f"Unexpected phase for seed {expected_seed}.",
    )
    _require(
        payload.get("confirmatory_execution") is True,
        f"Seed {expected_seed} is not confirmatory.",
    )
    _require(
        payload.get("seed") == expected_seed,
        f"Seed identity mismatch for {expected_seed}.",
    )
    _require(
        payload.get("final_latent_event_count") == 2000,
        f"Latent-event count mismatch for seed {expected_seed}.",
    )
    _require(
        payload.get("condition_view_count") == 14000,
        f"Condition-view count mismatch for seed {expected_seed}.",
    )
    _require(
        payload.get("system_count") == len(SYSTEMS),
        f"System count mismatch for seed {expected_seed}.",
    )
    _require(
        payload.get("system_evaluation_count") == 126000,
        f"Evaluation count mismatch for seed {expected_seed}.",
    )

    integrity = payload.get("integrity")
    _require(
        integrity == EXPECTED_INTEGRITY,
        f"Integrity record mismatch for seed {expected_seed}.",
    )

    exported_lock = payload.get("locked_values")
    _require(
        isinstance(exported_lock, Mapping),
        f"Missing locked values for seed {expected_seed}.",
    )

    for field in LOCK_FIELDS:
        _require(
            field in exported_lock,
            f"Missing exported {field} for seed {expected_seed}.",
        )
        _require(
            _close(
                float(exported_lock[field]),
                locked_values[field],
            ),
            f"Lock mismatch for seed {expected_seed}: {field}.",
        )

    records = payload.get("view_results")
    _require(
        isinstance(records, list),
        f"Missing view results for seed {expected_seed}.",
    )
    _require(
        len(records) == 126000,
        f"Raw record count mismatch for seed {expected_seed}.",
    )

    unique_records: set[
        tuple[str, str, str]
    ] = set()
    event_identity: dict[
        str,
        tuple[str, str, str],
    ] = {}
    event_conditions: dict[
        str,
        set[str],
    ] = defaultdict(set)
    view_systems: dict[
        tuple[str, str],
        set[str],
    ] = defaultdict(set)
    view_ids: dict[
        tuple[str, str],
        str,
    ] = {}
    view_support_trace: dict[
        tuple[str, str],
        tuple[float, bool, str],
    ] = {}
    trace_samples: dict[
        str,
        dict[str, Any],
    ] = {}

    for index, raw in enumerate(records):
        _require(
            isinstance(raw, Mapping),
            f"Record {index} is not a mapping.",
        )

        latent_event_id = str(
            raw["latent_event_id"]
        )
        view_id = str(raw["view_id"])
        condition = str(raw["condition"])
        system = str(raw["system"])
        regime = str(raw["support_regime"])
        target_item_id = str(
            raw["target_item_id"]
        )
        target_family_id = str(
            raw["target_family_id"]
        )

        _require(
            condition in CONDITIONS,
            f"Unknown condition in seed {expected_seed}.",
        )
        _require(
            system in SYSTEMS,
            f"Unknown system in seed {expected_seed}.",
        )
        _require(
            regime in SUPPORT_REGIMES,
            f"Unknown support regime in seed {expected_seed}.",
        )
        _require(
            raw.get("statistical_unit")
            == "latent_event_id",
            f"Wrong statistical unit in seed {expected_seed}.",
        )

        record_key = (
            latent_event_id,
            condition,
            system,
        )
        _require(
            record_key not in unique_records,
            f"Duplicate paired record in seed {expected_seed}.",
        )
        unique_records.add(record_key)

        identity = (
            regime,
            target_item_id,
            target_family_id,
        )
        prior_identity = event_identity.get(
            latent_event_id
        )

        if prior_identity is None:
            event_identity[latent_event_id] = identity
        else:
            _require(
                prior_identity == identity,
                f"Event identity changed in seed {expected_seed}.",
            )

        view_key = (
            latent_event_id,
            condition,
        )
        prior_view_id = view_ids.get(view_key)

        if prior_view_id is None:
            view_ids[view_key] = view_id
        else:
            _require(
                prior_view_id == view_id,
                f"View ID changed across systems in seed {expected_seed}.",
            )

        _require(
            system not in view_systems[view_key],
            f"Duplicate system view in seed {expected_seed}.",
        )
        view_systems[view_key].add(system)
        event_conditions[latent_event_id].add(
            condition
        )

        abstained = raw["abstained"]
        ranking = raw["ranking"]
        scores = raw["scores"]

        _require(
            isinstance(abstained, bool),
            f"Non-Boolean abstention in seed {expected_seed}.",
        )
        _require(
            isinstance(ranking, list)
            and isinstance(scores, list),
            f"Invalid ranking or scores in seed {expected_seed}.",
        )
        _require(
            len(ranking) == len(scores),
            f"Ranking-score length mismatch in seed {expected_seed}.",
        )

        expected_correct = bool(
            not abstained
            and ranking
            and ranking[0] == target_item_id
        )
        _require(
            raw["correct"] is expected_correct,
            f"Correctness mismatch in seed {expected_seed}.",
        )

        expected_rr = _reciprocal_rank(
            ranking,
            target_item_id,
        )
        _require(
            _close(
                float(raw["reciprocal_rank"]),
                expected_rr,
            ),
            f"Reciprocal-rank mismatch in seed {expected_seed}.",
        )

        expected_confidence = score_to_confidence(
            top_score=(
                float(scores[0])
                if scores
                else None
            ),
            abstained=abstained,
        )
        _require(
            _close(
                float(raw["confidence"]),
                expected_confidence,
            ),
            f"Confidence mismatch in seed {expected_seed}.",
        )

        _require(
            raw["predicted_supported"]
            is (not abstained),
            f"Prediction-abstention mismatch in seed {expected_seed}.",
        )
        _require(
            raw["true_supported"]
            is (regime == "seen_item"),
            f"Scoring label mismatch in seed {expected_seed}.",
        )

        support_score = float(
            raw["support_score"]
        )
        decision = apply_locked_support_threshold(
            event_id=view_id,
            method=SupportMethod.MAHALANOBIS,
            support_score=support_score,
            threshold=locked_values[
                "support_threshold"
            ],
            uncertainty_lower=locked_values[
                "support_uncertainty_lower"
            ],
            uncertainty_upper=locked_values[
                "support_uncertainty_upper"
            ],
        )

        support_signature = (
            support_score,
            decision.is_supported,
            decision.uncertainty_status.value,
        )
        prior_support = view_support_trace.get(
            view_key
        )

        if prior_support is None:
            view_support_trace[
                view_key
            ] = support_signature
        else:
            _require(
                prior_support == support_signature,
                f"Support trace changed across systems in seed "
                f"{expected_seed}.",
            )

        _require(
            raw["support_gate_predicted_supported"]
            is decision.is_supported,
            f"Locked support decision mismatch in seed {expected_seed}.",
        )
        _require(
            raw["support_uncertainty_status"]
            == decision.uncertainty_status.value,
            f"Uncertainty status mismatch in seed {expected_seed}.",
        )

        if (
            condition == "clean"
            and system == "support_gate_odor_only"
            and regime not in trace_samples
        ):
            trace_samples[regime] = {
                "latent_event_id": latent_event_id,
                "view_id": view_id,
                "support_regime": regime,
                "target_item_id": target_item_id,
                "target_family_id": target_family_id,
                "support_score": support_score,
                "locked_support_threshold": locked_values[
                    "support_threshold"
                ],
                "support_gate_predicted_supported": (
                    decision.is_supported
                ),
                "support_uncertainty_status": (
                    decision.uncertainty_status.value
                ),
                "abstained": abstained,
                "correct": expected_correct,
                "reciprocal_rank": expected_rr,
                "confidence": expected_confidence,
            }

    _require(
        len(event_identity) == 2000,
        f"Unique event count mismatch for seed {expected_seed}.",
    )
    _require(
        len(view_systems) == 14000,
        f"Unique paired-view count mismatch for seed {expected_seed}.",
    )
    _require(
        len(unique_records) == 126000,
        f"Unique record count mismatch for seed {expected_seed}.",
    )

    expected_condition_set = set(CONDITIONS)
    expected_system_set = set(SYSTEMS)

    for event_id, conditions in event_conditions.items():
        _require(
            conditions == expected_condition_set,
            f"Incomplete paired conditions for {event_id}.",
        )

    for view_key, systems in view_systems.items():
        _require(
            systems == expected_system_set,
            f"Incomplete system set for {view_key}.",
        )

    regime_counts = Counter(
        identity[0]
        for identity in event_identity.values()
    )
    _require(
        dict(regime_counts)
        == EXPECTED_REGIME_COUNTS,
        f"Support allocation mismatch for seed {expected_seed}.",
    )

    condition_counts = Counter(
        condition
        for _, condition in view_systems
    )
    _require(
        all(
            condition_counts[name] == 2000
            for name in CONDITIONS
        ),
        f"Condition pairing mismatch for seed {expected_seed}.",
    )

    _require(
        set(trace_samples) == set(SUPPORT_REGIMES),
        f"Missing trace samples for seed {expected_seed}.",
    )

    return {
        "seed": expected_seed,
        "raw_seed_sha256": raw_sha256,
        "raw_seed_hash_verified": True,
        "validation_lock_values_matched": True,
        "integrity_record_matched": True,
        "support_decisions_reproduced": True,
        "scoring_values_reproduced": True,
        "paired_condition_structure_verified": True,
        "system_alignment_verified": True,
        "latent_event_count": len(event_identity),
        "condition_view_count": len(view_systems),
        "system_evaluation_count": len(unique_records),
        "support_regime_counts": {
            name: regime_counts[name]
            for name in SUPPORT_REGIMES
        },
        "trace_samples": [
            trace_samples[name]
            for name in SUPPORT_REGIMES
        ],
    }


def build_trace_audit(
    *,
    results_directory: Path,
    validation_lock_path: Path,
    release_artifact_directory: Path,
) -> dict[str, Any]:
    validation_lock_sha256 = _verify_sidecar(
        validation_lock_path
    )
    validation_lock = yaml.safe_load(
        validation_lock_path.read_text(
            encoding="utf-8"
        )
    )

    _require(
        isinstance(validation_lock, Mapping),
        "Validation lock root must be a mapping.",
    )

    seed_audits = []

    for seed in REGISTERED_SEEDS:
        seed_path = (
            results_directory
            / f"seed-{seed}.json"
        )
        payload, raw_sha256 = (
            _load_verified_json(seed_path)
        )
        audit = audit_seed_payload(
            payload=payload,
            raw_sha256=raw_sha256,
            expected_seed=seed,
            locked_values=_seed_lock(
                validation_lock,
                seed,
            ),
        )
        seed_audits.append(audit)
        print(
            f"TRACE AUDIT SEED {seed}: PASS",
            flush=True,
        )

    aggregate_path = (
        release_artifact_directory
        / "aggregate.json"
    )
    aggregate, aggregate_sha256 = (
        _load_verified_json(aggregate_path)
    )

    _require(
        aggregate.get("completed_seeds")
        == list(REGISTERED_SEEDS),
        "Aggregate completed-seed list mismatch.",
    )
    _require(
        aggregate.get("seed_count")
        == len(REGISTERED_SEEDS),
        "Aggregate seed count mismatch.",
    )

    posthoc_path = (
        release_artifact_directory
        / "posthoc_sensitivity.json"
    )
    posthoc, posthoc_sha256 = (
        _load_verified_json(posthoc_path)
    )

    reproduction = posthoc.get(
        "point_estimate_reproduction"
    )
    _require(
        isinstance(reproduction, Mapping)
        and reproduction,
        "Missing post-hoc point-estimate reproduction.",
    )
    _require(
        all(
            item.get("matched") is True
            for item in reproduction.values()
            if isinstance(item, Mapping)
        ),
        "Aggregate point-estimate reproduction failed.",
    )

    total_records = sum(
        item["system_evaluation_count"]
        for item in seed_audits
    )
    _require(
        total_records == 1260000,
        "Total system-evaluation count mismatch.",
    )

    return {
        "schema_version": (
            "noi-v0.3-posthoc-trace-audit-v1"
        ),
        "study_phase": "post_confirmatory",
        "analysis_classification": (
            "post_hoc_read_only_integrity_audit"
        ),
        "confirmatory_results_modified": False,
        "confirmatory_hypothesis_statuses_modified": False,
        "registered_seeds": list(
            REGISTERED_SEEDS
        ),
        "audit_chain": [
            "verified_raw_seed_artifact",
            "seedwise_validation_lock",
            "locked_support_application",
            "system_inference_export",
            "label_based_scoring",
            "aggregate_point_estimate_reproduction",
        ],
        "verified_boundaries": {
            "training_only_model_fitting": True,
            "validation_only_threshold_derivation": True,
            "final_test_labels_used_for_scoring_only": True,
            "target_labels_used_as_inference_input": False,
            "condition_metadata_used_as_model_input": False,
            "quality_metadata_used_as_model_input": False,
            "thresholds_changed_from_final_test": False,
            "paired_views_treated_as_independent": False,
        },
        "aggregate_linkage": {
            "aggregate_sha256": aggregate_sha256,
            "posthoc_sensitivity_sha256": (
                posthoc_sha256
            ),
            "point_estimates_reproduced": True,
            "total_system_evaluation_count": (
                total_records
            ),
        },
        "validation_lock": {
            "path": str(validation_lock_path),
            "sha256": validation_lock_sha256,
            "seedwise_values_matched": True,
        },
        "seed_audits": seed_audits,
        "interpretation_limits": {
            "proof_of_no_possible_leakage": False,
            "meaning": (
                "The audit found no violation of the "
                "registered and testable leakage boundaries; "
                "it is not a mathematical proof that no "
                "conceivable leakage channel exists."
            ),
            "new_confirmatory_claim_created": False,
        },
    }


def export_trace_audit(
    payload: Mapping[str, Any],
    output_path: Path,
    *,
    overwrite: bool,
) -> str:
    sidecar = Path(f"{output_path}.sha256")

    if (
        not overwrite
        and (
            output_path.exists()
            or sidecar.exists()
        )
    ):
        raise TraceAuditError(
            f"Refusing to overwrite: {output_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    output_path.write_text(
        content,
        encoding="utf-8",
    )
    digest = _sha256(output_path)
    sidecar.write_text(
        f"{digest}  {output_path.name}\n",
        encoding="utf-8",
    )

    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Perform a read-only trace audit of the "
            "completed NOI v0.3 confirmatory artifacts."
        )
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path(
            "results/v0.3-confirmatory"
        ),
    )
    parser.add_argument(
        "--validation-lock",
        type=Path,
        default=Path(
            "configs/noi_v0.3_validation_lock.yaml"
        ),
    )
    parser.add_argument(
        "--release-artifact-directory",
        type=Path,
        default=Path(
            "artifacts/noi_v0.3_confirmatory"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/noi_v0.3_confirmatory/"
            "trace_audit.json"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()

    payload = build_trace_audit(
        results_directory=(
            args.results_directory
        ),
        validation_lock_path=(
            args.validation_lock
        ),
        release_artifact_directory=(
            args.release_artifact_directory
        ),
    )
    digest = export_trace_audit(
        payload,
        args.output,
        overwrite=args.overwrite,
    )

    print("CONFIRMATORY TRACE AUDIT: PASS")
    print(f"OUTPUT: {args.output}")
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
