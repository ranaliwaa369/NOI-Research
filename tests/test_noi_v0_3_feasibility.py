"""Integration tests for the non-confirmatory NOI v0.3 feasibility pilot."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from experiments.run_noi_v0_3_feasibility import (
    FeasibilityPilotError,
    build_feasibility_payload,
    export_feasibility_payload,
    run_feasibility_pilot,
)


def test_feasibility_pilot_is_deterministic() -> None:
    """Repeated feasibility execution returns identical scientific content."""

    first = run_feasibility_pilot()
    second = run_feasibility_pilot()

    assert first == second


def test_payload_has_explicit_nonconfirmatory_identity() -> None:
    """The artifact cannot be confused with confirmatory evidence."""

    report = run_feasibility_pilot()
    payload = build_feasibility_payload(report)

    assert payload["schema_version"] == "noi-v0.3-feasibility-v1"
    assert payload["study_phase"] == "feasibility"
    assert payload["confirmatory"] is False
    assert payload["seed"] == 1301
    assert payload["generator_version"] == "0.3.0-feasibility"


def test_pilot_uses_scaled_split_allocation() -> None:
    """The pilot preserves the documented 70/10/20 ratios."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )

    assert payload["allocation"] == {
        "train": 70,
        "validation": 10,
        "final_test": 20,
    }
    assert payload["final_support_allocation"] == {
        "seen_item": 8,
        "known_family_unseen_item": 6,
        "unseen_family": 6,
    }


def test_all_seven_conditions_are_generated() -> None:
    """Every final-test latent event receives all stress views."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )

    assert payload["condition_audit"]["latent_event_count"] == 20
    assert payload["condition_audit"]["view_count"] == 140
    assert payload["condition_audit"]["views_per_event"] == 7
    assert payload["condition_audit"]["all_conditions_present"] is True
    assert payload["condition_audit"]["ground_truth_preserved"] is True
    assert payload["condition_audit"]["support_regimes_preserved"] is True


def test_leakage_audit_passes_all_registered_boundaries() -> None:
    """Training and unknown-family partitions remain separated."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )
    audit = payload["leakage_audit"]

    assert audit["seen_items_reachable"] is True
    assert audit["known_family_items_absent_from_training"] is True
    assert audit["known_families_reachable"] is True
    assert audit["unseen_families_absent_from_training"] is True
    assert audit[
        "validation_unknown_families_disjoint_from_final"
    ] is True
    assert audit["passed"] is True


def test_all_three_support_methods_run() -> None:
    """The pilot checks mechanics for every preregistered support method."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )
    methods = payload["support_gate_methods"]

    assert set(methods) == {
        "mahalanobis",
        "cosine_margin",
        "nearest_prototype_distance",
    }

    for method_payload in methods.values():
        assert method_payload["fitted_on_split"] == "train"
        assert method_payload["calibrated_on_split"] == "validation"
        assert method_payload["training_event_count"] == 70
        assert method_payload["validation_event_count"] == 10
        assert method_payload["final_test_labels_used_for_calibration"] is False
        assert math.isfinite(method_payload["threshold"])
        assert math.isfinite(method_payload["balanced_accuracy"])
        assert math.isfinite(method_payload["final_auroc"])
        assert math.isfinite(method_payload["final_false_known_rate"])


def test_metric_behavior_is_finite_and_bounded() -> None:
    """Feasibility summaries prove mechanics without implying efficacy."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )
    metrics = payload["metric_behavior"]

    for value in metrics.values():
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_fusion_mechanics_cover_registered_methods() -> None:
    """Proposed and baseline fusion methods produce auditable actions."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )
    fusion = payload["fusion_audit"]

    assert set(fusion) == {
        "reliability_gated",
        "naive_concatenation",
        "fixed_equal",
    }

    for method_payload in fusion.values():
        assert method_payload["decision_count"] == 140
        assert sum(
            method_payload["action_counts"].values()
        ) == 140
        assert method_payload["unavailable_modalities_received_weight"] is False


def test_hypotheses_are_explicitly_not_tested() -> None:
    """Feasibility results cannot support H6, H7, or H8."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )

    assert payload["hypotheses"] == {
        "H6": "not_tested",
        "H7": "not_tested",
        "H8": "not_tested",
    }
    assert payload["supports_h6"] is False
    assert payload["supports_h7"] is False
    assert payload["supports_h8"] is False


def test_integrity_flags_prohibit_test_based_tuning() -> None:
    """The payload records the pilot's scientific boundaries."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )
    integrity = payload["integrity"]

    assert integrity["test_labels_for_training_used"] is False
    assert integrity["test_labels_for_calibration_used"] is False
    assert integrity["thresholds_changed_from_final_test"] is False
    assert integrity["confirmatory_claims_allowed"] is False


def test_export_writes_canonical_json(
    tmp_path: Path,
) -> None:
    """The pilot artifact is machine-readable and reproducible."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )
    output_path = tmp_path / "feasibility.json"

    written = export_feasibility_payload(
        payload,
        output_path,
    )

    assert written == output_path
    assert output_path.is_file()

    loaded = json.loads(
        output_path.read_text(encoding="utf-8"),
    )

    assert loaded == payload
    assert output_path.read_text(
        encoding="utf-8",
    ).endswith("\n")


def test_export_refuses_silent_overwrite(
    tmp_path: Path,
) -> None:
    """Existing pilot evidence cannot be replaced accidentally."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )
    output_path = tmp_path / "feasibility.json"

    export_feasibility_payload(
        payload,
        output_path,
    )

    with pytest.raises(
        FeasibilityPilotError,
        match="already exists",
    ):
        export_feasibility_payload(
            payload,
            output_path,
        )


def test_export_allows_explicit_overwrite(
    tmp_path: Path,
) -> None:
    """Intentional replacement requires an explicit flag."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )
    output_path = tmp_path / "feasibility.json"

    export_feasibility_payload(
        payload,
        output_path,
    )
    export_feasibility_payload(
        payload,
        output_path,
        overwrite=True,
    )

    loaded = json.loads(
        output_path.read_text(encoding="utf-8"),
    )

    assert loaded == payload


def test_provenance_document_contains_required_limits() -> None:
    """Human-readable provenance prevents overclaiming the pilot."""

    path = Path(
        "docs/noi_v0.3_feasibility_provenance.md",
    )

    assert path.is_file()

    text = path.read_text(encoding="utf-8")

    for required in (
        "Feasibility",
        "H6",
        "H7",
        "H8",
        "not tested",
        "synthetic computational",
        "validation",
        "final-test labels",
        "confirmatory",
    ):
        assert required.lower() in text.lower()


def test_payload_contains_no_runtime_in_deterministic_core() -> None:
    """Wall-clock variation cannot change the scientific payload."""

    payload = build_feasibility_payload(
        run_feasibility_pilot(),
    )

    assert "runtime_seconds" not in payload
    assert "timestamp" not in payload


def test_unknown_payload_type_is_rejected(
    tmp_path: Path,
) -> None:
    """Export accepts only the validated pilot payload schema."""

    with pytest.raises(FeasibilityPilotError):
        export_feasibility_payload(
            {"schema_version": "wrong"},
            tmp_path / "bad.json",
        )
