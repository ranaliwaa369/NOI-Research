"""Tests for the locked NOI v0.3 execution specification."""

from pathlib import Path

import yaml


SPEC_PATH = Path("configs/noi_v0.3_execution_spec.yaml")

REGISTERED_SYSTEMS = {
    "odor_only_cosine",
    "touch_only_cosine",
    "odor_only_ridge",
    "touch_only_ridge",
    "naive_concatenation",
    "fixed_weight_fusion",
    "support_gate_odor_only",
    "reliability_gated_olfactory_tactile_fusion",
    "support_gate_reliability_fusion_with_abstention",
}


def load_spec() -> dict:
    """Load the versioned execution specification."""

    payload = yaml.safe_load(
        SPEC_PATH.read_text(encoding="utf-8"),
    )

    assert isinstance(payload, dict)
    return payload


def test_execution_spec_is_execution_locked() -> None:
    """Execution must be locked before final-test use."""

    specification = load_spec()[
        "execution_specification"
    ]

    assert specification["status"] == "execution_locked"
    assert specification[
        "parent_validation_lock_tag"
    ] == "noi-v0.3-validation-lock"
    assert specification["implementation_commit"] == (
        "6f12fcd77f18897c794a47fcdb98224d5c36727d"
    )
    assert specification["execution_lock_tag"] == (
        "noi-v0.3-confirmatory-execution-lock"
    )
    assert specification[
        "classification"
    ]["confirmatory_execution_started"] is False


def test_no_locked_research_decision_is_changed() -> None:
    """Execution clarification cannot alter the registered study."""

    classification = load_spec()[
        "execution_specification"
    ]["classification"]

    assert classification[
        "protocol_hypotheses_changed"
    ] is False
    assert classification[
        "registered_systems_changed"
    ] is False
    assert classification[
        "registered_seeds_changed"
    ] is False
    assert classification[
        "locked_thresholds_changed"
    ] is False


def test_no_final_test_information_was_inspected() -> None:
    """The execution rules must be fixed without final-test feedback."""

    classification = load_spec()[
        "execution_specification"
    ]["classification"]

    assert classification[
        "final_test_events_inspected"
    ] is False
    assert classification[
        "final_test_labels_inspected"
    ] is False
    assert classification[
        "final_test_metrics_inspected"
    ] is False


def test_candidate_library_is_training_only() -> None:
    """Final and validation targets cannot enter retrieval memory."""

    library = load_spec()["candidate_library"]

    assert library["source_split"] == "training_only"
    assert library["deterministic_order"] == (
        "ascending_item_id"
    )
    assert library[
        "target_labels_used_at_inference"
    ] is False


def test_all_nine_registered_systems_are_defined() -> None:
    """Every preregistered deployable system needs exact mechanics."""

    assert set(load_spec()["systems"]) == REGISTERED_SYSTEMS


def test_cosine_systems_use_their_own_modalities() -> None:
    """Unimodal cosine baselines cannot borrow another modality."""

    systems = load_spec()["systems"]

    assert systems["odor_only_cosine"]["query"] == (
        "available_16d_olfactory_vector"
    )
    assert systems["touch_only_cosine"]["query"] == (
        "available_8d_tactile_vector"
    )
    assert systems["odor_only_cosine"]["score"] == (
        "cosine_similarity"
    )
    assert systems["touch_only_cosine"]["score"] == (
        "cosine_similarity"
    )


def test_ridge_models_fit_training_only() -> None:
    """Both ridge baselines must exclude validation and final data."""

    systems = load_spec()["systems"]

    for name in (
        "odor_only_ridge",
        "touch_only_ridge",
    ):
        assert systems[name]["fit_split"] == "training_only"
        assert systems[name]["alpha"] == 1.0
        assert systems[name]["intercept"] is True


def test_fixed_fusion_uses_registered_equal_weights() -> None:
    """The fixed baseline must retain the locked 0.5/0.5 rule."""

    fixed = load_spec()["systems"][
        "fixed_weight_fusion"
    ]

    assert fixed["odor_block_weight"] == 0.5
    assert fixed["touch_block_weight"] == 0.5
    assert fixed["unavailable_modality_weight"] == 0.0


def test_reliability_fusion_uses_locked_evidence() -> None:
    """Dynamic fusion must use evidence and seedwise lock values."""

    system = load_spec()["systems"][
        "reliability_gated_olfactory_tactile_fusion"
    ]

    assert system["action_source"] == (
        "locked_evidence_fusion"
    )
    assert system["reliability_threshold_source"] == (
        "seedwise_validation_lock"
    )
    assert system["conflict_threshold_source"] == (
        "seedwise_validation_lock"
    )
    assert system["ranking"]["abstain"] == (
        "no_identity_ranking"
    )


def test_combined_system_abstains_when_certainly_unsupported() -> None:
    """Unsupported identities cannot receive a forced ranking."""

    system = load_spec()["systems"][
        "support_gate_reliability_fusion_with_abstention"
    ]

    assert system["certain_unsupported_action"] == "abstain"
    assert system["final_ranking"]["abstain"] == (
        "no_identity_ranking"
    )


def test_model_inputs_exclude_oracle_metadata() -> None:
    """Labels and condition metadata cannot influence inference."""

    prohibited = set(
        load_spec()["inference_integrity"][
            "prohibited_model_inputs"
        ]
    )

    assert {
        "condition label",
        "support regime",
        "modality_conflict flag",
        "olfactory quality metadata",
        "tactile quality metadata",
        "target item id",
        "target family id",
        "final-test correctness",
    } <= prohibited


def test_labels_are_used_only_for_posthoc_scoring() -> None:
    """Ground truth may score outputs but cannot generate them."""

    integrity = load_spec()["inference_integrity"]

    assert "target item id" not in integrity[
        "permitted_inputs"
    ]
    assert "target family id" not in integrity[
        "permitted_inputs"
    ]
    assert set(
        integrity["labels_permitted_for_posthoc_scoring_only"]
    ) == {
        "target item id",
        "target family id",
        "support regime",
        "condition label",
    }


def test_confirmatory_controls_remain_registered() -> None:
    """Execution must preserve pairing and bootstrap controls."""

    evaluation = load_spec()["evaluation"]

    assert evaluation["top_k"] == 10
    assert evaluation["statistical_unit"] == (
        "latent_event_id"
    )
    assert evaluation[
        "paired_views_are_independent_samples"
    ] is False
    assert evaluation[
        "execute_each_registered_seed_exactly_once"
    ] is True
    assert evaluation["bootstrap_seed"] == 4242
    assert evaluation["bootstrap_resamples"] == 10000
    assert evaluation["confidence_level"] == 0.95
    assert evaluation[
        "secondary_comparison_correction"
    ] == "Holm"


def test_execution_requires_a_separate_lock() -> None:
    """Implementation and hashes must precede final evaluation."""

    locking = load_spec()["locking"]

    assert locking[
        "separate_execution_lock_commit_required"
    ] is True
    assert locking[
        "separate_execution_lock_tag_required"
    ] is True
    assert locking[
        "configuration_hash_required"
    ] is True
    assert locking[
        "implementation_committed_before_final_test"
    ] is True
    assert locking["threshold_changes_allowed"] is False
    assert locking["final_test_tuning_allowed"] is False

def test_confidence_mapping_is_fixed_before_execution() -> None:
    """Final scores cannot select or recalibrate confidence rules."""

    confidence = load_spec()["confidence_policy"]

    assert confidence["source"] == (
        "top_ranked_weighted_cosine_score"
    )
    assert confidence["transformation"] == (
        "clip((top_score + 1) / 2, 0, 1)"
    )
    assert confidence["abstention_confidence"] == 0.0
    assert confidence["false_confident_threshold"] == 0.80
    assert confidence["calibration_status"] == (
        "evaluated_not_refit_on_final_test"
    )
    assert confidence[
        "final_test_calibration_allowed"
    ] is False

def test_hypothesis_comparisons_are_fixed_before_results() -> None:
    """Comparator selection cannot be chosen after inspecting outcomes."""

    comparisons = load_spec()["confirmatory_comparisons"]

    assert set(comparisons) == {"H6", "H7", "H8"}

    assert comparisons["H6"]["proposed_system"] == (
        "support_gate_odor_only"
    )
    assert comparisons["H6"]["comparator_selection"] == {
        "metric": "seen_item_clean_mean_reciprocal_rank",
        "rule": "maximum",
        "deterministic_tie_break": "ascending_system_name",
    }

    assert comparisons["H7"]["eligible_conditions"] == [
        "degraded_odor",
        "missing_odor",
    ]
    assert comparisons["H7"]["improvement_rule"] == (
        "absolute_or_relative"
    )

    assert comparisons["H8"]["required_comparators"] == [
        "naive_concatenation",
        "fixed_weight_fusion",
    ]
    assert comparisons["H8"]["both_comparators_must_pass"] is True
    assert comparisons["H8"]["eligible_conditions"] == [
        "degraded_odor",
        "degraded_touch",
        "missing_touch",
        "missing_odor",
        "contradictory_modalities",
        "temporal_misalignment",
    ]

def test_execution_specification_hash_is_current() -> None:
    """The locked execution specification must match its SHA-256."""

    import hashlib

    source = Path("configs/noi_v0.3_execution_spec.yaml")
    recorded = Path(
        "configs/noi_v0.3_execution_spec.sha256"
    )

    expected = recorded.read_text(
        encoding="utf-8"
    ).split()[0]
    observed = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()

    assert expected == observed
