"""Tests for one leakage-resistant Track B seed."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.baselines.retrieval_baselines import (
    BaselineKind,
)
from src.evaluation.graded_ood import OODTier
from src.evaluation.paired_ood_source_generator import (
    generate_paired_graded_ood_bundle,
)
from src.evaluation.track_b_config import (
    load_track_b_configuration,
)
from src.evaluation.track_b_seed_experiment import (
    TrackBSeedExperiment,
    TrackBSeedExperimentError,
    run_track_b_seed_experiment,
)
from src.system.noi_pipeline import (
    load_noi_system_configuration,
)


TRACK_B_CONFIG_PATH = Path(
    "configs/"
    "track_b_unseen_family_evaluation_v0.2.2.yaml"
)
TRACK_B_HASH_PATH = Path(
    "configs/"
    "track_b_unseen_family_evaluation_v0.2.2.sha256"
)
TRAINED_AT = datetime(
    2026,
    8,
    27,
    12,
    0,
    tzinfo=timezone.utc,
)


@pytest.fixture(scope="module")
def track_b_config():
    return load_track_b_configuration(
        TRACK_B_CONFIG_PATH,
        TRACK_B_HASH_PATH,
    )


@pytest.fixture(scope="module")
def system_configuration() -> dict:
    return load_noi_system_configuration(
        "configs/noi_system_v0.1.yaml"
    )


@pytest.fixture(scope="module")
def policy_configuration() -> dict:
    with Path(
        "configs/policy_rules.yaml"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(handle)

    assert isinstance(payload, dict)
    return payload


@pytest.fixture(scope="module")
def bundle():
    return generate_paired_graded_ood_bundle(
        "configs/synthetic_data.yaml",
        "configs/research_protocol.yaml",
        "configs/protocol_amendment_v0.2.yaml",
        "configs/graded_ood_generation.yaml",
        event_count=200,
        generator_seed=1101,
        ood_seed=9101,
    )


@pytest.fixture(scope="module")
def experiment(
    track_b_config,
    system_configuration,
    policy_configuration,
    bundle,
) -> TrackBSeedExperiment:
    return run_track_b_seed_experiment(
        bundle=bundle,
        track_b_config=track_b_config,
        run=track_b_config.runs[0],
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=(
            track_b_config.configuration_sha256
        ),
        trained_at_utc=TRAINED_AT,
    )


def test_seed_metadata_and_counts(
    experiment: TrackBSeedExperiment,
) -> None:
    assert experiment.run_id == "track-b-seed-01"
    assert experiment.generator_seed == 1101
    assert experiment.ood_seed == 9101
    assert experiment.training_event_count == 140
    assert experiment.validation_event_count == 20
    assert experiment.latent_event_count == 40
    assert experiment.observed_event_count == 120


def test_all_ood_targets_are_unreachable(
    experiment: TrackBSeedExperiment,
) -> None:
    assert experiment.reachable_target_fraction == 0.0
    assert experiment.reachable_event_fraction == 0.0
    assert experiment.strict_family_separation_verified is True
    assert experiment.all_ood_targets_unreachable is True


def test_calibration_uses_validation_only(
    experiment: TrackBSeedExperiment,
) -> None:
    calibration = experiment.calibration

    assert experiment.validation_event_count == 20
    assert calibration.training_event_count == 140
    assert calibration.calibration_event_count == 13
    assert (
        experiment.validation_event_count
        - calibration.calibration_event_count
        == 7
    )
    assert calibration.excluded_event_count == 7
    assert (
        calibration.achieved_reachable_coverage
        >= calibration.minimum_reachable_coverage
        == 0.95
    )
    assert calibration.target_identifier_used is False
    assert calibration.family_identifier_used is False
    assert calibration.ood_oracle_used is False
    assert calibration.final_test_tuning_used is False


def test_locked_baselines_cover_every_tier(
    experiment: TrackBSeedExperiment,
) -> None:
    assert len(
        experiment.graded_baselines.evaluations
    ) == 12

    assert {
        (evaluation.tier, evaluation.baseline)
        for evaluation
        in experiment.graded_baselines.evaluations
    } == {
        (tier, baseline)
        for tier in OODTier
        for baseline in BaselineKind
    }


def test_noi_evaluations_cover_every_tier(
    experiment: TrackBSeedExperiment,
) -> None:
    assert len(experiment.full_noi_evaluations) == 3
    assert len(
        experiment.memory_only_evaluations
    ) == 3
    assert len(
        experiment.selective_evaluations
    ) == 3

    assert {
        evaluation.tier
        for evaluation
        in experiment.full_noi_evaluations
    } == set(OODTier)

    assert {
        evaluation.tier
        for evaluation
        in experiment.memory_only_evaluations
    } == set(OODTier)

    assert {
        evaluation.tier
        for evaluation
        in experiment.selective_evaluations
    } == set(OODTier)


def test_retrieval_metrics_are_valid(
    experiment: TrackBSeedExperiment,
) -> None:
    evaluations = (
        experiment.full_noi_evaluations
        + experiment.memory_only_evaluations
    )

    for evaluation in evaluations:
        assert evaluation.event_count == 40
        assert len(evaluation.rankings) == 40
        assert all(
            len(ranking) == 10
            for ranking in evaluation.rankings
        )

        for metric in (
            evaluation.recall_at_1,
            evaluation.recall_at_10,
            evaluation.mean_reciprocal_rank,
            evaluation.ndcg_at_10,
        ):
            assert 0.0 <= metric <= 1.0


def test_selective_metrics_are_consistent(
    experiment: TrackBSeedExperiment,
) -> None:
    for evaluation in (
        experiment.selective_evaluations
    ):
        assert evaluation.event_count == 40
        assert evaluation.reachable_event_fraction == 0.0
        assert (
            evaluation.coverage
            + evaluation.abstention_rate
            == pytest.approx(1.0)
        )
        assert (
            evaluation.false_support_rate
            == evaluation.coverage
        )
        assert (
            evaluation.abstained_event_count
            + evaluation.supported_event_count
            == evaluation.event_count
        )

        if evaluation.supported_event_count == 0:
            assert (
                evaluation.selective_error_rate
                is None
            )
        else:
            assert (
                evaluation.selective_error_rate
                is not None
            )
            assert (
                0.0
                <= evaluation.selective_error_rate
                <= 1.0
            )


def test_no_oracle_or_final_tuning(
    experiment: TrackBSeedExperiment,
) -> None:
    assert experiment.oracle_used is False
    assert experiment.final_test_tuning_used is False
    assert experiment.target_identifier_used_in_support is False
    assert experiment.family_identifier_used_in_support is False
    assert 0.0 <= experiment.selected_hybrid_alpha <= 1.0


def test_experiment_is_deterministic(
    experiment,
    track_b_config,
    system_configuration,
    policy_configuration,
    bundle,
) -> None:
    repeated = run_track_b_seed_experiment(
        bundle=bundle,
        track_b_config=track_b_config,
        run=track_b_config.runs[0],
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=(
            track_b_config.configuration_sha256
        ),
        trained_at_utc=TRAINED_AT,
    )

    assert repeated == experiment


def test_mismatched_run_seed_is_rejected(
    track_b_config,
    system_configuration,
    policy_configuration,
    bundle,
) -> None:
    with pytest.raises(
        TrackBSeedExperimentError,
        match="seed",
    ):
        run_track_b_seed_experiment(
            bundle=bundle,
            track_b_config=track_b_config,
            run=track_b_config.runs[1],
            system_configuration=system_configuration,
            policy_configuration=policy_configuration,
            protocol_hash=(
                track_b_config.configuration_sha256
            ),
            trained_at_utc=TRAINED_AT,
        )


def test_experiment_is_immutable(
    experiment: TrackBSeedExperiment,
) -> None:
    with pytest.raises(FrozenInstanceError):
        experiment.oracle_used = True  # type: ignore[misc]
