"""Tests for deterministic Track B seed export."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from src.evaluation.paired_ood_source_generator import (
    generate_paired_graded_ood_bundle,
)
from src.evaluation.track_b_config import (
    load_track_b_configuration,
)
from src.evaluation.track_b_seed_experiment import (
    run_track_b_seed_experiment,
)
from src.evaluation.track_b_seed_export import (
    TrackBSeedExportError,
    export_track_b_seed_experiment,
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
def experiment():
    track_b_config = load_track_b_configuration(
        TRACK_B_CONFIG_PATH,
        TRACK_B_HASH_PATH,
    )
    system_configuration = (
        load_noi_system_configuration(
            "configs/noi_system_v0.1.yaml"
        )
    )

    with Path(
        "configs/policy_rules.yaml"
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:
        policy_configuration = yaml.safe_load(
            handle
        )

    bundle = generate_paired_graded_ood_bundle(
        "configs/synthetic_data.yaml",
        "configs/research_protocol.yaml",
        "configs/protocol_amendment_v0.2.yaml",
        "configs/graded_ood_generation.yaml",
        event_count=200,
        generator_seed=1101,
        ood_seed=9101,
    )

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


def test_export_writes_json_and_hash(
    experiment,
    tmp_path,
) -> None:
    exported = export_track_b_seed_experiment(
        experiment,
        tmp_path / "track-b-seed-01.json",
    )

    assert exported.json_path.is_file()
    assert exported.sha256_path.is_file()

    observed = hashlib.sha256(
        exported.json_path.read_bytes()
    ).hexdigest()
    recorded = exported.sha256_path.read_text(
        encoding="utf-8"
    ).strip()

    assert exported.sha256 == observed
    assert recorded == observed


def test_export_contains_locked_metadata(
    experiment,
    tmp_path,
) -> None:
    exported = export_track_b_seed_experiment(
        experiment,
        tmp_path / "track-b-seed-01.json",
    )
    payload = json.loads(
        exported.json_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["artifact_type"] == (
        "track_b_unseen_family_seed_result"
    )
    assert payload["run_id"] == "track-b-seed-01"
    assert payload["seeds"] == {
        "generator_seed": 1101,
        "ood_seed": 9101,
    }
    assert payload["protocol_sha256"] == (
        experiment.protocol_hash
    )
    assert payload["counts"] == {
        "training_events": 140,
        "validation_events": 20,
        "latent_ood_events": 40,
        "observed_ood_events": 120,
    }


def test_export_preserves_governance(
    experiment,
    tmp_path,
) -> None:
    exported = export_track_b_seed_experiment(
        experiment,
        tmp_path / "track-b-seed-01.json",
    )
    payload = json.loads(
        exported.json_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["governance"] == {
        "oracle_used": False,
        "final_test_tuning_used": False,
        "target_identifier_used_in_support": False,
        "family_identifier_used_in_support": False,
        "strict_family_separation_verified": True,
        "all_ood_targets_unreachable": True,
    }
    assert (
        payload["calibration"][
            "ood_oracle_used"
        ]
        is False
    )
    assert (
        payload["calibration"][
            "final_test_tuning_used"
        ]
        is False
    )


def test_export_contains_all_tiers_and_systems(
    experiment,
    tmp_path,
) -> None:
    exported = export_track_b_seed_experiment(
        experiment,
        tmp_path / "track-b-seed-01.json",
    )
    payload = json.loads(
        exported.json_path.read_text(
            encoding="utf-8"
        )
    )

    assert {
        item["tier"]
        for item in payload["selective_evaluations"]
    } == {"mild", "moderate", "severe"}

    assert {
        item["tier"]
        for item in payload["full_noi_evaluations"]
    } == {"mild", "moderate", "severe"}

    assert {
        item["tier"]
        for item in payload["memory_only_evaluations"]
    } == {"mild", "moderate", "severe"}

    assert len(
        payload["graded_baselines"][
            "evaluations"
        ]
    ) == 12


def test_repeated_export_is_deterministic(
    experiment,
    tmp_path,
) -> None:
    first = export_track_b_seed_experiment(
        experiment,
        tmp_path / "first" /
        "track-b-seed-01.json",
    )
    second = export_track_b_seed_experiment(
        experiment,
        tmp_path / "second" /
        "track-b-seed-01.json",
    )

    assert (
        first.json_path.read_bytes()
        == second.json_path.read_bytes()
    )
    assert first.sha256 == second.sha256


def test_existing_artifact_requires_overwrite(
    experiment,
    tmp_path,
) -> None:
    output_path = (
        tmp_path / "track-b-seed-01.json"
    )

    export_track_b_seed_experiment(
        experiment,
        output_path,
    )

    with pytest.raises(
        TrackBSeedExportError,
        match="exists",
    ):
        export_track_b_seed_experiment(
            experiment,
            output_path,
        )

    repeated = export_track_b_seed_experiment(
        experiment,
        output_path,
        overwrite=True,
    )

    assert repeated.json_path == output_path


def test_wrong_filename_is_rejected(
    experiment,
    tmp_path,
) -> None:
    with pytest.raises(
        TrackBSeedExportError,
        match="run ID",
    ):
        export_track_b_seed_experiment(
            experiment,
            tmp_path / "wrong-seed.json",
        )


def test_non_json_output_is_rejected(
    experiment,
    tmp_path,
) -> None:
    with pytest.raises(
        TrackBSeedExportError,
        match=".json",
    ):
        export_track_b_seed_experiment(
            experiment,
            tmp_path / "track-b-seed-01.txt",
        )
