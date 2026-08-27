"""Tests for per-seed repeated Track A checkpoints."""

from hashlib import sha256
import json

import pytest

from src.evaluation.repeated_seed_export import (
    RepeatedSeedExportError,
    export_repeated_seed_result,
)
from src.evaluation.seen_item_final_experiment import (
    SeenItemFinalExperiment,
)
from src.evaluation.seen_item_memory_experiment import (
    SeenItemEvaluation,
    SeenItemSystem,
)
from src.evaluation.seen_item_repeated_config import (
    RepeatedSeedSpec,
)
from src.evaluation.seen_item_repeated_runner import (
    RepeatedSeedRunResult,
)


def make_evaluation(
    system: SeenItemSystem,
    alpha: float,
) -> SeenItemEvaluation:
    return SeenItemEvaluation(
        system=system,
        alpha=alpha,
        event_ids=("validation-000001",),
        rankings=(("odor-a", "odor-b"),),
        relevant_items=(frozenset(("odor-a",)),),
        recall_at_1=1.0,
        recall_at_10=1.0,
        mean_reciprocal_rank=1.0,
        ndcg_at_10=1.0,
    )


def make_result() -> RepeatedSeedRunResult:
    experiment = SeenItemFinalExperiment(
        training_event_count=10,
        calibration_event_count=2,
        final_test_event_count=1,
        raw_final_test_event_count=1,
        reachable_event_fraction=1.0,
        calibration_template_ids=(1,),
        final_test_template_ids=(2,),
        final_test_event_ids=("validation-000001",),
        selected_hybrid_alpha=0.5,
        evaluations=(
            make_evaluation(
                SeenItemSystem.MEMORY_ONLY,
                0.0,
            ),
            make_evaluation(
                SeenItemSystem.RIDGE_ONLY,
                1.0,
            ),
            make_evaluation(
                SeenItemSystem.HYBRID,
                0.5,
            ),
        ),
        oracle_used=False,
        final_test_tuning_used=False,
        protocol_hash="protocol-hash",
    )

    return RepeatedSeedRunResult(
        run_spec=RepeatedSeedSpec(
            run_id="seed-01",
            generator_seed=1101,
            ood_seed=9101,
            partition_seed=2201,
        ),
        experiment=experiment,
    )


def test_checkpoint_contains_run_identity_and_seeds(
    tmp_path,
) -> None:
    exported = export_repeated_seed_result(
        make_result(),
        tmp_path,
        repeated_protocol_sha256="a" * 64,
    )

    payload = json.loads(
        exported.json_path.read_text(
            encoding="utf-8"
        )
    )

    assert payload["schema_version"] == "0.2.1"
    assert payload["repeated_run"] == {
        "run_id": "seed-01",
        "generator_seed": 1101,
        "ood_seed": 9101,
        "partition_seed": 2201,
    }
    assert payload[
        "repeated_protocol_sha256"
    ] == "a" * 64


def test_checkpoint_hash_matches_json(
    tmp_path,
) -> None:
    exported = export_repeated_seed_result(
        make_result(),
        tmp_path,
        repeated_protocol_sha256="a" * 64,
    )

    observed = sha256(
        exported.json_path.read_bytes()
    ).hexdigest()

    assert exported.sha256 == observed
    assert (
        exported.sha256_path.read_text(
            encoding="utf-8"
        ).strip()
        == observed
    )


def test_existing_checkpoint_requires_explicit_overwrite(
    tmp_path,
) -> None:
    result = make_result()

    export_repeated_seed_result(
        result,
        tmp_path,
        repeated_protocol_sha256="a" * 64,
    )

    with pytest.raises(
        RepeatedSeedExportError,
        match="already exists",
    ):
        export_repeated_seed_result(
            result,
            tmp_path,
            repeated_protocol_sha256="a" * 64,
        )


def test_explicit_overwrite_is_deterministic(
    tmp_path,
) -> None:
    result = make_result()

    first = export_repeated_seed_result(
        result,
        tmp_path,
        repeated_protocol_sha256="a" * 64,
    )
    first_bytes = first.json_path.read_bytes()

    second = export_repeated_seed_result(
        result,
        tmp_path,
        repeated_protocol_sha256="a" * 64,
        overwrite=True,
    )

    assert second.json_path.read_bytes() == first_bytes
    assert second.sha256 == first.sha256
