"""Tests for the locked NOI v0.3 confirmatory command runner."""

from pathlib import Path

import pytest

from experiments.run_noi_v0_3_confirmatory import (
    REGISTERED_SEEDS,
    ConfirmatoryExecutionError,
    condition_config_for_seed,
    generation_config_for_seed,
    load_seed_locked_values,
    run_registered_confirmatory_seed,
)


def test_registered_seed_set_is_exact() -> None:
    assert REGISTERED_SEEDS == tuple(range(1301, 1311))


@pytest.mark.parametrize(
    "seed",
    REGISTERED_SEEDS,
)
def test_generation_config_uses_full_locked_allocation(
    seed: int,
) -> None:
    config = generation_config_for_seed(seed)

    assert config.seed == seed
    assert config.train_event_count == 7000
    assert config.validation_event_count == 1000
    assert config.final_test_event_count == 2000
    assert config.final_seen_item_count == 800
    assert (
        config.final_known_family_unseen_item_count
        == 600
    )
    assert config.final_unseen_family_count == 600
    assert config.feasibility_only is False


@pytest.mark.parametrize(
    "seed",
    REGISTERED_SEEDS,
)
def test_condition_config_uses_locked_controls(
    seed: int,
) -> None:
    config = condition_config_for_seed(seed)

    assert config.seed == seed
    assert config.odor_noise_scale == 0.10
    assert config.tactile_noise_scale == 0.10
    assert config.degraded_quality == 0.40
    assert config.locked_temporal_offset_steps == 3


@pytest.mark.parametrize(
    "seed",
    (
        0,
        1300,
        1311,
        9999,
    ),
)
def test_unregistered_seed_is_rejected(
    seed: int,
) -> None:
    with pytest.raises(
        ConfirmatoryExecutionError,
        match="registered",
    ):
        generation_config_for_seed(seed)

    with pytest.raises(
        ConfirmatoryExecutionError,
        match="registered",
    ):
        condition_config_for_seed(seed)

    with pytest.raises(
        ConfirmatoryExecutionError,
        match="registered",
    ):
        load_seed_locked_values(seed)


@pytest.mark.parametrize(
    "seed",
    REGISTERED_SEEDS,
)
def test_each_seed_loads_all_five_frozen_values(
    seed: int,
) -> None:
    values = load_seed_locked_values(seed)

    assert set(values) == {
        "support_threshold",
        "support_uncertainty_lower",
        "support_uncertainty_upper",
        "reliability_threshold",
        "conflict_threshold",
    }
    assert (
        values["support_uncertainty_lower"]
        <= values["support_threshold"]
        <= values["support_uncertainty_upper"]
    )


def test_existing_result_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    output = tmp_path / "seed-1301.json"
    output.write_text(
        '{"existing": true}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ConfirmatoryExecutionError,
        match="already exists",
    ):
        run_registered_confirmatory_seed(
            seed=1301,
            output_path=output,
        )

    assert output.read_text(encoding="utf-8") == (
        '{"existing": true}\n'
    )


def test_result_hash_collision_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "seed-1301.json"
    hash_path = Path(f"{output}.sha256")
    hash_path.write_text(
        "existing\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfirmatoryExecutionError,
        match="already exists",
    ):
        run_registered_confirmatory_seed(
            seed=1301,
            output_path=output,
        )


def test_lock_file_is_validation_locked() -> None:
    values = load_seed_locked_values(1301)

    assert values["support_threshold"] == pytest.approx(
        -22.13686595194035
    )
    assert values["reliability_threshold"] == pytest.approx(
        0.177222364642993
    )
    assert values["conflict_threshold"] == pytest.approx(
        0.15928212004378728
    )
