"""Locked final robustness configuration loader."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import yaml


class RobustnessConfigurationError(ValueError):
    """Raised when the robustness lock is invalid."""


@dataclass(frozen=True)
class RobustnessRun:
    run_id: str
    generator_seed: int
    ood_seed: int


@dataclass(frozen=True)
class MissingModalityCondition:
    condition_id: str
    missing_count: int
    missing_modalities: tuple[str, ...]


@dataclass(frozen=True)
class RobustnessConfiguration:
    version: str
    event_count_per_run: int
    runs: tuple[RobustnessRun, ...]
    severity_tiers: tuple[str, ...]
    missing_conditions: tuple[
        MissingModalityCondition,
        ...,
    ]
    temporal_displacement_days: tuple[int, ...]
    systems: tuple[str, ...]
    baseline_systems: tuple[str, ...]
    full_system: str
    primary_metric: str
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float
    configuration_sha256: str


LOCKED_MODALITIES = (
    "text",
    "image",
    "audio",
)
LOCKED_TEMPORAL_DAYS = (
    0,
    1,
    7,
    30,
    90,
)
LOCKED_SYSTEMS = (
    "ridge_only",
    "memory_only",
    "hybrid_without_temporal_decay",
    "full_hybrid",
)
LOCKED_BASELINES = (
    "ridge_only",
    "memory_only",
    "hybrid_without_temporal_decay",
)
LOCKED_CONDITIONS = (
    ("all-present", 0, ()),
    ("missing-text", 1, ("text",)),
    ("missing-image", 1, ("image",)),
    ("missing-audio", 1, ("audio",)),
    (
        "missing-text-image",
        2,
        ("text", "image"),
    ),
    (
        "missing-text-audio",
        2,
        ("text", "audio"),
    ),
    (
        "missing-image-audio",
        2,
        ("image", "audio"),
    ),
)
LOCKED_PARENT_HASH = (
    "da88cbf40fcca2cfee53fab247a2d9e"
    "509598a1319129b53cc5045ba8f857fd0"
)


def load_robustness_configuration(
    config_path: str | Path,
    hash_path: str | Path,
) -> RobustnessConfiguration:
    """Load and strictly validate the robustness lock."""

    config_path = Path(config_path)
    hash_path = Path(hash_path)

    try:
        data = config_path.read_bytes()
        recorded = hash_path.read_text(
            encoding="utf-8"
        ).strip().split()[0]
    except OSError as exc:
        raise RobustnessConfigurationError(
            "Unable to read robustness configuration."
        ) from exc

    observed = sha256(data).hexdigest()

    if recorded != observed:
        raise RobustnessConfigurationError(
            "Robustness configuration SHA-256 "
            "does not match."
        )

    try:
        payload = yaml.safe_load(
            data.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        yaml.YAMLError,
    ) as exc:
        raise RobustnessConfigurationError(
            "Invalid robustness YAML."
        ) from exc

    if not isinstance(payload, Mapping):
        raise RobustnessConfigurationError(
            "Robustness YAML must be a mapping."
        )

    root = _mapping(
        payload.get("robustness_evaluation"),
        "robustness_evaluation",
    )
    dataset = _mapping(
        payload.get("dataset"),
        "dataset",
    )
    axes = _mapping(
        payload.get("evaluation_axes"),
        "evaluation_axes",
    )
    missing = _mapping(
        payload.get("missing_modality"),
        "missing_modality",
    )
    temporal = _mapping(
        payload.get("temporal_displacement"),
        "temporal_displacement",
    )
    systems_payload = _mapping(
        payload.get("systems"),
        "systems",
    )
    metrics = _mapping(
        payload.get("metrics"),
        "metrics",
    )
    baseline_rule = _mapping(
        payload.get("strongest_baseline_rule"),
        "strongest_baseline_rule",
    )
    statistics = _mapping(
        payload.get("statistical_analysis"),
        "statistical_analysis",
    )
    governance = _mapping(
        payload.get("governance"),
        "governance",
    )

    if root.get("version") != "0.2.3":
        raise RobustnessConfigurationError(
            "Version must equal 0.2.3."
        )

    if root.get("status") != (
        "prespecified_before_final_"
        "robustness_results"
    ):
        raise RobustnessConfigurationError(
            "Registration status is invalid."
        )

    if root.get("parent_track_b_sha256") != (
        LOCKED_PARENT_HASH
    ):
        raise RobustnessConfigurationError(
            "Parent Track B SHA-256 is invalid."
        )

    if dataset.get("event_count_per_run") != 10000:
        raise RobustnessConfigurationError(
            "Each run must contain 10000 events."
        )

    if dataset.get("independent_run_count") != 10:
        raise RobustnessConfigurationError(
            "Independent run count must equal 10."
        )

    raw_runs = dataset.get("runs")

    if (
        not isinstance(raw_runs, list)
        or len(raw_runs) != 10
    ):
        raise RobustnessConfigurationError(
            "Exactly ten locked runs are required."
        )

    runs = tuple(
        _parse_run(item, index)
        for index, item in enumerate(
            raw_runs,
            start=1,
        )
    )

    expected_runs = tuple(
        RobustnessRun(
            run_id=(
                f"robustness-seed-{number:02d}"
            ),
            generator_seed=(
                1001 + number * 100
            ),
            ood_seed=(
                9001 + number * 100
            ),
        )
        for number in range(1, 11)
    )

    if runs != expected_runs:
        raise RobustnessConfigurationError(
            "Run identifiers or seeds are not locked."
        )

    severity_tiers = tuple(
        dataset.get("severity_tiers", ())
    )

    if severity_tiers != (
        "mild",
        "moderate",
        "severe",
    ):
        raise RobustnessConfigurationError(
            "Severity tiers are not locked."
        )

    for key in (
        "same_latent_events_across_systems",
        "strict_held_out_odor_families",
        "strict_held_out_context_templates",
    ):
        if dataset.get(key) is not True:
            raise RobustnessConfigurationError(
                f"{key} must be true."
            )

    if dataset.get("paired_analysis_unit") != (
        "latent_event_id"
    ):
        raise RobustnessConfigurationError(
            "Paired analysis unit is invalid."
        )

    if (
        axes.get("axes_analyzed_separately")
        is not True
        or axes.get(
            "full_factorial_crossing_prohibited"
        )
        is not True
    ):
        raise RobustnessConfigurationError(
            "Robustness axes must remain separate."
        )

    if tuple(
        missing.get("available_modalities", ())
    ) != LOCKED_MODALITIES:
        raise RobustnessConfigurationError(
            "Available modalities are not locked."
        )

    raw_conditions = missing.get("conditions")

    if not isinstance(raw_conditions, list):
        raise RobustnessConfigurationError(
            "Missing-modality conditions are required."
        )

    conditions = tuple(
        _parse_condition(item)
        for item in raw_conditions
    )

    observed_conditions = tuple(
        (
            item.condition_id,
            item.missing_count,
            item.missing_modalities,
        )
        for item in conditions
    )

    if observed_conditions != LOCKED_CONDITIONS:
        raise RobustnessConfigurationError(
            "Missing-modality conditions differ "
            "from the exhaustive lock."
        )

    if (
        missing.get(
            "all_combinations_for_locked_counts_required"
        )
        is not True
        or missing.get(
            "temporal_displacement_days"
        )
        != 0
    ):
        raise RobustnessConfigurationError(
            "Missing-modality axis is invalid."
        )

    temporal_days = tuple(
        temporal.get("days", ())
    )

    if temporal_days != LOCKED_TEMPORAL_DAYS:
        raise RobustnessConfigurationError(
            "Temporal displacement values are invalid."
        )

    if (
        temporal.get(
            "all_modalities_present_required"
        )
        is not True
        or tuple(
            temporal.get(
                "missing_modalities",
                (),
            )
        )
        != ()
    ):
        raise RobustnessConfigurationError(
            "Temporal axis must retain all modalities."
        )

    systems = tuple(systems_payload)

    if systems != LOCKED_SYSTEMS:
        raise RobustnessConfigurationError(
            "System order is not locked."
        )

    baseline_systems = tuple(
        baseline_rule.get(
            "eligible_systems",
            (),
        )
    )

    if baseline_systems != LOCKED_BASELINES:
        raise RobustnessConfigurationError(
            "Strongest baseline set is invalid."
        )

    if metrics.get("primary_metric") != (
        "mean_reciprocal_rank"
    ):
        raise RobustnessConfigurationError(
            "Primary metric must be MRR."
        )

    if statistics.get(
        "bootstrap_resamples"
    ) != 10000:
        raise RobustnessConfigurationError(
            "Bootstrap resamples must equal 10000."
        )

    if statistics.get("bootstrap_seed") != 4245:
        raise RobustnessConfigurationError(
            "Bootstrap seed must equal 4245."
        )

    confidence_level = statistics.get(
        "confidence_level"
    )

    if confidence_level != 0.95:
        raise RobustnessConfigurationError(
            "Confidence level must equal 0.95."
        )

    required_false = (
        "oracle_used",
        "target_identifier_used_as_feature",
        "family_identifier_used_as_feature",
    )
    required_true = (
        "ood_model_fitting_prohibited",
        "ood_alpha_tuning_prohibited",
        "final_test_tuning_prohibited",
        "policy_gate_changes_ranking_prohibited",
        "all_conditions_must_be_reported",
        "all_runs_must_be_reported",
        "all_failures_must_be_retained",
    )

    for key in required_false:
        if governance.get(key) is not False:
            raise RobustnessConfigurationError(
                f"{key} must be false."
            )

    for key in required_true:
        if governance.get(key) is not True:
            raise RobustnessConfigurationError(
                f"{key} must be true."
            )

    return RobustnessConfiguration(
        version="0.2.3",
        event_count_per_run=10000,
        runs=runs,
        severity_tiers=severity_tiers,
        missing_conditions=conditions,
        temporal_displacement_days=(
            temporal_days
        ),
        systems=systems,
        baseline_systems=baseline_systems,
        full_system="full_hybrid",
        primary_metric=(
            "mean_reciprocal_rank"
        ),
        bootstrap_resamples=10000,
        bootstrap_seed=4245,
        confidence_level=0.95,
        configuration_sha256=observed,
    )


def _parse_run(
    value: Any,
    index: int,
) -> RobustnessRun:
    mapping = _mapping(
        value,
        f"runs[{index}]",
    )

    return RobustnessRun(
        run_id=_string(
            mapping.get("run_id"),
            "run_id",
        ),
        generator_seed=_seed(
            mapping.get("generator_seed"),
            "generator_seed",
        ),
        ood_seed=_seed(
            mapping.get("ood_seed"),
            "ood_seed",
        ),
    )


def _parse_condition(
    value: Any,
) -> MissingModalityCondition:
    mapping = _mapping(
        value,
        "missing condition",
    )
    modalities = tuple(
        mapping.get(
            "missing_modalities",
            (),
        )
    )

    if any(
        modality not in LOCKED_MODALITIES
        for modality in modalities
    ):
        raise RobustnessConfigurationError(
            "Unknown missing modality."
        )

    missing_count = mapping.get(
        "missing_count"
    )

    if (
        isinstance(missing_count, bool)
        or not isinstance(missing_count, int)
        or missing_count != len(modalities)
    ):
        raise RobustnessConfigurationError(
            "Missing modality count is invalid."
        )

    return MissingModalityCondition(
        condition_id=_string(
            mapping.get("condition_id"),
            "condition_id",
        ),
        missing_count=missing_count,
        missing_modalities=modalities,
    )


def _mapping(
    value: Any,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RobustnessConfigurationError(
            f"{name} must be a mapping."
        )

    return value


def _string(
    value: Any,
    name: str,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise RobustnessConfigurationError(
            f"{name} must be a nonempty string."
        )

    return value


def _seed(
    value: Any,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise RobustnessConfigurationError(
            f"{name} must be a nonnegative integer."
        )

    return value
