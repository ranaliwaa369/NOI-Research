"""Validated loader for the locked Track B protocol."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import yaml


class TrackBConfigurationError(ValueError):
    """Raised when the Track B protocol is invalid."""


@dataclass(frozen=True)
class TrackBRun:
    """One prespecified independent Track B run."""

    run_id: str
    generator_seed: int
    ood_seed: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise TrackBConfigurationError(
                "run_id must be a nonempty string."
            )

        for name, value in (
            ("generator_seed", self.generator_seed),
            ("ood_seed", self.ood_seed),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise TrackBConfigurationError(
                    f"{name} must be a nonnegative integer."
                )

        if self.generator_seed == self.ood_seed:
            raise TrackBConfigurationError(
                "generator_seed and ood_seed must differ."
            )


@dataclass(frozen=True)
class TrackBConfiguration:
    """Immutable validated Track B configuration."""

    version: str
    track: str
    independent_run_count: int
    runs: tuple[TrackBRun, ...]
    severity_tiers: tuple[str, ...]
    minimum_reachable_coverage: float
    minimum_ood_abstention_rate: float
    bootstrap_resamples: int
    bootstrap_seed: int
    ood_events_used_for_threshold: bool
    final_test_tuning_prohibited: bool
    require_all_ood_unreachable: bool
    strict_family_separation: bool
    target_identifier_used: bool
    family_identifier_used: bool
    ood_oracle_used: bool
    pool_seen_and_unseen_metrics: bool
    configuration_sha256: str


def load_track_b_configuration(
    configuration_path: str | Path,
    hash_path: str | Path,
) -> TrackBConfiguration:
    """Load and validate the prespecified Track B protocol."""

    config_path = Path(configuration_path)
    digest_path = Path(hash_path)

    if not config_path.is_file():
        raise TrackBConfigurationError(
            f"Configuration does not exist: {config_path}"
        )

    if not digest_path.is_file():
        raise TrackBConfigurationError(
            f"SHA-256 file does not exist: {digest_path}"
        )

    try:
        config_bytes = config_path.read_bytes()
        digest_text = digest_path.read_text(
            encoding="utf-8"
        )
    except OSError as error:
        raise TrackBConfigurationError(
            "Unable to read Track B configuration files."
        ) from error

    digest_parts = digest_text.strip().split()

    if not digest_parts:
        raise TrackBConfigurationError(
            "SHA-256 file is empty."
        )

    recorded_digest = digest_parts[0].lower()
    _validate_digest(recorded_digest)

    observed_digest = sha256(
        config_bytes
    ).hexdigest()

    if observed_digest != recorded_digest:
        raise TrackBConfigurationError(
            "Track B configuration failed SHA-256 "
            "verification."
        )

    try:
        payload = yaml.safe_load(
            config_bytes.decode("utf-8")
        )
    except (
        UnicodeDecodeError,
        yaml.YAMLError,
    ) as error:
        raise TrackBConfigurationError(
            "Track B configuration is not valid YAML."
        ) from error

    if not isinstance(payload, dict):
        raise TrackBConfigurationError(
            "Track B configuration must be a mapping."
        )

    protocol = _mapping(payload, "protocol")
    scope = _mapping(payload, "scope")
    governance = _mapping(payload, "governance")
    dataset = _mapping(payload, "dataset")
    primary_runs = _mapping(
        payload,
        "primary_runs",
    )
    memory_support = _mapping(
        payload,
        "memory_support",
    )
    calibration = _mapping(
        payload,
        "threshold_calibration",
    )
    criterion = _mapping(
        payload,
        "confirmatory_abstention_criterion",
    )
    statistics = _mapping(payload, "statistics")

    _require_equal(
        protocol.get("name"),
        "NOI Unseen-Family Generalization "
        "and Abstention",
        "protocol.name",
    )
    _require_equal(
        protocol.get("version"),
        "0.2.2",
        "protocol.version",
    )
    _require_equal(
        protocol.get("status"),
        "prespecified_preimplementation",
        "protocol.status",
    )
    _require_equal(
        protocol.get("owner"),
        "GUARDIANX LLC",
        "protocol.owner",
    )

    track = scope.get("evaluation_track")

    _require_equal(
        track,
        "Track B: Unseen-family generalization",
        "scope.evaluation_track",
    )

    pooling = scope.get(
        "pools_seen_and_unseen_metrics"
    )

    if pooling is not False:
        raise TrackBConfigurationError(
            "Seen and unseen metrics must not be pooled."
        )

    final_tuning_prohibited = governance.get(
        "prohibit_final_test_tuning"
    )

    if final_tuning_prohibited is not True:
        raise TrackBConfigurationError(
            "The protocol must prohibit final-test tuning."
        )

    for key in (
        "prohibit_result_inspection_before_implementation_lock",
        "retain_all_runs",
        "hash_every_export",
    ):
        if governance.get(key) is not True:
            raise TrackBConfigurationError(
                f"governance.{key} must be true."
            )

    _require_integer(
        dataset.get("events_per_run"),
        expected=10000,
        name="dataset.events_per_run",
    )
    _require_integer(
        dataset.get("base_latent_ood_events"),
        expected=2000,
        name="dataset.base_latent_ood_events",
    )
    _require_integer(
        dataset.get("severity_views_per_latent_event"),
        expected=3,
        name=(
            "dataset."
            "severity_views_per_latent_event"
        ),
    )
    _require_integer(
        dataset.get("expected_observed_ood_rows"),
        expected=6000,
        name="dataset.expected_observed_ood_rows",
    )

    severity_tiers_raw = dataset.get(
        "severity_tiers"
    )

    if severity_tiers_raw != [
        "mild",
        "moderate",
        "severe",
    ]:
        raise TrackBConfigurationError(
            "Severity tiers must be mild, moderate, "
            "and severe."
        )

    strict_family_separation = dataset.get(
        "strict_odor_family_separation"
    )

    if strict_family_separation is not True:
        raise TrackBConfigurationError(
            "Strict odor-family separation is required."
        )

    require_all_unreachable = dataset.get(
        "require_all_final_ood_targets_unreachable"
    )

    if require_all_unreachable is not True:
        raise TrackBConfigurationError(
            "All final OOD targets must be unreachable."
        )

    for key in (
        "report_each_severity_separately",
        "prohibit_seen_unseen_pooling",
    ):
        if dataset.get(key) is not True:
            raise TrackBConfigurationError(
                f"dataset.{key} must be true."
            )

    independent_run_count = _require_integer(
        primary_runs.get("independent_run_count"),
        expected=10,
        name="primary_runs.independent_run_count",
    )

    raw_runs = primary_runs.get("runs")

    if not isinstance(raw_runs, list):
        raise TrackBConfigurationError(
            "primary_runs.runs must be a list."
        )

    if len(raw_runs) != independent_run_count:
        raise TrackBConfigurationError(
            "Run count must equal independent_run_count."
        )

    runs: list[TrackBRun] = []

    for index, raw_run in enumerate(
        raw_runs,
        start=1,
    ):
        if not isinstance(raw_run, dict):
            raise TrackBConfigurationError(
                "Every Track B run must be a mapping."
            )

        expected_id = (
            f"track-b-seed-{index:02d}"
        )

        _require_equal(
            raw_run.get("run_id"),
            expected_id,
            f"run {index} run_id",
        )

        run = TrackBRun(
            run_id=raw_run.get("run_id"),
            generator_seed=raw_run.get(
                "generator_seed"
            ),
            ood_seed=raw_run.get("ood_seed"),
        )
        runs.append(run)

    _require_unique(
        tuple(run.run_id for run in runs),
        "run_id",
    )
    _require_unique(
        tuple(
            run.generator_seed
            for run in runs
        ),
        "generator_seed",
    )
    _require_unique(
        tuple(run.ood_seed for run in runs),
        "ood_seed",
    )

    _require_equal(
        memory_support.get("signal"),
        (
            "maximum clipped cosine similarity "
            "to any active training memory"
        ),
        "memory_support.signal",
    )
    _require_equal(
        memory_support.get("feature_source"),
        "mean_fused_context",
        "memory_support.feature_source",
    )

    target_identifier_used = memory_support.get(
        "target_item_identifier_used"
    )
    family_identifier_used = memory_support.get(
        "target_family_identifier_used"
    )
    ood_oracle_used = memory_support.get(
        "ood_oracle_used"
    )

    if target_identifier_used is not False:
        raise TrackBConfigurationError(
            "Target identifiers must not be used."
        )

    if family_identifier_used is not False:
        raise TrackBConfigurationError(
            "Family identifiers must not be used."
        )

    if ood_oracle_used is not False:
        raise TrackBConfigurationError(
            "The OOD oracle must not be used."
        )

    if (
        memory_support.get(
            "temporal_decay_applied"
        )
        is not False
    ):
        raise TrackBConfigurationError(
            "Temporal decay must be disabled in "
            "the Track B support score."
        )

    _require_equal(
        calibration.get("source_split"),
        "validation",
        "threshold_calibration.source_split",
    )

    if (
        calibration.get(
            "require_memory_reachable_events"
        )
        is not True
    ):
        raise TrackBConfigurationError(
            "Threshold calibration requires "
            "memory-reachable events."
        )

    if (
        calibration.get(
            "exclude_memory_unreachable_events"
        )
        is not True
    ):
        raise TrackBConfigurationError(
            "Unreachable validation events must "
            "be excluded from threshold calibration."
        )

    ood_events_used = calibration.get(
        "ood_events_used"
    )

    if ood_events_used is not False:
        raise TrackBConfigurationError(
            "OOD events must not be used for "
            "threshold calibration."
        )

    if (
        calibration.get(
            "final_test_events_used"
        )
        is not False
    ):
        raise TrackBConfigurationError(
            "Final-test events must not be used "
            "for threshold calibration."
        )

    minimum_coverage = _require_probability(
        calibration.get(
            "minimum_reachable_coverage"
        ),
        "minimum_reachable_coverage",
    )

    if minimum_coverage != 0.95:
        raise TrackBConfigurationError(
            "Minimum reachable coverage must be 0.95."
        )

    _require_equal(
        calibration.get("support_rule"),
        "support_score >= threshold",
        "threshold_calibration.support_rule",
    )
    _require_equal(
        calibration.get("abstention_rule"),
        "support_score < threshold",
        "threshold_calibration.abstention_rule",
    )

    if calibration.get("ties_are_supported") is not True:
        raise TrackBConfigurationError(
            "Threshold ties must be supported."
        )

    criterion_coverage = _require_probability(
        criterion.get(
            "calibration_reachable_coverage_minimum"
        ),
        (
            "confirmatory calibration "
            "reachable coverage"
        ),
    )

    if criterion_coverage != 0.95:
        raise TrackBConfigurationError(
            "Confirmatory reachable coverage "
            "must be 0.95."
        )

    minimum_abstention = _require_probability(
        criterion.get(
            "unseen_family_abstention_rate_minimum"
        ),
        "unseen-family abstention rate",
    )

    if minimum_abstention != 0.80:
        raise TrackBConfigurationError(
            "Minimum unseen-family abstention "
            "rate must be 0.80."
        )

    if criterion.get(
        "require_both_conditions"
    ) is not True:
        raise TrackBConfigurationError(
            "Both confirmatory conditions are required."
        )

    _require_equal(
        statistics.get("confidence_level"),
        0.95,
        "statistics.confidence_level",
    )

    bootstrap_resamples = _require_integer(
        statistics.get("bootstrap_resamples"),
        expected=10000,
        name="statistics.bootstrap_resamples",
    )
    bootstrap_seed = _require_integer(
        statistics.get("bootstrap_seed"),
        expected=4244,
        name="statistics.bootstrap_seed",
    )

    _require_equal(
        statistics.get("bootstrap_unit"),
        "latent_event_id",
        "statistics.bootstrap_unit",
    )

    return TrackBConfiguration(
        version="0.2.2",
        track=track,
        independent_run_count=(
            independent_run_count
        ),
        runs=tuple(runs),
        severity_tiers=tuple(
            severity_tiers_raw
        ),
        minimum_reachable_coverage=(
            minimum_coverage
        ),
        minimum_ood_abstention_rate=(
            minimum_abstention
        ),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        ood_events_used_for_threshold=(
            ood_events_used
        ),
        final_test_tuning_prohibited=(
            final_tuning_prohibited
        ),
        require_all_ood_unreachable=(
            require_all_unreachable
        ),
        strict_family_separation=(
            strict_family_separation
        ),
        target_identifier_used=(
            target_identifier_used
        ),
        family_identifier_used=(
            family_identifier_used
        ),
        ood_oracle_used=ood_oracle_used,
        pool_seen_and_unseen_metrics=pooling,
        configuration_sha256=observed_digest,
    )


def _mapping(
    payload: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    value = payload.get(key)

    if not isinstance(value, dict):
        raise TrackBConfigurationError(
            f"{key} must be a mapping."
        )

    return value


def _validate_digest(value: str) -> None:
    if (
        len(value) != 64
        or any(
            character
            not in "0123456789abcdef"
            for character in value
        )
    ):
        raise TrackBConfigurationError(
            "Recorded SHA-256 is invalid."
        )


def _require_equal(
    observed: Any,
    expected: Any,
    name: str,
) -> None:
    if observed != expected:
        raise TrackBConfigurationError(
            f"{name} must equal {expected!r}."
        )


def _require_integer(
    value: Any,
    *,
    expected: int,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != expected
    ):
        raise TrackBConfigurationError(
            f"{name} must equal {expected}."
        )

    return value


def _require_probability(
    value: Any,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TrackBConfigurationError(
            f"{name} must be numeric."
        )

    converted = float(value)

    if (
        not isfinite(converted)
        or not 0.0 <= converted <= 1.0
    ):
        raise TrackBConfigurationError(
            f"{name} must be in [0, 1]."
        )

    return converted


def _require_unique(
    values: tuple[Any, ...],
    name: str,
) -> None:
    if len(set(values)) != len(values):
        raise TrackBConfigurationError(
            f"{name} values must be unique."
        )
