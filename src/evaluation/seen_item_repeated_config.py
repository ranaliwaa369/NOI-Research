"""Locked configuration loader for repeated NOI Track A runs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import yaml


class SeenItemRepeatedConfigError(ValueError):
    """Raised when the repeated-run protocol is invalid."""


@dataclass(frozen=True, slots=True)
class RepeatedSeedSpec:
    """One independent prespecified seed combination."""

    run_id: str
    generator_seed: int
    ood_seed: int
    partition_seed: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise SeenItemRepeatedConfigError(
                "run_id must not be empty."
            )

        for name, value in (
            ("generator_seed", self.generator_seed),
            ("ood_seed", self.ood_seed),
            ("partition_seed", self.partition_seed),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise SeenItemRepeatedConfigError(
                    f"{name} must be a nonnegative integer."
                )

        if self.generator_seed == self.ood_seed:
            raise SeenItemRepeatedConfigError(
                "Generator and OOD seeds must differ."
            )


EXPECTED_REPEATED_RUNS = (
    RepeatedSeedSpec("seed-01", 1101, 9101, 2201),
    RepeatedSeedSpec("seed-02", 1201, 9201, 2301),
    RepeatedSeedSpec("seed-03", 1301, 9301, 2401),
    RepeatedSeedSpec("seed-04", 1401, 9401, 2501),
    RepeatedSeedSpec("seed-05", 1501, 9501, 2601),
    RepeatedSeedSpec("seed-06", 1601, 9601, 2701),
    RepeatedSeedSpec("seed-07", 1701, 9701, 2801),
    RepeatedSeedSpec("seed-08", 1801, 9801, 2901),
    RepeatedSeedSpec("seed-09", 1901, 9901, 3001),
    RepeatedSeedSpec("seed-10", 2001, 10001, 3101),
)

EXPECTED_TOP_LEVEL_SECTIONS = frozenset(
    {
        "protocol",
        "scope",
        "provenance",
        "pilot_reference",
        "primary_runs",
        "dataset",
        "systems",
        "metrics",
        "primary_comparison",
        "statistics",
        "reproducibility",
        "prohibited_claims",
    }
)


@dataclass(frozen=True, slots=True)
class SeenItemRepeatedConfig:
    """Validated repeated-run settings."""

    version: str
    event_count: int
    include_pilot_reference: bool
    runs: tuple[RepeatedSeedSpec, ...]
    confidence_level: float
    bootstrap_seed: int
    bootstrap_resamples: int

    def __post_init__(self) -> None:
        if self.version != "0.2.1":
            raise SeenItemRepeatedConfigError(
                "Version must be 0.2.1."
            )

        if self.event_count != 10000:
            raise SeenItemRepeatedConfigError(
                "Each run must contain 10000 events."
            )

        if self.include_pilot_reference is not False:
            raise SeenItemRepeatedConfigError(
                "The pilot reference cannot enter primary runs."
            )

        if self.runs != EXPECTED_REPEATED_RUNS:
            raise SeenItemRepeatedConfigError(
                "Runs differ from the locked seed schedule."
            )

        if self.confidence_level != 0.95:
            raise SeenItemRepeatedConfigError(
                "Confidence level must remain 0.95."
            )

        if self.bootstrap_seed != 4243:
            raise SeenItemRepeatedConfigError(
                "Bootstrap seed must remain 4243."
            )

        if self.bootstrap_resamples != 10000:
            raise SeenItemRepeatedConfigError(
                "Bootstrap resamples must remain 10000."
            )


def load_seen_item_repeated_config(
    configuration_path: str | Path,
    sha256_path: str | Path,
) -> SeenItemRepeatedConfig:
    """Load a hash-verified, strictly validated protocol."""

    config_path = Path(configuration_path)
    hash_path = Path(sha256_path)

    if not config_path.is_file():
        raise SeenItemRepeatedConfigError(
            f"Configuration file not found: {config_path}"
        )

    if not hash_path.is_file():
        raise SeenItemRepeatedConfigError(
            f"SHA-256 file not found: {hash_path}"
        )

    raw_bytes = config_path.read_bytes()
    actual_digest = sha256(raw_bytes).hexdigest()

    hash_text = hash_path.read_text(
        encoding="utf-8"
    ).strip()

    expected_digest = hash_text.split()[0]

    if (
        len(expected_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_digest
        )
    ):
        raise SeenItemRepeatedConfigError(
            "SHA-256 file contains an invalid digest."
        )

    if actual_digest != expected_digest:
        raise SeenItemRepeatedConfigError(
            "Configuration failed SHA-256 verification."
        )

    try:
        raw = yaml.safe_load(
            raw_bytes.decode("utf-8")
        )
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SeenItemRepeatedConfigError(
            "Configuration is not valid UTF-8 YAML."
        ) from exc

    if not isinstance(raw, Mapping):
        raise SeenItemRepeatedConfigError(
            "Configuration root must be a mapping."
        )

    sections = set(raw)
    missing = sorted(
        EXPECTED_TOP_LEVEL_SECTIONS - sections
    )
    unexpected = sorted(
        sections - EXPECTED_TOP_LEVEL_SECTIONS
    )

    if missing:
        raise SeenItemRepeatedConfigError(
            f"Missing protocol sections: {missing}"
        )

    if unexpected:
        raise SeenItemRepeatedConfigError(
            f"Unexpected protocol sections: {unexpected}"
        )

    protocol = _mapping(raw, "protocol")
    scope = _mapping(raw, "scope")
    primary_runs = _mapping(raw, "primary_runs")
    dataset = _mapping(raw, "dataset")
    comparison = _mapping(
        raw,
        "primary_comparison",
    )
    statistics = _mapping(raw, "statistics")
    reproducibility = _mapping(
        raw,
        "reproducibility",
    )

    if (
        protocol.get("status")
        != "prespecified_preimplementation"
    ):
        raise SeenItemRepeatedConfigError(
            "Protocol status must remain prespecified."
        )

    if scope.get("modifies_v0_1_results") is not False:
        raise SeenItemRepeatedConfigError(
            "v0.1 results must remain unchanged."
        )

    if (
        scope.get("modifies_existing_track_a_result")
        is not False
    ):
        raise SeenItemRepeatedConfigError(
            "The existing Track A result must remain unchanged."
        )

    include_pilot = primary_runs.get(
        "include_pilot_reference"
    )

    if include_pilot is not False:
        raise SeenItemRepeatedConfigError(
            "The pilot reference cannot enter primary runs."
        )

    if primary_runs.get("independent_run_count") != 10:
        raise SeenItemRepeatedConfigError(
            "Exactly ten independent runs are required."
        )

    raw_runs = primary_runs.get("runs")

    if (
        not isinstance(raw_runs, list)
        or len(raw_runs) != 10
    ):
        raise SeenItemRepeatedConfigError(
            "Exactly ten run definitions are required."
        )

    runs = tuple(
        _parse_run(run)
        for run in raw_runs
    )

    if runs != EXPECTED_REPEATED_RUNS:
        raise SeenItemRepeatedConfigError(
            "Runs differ from the locked seed schedule."
        )

    if len({run.run_id for run in runs}) != 10:
        raise SeenItemRepeatedConfigError(
            "Run identifiers must be unique."
        )

    for attribute in (
        "generator_seed",
        "ood_seed",
        "partition_seed",
    ):
        values = {
            getattr(run, attribute)
            for run in runs
        }

        if len(values) != 10:
            raise SeenItemRepeatedConfigError(
                f"{attribute} values must be unique."
            )

    if (
        comparison.get("independent_replication_unit")
        != "seed"
    ):
        raise SeenItemRepeatedConfigError(
            "The independent replication unit must be seed."
        )

    if (
        comparison.get("pairing")
        != "within identical final-test events"
    ):
        raise SeenItemRepeatedConfigError(
            "The within-event pairing rule must remain locked."
        )

    if reproducibility.get("deterministic") is not True:
        raise SeenItemRepeatedConfigError(
            "Repeated evaluation must be deterministic."
        )

    if (
        reproducibility.get("oracle_use_prohibited")
        is not True
    ):
        raise SeenItemRepeatedConfigError(
            "Oracle use must remain prohibited."
        )

    return SeenItemRepeatedConfig(
        version=protocol.get("version"),
        event_count=dataset.get("events_per_run"),
        include_pilot_reference=include_pilot,
        runs=runs,
        confidence_level=statistics.get(
            "confidence_level"
        ),
        bootstrap_seed=statistics.get(
            "bootstrap_seed"
        ),
        bootstrap_resamples=statistics.get(
            "bootstrap_resamples"
        ),
    )


def _mapping(
    parent: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = parent.get(key)

    if not isinstance(value, Mapping):
        raise SeenItemRepeatedConfigError(
            f"{key} must be a mapping."
        )

    return value


def _parse_run(
    raw: Any,
) -> RepeatedSeedSpec:
    if not isinstance(raw, Mapping):
        raise SeenItemRepeatedConfigError(
            "Every run definition must be a mapping."
        )

    try:
        return RepeatedSeedSpec(
            run_id=raw["run_id"],
            generator_seed=raw["generator_seed"],
            ood_seed=raw["ood_seed"],
            partition_seed=raw["partition_seed"],
        )
    except KeyError as exc:
        raise SeenItemRepeatedConfigError(
            f"Missing run field: {exc.args[0]}"
        ) from exc
