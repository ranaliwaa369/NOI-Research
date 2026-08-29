"""Paired bootstrap and Holm correction for NOI v0.3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


class NOIV03AnalysisError(ValueError):
    """Raised when a v0.3 statistical-analysis contract is violated."""


@dataclass(frozen=True, slots=True)
class PairedObservation:
    """One baseline/proposed comparison keyed by latent event."""

    latent_event_id: str
    baseline_value: float
    proposed_value: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.latent_event_id, str)
            or not self.latent_event_id.strip()
        ):
            raise NOIV03AnalysisError(
                "latent_event_id must be a nonempty string."
            )

        for name, value in (
            ("baseline_value", self.baseline_value),
            ("proposed_value", self.proposed_value),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise NOIV03AnalysisError(
                    f"{name} must be a finite numeric value."
                )

    @property
    def difference(self) -> float:
        """Return proposed minus baseline for this observation."""

        return float(self.proposed_value) - float(
            self.baseline_value,
        )


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Immutable latent-event-level paired bootstrap result."""

    event_count: int
    observation_count: int
    mean_difference: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    two_sided_p_value: float
    confidence_level: float
    bootstrap_seed: int
    bootstrap_resamples: int
    resampling_unit: str


@dataclass(frozen=True, slots=True)
class HypothesisTest:
    """One named raw p-value submitted to Holm correction."""

    name: str
    p_value: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise NOIV03AnalysisError(
                "Hypothesis-test name must be a nonempty string."
            )

        if (
            isinstance(self.p_value, bool)
            or not isinstance(self.p_value, Real)
            or not math.isfinite(float(self.p_value))
            or not 0.0 <= float(self.p_value) <= 1.0
        ):
            raise NOIV03AnalysisError(
                "p_value must be finite and between 0 and 1."
            )


@dataclass(frozen=True, slots=True)
class HolmComparison:
    """One raw and multiplicity-adjusted secondary comparison."""

    name: str
    raw_p_value: float
    adjusted_p_value: float
    rejected: bool
    sorted_rank: int


@dataclass(frozen=True, slots=True)
class HolmCorrectionResult:
    """Holm-corrected comparisons in original reporting order."""

    alpha: float
    comparison_count: int
    comparisons: tuple[HolmComparison, ...]


def _validate_bootstrap_controls(
    *,
    bootstrap_seed: object,
    bootstrap_resamples: object,
    confidence_level: object,
) -> tuple[int, int, float]:
    """Validate deterministic bootstrap controls."""

    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise NOIV03AnalysisError(
            "bootstrap_seed must be a nonnegative integer."
        )

    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples <= 0
    ):
        raise NOIV03AnalysisError(
            "bootstrap_resamples must be a positive integer."
        )

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, Real)
        or not math.isfinite(float(confidence_level))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise NOIV03AnalysisError(
            "confidence_level must be finite and between 0 and 1."
        )

    return (
        bootstrap_seed,
        bootstrap_resamples,
        float(confidence_level),
    )


def _event_level_differences(
    observations: tuple[PairedObservation, ...],
) -> tuple[float, ...]:
    """Average condition rows within each latent event before resampling."""

    grouped: dict[str, list[float]] = {}
    order: list[str] = []

    for observation in observations:
        if not isinstance(observation, PairedObservation):
            raise NOIV03AnalysisError(
                "observations must contain PairedObservation records."
            )

        if observation.latent_event_id not in grouped:
            grouped[observation.latent_event_id] = []
            order.append(observation.latent_event_id)

        grouped[observation.latent_event_id].append(
            observation.difference,
        )

    return tuple(
        sum(grouped[event_id]) / len(grouped[event_id])
        for event_id in order
    )


def paired_bootstrap(
    observations: tuple[PairedObservation, ...],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int = 10_000,
    confidence_level: float = 0.95,
) -> PairedBootstrapResult:
    """Bootstrap paired differences using latent event as the sampling unit."""

    if not isinstance(observations, tuple) or not observations:
        raise NOIV03AnalysisError(
            "observations must be a nonempty tuple."
        )

    seed, resamples, confidence = _validate_bootstrap_controls(
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
    )

    event_differences = _event_level_differences(
        observations,
    )
    event_count = len(event_differences)
    difference_array = np.asarray(
        event_differences,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(difference_array)):
        raise NOIV03AnalysisError(
            "Event-level paired differences must be finite."
        )

    observed_mean = float(
        difference_array.mean(),
    )
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(
        resamples,
        dtype=np.float64,
    )

    for index in range(resamples):
        sampled_indices = generator.integers(
            low=0,
            high=event_count,
            size=event_count,
        )
        bootstrap_means[index] = float(
            difference_array[sampled_indices].mean(),
        )

    tail_probability = (
        1.0 - confidence
    ) / 2.0
    lower = float(
        np.quantile(
            bootstrap_means,
            tail_probability,
            method="linear",
        ),
    )
    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - tail_probability,
            method="linear",
        ),
    )

    nonpositive = int(
        np.count_nonzero(
            bootstrap_means <= 0.0,
        ),
    )
    nonnegative = int(
        np.count_nonzero(
            bootstrap_means >= 0.0,
        ),
    )
    two_sided_p_value = min(
        1.0,
        2.0
        * min(
            (nonpositive + 1) / (resamples + 1),
            (nonnegative + 1) / (resamples + 1),
        ),
    )

    values = (
        observed_mean,
        lower,
        upper,
        two_sided_p_value,
    )

    if not all(math.isfinite(value) for value in values):
        raise NOIV03AnalysisError(
            "Bootstrap output must contain only finite values."
        )

    return PairedBootstrapResult(
        event_count=event_count,
        observation_count=len(observations),
        mean_difference=observed_mean,
        confidence_interval_lower=lower,
        confidence_interval_upper=upper,
        two_sided_p_value=two_sided_p_value,
        confidence_level=confidence,
        bootstrap_seed=seed,
        bootstrap_resamples=resamples,
        resampling_unit="latent_event_id",
    )


def holm_correction(
    tests: tuple[HypothesisTest, ...],
    *,
    alpha: float,
) -> HolmCorrectionResult:
    """Apply Holm's step-down family-wise error correction."""

    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, Real)
        or not math.isfinite(float(alpha))
        or not 0.0 < float(alpha) < 1.0
    ):
        raise NOIV03AnalysisError(
            "alpha must be finite and between 0 and 1."
        )

    if (
        not isinstance(tests, tuple)
        or not tests
        or not all(
            isinstance(test, HypothesisTest)
            for test in tests
        )
    ):
        raise NOIV03AnalysisError(
            "tests must be a nonempty tuple of HypothesisTest records."
        )

    names = tuple(test.name for test in tests)

    if len(names) != len(set(names)):
        raise NOIV03AnalysisError(
            "Hypothesis-test names must be unique."
        )

    comparison_count = len(tests)
    sorted_entries = tuple(
        sorted(
            enumerate(tests),
            key=lambda entry: (
                float(entry[1].p_value),
                entry[1].name,
            ),
        ),
    )

    adjusted_by_original_index: dict[int, float] = {}
    rejected_by_original_index: dict[int, bool] = {}
    rank_by_original_index: dict[int, int] = {}

    previous_adjusted = 0.0
    rejection_open = True

    for zero_rank, (original_index, test) in enumerate(
        sorted_entries,
    ):
        rank = zero_rank + 1
        multiplier = comparison_count - zero_rank
        adjusted = min(
            1.0,
            max(
                previous_adjusted,
                float(test.p_value) * multiplier,
            ),
        )
        previous_adjusted = adjusted

        local_threshold = float(alpha) / multiplier
        rejected = (
            rejection_open
            and float(test.p_value) <= local_threshold
        )

        if not rejected:
            rejection_open = False

        adjusted_by_original_index[original_index] = adjusted
        rejected_by_original_index[original_index] = rejected
        rank_by_original_index[original_index] = rank

    comparisons = tuple(
        HolmComparison(
            name=test.name,
            raw_p_value=float(test.p_value),
            adjusted_p_value=(
                adjusted_by_original_index[index]
            ),
            rejected=rejected_by_original_index[index],
            sorted_rank=rank_by_original_index[index],
        )
        for index, test in enumerate(tests)
    )

    return HolmCorrectionResult(
        alpha=float(alpha),
        comparison_count=comparison_count,
        comparisons=comparisons,
    )
