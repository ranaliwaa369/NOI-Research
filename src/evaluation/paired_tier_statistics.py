"""Paired bootstrap statistics across graded-OOD severity tiers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from src.baselines.retrieval_baselines import BaselineKind
from src.evaluation.graded_ood import OODTier
from src.evaluation.graded_ood_experiment import (
    GradedOODEvaluation,
    GradedOODExperiment,
)


LOCKED_BOOTSTRAP_SEED = 4242
LOCKED_BOOTSTRAP_RESAMPLES = 10_000
LOCKED_CONFIDENCE_LEVEL = 0.95

LOCKED_TIER_CONTRASTS = (
    (OODTier.MILD, OODTier.MODERATE),
    (OODTier.MODERATE, OODTier.SEVERE),
    (OODTier.MILD, OODTier.SEVERE),
)


class PairedTierStatisticsError(ValueError):
    """Raised when paired tier statistics cannot be computed fairly."""


@dataclass(frozen=True)
class PairedTierComparison:
    """Paired reciprocal-rank contrast between two OOD tiers."""

    baseline: BaselineKind
    lower_severity_tier: OODTier
    higher_severity_tier: OODTier
    latent_event_ids: tuple[str, ...]
    reciprocal_rank_differences: tuple[float, ...]
    mean_mrr_difference: float
    standard_deviation_difference: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    confidence_level: float
    bootstrap_seed: int
    bootstrap_resamples: int
    improved_count: int
    tied_count: int
    worsened_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.baseline, BaselineKind):
            raise PairedTierStatisticsError(
                "baseline must be a BaselineKind."
            )

        if not isinstance(self.lower_severity_tier, OODTier):
            raise PairedTierStatisticsError(
                "lower_severity_tier must be an OODTier."
            )

        if not isinstance(self.higher_severity_tier, OODTier):
            raise PairedTierStatisticsError(
                "higher_severity_tier must be an OODTier."
            )

        if (
            self.lower_severity_tier,
            self.higher_severity_tier,
        ) not in LOCKED_TIER_CONTRASTS:
            raise PairedTierStatisticsError(
                "The tier contrast was not preregistered."
            )

        count = len(self.latent_event_ids)

        if count < 2:
            raise PairedTierStatisticsError(
                "At least two paired latent events are required."
            )

        if len(set(self.latent_event_ids)) != count:
            raise PairedTierStatisticsError(
                "latent_event_ids must be unique."
            )

        if len(self.reciprocal_rank_differences) != count:
            raise PairedTierStatisticsError(
                "Every latent event must have one paired difference."
            )

        values = (
            self.mean_mrr_difference,
            self.standard_deviation_difference,
            self.bootstrap_ci_lower,
            self.bootstrap_ci_upper,
            self.confidence_level,
            *self.reciprocal_rank_differences,
        )

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            for value in values
        ):
            raise PairedTierStatisticsError(
                "All paired statistics must be finite numbers."
            )

        if self.standard_deviation_difference < 0.0:
            raise PairedTierStatisticsError(
                "standard_deviation_difference cannot be negative."
            )

        if self.bootstrap_ci_lower > self.bootstrap_ci_upper:
            raise PairedTierStatisticsError(
                "Bootstrap confidence bounds are reversed."
            )

        if self.confidence_level != LOCKED_CONFIDENCE_LEVEL:
            raise PairedTierStatisticsError(
                "confidence_level must remain locked at 0.95."
            )

        if self.bootstrap_seed != LOCKED_BOOTSTRAP_SEED:
            raise PairedTierStatisticsError(
                "bootstrap_seed must remain locked at 4242."
            )

        if self.bootstrap_resamples != LOCKED_BOOTSTRAP_RESAMPLES:
            raise PairedTierStatisticsError(
                "bootstrap_resamples must remain locked at 10000."
            )

        counts = (
            self.improved_count,
            self.tied_count,
            self.worsened_count,
        )

        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in counts
        ):
            raise PairedTierStatisticsError(
                "Outcome counts must be nonnegative integers."
            )

        if sum(counts) != count:
            raise PairedTierStatisticsError(
                "Outcome counts must equal the latent-event count."
            )

        tolerance = 1e-15

        observed_improved = sum(
            difference > tolerance
            for difference in self.reciprocal_rank_differences
        )
        observed_worsened = sum(
            difference < -tolerance
            for difference in self.reciprocal_rank_differences
        )
        observed_tied = (
            count
            - observed_improved
            - observed_worsened
        )

        if (
            self.improved_count != observed_improved
            or self.worsened_count != observed_worsened
            or self.tied_count != observed_tied
        ):
            raise PairedTierStatisticsError(
                "Outcome counts do not match paired differences."
            )

    @property
    def paired_event_count(self) -> int:
        """Return the number of independent paired latent units."""

        return len(self.latent_event_ids)

    @property
    def contrast_name(self) -> str:
        """Return an explicit lower-minus-higher contrast label."""

        return (
            f"{self.lower_severity_tier.value}"
            f"_minus_{self.higher_severity_tier.value}"
        )


@dataclass(frozen=True)
class PairedTierStatistics:
    """Complete paired statistics for all baselines and contrasts."""

    comparisons: tuple[PairedTierComparison, ...]
    paired_analysis_unit: str
    bootstrap_seed: int
    bootstrap_resamples: int
    confidence_level: float
    oracle_used: bool

    def __post_init__(self) -> None:
        expected_count = (
            len(BaselineKind)
            * len(LOCKED_TIER_CONTRASTS)
        )

        if len(self.comparisons) != expected_count:
            raise PairedTierStatisticsError(
                f"Expected exactly {expected_count} paired comparisons."
            )

        keys = {
            (
                comparison.baseline,
                comparison.lower_severity_tier,
                comparison.higher_severity_tier,
            )
            for comparison in self.comparisons
        }

        if len(keys) != expected_count:
            raise PairedTierStatisticsError(
                "Every baseline-contrast combination must be unique."
            )

        if self.paired_analysis_unit != "latent_event_id":
            raise PairedTierStatisticsError(
                "paired_analysis_unit must be latent_event_id."
            )

        if self.bootstrap_seed != LOCKED_BOOTSTRAP_SEED:
            raise PairedTierStatisticsError(
                "bootstrap_seed must remain locked at 4242."
            )

        if self.bootstrap_resamples != LOCKED_BOOTSTRAP_RESAMPLES:
            raise PairedTierStatisticsError(
                "bootstrap_resamples must remain locked at 10000."
            )

        if self.confidence_level != LOCKED_CONFIDENCE_LEVEL:
            raise PairedTierStatisticsError(
                "confidence_level must remain locked at 0.95."
            )

        if self.oracle_used is not False:
            raise PairedTierStatisticsError(
                "OOD oracle use is prohibited."
            )

    def get(
        self,
        *,
        baseline: BaselineKind,
        lower_severity_tier: OODTier,
        higher_severity_tier: OODTier,
    ) -> PairedTierComparison:
        """Return one registered baseline-tier contrast."""

        if not isinstance(baseline, BaselineKind):
            raise PairedTierStatisticsError(
                "baseline must be a BaselineKind."
            )

        requested = (
            lower_severity_tier,
            higher_severity_tier,
        )

        if requested not in LOCKED_TIER_CONTRASTS:
            raise PairedTierStatisticsError(
                "The requested tier contrast was not preregistered."
            )

        for comparison in self.comparisons:
            if (
                comparison.baseline is baseline
                and comparison.lower_severity_tier
                is lower_severity_tier
                and comparison.higher_severity_tier
                is higher_severity_tier
            ):
                return comparison

        raise PairedTierStatisticsError(
            "The requested paired comparison is missing."
        )


def compute_paired_tier_statistics(
    experiment: GradedOODExperiment,
    *,
    bootstrap_seed: int = LOCKED_BOOTSTRAP_SEED,
    bootstrap_resamples: int = LOCKED_BOOTSTRAP_RESAMPLES,
    confidence_level: float = LOCKED_CONFIDENCE_LEVEL,
) -> PairedTierStatistics:
    """Compute locked latent-event bootstrap contrasts for every baseline."""

    if not isinstance(experiment, GradedOODExperiment):
        raise PairedTierStatisticsError(
            "experiment must be a GradedOODExperiment."
        )

    if bootstrap_seed != LOCKED_BOOTSTRAP_SEED:
        raise PairedTierStatisticsError(
            "bootstrap_seed must remain locked at 4242."
        )

    if bootstrap_resamples != LOCKED_BOOTSTRAP_RESAMPLES:
        raise PairedTierStatisticsError(
            "bootstrap_resamples must remain locked at 10000."
        )

    if confidence_level != LOCKED_CONFIDENCE_LEVEL:
        raise PairedTierStatisticsError(
            "confidence_level must remain locked at 0.95."
        )

    comparisons: list[PairedTierComparison] = []

    for baseline in BaselineKind:
        for lower_tier, higher_tier in LOCKED_TIER_CONTRASTS:
            lower = experiment.get(
                tier=lower_tier,
                baseline=baseline,
            )
            higher = experiment.get(
                tier=higher_tier,
                baseline=baseline,
            )

            comparisons.append(
                _compare_evaluations(
                    lower,
                    higher,
                    bootstrap_seed=bootstrap_seed,
                    bootstrap_resamples=bootstrap_resamples,
                    confidence_level=confidence_level,
                )
            )

    return PairedTierStatistics(
        comparisons=tuple(comparisons),
        paired_analysis_unit="latent_event_id",
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        confidence_level=confidence_level,
        oracle_used=False,
    )


def _compare_evaluations(
    lower: GradedOODEvaluation,
    higher: GradedOODEvaluation,
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    confidence_level: float,
) -> PairedTierComparison:
    """Compare reciprocal ranks using aligned latent-event units."""

    if lower.baseline is not higher.baseline:
        raise PairedTierStatisticsError(
            "Paired evaluations must use the same baseline."
        )

    if lower.latent_event_ids != higher.latent_event_ids:
        raise PairedTierStatisticsError(
            "Paired evaluations must have identical latent-event order."
        )

    lower_rr = np.asarray(
        [
            _reciprocal_rank(ranking, relevant)
            for ranking, relevant in zip(
                lower.rankings,
                lower.relevant_items,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    higher_rr = np.asarray(
        [
            _reciprocal_rank(ranking, relevant)
            for ranking, relevant in zip(
                higher.rankings,
                higher.relevant_items,
                strict=True,
            )
        ],
        dtype=np.float64,
    )

    differences = lower_rr - higher_rr
    mean_difference = float(np.mean(differences))
    standard_deviation = float(
        np.std(differences, ddof=1)
    )

    generator = np.random.default_rng(
        bootstrap_seed
    )
    indices = generator.integers(
        0,
        differences.shape[0],
        size=(
            bootstrap_resamples,
            differences.shape[0],
        ),
    )
    bootstrap_means = differences[indices].mean(
        axis=1
    )

    tail_probability = (
        1.0 - confidence_level
    ) / 2.0

    lower_quantile = 100.0 * tail_probability
    upper_quantile = 100.0 * (
        1.0 - tail_probability
    )

    ci_lower, ci_upper = np.percentile(
        bootstrap_means,
        (
            lower_quantile,
            upper_quantile,
        ),
    )

    tolerance = 1e-15
    improved = int(
        np.sum(differences > tolerance)
    )
    worsened = int(
        np.sum(differences < -tolerance)
    )
    tied = int(
        differences.shape[0]
        - improved
        - worsened
    )

    return PairedTierComparison(
        baseline=lower.baseline,
        lower_severity_tier=lower.tier,
        higher_severity_tier=higher.tier,
        latent_event_ids=lower.latent_event_ids,
        reciprocal_rank_differences=tuple(
            float(value)
            for value in differences
        ),
        mean_mrr_difference=mean_difference,
        standard_deviation_difference=standard_deviation,
        bootstrap_ci_lower=float(ci_lower),
        bootstrap_ci_upper=float(ci_upper),
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
        improved_count=improved,
        tied_count=tied,
        worsened_count=worsened,
    )


def _reciprocal_rank(
    ranking: tuple[str, ...],
    relevant_items: frozenset[str],
) -> float:
    """Return reciprocal rank for one locked top-ten ranking."""

    for rank, item_id in enumerate(
        ranking,
        start=1,
    ):
        if item_id in relevant_items:
            return 1.0 / rank

    return 0.0
