"""Tests for NOI v0.3 paired bootstrap and Holm correction."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest

from src.evaluation.noi_v0_3_analysis import (
    HolmComparison,
    HolmCorrectionResult,
    HypothesisTest,
    NOIV03AnalysisError,
    PairedBootstrapResult,
    PairedObservation,
    holm_correction,
    paired_bootstrap,
)


def observations() -> tuple[PairedObservation, ...]:
    """Return a small paired improvement sample."""

    return (
        PairedObservation(
            latent_event_id="event-001",
            baseline_value=0.10,
            proposed_value=0.40,
        ),
        PairedObservation(
            latent_event_id="event-002",
            baseline_value=0.20,
            proposed_value=0.50,
        ),
        PairedObservation(
            latent_event_id="event-003",
            baseline_value=0.30,
            proposed_value=0.60,
        ),
        PairedObservation(
            latent_event_id="event-004",
            baseline_value=0.40,
            proposed_value=0.70,
        ),
    )


def test_paired_observation_exposes_difference() -> None:
    """Each record preserves the within-event comparison."""

    observation = observations()[0]

    assert observation.difference == pytest.approx(0.30)


def test_bootstrap_uses_latent_event_as_resampling_unit() -> None:
    """Result provenance identifies the registered resampling unit."""

    result = paired_bootstrap(
        observations(),
        bootstrap_seed=1301,
        bootstrap_resamples=1000,
        confidence_level=0.95,
    )

    assert isinstance(result, PairedBootstrapResult)
    assert result.event_count == 4
    assert result.observation_count == 4
    assert result.resampling_unit == "latent_event_id"
    assert result.bootstrap_seed == 1301
    assert result.bootstrap_resamples == 1000
    assert result.confidence_level == 0.95


def test_mean_difference_is_paired_and_exact() -> None:
    """The effect is proposed minus baseline within each event."""

    result = paired_bootstrap(
        observations(),
        bootstrap_seed=1301,
        bootstrap_resamples=1000,
        confidence_level=0.95,
    )

    assert result.mean_difference == pytest.approx(0.30)
    assert result.confidence_interval_lower == pytest.approx(0.30)
    assert result.confidence_interval_upper == pytest.approx(0.30)
    assert 0.0 <= result.two_sided_p_value <= 1.0


def test_bootstrap_is_deterministic_for_one_seed() -> None:
    """Repeated analysis with one seed is byte-for-byte stable."""

    first = paired_bootstrap(
        observations(),
        bootstrap_seed=1301,
        bootstrap_resamples=1000,
        confidence_level=0.95,
    )
    second = paired_bootstrap(
        observations(),
        bootstrap_seed=1301,
        bootstrap_resamples=1000,
        confidence_level=0.95,
    )

    assert first == second


def test_multiple_rows_are_grouped_within_latent_event() -> None:
    """Condition rows are averaged before resampling event identifiers."""

    rows = (
        PairedObservation(
            latent_event_id="event-001",
            baseline_value=0.0,
            proposed_value=0.2,
        ),
        PairedObservation(
            latent_event_id="event-001",
            baseline_value=0.0,
            proposed_value=0.4,
        ),
        PairedObservation(
            latent_event_id="event-002",
            baseline_value=0.5,
            proposed_value=0.7,
        ),
    )

    result = paired_bootstrap(
        rows,
        bootstrap_seed=1301,
        bootstrap_resamples=1000,
        confidence_level=0.95,
    )

    assert result.event_count == 2
    assert result.observation_count == 3
    assert result.mean_difference == pytest.approx(0.25)


def test_ci_contains_observed_mean_for_variable_effects() -> None:
    """The percentile interval is ordered around the point estimate."""

    rows = (
        PairedObservation("a", 0.1, 0.0),
        PairedObservation("b", 0.1, 0.2),
        PairedObservation("c", 0.1, 0.4),
        PairedObservation("d", 0.1, 0.8),
    )

    result = paired_bootstrap(
        rows,
        bootstrap_seed=1301,
        bootstrap_resamples=2000,
        confidence_level=0.95,
    )

    assert (
        result.confidence_interval_lower
        <= result.mean_difference
        <= result.confidence_interval_upper
    )
    assert math.isfinite(result.mean_difference)
    assert math.isfinite(result.confidence_interval_lower)
    assert math.isfinite(result.confidence_interval_upper)


def test_default_analysis_uses_registered_10000_resamples() -> None:
    """The public default follows the preregistered bootstrap count."""

    result = paired_bootstrap(
        observations(),
        bootstrap_seed=1301,
    )

    assert result.bootstrap_resamples == 10_000
    assert result.confidence_level == 0.95


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("latent_event_id", ""),
        ("latent_event_id", " "),
        ("baseline_value", float("nan")),
        ("proposed_value", float("inf")),
        ("baseline_value", True),
    ),
)
def test_invalid_paired_observation_is_rejected(
    field: str,
    value: object,
) -> None:
    """Identifiers and paired values must be explicit and finite."""

    values: dict[str, object] = {
        "latent_event_id": "event-001",
        "baseline_value": 0.2,
        "proposed_value": 0.3,
    }
    values[field] = value

    with pytest.raises(NOIV03AnalysisError):
        PairedObservation(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("bootstrap_seed", -1),
        ("bootstrap_seed", True),
        ("bootstrap_resamples", 0),
        ("bootstrap_resamples", True),
        ("confidence_level", 0.0),
        ("confidence_level", 1.0),
        ("confidence_level", float("nan")),
    ),
)
def test_invalid_bootstrap_configuration_is_rejected(
    field: str,
    value: object,
) -> None:
    """Bootstrap controls obey strict reproducibility contracts."""

    arguments: dict[str, object] = {
        "bootstrap_seed": 1301,
        "bootstrap_resamples": 1000,
        "confidence_level": 0.95,
    }
    arguments[field] = value

    with pytest.raises(NOIV03AnalysisError):
        paired_bootstrap(
            observations(),
            **arguments,
        )


def test_empty_observations_are_rejected() -> None:
    """A paired effect cannot be estimated without events."""

    with pytest.raises(
        NOIV03AnalysisError,
        match="nonempty",
    ):
        paired_bootstrap(
            (),
            bootstrap_seed=1301,
        )


def test_holm_correction_matches_manual_example() -> None:
    """Adjusted p-values follow Holm's step-down procedure."""

    result = holm_correction(
        (
            HypothesisTest("H7-a", 0.01),
            HypothesisTest("H7-b", 0.04),
            HypothesisTest("H8-a", 0.03),
        ),
        alpha=0.05,
    )

    assert isinstance(result, HolmCorrectionResult)
    assert result.alpha == 0.05
    assert result.comparison_count == 3

    comparisons = {
        comparison.name: comparison
        for comparison in result.comparisons
    }

    assert comparisons["H7-a"].adjusted_p_value == pytest.approx(0.03)
    assert comparisons["H7-b"].adjusted_p_value == pytest.approx(0.06)
    assert comparisons["H8-a"].adjusted_p_value == pytest.approx(0.06)
    assert comparisons["H7-a"].rejected is True
    assert comparisons["H7-b"].rejected is False
    assert comparisons["H8-a"].rejected is False


def test_holm_output_preserves_input_order() -> None:
    """Reporting order remains stable even though correction sorts internally."""

    tests = (
        HypothesisTest("third", 0.03),
        HypothesisTest("first", 0.01),
        HypothesisTest("second", 0.02),
    )

    result = holm_correction(
        tests,
        alpha=0.05,
    )

    assert tuple(
        comparison.name
        for comparison in result.comparisons
    ) == ("third", "first", "second")
    assert all(
        isinstance(comparison, HolmComparison)
        for comparison in result.comparisons
    )


def test_holm_adjusted_values_are_bounded() -> None:
    """Multiplicity correction always returns probabilities."""

    result = holm_correction(
        (
            HypothesisTest("a", 0.8),
            HypothesisTest("b", 0.9),
            HypothesisTest("c", 1.0),
        ),
        alpha=0.05,
    )

    assert all(
        0.0 <= comparison.adjusted_p_value <= 1.0
        for comparison in result.comparisons
    )


def test_duplicate_hypothesis_names_are_rejected() -> None:
    """Every secondary comparison must have a unique audit label."""

    with pytest.raises(
        NOIV03AnalysisError,
        match="unique",
    ):
        holm_correction(
            (
                HypothesisTest("H7", 0.01),
                HypothesisTest("H7", 0.02),
            ),
            alpha=0.05,
        )


@pytest.mark.parametrize(
    "p_value",
    (
        -0.1,
        1.1,
        float("nan"),
        True,
    ),
)
def test_invalid_raw_p_value_is_rejected(
    p_value: object,
) -> None:
    """Raw p-values must be finite probabilities."""

    with pytest.raises(
        NOIV03AnalysisError,
        match="p_value",
    ):
        HypothesisTest(
            name="H7",
            p_value=p_value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "alpha",
    (
        0.0,
        1.0,
        -0.1,
        float("nan"),
        True,
    ),
)
def test_invalid_holm_alpha_is_rejected(
    alpha: object,
) -> None:
    """Family-wise alpha must be strictly between zero and one."""

    with pytest.raises(
        NOIV03AnalysisError,
        match="alpha",
    ):
        holm_correction(
            (HypothesisTest("H7", 0.01),),
            alpha=alpha,  # type: ignore[arg-type]
        )


def test_analysis_records_are_immutable() -> None:
    """Published statistics cannot be silently altered."""

    observation = observations()[0]

    with pytest.raises(AttributeError):
        observation.proposed_value = 1.0  # type: ignore[misc]

    corrected = holm_correction(
        (HypothesisTest("H7", 0.01),),
        alpha=0.05,
    )

    with pytest.raises(AttributeError):
        corrected.alpha = 0.10  # type: ignore[misc]


def test_replace_revalidates_hypothesis_test() -> None:
    """Dataclass replacement cannot bypass probability validation."""

    test = HypothesisTest("H7", 0.01)

    with pytest.raises(NOIV03AnalysisError):
        replace(
            test,
            p_value=2.0,
        )
