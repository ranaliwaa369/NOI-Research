"""Tests for post-hoc NOI v0.3 sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.analyze_noi_v0_3_posthoc_sensitivity import (
    ALL_CONDITIONS,
    PosthocSensitivityError,
    _distribution,
    _event_differences,
    _support_diagnostic,
    export_posthoc_sensitivity,
    hierarchical_bootstrap,
)
from src.evaluation.noi_v0_3_analysis import (
    PairedObservation,
)


def test_event_differences_average_condition_views() -> None:
    pairs = (
        PairedObservation(
            latent_event_id="1301:event-1",
            baseline_value=0.0,
            proposed_value=1.0,
        ),
        PairedObservation(
            latent_event_id="1301:event-1",
            baseline_value=0.0,
            proposed_value=3.0,
        ),
        PairedObservation(
            latent_event_id="1301:event-2",
            baseline_value=1.0,
            proposed_value=0.0,
        ),
    )

    observed = _event_differences(pairs)

    assert np.array_equal(
        observed,
        np.asarray(
            [2.0, -1.0],
            dtype=np.float64,
        ),
    )


def test_distribution_is_deterministic() -> None:
    observed = _distribution(
        (1.0, 2.0, 3.0, 4.0)
    )

    assert observed["count"] == 4
    assert observed["minimum"] == pytest.approx(1.0)
    assert observed["median"] == pytest.approx(2.5)
    assert observed["maximum"] == pytest.approx(4.0)
    assert observed["mean"] == pytest.approx(2.5)


def test_hierarchical_bootstrap_resamples_two_levels() -> None:
    arrays = (
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, 1.0]),
    )

    first = hierarchical_bootstrap(
        arrays,
        bootstrap_seed=41,
        bootstrap_resamples=100,
        confidence_level=0.95,
    )
    second = hierarchical_bootstrap(
        arrays,
        bootstrap_seed=41,
        bootstrap_resamples=100,
        confidence_level=0.95,
    )

    assert first == second
    assert first["observed_mean_difference"] == (
        pytest.approx(1.0)
    )
    assert first["confidence_interval_lower"] == (
        pytest.approx(1.0)
    )
    assert first["confidence_interval_upper"] == (
        pytest.approx(1.0)
    )
    assert first["resampling_hierarchy"] == [
        "seed",
        "latent_event_id",
    ]
    assert first["classification"] == (
        "post_hoc_exploratory"
    )


def test_hierarchical_bootstrap_rejects_empty_input() -> None:
    with pytest.raises(
        PosthocSensitivityError,
        match="must not be empty",
    ):
        hierarchical_bootstrap(
            (),
            bootstrap_seed=41,
            bootstrap_resamples=100,
            confidence_level=0.95,
        )


def test_support_diagnostic_reports_score_gap() -> None:
    records = []

    regime_values = {
        "seen_item": (1.0, True),
        "known_family_unseen_item": (-2.0, True),
        "unseen_family": (-10.0, False),
    }

    for condition in ALL_CONDITIONS:
        for regime, (score, predicted) in (
            regime_values.items()
        ):
            records.append(
                {
                    "system": "support_gate_odor_only",
                    "condition": condition,
                    "support_regime": regime,
                    "support_score": score,
                    "predicted_supported": predicted,
                    (
                        "support_gate_predicted_supported"
                    ): predicted,
                }
            )

    diagnostic, pooled = _support_diagnostic(
        records,
        threshold=-5.0,
    )

    clean = diagnostic["clean"]

    assert clean["score_ranges_disjoint"] is True
    assert clean[
        "seen_minimum_minus_unseen_maximum"
    ] == pytest.approx(11.0)
    assert clean[
        "threshold_minus_unseen_maximum"
    ] == pytest.approx(5.0)
    assert clean[
        "seen_minimum_minus_threshold"
    ] == pytest.approx(6.0)
    assert clean["regimes"]["unseen_family"][
        "predicted_supported_rate"
    ] == pytest.approx(0.0)
    assert pooled["clean"]["seen_item"] == [1.0]


def test_export_writes_hash_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "posthoc.json"
    analysis = {
        "schema_version": (
            "noi-v0.3-posthoc-sensitivity-v1"
        ),
        "analysis_classification": (
            "post_hoc_exploratory"
        ),
    }

    export_posthoc_sensitivity(
        analysis,
        output,
        overwrite=False,
    )

    observed = json.loads(
        output.read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    recorded = Path(f"{output}.sha256").read_text(
        encoding="utf-8"
    ).split()[0]

    assert observed == analysis
    assert recorded == digest

    with pytest.raises(
        PosthocSensitivityError,
        match="already exists",
    ):
        export_posthoc_sensitivity(
            analysis,
            output,
            overwrite=False,
        )
