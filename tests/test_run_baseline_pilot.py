"""Tests for the exported NOI baseline pilot workflow."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from experiments.run_baseline_pilot import (
    BaselinePilotError,
    DEFAULT_EVENT_COUNT,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RIDGE_ALPHA,
    DEFAULT_TOP_K,
    build_argument_parser,
    run_and_export_baseline_pilot,
)


SYNTHETIC_CONFIG_PATH = "configs/synthetic_data.yaml"
PROTOCOL_PATH = "configs/research_protocol.yaml"


def run_pilot(output_directory: Path, *, overwrite: bool = False):
    """Run the standard baseline pilot in a temporary directory."""

    return run_and_export_baseline_pilot(
        synthetic_config_path=SYNTHETIC_CONFIG_PATH,
        research_protocol_path=PROTOCOL_PATH,
        output_directory=output_directory,
        event_count=200,
        random_seed=2026,
        ridge_alpha=1.0,
        top_k=10,
        overwrite=overwrite,
    )


def test_workflow_exports_verified_results(
    tmp_path: Path,
) -> None:
    """The workflow must produce one hash-verified result file."""

    exported = run_pilot(tmp_path / "pilot")

    assert exported.results_path.is_file()
    assert (
        sha256(exported.results_path.read_bytes()).hexdigest()
        == exported.results_sha256
    )
    assert len(exported.experiment.summaries) == 8


def test_export_contains_expected_structure(
    tmp_path: Path,
) -> None:
    """The result file must preserve settings and scientific scope."""

    exported = run_pilot(tmp_path / "pilot")

    payload = json.loads(
        exported.results_path.read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "1.0.0"
    assert payload["owner"] == "GUARDIANX LLC"
    assert payload["dataset"]["event_count"] == 200
    assert payload["dataset"]["odor_target_count"] == 200
    assert payload["dataset"]["leakage_audit_passed"] is True
    assert payload["settings"]["top_k"] == 10
    assert len(payload["summaries"]) == 8
    assert len(payload["event_level_rankings"]) == 8
    assert len(payload["scope_limitations"]) >= 4


def test_export_is_deterministic(
    tmp_path: Path,
) -> None:
    """Identical settings must produce identical result bytes."""

    first = run_pilot(tmp_path / "first")
    second = run_pilot(tmp_path / "second")

    assert first.results_sha256 == second.results_sha256
    assert (
        first.results_path.read_bytes()
        == second.results_path.read_bytes()
    )


def test_existing_results_are_protected(
    tmp_path: Path,
) -> None:
    """Existing results must not be overwritten implicitly."""

    output_directory = tmp_path / "pilot"
    run_pilot(output_directory)

    with pytest.raises(
        BaselinePilotError,
        match="Refusing to overwrite",
    ):
        run_pilot(output_directory)


def test_explicit_overwrite_is_deterministic(
    tmp_path: Path,
) -> None:
    """Explicit replacement must reproduce the same result hash."""

    output_directory = tmp_path / "pilot"

    first = run_pilot(output_directory)
    second = run_pilot(
        output_directory,
        overwrite=True,
    )

    assert first.results_sha256 == second.results_sha256


def test_non_directory_output_is_rejected(
    tmp_path: Path,
) -> None:
    """An ordinary file cannot be used as an output directory."""

    invalid_output = tmp_path / "ordinary-file"
    invalid_output.write_text(
        "not a directory",
        encoding="utf-8",
    )

    with pytest.raises(
        BaselinePilotError,
        match="not a directory",
    ):
        run_pilot(invalid_output)


def test_cli_defaults_match_locked_pilot() -> None:
    """CLI defaults must preserve the documented pilot settings."""

    arguments = build_argument_parser().parse_args([])

    assert arguments.event_count == DEFAULT_EVENT_COUNT
    assert arguments.output_directory == DEFAULT_OUTPUT_DIRECTORY
    assert arguments.random_seed == DEFAULT_RANDOM_SEED
    assert arguments.ridge_alpha == DEFAULT_RIDGE_ALPHA
    assert arguments.top_k == DEFAULT_TOP_K
    assert arguments.overwrite is False


@pytest.mark.parametrize(
    "invalid_overwrite",
    (
        1,
        "yes",
        None,
    ),
)
def test_nonboolean_overwrite_is_rejected(
    tmp_path: Path,
    invalid_overwrite,
) -> None:
    """Overwrite authorization must be explicitly boolean."""

    with pytest.raises(
        BaselinePilotError,
        match="overwrite must be boolean",
    ):
        run_and_export_baseline_pilot(
            synthetic_config_path=SYNTHETIC_CONFIG_PATH,
            research_protocol_path=PROTOCOL_PATH,
            output_directory=tmp_path / "pilot",
            event_count=200,
            overwrite=invalid_overwrite,
        )