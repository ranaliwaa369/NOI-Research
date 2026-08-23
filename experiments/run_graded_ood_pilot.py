"""Reproducible export workflow for the graded-OOD pilot v0.2."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from src.evaluation.amendment_config import file_sha256
from src.evaluation.graded_ood_experiment import (
    run_graded_ood_experiment,
)
from src.evaluation.paired_ood_source_generator import (
    generate_paired_graded_ood_bundle,
)
from src.evaluation.paired_tier_statistics import (
    compute_paired_tier_statistics,
)


DEFAULT_OUTPUT_DIRECTORY = Path(
    "results/graded-ood-pilot-v0.2"
)

SYNTHETIC_CONFIGURATION_PATH = Path(
    "configs/synthetic_data.yaml"
)
PROTOCOL_PATH = Path(
    "configs/research_protocol.yaml"
)
AMENDMENT_PATH = Path(
    "configs/protocol_amendment_v0.2.yaml"
)
GENERATION_DEFINITION_PATH = Path(
    "configs/graded_ood_generation.yaml"
)


class GradedOODPilotWorkflowError(ValueError):
    """Raised when the graded-OOD pilot cannot be exported safely."""


def run_graded_ood_pilot(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run, export, hash, and verify the exploratory paired pilot."""

    if not isinstance(overwrite, bool):
        raise GradedOODPilotWorkflowError(
            "overwrite must be boolean."
        )

    output_path = Path(output_directory)

    if output_path.exists() and not output_path.is_dir():
        raise GradedOODPilotWorkflowError(
            "Output path exists and is not a directory."
        )

    expected_files = (
        output_path / "graded_ood_results.json",
        output_path / "paired_statistics.json",
        output_path / "run_manifest.json",
    )

    existing = [
        path
        for path in expected_files
        if path.exists()
    ]

    if existing and not overwrite:
        raise GradedOODPilotWorkflowError(
            "Output files already exist; use overwrite=True "
            "for an intentional deterministic rerun."
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = generate_paired_graded_ood_bundle(
        SYNTHETIC_CONFIGURATION_PATH,
        PROTOCOL_PATH,
        AMENDMENT_PATH,
        GENERATION_DEFINITION_PATH,
        event_count=200,
    )

    experiment = run_graded_ood_experiment(
        bundle
    )
    statistics = compute_paired_tier_statistics(
        experiment
    )

    results_document = {
        "schema_version": "1.0.0",
        "project": "Neuro-Olfactive Interface",
        "owner": "GUARDIANX LLC",
        "dataset_type": "paired_graded_ood_synthetic_pilot",
        "scope": "exploratory_synthetic_implementation_evaluation",
        "latent_event_count": experiment.latent_event_count,
        "odor_library_size": experiment.odor_library_size,
        "top_k": experiment.top_k,
        "random_seed": experiment.random_seed,
        "ridge_alpha": experiment.ridge_alpha,
        "oracle_used": experiment.oracle_used,
        "paired_analysis_unit": experiment.paired_analysis_unit,
        "evaluations": [
            {
                "tier": evaluation.tier.value,
                "baseline": evaluation.baseline.value,
                "event_count": evaluation.event_count,
                "training_event_count": (
                    evaluation.training_event_count
                ),
                "recall_at_1": evaluation.recall_at_1,
                "recall_at_10": evaluation.recall_at_10,
                "mean_reciprocal_rank": (
                    evaluation.mean_reciprocal_rank
                ),
                "ndcg_at_10": evaluation.ndcg_at_10,
                "latent_event_ids": list(
                    evaluation.latent_event_ids
                ),
                "rankings": [
                    list(ranking)
                    for ranking in evaluation.rankings
                ],
                "relevant_items": [
                    sorted(relevant)
                    for relevant in evaluation.relevant_items
                ],
            }
            for evaluation in experiment.evaluations
        ],
    }

    statistics_document = {
        "schema_version": "1.0.0",
        "project": "Neuro-Olfactive Interface",
        "owner": "GUARDIANX LLC",
        "paired_analysis_unit": statistics.paired_analysis_unit,
        "bootstrap_seed": statistics.bootstrap_seed,
        "bootstrap_resamples": statistics.bootstrap_resamples,
        "confidence_level": statistics.confidence_level,
        "oracle_used": statistics.oracle_used,
        "comparisons": [
            {
                "baseline": comparison.baseline.value,
                "contrast": comparison.contrast_name,
                "lower_severity_tier": (
                    comparison.lower_severity_tier.value
                ),
                "higher_severity_tier": (
                    comparison.higher_severity_tier.value
                ),
                "paired_event_count": (
                    comparison.paired_event_count
                ),
                "mean_mrr_difference": (
                    comparison.mean_mrr_difference
                ),
                "standard_deviation_difference": (
                    comparison.standard_deviation_difference
                ),
                "bootstrap_ci_lower": (
                    comparison.bootstrap_ci_lower
                ),
                "bootstrap_ci_upper": (
                    comparison.bootstrap_ci_upper
                ),
                "improved_count": comparison.improved_count,
                "tied_count": comparison.tied_count,
                "worsened_count": comparison.worsened_count,
                "latent_event_ids": list(
                    comparison.latent_event_ids
                ),
                "reciprocal_rank_differences": list(
                    comparison.reciprocal_rank_differences
                ),
            }
            for comparison in statistics.comparisons
        ],
    }

    results_path = expected_files[0]
    statistics_path = expected_files[1]
    manifest_path = expected_files[2]

    _write_json(
        results_path,
        results_document,
    )
    _write_json(
        statistics_path,
        statistics_document,
    )

    results_hash = file_sha256(
        results_path
    )
    statistics_hash = file_sha256(
        statistics_path
    )

    manifest = {
        "schema_version": "1.0.0",
        "project": "Neuro-Olfactive Interface",
        "owner": "GUARDIANX LLC",
        "release": "graded-ood-pilot-v0.2",
        "status": "exploratory",
        "dataset_type": "synthetic_implementation_evaluation",
        "counts": {
            "original_events": len(
                bundle.original_dataset.events
            ),
            "odor_targets": len(bundle.odor_targets),
            "latent_ood_events": bundle.source_count,
            "observed_graded_ood_rows": (
                bundle.graded_dataset.observed_event_count
            ),
            "tier_rows": bundle.graded_dataset.tier_counts,
            "baseline_tier_evaluations": len(
                experiment.evaluations
            ),
            "paired_statistical_comparisons": len(
                statistics.comparisons
            ),
        },
        "verification": {
            "severe_reference_replay": "PASSED",
            "paired_ground_truth": "PASSED",
            "oracle_used": False,
            "analysis_unit": "latent_event_id",
            "tests_required_before_release": True,
        },
        "configuration_hashes": {
            "research_protocol.yaml": file_sha256(
                PROTOCOL_PATH
            ),
            "protocol_amendment_v0.2.yaml": file_sha256(
                AMENDMENT_PATH
            ),
            "graded_ood_generation.yaml": file_sha256(
                GENERATION_DEFINITION_PATH
            ),
            "synthetic_data.yaml": file_sha256(
                SYNTHETIC_CONFIGURATION_PATH
            ),
        },
        "output_files": {
            "graded_ood_results.json": {
                "sha256": results_hash,
                "evaluations": len(
                    experiment.evaluations
                ),
            },
            "paired_statistics.json": {
                "sha256": statistics_hash,
                "comparisons": len(
                    statistics.comparisons
                ),
            },
        },
        "randomness": {
            "generator_seed": bundle.generator_seed,
            "ood_seed": bundle.ood_seed,
            "baseline_random_seed": experiment.random_seed,
            "bootstrap_seed": statistics.bootstrap_seed,
            "bootstrap_resamples": (
                statistics.bootstrap_resamples
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
            "pyyaml": version("PyYAML"),
        },
        "scope_limitations": [
            "Exploratory synthetic implementation evaluation only.",
            "Not a final confirmatory result.",
            "No human perceptual validity claim.",
            "No clinical or diagnostic claim.",
            "No physical odor-emission safety claim.",
            "No deployment-readiness claim.",
        ],
    }

    _write_json(
        manifest_path,
        manifest,
    )

    verification = verify_graded_ood_pilot_export(
        output_path
    )

    if verification["passed"] is not True:
        raise GradedOODPilotWorkflowError(
            "Export verification failed."
        )

    return {
        "output_directory": str(output_path),
        "results_path": str(results_path),
        "statistics_path": str(statistics_path),
        "manifest_path": str(manifest_path),
        "results_sha256": results_hash,
        "statistics_sha256": statistics_hash,
        "manifest_sha256": file_sha256(
            manifest_path
        ),
        "verification_passed": True,
        "severe_replay_passed": True,
        "latent_event_count": bundle.source_count,
        "observed_row_count": (
            bundle.graded_dataset.observed_event_count
        ),
    }


def verify_graded_ood_pilot_export(
    output_directory: str | Path,
) -> dict[str, Any]:
    """Verify exported file hashes and locked manifest safeguards."""

    output_path = Path(output_directory)
    manifest_path = output_path / "run_manifest.json"

    if not manifest_path.is_file():
        raise GradedOODPilotWorkflowError(
            "run_manifest.json is missing."
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise GradedOODPilotWorkflowError(
            "run_manifest.json is not valid JSON."
        ) from error

    failures: list[str] = []

    for filename, metadata in manifest.get(
        "output_files",
        {},
    ).items():
        path = output_path / filename

        if not path.is_file():
            failures.append(
                f"Missing output file: {filename}"
            )
            continue

        actual_hash = file_sha256(path)

        if actual_hash != metadata.get("sha256"):
            failures.append(
                f"SHA-256 mismatch: {filename}"
            )

    verification = manifest.get(
        "verification",
        {},
    )

    if verification.get(
        "severe_reference_replay"
    ) != "PASSED":
        failures.append(
            "Severe replay was not recorded as passed."
        )

    if verification.get("oracle_used") is not False:
        failures.append(
            "Oracle use must remain false."
        )

    if verification.get(
        "analysis_unit"
    ) != "latent_event_id":
        failures.append(
            "Analysis unit must be latent_event_id."
        )

    return {
        "passed": not failures,
        "failures": failures,
    }


def _write_json(
    path: Path,
    document: dict[str, Any],
) -> None:
    """Write deterministic UTF-8 JSON with a final newline."""

    serialized = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"

    path.write_text(
        serialized,
        encoding="utf-8",
    )


def main() -> None:
    """Run the command-line pilot workflow."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the exploratory paired graded-OOD pilot."
        )
    )
    parser.add_argument(
        "--output-directory",
        default=str(DEFAULT_OUTPUT_DIRECTORY),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    arguments = parser.parse_args()

    result = run_graded_ood_pilot(
        arguments.output_directory,
        overwrite=arguments.overwrite,
    )

    print("NOI graded-OOD pilot completed successfully")
    print(
        "Output directory:",
        result["output_directory"],
    )
    print(
        "Results SHA-256:",
        result["results_sha256"],
    )
    print(
        "Statistics SHA-256:",
        result["statistics_sha256"],
    )
    print(
        "Manifest SHA-256:",
        result["manifest_sha256"],
    )
    print(
        "Severe replay verification: PASSED"
    )
    print(
        "Export verification: PASSED"
    )
    print(
        "Scope: exploratory synthetic pilot only; "
        "not confirmatory or perceptual evidence."
    )


if __name__ == "__main__":
    main()
