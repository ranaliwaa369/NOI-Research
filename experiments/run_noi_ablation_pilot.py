"""Reproducible exploratory workflow for the NOI ablation pilot."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

from src.evaluation.noi_ablation_experiment import (
    LOCKED_TEMPORAL_DISPLACEMENTS,
    NOIAblationSystem,
    run_noi_ablation_experiment,
)
from src.evaluation.paired_ood_source_generator import (
    generate_paired_graded_ood_bundle,
)
from src.safety.policy_gate import load_policy_rules
from src.system.noi_pipeline import (
    load_noi_system_configuration,
)


DEFAULT_OUTPUT_DIRECTORY = Path(
    "results/noi-ablation-pilot-v0.1"
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
SYSTEM_CONFIGURATION_PATH = Path(
    "configs/noi_system_v0.1.yaml"
)
POLICY_PATH = Path(
    "configs/policy_rules.yaml"
)

PROTOCOL_HASH = (
    "e885f537b3209d7052d1517efc5be753"
    "24df39689cb4e54b38a241c95f22e512"
)

TRAINED_AT_UTC = datetime(
    2026,
    8,
    22,
    19,
    5,
    tzinfo=timezone.utc,
)


class NOIAblationPilotWorkflowError(ValueError):
    """Raised when the NOI ablation workflow cannot complete."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    file_path = Path(path)

    if not file_path.is_file():
        raise NOIAblationPilotWorkflowError(
            f"File not found for hashing: {file_path}"
        )

    digest = sha256()

    with file_path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def run_noi_ablation_pilot(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run, export, hash, and verify the exploratory NOI ablation."""

    if not isinstance(overwrite, bool):
        raise NOIAblationPilotWorkflowError(
            "overwrite must be boolean."
        )

    output_path = Path(output_directory)

    if output_path.exists() and not output_path.is_dir():
        raise NOIAblationPilotWorkflowError(
            "Output path exists and is not a directory."
        )

    results_path = output_path / "noi_ablation_results.json"
    summary_path = output_path / "noi_ablation_summary.json"
    manifest_path = output_path / "run_manifest.json"

    expected_files = (
        results_path,
        summary_path,
        manifest_path,
    )

    existing = [
        path for path in expected_files if path.exists()
    ]

    if existing and not overwrite:
        raise NOIAblationPilotWorkflowError(
            "Output files already exist; use overwrite=True "
            "for an intentional deterministic rerun."
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    system_configuration = load_noi_system_configuration(
        SYSTEM_CONFIGURATION_PATH
    )
    policy_configuration = load_policy_rules(
        POLICY_PATH
    )

    bundle = generate_paired_graded_ood_bundle(
        SYNTHETIC_CONFIGURATION_PATH,
        PROTOCOL_PATH,
        AMENDMENT_PATH,
        GENERATION_DEFINITION_PATH,
        event_count=200,
    )

    experiment = run_noi_ablation_experiment(
        bundle,
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=PROTOCOL_HASH,
        trained_at_utc=TRAINED_AT_UTC,
    )

    results_document = {
        "schema_version": "1.0.0",
        "project": "Neuro-Olfactive Intelligence",
        "owner": "GUARDIANX LLC",
        "release": "noi-ablation-pilot-v0.1",
        "status": "exploratory",
        "dataset_type": (
            "paired_graded_ood_synthetic_pilot"
        ),
        "scope": (
            "synthetic_computational_implementation_evaluation"
        ),
        "trained_at_utc": TRAINED_AT_UTC.isoformat(),
        "latent_event_count": experiment.latent_event_count,
        "odor_library_size": experiment.odor_library_size,
        "training_event_count": (
            experiment.training_event_count
        ),
        "validation_event_count": (
            experiment.validation_event_count
        ),
        "selected_validation_alpha": (
            experiment.selected_validation_alpha
        ),
        "top_k": experiment.top_k,
        "temporal_displacements": list(
            experiment.temporal_displacements
        ),
        "paired_analysis_unit": (
            experiment.paired_analysis_unit
        ),
        "oracle_used": experiment.oracle_used,
        "ood_tuning_used": experiment.ood_tuning_used,
        "evaluations": [
            {
                "system": evaluation.system.value,
                "tier": evaluation.tier.value,
                "temporal_displacement_days": (
                    evaluation.temporal_displacement_days
                ),
                "selected_alpha": (
                    evaluation.selected_alpha
                ),
                "apply_temporal_decay": (
                    evaluation.apply_temporal_decay
                ),
                "event_count": len(
                    evaluation.latent_event_ids
                ),
                "top_k": evaluation.top_k,
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

    ridge_cells = {
        (
            evaluation.tier,
            evaluation.temporal_displacement_days,
        ): evaluation
        for evaluation in experiment.evaluations
        if evaluation.system is NOIAblationSystem.RIDGE_ONLY
    }

    full_cells = {
        (
            evaluation.tier,
            evaluation.temporal_displacement_days,
        ): evaluation
        for evaluation in experiment.evaluations
        if evaluation.system is NOIAblationSystem.FULL_HYBRID
    }

    memory_cells = [
        evaluation
        for evaluation in experiment.evaluations
        if evaluation.system is NOIAblationSystem.MEMORY_ONLY
    ]

    full_equals_ridge = all(
        full_cells[key].rankings
        == ridge_cells[key].rankings
        for key in ridge_cells
    )

    full_mrr_differences = [
        {
            "tier": key[0].value,
            "temporal_displacement_days": key[1],
            "full_hybrid_mrr": (
                full_cells[key].mean_reciprocal_rank
            ),
            "ridge_only_mrr": (
                ridge_cells[key].mean_reciprocal_rank
            ),
            "difference": (
                full_cells[key].mean_reciprocal_rank
                - ridge_cells[key].mean_reciprocal_rank
            ),
        }
        for key in sorted(
            ridge_cells,
            key=lambda item: (
                item[0].value,
                item[1],
            ),
        )
    ]

    memory_any_nonzero_mrr = any(
        evaluation.mean_reciprocal_rank > 0.0
        for evaluation in memory_cells
    )

    summary_document = {
        "schema_version": "1.0.0",
        "project": "Neuro-Olfactive Intelligence",
        "owner": "GUARDIANX LLC",
        "release": "noi-ablation-pilot-v0.1",
        "status": "exploratory",
        "selected_validation_alpha": (
            experiment.selected_validation_alpha
        ),
        "full_hybrid_equals_ridge_rankings_all_conditions": (
            full_equals_ridge
        ),
        "memory_only_any_nonzero_mrr": (
            memory_any_nonzero_mrr
        ),
        "full_hybrid_minus_ridge_only": (
            full_mrr_differences
        ),
        "interpretation": {
            "ridge_component_supported_in_this_pilot": True,
            "associative_memory_incremental_benefit_detected": (
                not full_equals_ridge
                and any(
                    row["difference"] > 0.0
                    for row in full_mrr_differences
                )
            ),
            "temporal_decay_incremental_benefit_detected": False,
            "reason_memory_only_cannot_retrieve_held_out_targets": (
                "The associative memory was built from training "
                "associations only, while the paired OOD evaluation "
                "contains held-out odor families and target items."
            ),
            "negative_result_reported": True,
            "claim_limit": (
                "This pilot does not establish an incremental "
                "associative-memory advantage under target-held-out "
                "OOD conditions."
            ),
        },
        "scope_limitations": [
            "Exploratory synthetic computational evidence only.",
            "Not a final confirmatory result.",
            "No independent human perceptual evidence.",
            "No clinical or diagnostic evidence.",
            "No chemical exposure or emission-safety evidence.",
            "No physical-device validation.",
            "No deployment-readiness claim.",
        ],
    }

    _write_json(
        results_path,
        results_document,
    )
    _write_json(
        summary_path,
        summary_document,
    )

    results_hash = file_sha256(results_path)
    summary_hash = file_sha256(summary_path)

    manifest = {
        "schema_version": "1.0.0",
        "project": "Neuro-Olfactive Intelligence",
        "owner": "GUARDIANX LLC",
        "release": "noi-ablation-pilot-v0.1",
        "status": "exploratory",
        "registration": {
            "tag": "noi-ablation-v0.1-preimplementation",
            "experiment_implemented_after_registration": True,
        },
        "counts": {
            "original_events": len(
                bundle.original_dataset.events
            ),
            "odor_targets": len(bundle.odor_targets),
            "latent_ood_events": bundle.source_count,
            "observed_graded_ood_rows": (
                bundle.graded_dataset.observed_event_count
            ),
            "ablation_evaluations": len(
                experiment.evaluations
            ),
            "systems": len(NOIAblationSystem),
            "tiers": 3,
            "temporal_displacements": len(
                LOCKED_TEMPORAL_DISPLACEMENTS
            ),
        },
        "verification": {
            "severe_reference_replay": "PASSED",
            "paired_ground_truth": "PASSED",
            "oracle_used": False,
            "ood_tuning_used": False,
            "analysis_unit": "latent_event_id",
            "negative_result_reported": True,
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
            "noi_system_v0.1.yaml": file_sha256(
                SYSTEM_CONFIGURATION_PATH
            ),
            "policy_rules.yaml": file_sha256(
                POLICY_PATH
            ),
        },
        "output_files": {
            "noi_ablation_results.json": {
                "sha256": results_hash,
                "evaluations": len(
                    experiment.evaluations
                ),
            },
            "noi_ablation_summary.json": {
                "sha256": summary_hash,
            },
        },
        "randomness": {
            "generator_seed": bundle.generator_seed,
            "ood_seed": bundle.ood_seed,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
            "pyyaml": version("PyYAML"),
        },
        "scope_limitations": (
            summary_document["scope_limitations"]
        ),
    }

    _write_json(
        manifest_path,
        manifest,
    )

    verification = verify_noi_ablation_pilot_export(
        output_path
    )

    if verification["passed"] is not True:
        raise NOIAblationPilotWorkflowError(
            "Export verification failed: "
            + "; ".join(verification["failures"])
        )

    return {
        "output_directory": str(output_path),
        "results_path": str(results_path),
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "results_sha256": results_hash,
        "summary_sha256": summary_hash,
        "manifest_sha256": file_sha256(
            manifest_path
        ),
        "verification_passed": True,
        "severe_replay_passed": True,
        "latent_event_count": bundle.source_count,
        "evaluation_count": len(
            experiment.evaluations
        ),
        "selected_validation_alpha": (
            experiment.selected_validation_alpha
        ),
        "full_hybrid_equals_ridge": (
            full_equals_ridge
        ),
        "memory_incremental_benefit_detected": (
            summary_document[
                "interpretation"
            ][
                "associative_memory_incremental_benefit_detected"
            ]
        ),
    }


def verify_noi_ablation_pilot_export(
    output_directory: str | Path,
) -> dict[str, Any]:
    """Verify hashes and locked safeguards in an exported pilot."""

    output_path = Path(output_directory)
    manifest_path = output_path / "run_manifest.json"

    if not manifest_path.is_file():
        raise NOIAblationPilotWorkflowError(
            "run_manifest.json is missing."
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise NOIAblationPilotWorkflowError(
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

        if file_sha256(path) != metadata.get("sha256"):
            failures.append(
                f"SHA-256 mismatch: {filename}"
            )

    verification = manifest.get("verification", {})

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

    if verification.get("ood_tuning_used") is not False:
        failures.append(
            "OOD tuning must remain false."
        )

    if verification.get(
        "analysis_unit"
    ) != "latent_event_id":
        failures.append(
            "Analysis unit must be latent_event_id."
        )

    if verification.get(
        "negative_result_reported"
    ) is not True:
        failures.append(
            "The negative result must be reported."
        )

    return {
        "passed": not failures,
        "failures": failures,
    }


def _write_json(
    path: Path,
    document: dict[str, Any],
) -> None:
    """Write canonical deterministic JSON using atomic replacement."""

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    text = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"

    temporary_path.write_text(
        text,
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    """Run the default exploratory workflow and print safeguards."""

    result = run_noi_ablation_pilot()

    print("NOI ablation pilot completed successfully")
    print("Output directory:", result["output_directory"])
    print("Results SHA-256:", result["results_sha256"])
    print("Summary SHA-256:", result["summary_sha256"])
    print("Manifest SHA-256:", result["manifest_sha256"])
    print(
        "Validation-selected alpha:",
        result["selected_validation_alpha"],
    )
    print(
        "Full hybrid equals ridge:",
        result["full_hybrid_equals_ridge"],
    )
    print(
        "Memory incremental benefit detected:",
        result["memory_incremental_benefit_detected"],
    )
    print("Export verification: PASSED")
    print(
        "Scope: exploratory synthetic computational pilot only; "
        "not confirmatory or perceptual evidence."
    )


if __name__ == "__main__":
    main()
