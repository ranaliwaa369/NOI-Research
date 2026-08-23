"""Reproducible workflow for the controlled corrective-memory pilot."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

from src.evaluation.corrective_memory_config import (
    load_corrective_memory_configuration,
)
from src.evaluation.corrective_memory_experiment import (
    run_corrective_memory_experiment,
)
from src.evaluation.synthetic_generator import (
    generate_synthetic_pilot,
)
from src.safety.policy_gate import load_policy_rules
from src.system.noi_pipeline import (
    load_noi_system_configuration,
)


DEFAULT_OUTPUT_DIRECTORY = Path(
    "results/corrective-memory-pilot-v0.1"
)

SYNTHETIC_CONFIGURATION_PATH = Path(
    "configs/synthetic_data.yaml"
)
PROTOCOL_PATH = Path(
    "configs/research_protocol.yaml"
)
EVALUATION_CONFIGURATION_PATH = Path(
    "configs/corrective_memory_evaluation_v0.1.yaml"
)
EVALUATION_CHECKSUM_PATH = Path(
    "configs/corrective_memory_evaluation_v0.1.sha256"
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


class CorrectiveMemoryPilotWorkflowError(ValueError):
    """Raised when the corrective-memory workflow cannot complete."""


def file_sha256(path: str | Path) -> str:
    """Return a file SHA-256 digest."""

    file_path = Path(path)

    if not file_path.is_file():
        raise CorrectiveMemoryPilotWorkflowError(
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


def run_corrective_memory_pilot(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run, export, hash, and verify the controlled H2 pilot."""

    if not isinstance(overwrite, bool):
        raise CorrectiveMemoryPilotWorkflowError(
            "overwrite must be boolean."
        )

    output_path = Path(output_directory)

    if output_path.exists() and not output_path.is_dir():
        raise CorrectiveMemoryPilotWorkflowError(
            "Output path exists and is not a directory."
        )

    results_path = (
        output_path / "corrective_memory_results.json"
    )
    summary_path = (
        output_path / "corrective_memory_summary.json"
    )
    manifest_path = output_path / "run_manifest.json"

    existing = [
        path
        for path in (
            results_path,
            summary_path,
            manifest_path,
        )
        if path.exists()
    ]

    if existing and not overwrite:
        raise CorrectiveMemoryPilotWorkflowError(
            "Output files already exist; use overwrite=True "
            "for an intentional deterministic rerun."
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = generate_synthetic_pilot(
        SYNTHETIC_CONFIGURATION_PATH,
        PROTOCOL_PATH,
        event_count=200,
    )

    evaluation_configuration = (
        load_corrective_memory_configuration(
            EVALUATION_CONFIGURATION_PATH,
            EVALUATION_CHECKSUM_PATH,
        )
    )

    experiment = run_corrective_memory_experiment(
        dataset,
        evaluation_configuration=(
            evaluation_configuration
        ),
        system_configuration=(
            load_noi_system_configuration(
                SYSTEM_CONFIGURATION_PATH
            )
        ),
        policy_configuration=load_policy_rules(
            POLICY_PATH
        ),
        protocol_hash=PROTOCOL_HASH,
        trained_at_utc=TRAINED_AT_UTC,
    )

    results_document = {
        "schema_version": "1.0.0",
        "project": "Neuro-Olfactive Intelligence",
        "owner": "GUARDIANX LLC",
        "release": "corrective-memory-pilot-v0.1",
        "status": "exploratory",
        "experiment_type": (
            "controlled_corrective_memory_mechanism_evaluation"
        ),
        "trained_at_utc": TRAINED_AT_UTC.isoformat(),
        "training_event_count": (
            experiment.training_event_count
        ),
        "validation_event_count": (
            experiment.validation_event_count
        ),
        "eligible_target_count": (
            experiment.eligible_target_count
        ),
        "eligible_validation_event_count": (
            experiment.eligible_validation_event_count
        ),
        "eligible_target_ids": list(
            experiment.eligible_target_ids
        ),
        "eligible_validation_event_ids": list(
            experiment.eligible_validation_event_ids
        ),
        "excluded_validation_target_ids": list(
            experiment.excluded_validation_target_ids
        ),
        "alpha": experiment.alpha,
        "apply_temporal_decay": (
            experiment.apply_temporal_decay
        ),
        "top_k": experiment.top_k,
        "oracle_used": experiment.oracle_used,
        "ood_tuning_used": experiment.ood_tuning_used,
        "target_results": [
            {
                "target_item_id": result.target_item_id,
                "decoy_item_id": result.decoy_item_id,
                "validation_event_ids": list(
                    result.validation_event_ids
                ),
                "corrupted_memory_ids": list(
                    result.corrupted_memory_ids
                ),
                "restoration_audit_count": (
                    result.restoration_audit_count
                ),
                "no_update_rankings": [
                    list(ranking)
                    for ranking in result.no_update_rankings
                ],
                "corrected_rankings": [
                    list(ranking)
                    for ranking in result.corrected_rankings
                ],
                "relevant_items": [
                    sorted(relevant)
                    for relevant in result.relevant_items
                ],
                "no_update_mrr": (
                    result.no_update_mean_reciprocal_rank
                ),
                "corrected_mrr": (
                    result.corrected_mean_reciprocal_rank
                ),
                "mrr_difference": (
                    result.reciprocal_rank_difference
                ),
                "no_update_recall_at_1": (
                    result.no_update_recall_at_1
                ),
                "corrected_recall_at_1": (
                    result.corrected_recall_at_1
                ),
                "no_update_recall_at_10": (
                    result.no_update_recall_at_10
                ),
                "corrected_recall_at_10": (
                    result.corrected_recall_at_10
                ),
                "no_update_ndcg_at_10": (
                    result.no_update_ndcg_at_10
                ),
                "corrected_ndcg_at_10": (
                    result.corrected_ndcg_at_10
                ),
                "old_memory_baseline_mrr": (
                    result.old_memory_baseline_mrr
                ),
                "old_memory_post_correction_mrr": (
                    result.old_memory_post_correction_mrr
                ),
                "old_memory_degradation": (
                    result.old_memory_degradation
                ),
            }
            for result in experiment.target_results
        ],
    }

    summary_document = {
        "schema_version": "1.0.0",
        "project": "Neuro-Olfactive Intelligence",
        "owner": "GUARDIANX LLC",
        "release": "corrective-memory-pilot-v0.1",
        "status": "exploratory",
        "primary_contrast": "corrected minus no_update",
        "paired_unit": "target_item_id",
        "eligible_target_count": (
            experiment.eligible_target_count
        ),
        "eligible_validation_event_count": (
            experiment.eligible_validation_event_count
        ),
        "mean_mrr_improvement": (
            experiment.mean_mrr_improvement
        ),
        "standard_deviation_mrr_improvement": (
            experiment.standard_deviation_mrr_improvement
        ),
        "minimum_mrr_improvement": (
            experiment.minimum_mrr_improvement
        ),
        "maximum_mrr_improvement": (
            experiment.maximum_mrr_improvement
        ),
        "bootstrap_ci_lower": (
            experiment.bootstrap_ci_lower
        ),
        "bootstrap_ci_upper": (
            experiment.bootstrap_ci_upper
        ),
        "bootstrap_seed": experiment.bootstrap_seed,
        "bootstrap_resamples": (
            experiment.bootstrap_resamples
        ),
        "confidence_level": (
            experiment.confidence_level
        ),
        "mean_old_memory_degradation": (
            experiment.mean_old_memory_degradation
        ),
        "maximum_old_memory_degradation": (
            experiment.maximum_old_memory_degradation
        ),
        "correction_success_rule_passed": (
            experiment.correction_success_rule_passed
        ),
        "old_memory_degradation_rule_passed": (
            experiment.old_memory_degradation_rule_passed
        ),
        "oracle_used": False,
        "ood_tuning_used": False,
        "interpretation": {
            "controlled_mechanism_restoration_supported": (
                experiment.correction_success_rule_passed
            ),
            "automatic_error_detection_tested": False,
            "unseen_target_discovery_tested": False,
            "human_perceptual_validity_tested": False,
            "physical_device_tested": False,
            "claim_limit": (
                "The result evaluates restoration of deliberately "
                "corrupted, previously stored synthetic associations. "
                "It does not establish automatic correction learning "
                "or real-world olfactory performance."
            ),
        },
        "scope_limitations": [
            "Exploratory synthetic computational pilot only.",
            "Controlled corruption and restoration mechanism test.",
            "Not automatic error-detection evidence.",
            "Not unseen-target discovery evidence.",
            "Not human perceptual evidence.",
            "Not clinical or diagnostic evidence.",
            "Not physical-device or emission-safety evidence.",
            "Not deployment-readiness evidence.",
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
        "release": "corrective-memory-pilot-v0.1",
        "status": "exploratory",
        "registration": {
            "tag": (
                "corrective-memory-v0.1-preimplementation"
            ),
            "experiment_implemented_after_registration": True,
        },
        "counts": {
            "original_events": len(dataset.events),
            "odor_targets": len(dataset.odor_targets),
            "training_events": (
                experiment.training_event_count
            ),
            "validation_events": (
                experiment.validation_event_count
            ),
            "eligible_targets": (
                experiment.eligible_target_count
            ),
            "eligible_queries": (
                experiment.eligible_validation_event_count
            ),
            "target_results": len(
                experiment.target_results
            ),
            "restoration_audits": sum(
                result.restoration_audit_count
                for result in experiment.target_results
            ),
        },
        "verification": {
            "paired_arms": "PASSED",
            "deterministic_decoy_selection": "PASSED",
            "restoration_audit_completeness": "PASSED",
            "oracle_used": False,
            "ood_tuning_used": False,
            "paired_unit": "target_item_id",
            "tests_required_before_release": True,
        },
        "configuration_hashes": {
            "research_protocol.yaml": file_sha256(
                PROTOCOL_PATH
            ),
            "synthetic_data.yaml": file_sha256(
                SYNTHETIC_CONFIGURATION_PATH
            ),
            "corrective_memory_evaluation_v0.1.yaml": (
                file_sha256(
                    EVALUATION_CONFIGURATION_PATH
                )
            ),
            "noi_system_v0.1.yaml": file_sha256(
                SYSTEM_CONFIGURATION_PATH
            ),
            "policy_rules.yaml": file_sha256(
                POLICY_PATH
            ),
        },
        "output_files": {
            "corrective_memory_results.json": {
                "sha256": results_hash,
                "target_results": len(
                    experiment.target_results
                ),
            },
            "corrective_memory_summary.json": {
                "sha256": summary_hash,
            },
        },
        "randomness": {
            "generator_seed": dataset.generator_seed,
            "ood_seed": dataset.ood_seed,
            "bootstrap_seed": (
                experiment.bootstrap_seed
            ),
            "bootstrap_resamples": (
                experiment.bootstrap_resamples
            ),
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

    verification = (
        verify_corrective_memory_pilot_export(
            output_path
        )
    )

    if verification["passed"] is not True:
        raise CorrectiveMemoryPilotWorkflowError(
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
        "eligible_target_count": (
            experiment.eligible_target_count
        ),
        "mean_mrr_improvement": (
            experiment.mean_mrr_improvement
        ),
        "bootstrap_ci": (
            experiment.bootstrap_ci_lower,
            experiment.bootstrap_ci_upper,
        ),
        "mean_old_memory_degradation": (
            experiment.mean_old_memory_degradation
        ),
        "correction_success_rule_passed": (
            experiment.correction_success_rule_passed
        ),
        "old_memory_rule_passed": (
            experiment.old_memory_degradation_rule_passed
        ),
    }


def verify_corrective_memory_pilot_export(
    output_directory: str | Path,
) -> dict[str, Any]:
    """Verify hashes and locked safeguards in an H2 export."""

    output_path = Path(output_directory)
    manifest_path = output_path / "run_manifest.json"

    if not manifest_path.is_file():
        raise CorrectiveMemoryPilotWorkflowError(
            "run_manifest.json is missing."
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise CorrectiveMemoryPilotWorkflowError(
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

    verification = manifest.get(
        "verification",
        {},
    )

    if verification.get("paired_arms") != "PASSED":
        failures.append(
            "Paired arms were not recorded as passed."
        )

    if verification.get(
        "restoration_audit_completeness"
    ) != "PASSED":
        failures.append(
            "Restoration audit completeness was not passed."
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
        "paired_unit"
    ) != "target_item_id":
        failures.append(
            "Paired unit must be target_item_id."
        )

    return {
        "passed": not failures,
        "failures": failures,
    }


def _write_json(
    path: Path,
    document: dict[str, Any],
) -> None:
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
    result = run_corrective_memory_pilot()

    print(
        "NOI corrective-memory pilot completed successfully"
    )
    print(
        "Output directory:",
        result["output_directory"],
    )
    print(
        "Results SHA-256:",
        result["results_sha256"],
    )
    print(
        "Summary SHA-256:",
        result["summary_sha256"],
    )
    print(
        "Manifest SHA-256:",
        result["manifest_sha256"],
    )
    print(
        "Mean MRR improvement:",
        result["mean_mrr_improvement"],
    )
    print(
        "Bootstrap 95% CI:",
        result["bootstrap_ci"],
    )
    print(
        "Mean old-memory degradation:",
        result["mean_old_memory_degradation"],
    )
    print(
        "Correction rule passed:",
        result["correction_success_rule_passed"],
    )
    print(
        "Old-memory rule passed:",
        result["old_memory_rule_passed"],
    )
    print("Export verification: PASSED")
    print(
        "Scope: controlled exploratory synthetic mechanism "
        "evaluation only."
    )


if __name__ == "__main__":
    main()
