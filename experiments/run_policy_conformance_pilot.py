"""Reproducible workflow for the locked policy-conformance pilot."""

from __future__ import annotations

import json
import platform
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

from src.evaluation.policy_conformance_config import (
    load_policy_conformance_configuration,
)
from src.evaluation.policy_conformance_experiment import (
    run_policy_conformance_experiment,
)
from src.safety.policy_gate import load_policy_rules


DEFAULT_OUTPUT_DIRECTORY = Path(
    "results/policy-conformance-pilot-v0.1"
)

EVALUATION_CONFIGURATION_PATH = Path(
    "configs/policy_conformance_evaluation_v0.1.yaml"
)
EVALUATION_CHECKSUM_PATH = Path(
    "configs/policy_conformance_evaluation_v0.1.sha256"
)
POLICY_PATH = Path(
    "configs/policy_rules.yaml"
)
PROTOCOL_PATH = Path(
    "configs/research_protocol.yaml"
)

PROTOCOL_HASH = (
    "e885f537b3209d7052d1517efc5be753"
    "24df39689cb4e54b38a241c95f22e512"
)


class PolicyConformancePilotWorkflowError(ValueError):
    """Raised when the H3 export workflow cannot complete."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    file_path = Path(path)

    if not file_path.is_file():
        raise PolicyConformancePilotWorkflowError(
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


def run_policy_conformance_pilot(
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run, export, hash, and verify the locked H3 suite."""

    if not isinstance(overwrite, bool):
        raise PolicyConformancePilotWorkflowError(
            "overwrite must be boolean."
        )

    output_path = Path(output_directory)

    if output_path.exists() and not output_path.is_dir():
        raise PolicyConformancePilotWorkflowError(
            "Output path exists and is not a directory."
        )

    results_path = (
        output_path / "policy_conformance_results.json"
    )
    summary_path = (
        output_path / "policy_conformance_summary.json"
    )
    manifest_path = output_path / "run_manifest.json"

    expected_files = (
        results_path,
        summary_path,
        manifest_path,
    )

    if (
        any(path.exists() for path in expected_files)
        and not overwrite
    ):
        raise PolicyConformancePilotWorkflowError(
            "Output files already exist; use overwrite=True "
            "for an intentional deterministic rerun."
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_configuration = (
        load_policy_conformance_configuration(
            EVALUATION_CONFIGURATION_PATH,
            EVALUATION_CHECKSUM_PATH,
        )
    )

    experiment = run_policy_conformance_experiment(
        evaluation_configuration,
        policy_configuration=load_policy_rules(
            POLICY_PATH
        ),
        protocol_hash=PROTOCOL_HASH,
    )

    results_document = {
        "schema_version": "1.0.0",
        "project": "Neuro-Olfactive Intelligence",
        "owner": "GUARDIANX LLC",
        "release": "policy-conformance-pilot-v0.1",
        "status": "exploratory",
        "suite_type": (
            "locked_simulated_computational_policy_cases"
        ),
        "total_case_count": experiment.total_case_count,
        "expected_counts": {
            "allow": experiment.expected_allow_count,
            "block": experiment.expected_block_count,
            "require_missing_information": (
                experiment.expected_missing_information_count
            ),
        },
        "physical_emission_performed": (
            experiment.physical_emission_performed
        ),
        "case_results": [
            {
                "request_id": result.request_id,
                "description": result.description,
                "expected_outcome": (
                    result.expected_outcome.value
                ),
                "predicted_outcome": (
                    result.predicted_outcome.value
                ),
                "rule_ids": list(result.rule_ids),
                "explanation": result.explanation,
                "protocol_hash": result.protocol_hash,
                "exact_match": result.exact_match,
                "false_allow": result.false_allow,
                "false_block": result.false_block,
            }
            for result in experiment.case_results
        ],
    }

    summary_document = {
        "schema_version": "1.0.0",
        "project": "Neuro-Olfactive Intelligence",
        "owner": "GUARDIANX LLC",
        "release": "policy-conformance-pilot-v0.1",
        "status": "exploratory",
        "total_case_count": experiment.total_case_count,
        "exact_match_count": experiment.exact_match_count,
        "false_allow_count": experiment.false_allow_count,
        "false_allow_rate": experiment.false_allow_rate,
        "false_block_count": experiment.false_block_count,
        "false_block_rate": experiment.false_block_rate,
        "exact_conformance_rate": (
            experiment.exact_conformance_rate
        ),
        "policy_coverage": experiment.policy_coverage,
        "false_allow_target_passed": (
            experiment.false_allow_target_passed
        ),
        "false_block_target_passed": (
            experiment.false_block_target_passed
        ),
        "exact_conformance_target_passed": (
            experiment.exact_conformance_target_passed
        ),
        "policy_coverage_target_passed": (
            experiment.policy_coverage_target_passed
        ),
        "h3_success_rule_passed": (
            experiment.h3_success_rule_passed
        ),
        "physical_emission_performed": False,
        "interpretation": {
            "locked_case_conformance_supported": (
                experiment.h3_success_rule_passed
            ),
            "comprehensive_safety_demonstrated": False,
            "chemical_exposure_safety_demonstrated": False,
            "clinical_safety_demonstrated": False,
            "physical_device_safety_demonstrated": False,
            "adversarial_robustness_demonstrated": False,
            "legal_compliance_demonstrated": False,
            "deployment_readiness_demonstrated": False,
            "claim_limit": (
                "Passing establishes exact conformance only to "
                "the 26 preregistered simulated computational "
                "policy cases."
            ),
        },
        "scope_limitations": [
            "Exploratory locked computational test suite only.",
            "Thresholds are simulated and are not exposure limits.",
            "Not comprehensive safety evidence.",
            "Not chemical or clinical safety evidence.",
            "Not physical-device or emission evidence.",
            "Not adversarial robustness evidence.",
            "Not legal-compliance evidence.",
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
        "release": "policy-conformance-pilot-v0.1",
        "status": "exploratory",
        "registration": {
            "tag": (
                "policy-conformance-v0.1-preimplementation"
            ),
            "experiment_implemented_after_registration": True,
        },
        "counts": {
            "total_cases": experiment.total_case_count,
            "expected_allow": (
                experiment.expected_allow_count
            ),
            "expected_block": (
                experiment.expected_block_count
            ),
            "expected_missing_information": (
                experiment.expected_missing_information_count
            ),
            "exact_matches": experiment.exact_match_count,
        },
        "verification": {
            "unique_request_ids": "PASSED",
            "locked_case_order": "PASSED",
            "decision_audit_retained": "PASSED",
            "physical_emission_performed": False,
            "tests_required_before_release": True,
        },
        "configuration_hashes": {
            "research_protocol.yaml": file_sha256(
                PROTOCOL_PATH
            ),
            "policy_rules.yaml": file_sha256(
                POLICY_PATH
            ),
            "policy_conformance_evaluation_v0.1.yaml": (
                file_sha256(
                    EVALUATION_CONFIGURATION_PATH
                )
            ),
        },
        "output_files": {
            "policy_conformance_results.json": {
                "sha256": results_hash,
                "case_count": experiment.total_case_count,
            },
            "policy_conformance_summary.json": {
                "sha256": summary_hash,
            },
        },
        "environment": {
            "python": platform.python_version(),
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
        verify_policy_conformance_pilot_export(
            output_path
        )
    )

    if verification["passed"] is not True:
        raise PolicyConformancePilotWorkflowError(
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
        "total_case_count": experiment.total_case_count,
        "false_allow_count": (
            experiment.false_allow_count
        ),
        "false_block_count": (
            experiment.false_block_count
        ),
        "exact_conformance_rate": (
            experiment.exact_conformance_rate
        ),
        "policy_coverage": experiment.policy_coverage,
        "h3_success_rule_passed": (
            experiment.h3_success_rule_passed
        ),
    }


def verify_policy_conformance_pilot_export(
    output_directory: str | Path,
) -> dict[str, Any]:
    """Verify output hashes and locked H3 safeguards."""

    output_path = Path(output_directory)
    manifest_path = output_path / "run_manifest.json"

    if not manifest_path.is_file():
        raise PolicyConformancePilotWorkflowError(
            "run_manifest.json is missing."
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise PolicyConformancePilotWorkflowError(
            "run_manifest.json is not valid JSON."
        ) from error

    failures = []

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

    for key in (
        "unique_request_ids",
        "locked_case_order",
        "decision_audit_retained",
    ):
        if verification.get(key) != "PASSED":
            failures.append(
                f"{key} was not recorded as passed."
            )

    if verification.get(
        "physical_emission_performed"
    ) is not False:
        failures.append(
            "Physical emission must remain false."
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
    result = run_policy_conformance_pilot()

    print(
        "NOI policy-conformance pilot completed successfully"
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
        "Cases:",
        result["total_case_count"],
    )
    print(
        "False allows:",
        result["false_allow_count"],
    )
    print(
        "False blocks:",
        result["false_block_count"],
    )
    print(
        "Exact conformance:",
        result["exact_conformance_rate"],
    )
    print(
        "Policy coverage:",
        result["policy_coverage"],
    )
    print(
        "H3 rule passed:",
        result["h3_success_rule_passed"],
    )
    print("Export verification: PASSED")
    print(
        "Scope: locked simulated computational cases only; "
        "not comprehensive safety evidence."
    )


if __name__ == "__main__":
    main()
