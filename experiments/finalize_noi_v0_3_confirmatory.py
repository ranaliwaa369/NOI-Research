"""Finalize locked NOI v0.3 confirmatory artifacts safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.analyze_noi_v0_3_confirmatory import (
    analyze_confirmatory_payloads,
    export_confirmatory_aggregate,
)


REGISTERED_SEEDS = tuple(range(1301, 1311))


class ConfirmatoryFinalizationError(ValueError):
    """Raised when confirmatory finalization is unsafe or invalid."""


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one regular file."""

    if not isinstance(path, Path):
        raise ConfirmatoryFinalizationError(
            "path must be a Path."
        )

    if not path.is_file():
        raise ConfirmatoryFinalizationError(
            f"Artifact is absent or not a file: {path}"
        )

    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sha256_sidecar(path: Path) -> str:
    """Verify an artifact against its adjacent SHA-256 sidecar."""

    digest = sha256_file(path)
    sidecar = Path(f"{path}.sha256")

    if not sidecar.is_file():
        raise ConfirmatoryFinalizationError(
            f"SHA-256 sidecar is absent: {sidecar}"
        )

    parts = sidecar.read_text(
        encoding="utf-8"
    ).strip().split()

    if not parts:
        raise ConfirmatoryFinalizationError(
            f"SHA-256 sidecar is empty: {sidecar}"
        )

    expected = parts[0].lower()

    if (
        len(expected) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected
        )
    ):
        raise ConfirmatoryFinalizationError(
            f"SHA-256 sidecar is invalid: {sidecar}"
        )

    if expected != digest:
        raise ConfirmatoryFinalizationError(
            f"SHA-256 mismatch for artifact: {path}"
        )

    return digest


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object with a controlled error surface."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryFinalizationError(
            f"Invalid JSON artifact: {path}"
        ) from error

    if not isinstance(payload, dict):
        raise ConfirmatoryFinalizationError(
            f"JSON artifact must contain an object: {path}"
        )

    return payload


def load_verified_seed_payloads(
    results_directory: Path,
) -> tuple[dict[str, Any], ...]:
    """Load exactly the registered seed payloads after hash checks."""

    if not isinstance(results_directory, Path):
        raise ConfirmatoryFinalizationError(
            "results_directory must be a Path."
        )

    if not results_directory.is_dir():
        raise ConfirmatoryFinalizationError(
            f"Results directory is absent: {results_directory}"
        )

    expected_names = {
        f"seed-{seed}.json"
        for seed in REGISTERED_SEEDS
    }
    observed_names = {
        item.name
        for item in results_directory.glob("seed-*.json")
        if item.is_file()
    }

    unexpected = sorted(observed_names - expected_names)

    if unexpected:
        raise ConfirmatoryFinalizationError(
            "Unexpected confirmatory seed files: "
            + ", ".join(unexpected)
        )

    payloads: list[dict[str, Any]] = []

    for seed in REGISTERED_SEEDS:
        path = results_directory / f"seed-{seed}.json"

        if not path.is_file():
            raise ConfirmatoryFinalizationError(
                f"Required seed-{seed}.json is absent."
            )

        verify_sha256_sidecar(path)
        payload = _load_json(path)

        if payload.get("schema_version") != (
            "noi-v0.3-confirmatory-seed-v1"
        ):
            raise ConfirmatoryFinalizationError(
                f"seed-{seed}.json has an invalid schema."
            )

        if payload.get("seed") != seed:
            raise ConfirmatoryFinalizationError(
                f"seed-{seed}.json has a mismatched seed."
            )

        if payload.get("confirmatory_execution") is not True:
            raise ConfirmatoryFinalizationError(
                f"seed-{seed}.json is not confirmatory."
            )

        payloads.append(payload)

    return tuple(payloads)


def _package_version(name: str) -> str:
    """Return an installed package version or an explicit marker."""

    try:
        return version(name)
    except PackageNotFoundError:
        return "not_installed"


def build_environment_manifest() -> dict[str, Any]:
    """Describe the execution environment without a runtime timestamp."""

    return {
        "schema_version": (
            "noi-v0.3-environment-manifest-v1"
        ),
        "study_phase": "confirmatory",
        "python_version": platform.python_version(),
        "python_implementation": (
            platform.python_implementation()
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_version": _package_version("numpy"),
        "pyyaml_version": _package_version("PyYAML"),
        "registered_seeds": list(REGISTERED_SEEDS),
        "final_test_tuning_used": False,
    }


def _status_label(value: object) -> str:
    """Convert a registered machine status into report text."""

    if value == "supported":
        return "Supported"

    if value == "not_supported":
        return "Not supported"

    raise ConfirmatoryFinalizationError(
        f"Invalid hypothesis status: {value!r}"
    )


def render_findings_markdown(
    aggregate: Mapping[str, Any],
) -> str:
    """Render complete positive, null, and negative findings."""

    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("schema_version") != (
            "noi-v0.3-confirmatory-aggregate-v1"
        )
    ):
        raise ConfirmatoryFinalizationError(
            "Aggregate has an invalid schema."
        )

    hypotheses = aggregate.get("hypotheses")

    if not isinstance(hypotheses, Mapping):
        raise ConfirmatoryFinalizationError(
            "Aggregate hypotheses are absent."
        )

    lines = [
        "# NOI v0.3 Confirmatory Findings",
        "",
        "## Scope",
        "",
        (
            "These findings come from a synthetic computational "
            "evaluation of simulated olfactory and tactile vectors."
        ),
        (
            "They do not demonstrate performance of a physical sensor, "
            "biological equivalence, chemical safety, clinical validity, "
            "or deployment readiness."
        ),
        "",
        "## Registered hypotheses",
        "",
    ]

    for name in ("H6", "H7", "H8"):
        item = hypotheses.get(name)

        if not isinstance(item, Mapping):
            raise ConfirmatoryFinalizationError(
                f"Aggregate hypothesis {name} is absent."
            )

        label = _status_label(item.get("status"))
        role = item.get("role", "unspecified")

        lines.extend(
            [
                f"### {name}: {label}",
                "",
                f"Registered role: {role}.",
                "",
            ]
        )

    integrity = aggregate.get("integrity")

    if not isinstance(integrity, Mapping):
        raise ConfirmatoryFinalizationError(
            "Aggregate integrity record is absent."
        )

    if (
        integrity.get("all_registered_seeds_retained")
        is not True
        or integrity.get("final_test_tuning_used")
        is not False
    ):
        raise ConfirmatoryFinalizationError(
            "Aggregate integrity controls did not pass."
        )

    lines.extend(
        [
            "## Integrity statement",
            "",
            (
                "All ten registered seeds were retained. Final-test "
                "results were not used to tune thresholds, models, "
                "comparators, confidence rules, or decision policies."
            ),
            (
                "Supported, unsupported, null, and negative outcomes "
                "must be reported without silent seed or condition "
                "removal."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def _canonical_json_bytes(
    payload: Mapping[str, Any],
) -> bytes:
    """Return canonical UTF-8 JSON bytes."""

    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes_exclusive(
    path: Path,
    content: bytes,
) -> None:
    """Write a new artifact without replacing an existing file."""

    if path.exists():
        raise ConfirmatoryFinalizationError(
            f"Output already exists: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_hash_sidecar(path: Path) -> Path:
    """Write a new adjacent SHA-256 sidecar."""

    digest = sha256_file(path)
    sidecar = Path(f"{path}.sha256")

    _write_bytes_exclusive(
        sidecar,
        f"{digest}  {path.name}\n".encode("utf-8"),
    )

    return sidecar


def finalize_confirmatory_results(
    results_directory: Path,
) -> dict[str, Path]:
    """Verify, aggregate, document, and hash final results once."""

    payloads = load_verified_seed_payloads(
        results_directory
    )
    aggregate = analyze_confirmatory_payloads(payloads)
    environment = build_environment_manifest()
    findings = render_findings_markdown(aggregate)

    aggregate_path = (
        results_directory / "aggregate.json"
    )
    environment_path = (
        results_directory / "environment_manifest.json"
    )
    findings_path = (
        results_directory / "findings.md"
    )

    outputs = (
        aggregate_path,
        Path(f"{aggregate_path}.sha256"),
        environment_path,
        Path(f"{environment_path}.sha256"),
        findings_path,
        Path(f"{findings_path}.sha256"),
    )

    existing = tuple(
        path for path in outputs if path.exists()
    )

    if existing:
        raise ConfirmatoryFinalizationError(
            "Finalization output already exists: "
            + ", ".join(str(path) for path in existing)
        )

    export_confirmatory_aggregate(
        aggregate,
        aggregate_path,
        overwrite=False,
    )
    _write_hash_sidecar(aggregate_path)

    _write_bytes_exclusive(
        environment_path,
        _canonical_json_bytes(environment),
    )
    _write_hash_sidecar(environment_path)

    _write_bytes_exclusive(
        findings_path,
        findings.encode("utf-8"),
    )
    _write_hash_sidecar(findings_path)

    return {
        "aggregate": aggregate_path,
        "aggregate_sha256": Path(
            f"{aggregate_path}.sha256"
        ),
        "environment_manifest": environment_path,
        "environment_manifest_sha256": Path(
            f"{environment_path}.sha256"
        ),
        "findings": findings_path,
        "findings_sha256": Path(
            f"{findings_path}.sha256"
        ),
    }


def main(arguments: Sequence[str] | None = None) -> None:
    """Run confirmatory finalization from the command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Verify and finalize all locked NOI v0.3 "
            "confirmatory seed results."
        )
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path("results/v0.3-confirmatory"),
    )
    parsed = parser.parse_args(arguments)

    outputs = finalize_confirmatory_results(
        parsed.results_directory
    )

    print("CONFIRMATORY FINALIZATION: PASS")

    for name, path in outputs.items():
        print(f"{name.upper()}: {path}")


if __name__ == "__main__":
    main()
