"""Run the Stage 0 NOI v0.4 integrity audit on a local SmellNet copy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from src.evaluation.smellnet_adapter import (
    SmellNetAdapterError,
    audit_smellnet_recordings,
    discover_smellnet_base,
)


def build_audit_payload(dataset_root: str | Path) -> dict[str, object]:
    """Discover the dataset and return a stable, serializable audit result."""

    recordings = discover_smellnet_base(dataset_root)
    report = audit_smellnet_recordings(recordings)
    return {
        "schema": "noi-v0.4-smellnet-integrity-audit-v1",
        "stage": "ingestion_and_integrity_audit",
        "claim_boundary": (
            "Passing this audit establishes ingestion integrity only; it "
            "does not establish recognition, robotic, neural, safety, or "
            "deployment performance."
        ),
        "audit": asdict(report),
    }


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Audit an official-layout SmellNet-Base directory.",
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Directory containing the training/ and testing/ folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; stdout is always emitted.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute the auditable Stage 0 command."""

    options = parse_args(arguments)
    try:
        payload = build_audit_payload(options.dataset_root)
    except SmellNetAdapterError as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if options.output is not None:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["audit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
