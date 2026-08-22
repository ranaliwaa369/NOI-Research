"""Run and export the prespecified NOI baseline pilot experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.evaluation.baseline_experiment import (
    BaselineExperimentResult,
    run_baseline_experiment,
)
from src.evaluation.leakage_audit import audit_synthetic_dataset
from src.evaluation.synthetic_generator import generate_synthetic_pilot
from src.evaluation.synthetic_records import SplitLabel


DEFAULT_SYNTHETIC_CONFIG = Path("configs/synthetic_data.yaml")
DEFAULT_RESEARCH_PROTOCOL = Path("configs/research_protocol.yaml")
DEFAULT_OUTPUT_DIRECTORY = Path("results/baseline-pilot-v0.1")
DEFAULT_EVENT_COUNT = 200
DEFAULT_RANDOM_SEED = 2026
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_TOP_K = 10


class BaselinePilotError(RuntimeError):
    """Raised when the baseline pilot cannot be exported safely."""


@dataclass(frozen=True)
class BaselinePilotExport:
    """Metadata for one exported baseline pilot."""

    output_directory: Path
    results_path: Path
    results_sha256: str
    experiment: BaselineExperimentResult


def run_and_export_baseline_pilot(
    *,
    synthetic_config_path: str | Path = DEFAULT_SYNTHETIC_CONFIG,
    research_protocol_path: str | Path = DEFAULT_RESEARCH_PROTOCOL,
    output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
    event_count: int = DEFAULT_EVENT_COUNT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    ridge_alpha: float = DEFAULT_RIDGE_ALPHA,
    top_k: int = DEFAULT_TOP_K,
    overwrite: bool = False,
) -> BaselinePilotExport:
    """Generate data, run baselines, and export deterministic results."""

    if not isinstance(overwrite, bool):
        raise BaselinePilotError(
            "overwrite must be boolean."
        )

    output_path = Path(output_directory)

    if output_path.exists() and not output_path.is_dir():
        raise BaselinePilotError(
            "output_directory exists but is not a directory."
        )

    results_path = output_path / "baseline_results.json"

    if results_path.exists() and not overwrite:
        raise BaselinePilotError(
            "Refusing to overwrite existing baseline results."
        )

    dataset = generate_synthetic_pilot(
        synthetic_config_path,
        research_protocol_path,
        event_count=event_count,
    )

    audit_report = audit_synthetic_dataset(
        dataset,
        raise_on_failure=True,
    )

    experiment = run_baseline_experiment(
        dataset,
        top_k=top_k,
        random_seed=random_seed,
        ridge_alpha=ridge_alpha,
    )

    validation_strongest = experiment.strongest_baseline(
        SplitLabel.VALIDATION
    )
    ood_strongest = experiment.strongest_baseline(
        SplitLabel.OOD_TEST
    )

    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment": "NOI baseline pilot",
        "owner": "GUARDIANX LLC",
        "dataset": {
            "type": "synthetic_implementation_evaluation",
            "event_count": len(dataset.events),
            "odor_target_count": len(dataset.odor_targets),
            "generator_version": dataset.generator_version,
            "generator_seed": dataset.generator_seed,
            "ood_seed": dataset.ood_seed,
            "leakage_audit_passed": audit_report.passed,
        },
        "settings": {
            "top_k": top_k,
            "random_seed": random_seed,
            "ridge_alpha": ridge_alpha,
        },
        "summaries": list(experiment.to_records()),
        "strongest_observed_baselines": {
            "validation": {
                "baseline": validation_strongest.baseline.value,
                "mean_reciprocal_rank": (
                    validation_strongest.mean_reciprocal_rank
                ),
                "recall_at_10": (
                    validation_strongest.recall_at_10
                ),
            },
            "ood_test": {
                "baseline": ood_strongest.baseline.value,
                "mean_reciprocal_rank": (
                    ood_strongest.mean_reciprocal_rank
                ),
                "recall_at_10": ood_strongest.recall_at_10,
            },
        },
        "event_level_rankings": [
            {
                "baseline": evaluation.baseline.value,
                "split": evaluation.split.value,
                "events": [
                    {
                        "event_id": event_id,
                        "ranking": list(ranking),
                        "relevant_items": sorted(relevant),
                    }
                    for event_id, ranking, relevant in zip(
                        evaluation.event_ids,
                        evaluation.rankings,
                        evaluation.relevant_items,
                        strict=True,
                    )
                ],
            }
            for evaluation in experiment.evaluations
        ],
        "scope_limitations": [
            "Synthetic implementation evaluation only.",
            "No human perceptual validity claim.",
            "No clinical or diagnostic claim.",
            "No physical odor-emission safety claim.",
            "Pilot results are not final confirmatory evidence.",
        ],
    }

    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    results_hash = sha256(content).hexdigest()

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path / ".baseline_results.json.tmp"

    try:
        temporary_path.write_bytes(content)
        temporary_path.replace(results_path)
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise BaselinePilotError(
            f"Baseline pilot export failed: {error}"
        ) from error

    if sha256(results_path.read_bytes()).hexdigest() != results_hash:
        raise BaselinePilotError(
            "Exported baseline results failed SHA-256 verification."
        )

    return BaselinePilotExport(
        output_directory=output_path,
        results_path=results_path,
        results_sha256=results_hash,
        experiment=experiment,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the baseline-pilot command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run and export the prespecified synthetic NOI "
            "retrieval baseline pilot."
        )
    )

    parser.add_argument(
        "--synthetic-config",
        type=Path,
        default=DEFAULT_SYNTHETIC_CONFIG,
    )
    parser.add_argument(
        "--research-protocol",
        type=Path,
        default=DEFAULT_RESEARCH_PROTOCOL,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--event-count",
        type=int,
        default=DEFAULT_EVENT_COUNT,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=DEFAULT_RIDGE_ALPHA,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> int:
    """Run the baseline pilot from the command line."""

    arguments = build_argument_parser().parse_args()

    exported = run_and_export_baseline_pilot(
        synthetic_config_path=arguments.synthetic_config,
        research_protocol_path=arguments.research_protocol,
        output_directory=arguments.output_directory,
        event_count=arguments.event_count,
        random_seed=arguments.random_seed,
        ridge_alpha=arguments.ridge_alpha,
        top_k=arguments.top_k,
        overwrite=arguments.overwrite,
    )

    print("NOI baseline pilot completed successfully")
    print(f"Results file: {exported.results_path}")
    print(f"Results SHA-256: {exported.results_sha256}")

    print("\nBaseline summary:")
    for record in exported.experiment.to_records():
        print(
            f"{record['split']:10} "
            f"{record['baseline']:20} "
            f"R@1={record['recall_at_1']:.4f} "
            f"R@10={record['recall_at_10']:.4f} "
            f"MRR={record['mean_reciprocal_rank']:.4f} "
            f"nDCG@10={record['ndcg_at_10']:.4f}"
        )

    print(
        "\nScope: synthetic pilot only; "
        "not final confirmatory or perceptual evidence."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())