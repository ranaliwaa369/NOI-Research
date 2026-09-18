"""Execute the locked NOI v0.5 SmellNet final evaluation exactly once.

The command has a read-only preflight mode and an explicitly confirmed final
mode.  Final mode creates an exclusive execution marker before loading the
dataset, applies only the decisions fixed in the validation lock, and emits
hash-addressed summary and per-window evidence artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from statistics import mean
from typing import Iterable

import pandas as pd
import yaml

from src.evaluation.smellnet_adapter import discover_smellnet_base
from src.evaluation.smellnet_novel_odor_protocol import (
    NovelOdorFold,
    TemporalSensorWindow,
    build_locked_novel_odor_folds,
    create_disjoint_repeat_sensing_pairs,
    create_nonoverlapping_windows,
)
from src.evaluation.smellnet_odor_sentinel import (
    OdorSentinel,
    SentinelDecision,
    evaluate_sentinel_decisions,
)


SCHEMA = "noi-v0.5-smellnet-locked-final-v1"
CONFIRMATION = "EXECUTE-NOI-V0.5-FINAL-ONCE"
DEFAULT_LOCK = Path("configs/noi_v0.5_validation_lock.yaml")
DEFAULT_LOCK_DIGEST = Path("configs/noi_v0.5_validation_lock.sha256")
DEFAULT_OUTPUT = Path("results/noi_v0.5_locked_final")


class LockedFinalError(RuntimeError):
    """Raised when a locked-final safety or provenance check fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_sidecar_digest(path: Path) -> str:
    tokens = path.read_text(encoding="utf-8").strip().split()
    if not tokens or len(tokens[0]) != 64:
        raise LockedFinalError(f"Invalid SHA-256 sidecar: {path}")
    return tokens[0].lower()


def verify_locked_file(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise LockedFinalError(f"Missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected.lower():
        raise LockedFinalError(
            f"{label} SHA-256 mismatch: expected {expected}, got {observed}"
        )


def load_and_verify_lock(
    lock_path: Path = DEFAULT_LOCK,
    digest_path: Path = DEFAULT_LOCK_DIGEST,
) -> dict[str, object]:
    verify_locked_file(
        lock_path,
        _expected_sidecar_digest(digest_path),
        label="validation lock",
    )
    value = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LockedFinalError("Validation lock must be a YAML mapping.")
    state = value.get("validation_lock", {})
    if not isinstance(state, dict) or state.get("status") != "validation_locked":
        raise LockedFinalError("Protocol is not validation-locked.")
    if state.get("final_test_executed") is not False:
        raise LockedFinalError("Lock does not authorize an unexecuted final test.")
    return value


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise LockedFinalError(result.stderr.strip() or "Git check failed.")
    return result.stdout.strip()


def run_preflight(
    lock: dict[str, object],
    *,
    output_dir: Path,
    require_clean_git: bool = True,
) -> dict[str, object]:
    provenance = lock["provenance"]
    if not isinstance(provenance, dict):
        raise LockedFinalError("Lock provenance is malformed.")

    artifact_checks = (
        ("dataset_integrity_artifact", "dataset_integrity_sha256"),
        ("development_summary_artifact", "development_summary_sha256"),
        ("development_evidence_artifact", "development_evidence_sha256"),
    )
    verified: dict[str, str] = {}
    for path_key, digest_key in artifact_checks:
        path = Path(str(provenance[path_key]))
        expected = str(provenance[digest_key])
        verify_locked_file(path, expected, label=path_key)
        verified[str(path)] = expected

    implementation_commit = str(provenance["implementation_commit"])
    _git("merge-base", "--is-ancestor", implementation_commit, "HEAD")
    head = _git("rev-parse", "HEAD")
    if require_clean_git and _git("status", "--porcelain", "--untracked-files=no"):
        raise LockedFinalError(
            "Tracked working tree changes exist; commit or restore them first."
        )

    state_path = output_dir / "noi_v0.5_final_execution_state.json"
    summary_path = output_dir / "noi_v0.5_locked_final.json"
    evidence_path = output_dir / "noi_v0.5_locked_final_evidence.csv"
    existing = [path for path in (state_path, summary_path, evidence_path) if path.exists()]
    if existing:
        raise LockedFinalError(
            "Final execution is already started or completed: "
            + ", ".join(str(path) for path in existing)
        )

    return {
        "status": "ready",
        "head_commit": head,
        "implementation_commit": implementation_commit,
        "verified_artifacts": verified,
        "output_dir": str(output_dir),
        "confirmation_required": CONFIRMATION,
    }


def _write_exclusive_state(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except BaseException:
        raise


def _windows_for_ids(
    ids: Iterable[str],
    *,
    by_id: dict[str, object],
    window_size: int,
) -> tuple[TemporalSensorWindow, ...]:
    windows: list[TemporalSensorWindow] = []
    for recording_id in sorted(ids):
        windows.extend(
            create_nonoverlapping_windows(
                by_id[recording_id],  # type: ignore[arg-type]
                window_size=window_size,
            )
        )
    return tuple(windows)


def _terminal(
    model: OdorSentinel,
    windows: Iterable[TemporalSensorWindow],
) -> tuple[SentinelDecision, ...]:
    return tuple(model.decide(window, allow_repeat=False) for window in windows)


def _forced_closed_set(
    decisions: Iterable[SentinelDecision],
) -> tuple[SentinelDecision, ...]:
    return tuple(replace(value, action="recognize") for value in decisions)


def _calibrate_confidence_threshold(
    decisions: Iterable[SentinelDecision],
    *,
    required_known_acceptance: float,
) -> float:
    scores = sorted(value.classifier_confidence for value in decisions)
    if not scores:
        raise LockedFinalError("Confidence calibration requires known windows.")
    index = min(
        len(scores) - 1,
        int(max(0.0, 1.0 - required_known_acceptance) * len(scores)),
    )
    return float(scores[index])


def _apply_confidence_threshold(
    decisions: Iterable[SentinelDecision],
    *,
    threshold: float,
) -> tuple[SentinelDecision, ...]:
    return tuple(
        replace(
            value,
            action=(
                "recognize"
                if value.classifier_confidence >= threshold
                else "abstain_unknown"
            ),
            combined_support=value.classifier_confidence,
            novelty_score=1.0 - value.classifier_confidence,
        )
        for value in decisions
    )


def _evidence_rows(
    fold: NovelOdorFold,
    strategy: str,
    decisions: Iterable[SentinelDecision],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for decision in decisions:
        row = decision.to_dict()
        row.update(
            {
                "fold_id": fold.fold_id,
                "strategy": strategy,
                "unknown_family": fold.final_unknown_family,
                "is_unknown": decision.true_family == fold.final_unknown_family,
            }
        )
        rows.append(row)
    return rows


def score_hypotheses(
    aggregate: dict[str, float],
    *,
    required_known_acceptance: float,
    maximum_known_acceptance_loss: float,
) -> dict[str, dict[str, object]]:
    strongest_baseline = min(
        aggregate["closed_set_false_known_rate"],
        aggregate["confidence_baseline_false_known_rate"],
    )
    h10_checks = {
        "false_known_below_strongest_baseline": (
            aggregate["sentinel_false_known_rate"] < strongest_baseline
        ),
        "known_acceptance_at_least_required": (
            aggregate["sentinel_known_acceptance_rate"]
            >= required_known_acceptance
        ),
    }
    h11_checks = {
        "veto_utility_strictly_higher": (
            aggregate["asymmetric_veto_utility_change"] > 0.0
        ),
        "veto_false_known_not_higher": (
            aggregate["asymmetric_veto_false_known_rate"]
            <= aggregate["pair_initial_false_known_rate"]
        ),
        "known_acceptance_loss_within_limit": (
            aggregate["pair_initial_known_acceptance_rate"]
            - aggregate["asymmetric_veto_known_acceptance_rate"]
            <= maximum_known_acceptance_loss + 1e-12
        ),
    }
    h12_checks = {
        "memory_top1_strictly_higher": aggregate["memory_top1_change"] > 0.0,
        "memory_sentinel_false_known_not_higher": (
            aggregate["sentinel_false_known_rate"]
            <= aggregate["confidence_baseline_false_known_rate"]
        ),
    }

    def result(checks: dict[str, bool]) -> dict[str, object]:
        return {
            "passed": all(checks.values()),
            "checks": checks,
        }

    return {"H10": result(h10_checks), "H11": result(h11_checks), "H12": result(h12_checks)}


def run_locked_final(
    dataset_root: Path,
    lock: dict[str, object],
    *,
    head_commit: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    representation = lock["locked_representation"]
    calibration_lock = lock["locked_calibration"]
    hypothesis_lock = lock["hypotheses"]
    if not all(
        isinstance(value, dict)
        for value in (representation, calibration_lock, hypothesis_lock)
    ):
        raise LockedFinalError("Locked execution fields are malformed.")
    window_size = int(representation["window_size"])  # type: ignore[index]
    required_acceptance = float(
        calibration_lock["required_known_acceptance"]  # type: ignore[index]
    )
    repeat_margin = float(calibration_lock["repeat_margin"])  # type: ignore[index]

    recordings = discover_smellnet_base(dataset_root)
    folds = build_locked_novel_odor_folds(recordings)
    by_id = {value.recording_id: value for value in recordings}
    fold_reports: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []

    for fold_index, fold in enumerate(folds):
        train = _windows_for_ids(
            fold.model_train_ids, by_id=by_id, window_size=window_size
        )
        validation_known = _windows_for_ids(
            fold.validation_known_ids, by_id=by_id, window_size=window_size
        )
        validation_unknown = _windows_for_ids(
            fold.validation_unknown_ids, by_id=by_id, window_size=window_size
        )
        final_known = _windows_for_ids(
            fold.final_known_ids, by_id=by_id, window_size=window_size
        )
        final_unknown = _windows_for_ids(
            fold.final_unknown_ids, by_id=by_id, window_size=window_size
        )
        final_windows = final_known + final_unknown

        model = OdorSentinel(random_state=1729 + fold_index).fit(train)
        sentinel_calibration = model.calibrate(
            validation_known,
            validation_unknown,
            required_known_acceptance=required_acceptance,
            repeat_margin=repeat_margin,
        )
        validation_known_decisions = _terminal(model, validation_known)
        confidence_threshold = _calibrate_confidence_threshold(
            validation_known_decisions,
            required_known_acceptance=required_acceptance,
        )

        sentinel = _terminal(model, final_windows)
        closed_set = _forced_closed_set(sentinel)
        confidence = _apply_confidence_threshold(
            sentinel, threshold=confidence_threshold
        )
        pairs = create_disjoint_repeat_sensing_pairs(final_windows)
        pair_initial = tuple(
            model.decide(pair.initial, allow_repeat=False) for pair in pairs
        )
        veto = tuple(model.decide_pair_asymmetric_veto(pair) for pair in pairs)

        strategies = {
            "closed_set": closed_set,
            "confidence_threshold": confidence,
            "sentinel_initial": sentinel,
            "sentinel_pair_initial": pair_initial,
            "sentinel_asymmetric_repeat_veto": veto,
        }
        metrics = {
            name: evaluate_sentinel_decisions(
                values, unknown_family=fold.final_unknown_family
            ).to_dict()
            for name, values in strategies.items()
        }
        classifier_top1 = mean(
            value.predicted_family == value.true_family
            for value in sentinel
            if value.true_family != fold.final_unknown_family
        )
        memory_top1 = mean(
            model.predict_memory_family(window) == window.family
            for window in final_known
        )
        fold_reports.append(
            {
                "fold_id": fold.fold_id,
                "known_families": list(fold.known_families),
                "validation_unknown_family": fold.validation_unknown_family,
                "final_unknown_family": fold.final_unknown_family,
                "window_counts": {
                    "training": len(train),
                    "validation_known": len(validation_known),
                    "validation_unknown": len(validation_unknown),
                    "final_known": len(final_known),
                    "final_unknown": len(final_unknown),
                },
                "calibration": asdict(sentinel_calibration),
                "confidence_baseline_threshold": confidence_threshold,
                "metrics": metrics,
                "memory_diagnostic": {
                    "classifier_known_top1": classifier_top1,
                    "nearest_memory_known_top1": memory_top1,
                    "difference_memory_minus_classifier": memory_top1
                    - classifier_top1,
                },
                "repeat_diagnostic": {
                    "available_pairs": len(pairs),
                    "asymmetric_veto_utility_change": (
                        metrics["sentinel_asymmetric_repeat_veto"][
                            "decision_utility"
                        ]
                        - metrics["sentinel_pair_initial"]["decision_utility"]
                    ),
                },
            }
        )
        for name, values in strategies.items():
            evidence.extend(_evidence_rows(fold, name, values))

    def metric_mean(strategy: str, metric: str) -> float:
        return mean(
            float(report["metrics"][strategy][metric])  # type: ignore[index]
            for report in fold_reports
        )

    aggregate = {
        "closed_set_false_known_rate": metric_mean(
            "closed_set", "false_known_rate"
        ),
        "confidence_baseline_false_known_rate": metric_mean(
            "confidence_threshold", "false_known_rate"
        ),
        "sentinel_false_known_rate": metric_mean(
            "sentinel_initial", "false_known_rate"
        ),
        "sentinel_known_acceptance_rate": metric_mean(
            "sentinel_initial", "known_acceptance_rate"
        ),
        "sentinel_unknown_auroc": metric_mean(
            "sentinel_initial", "unknown_auroc"
        ),
        "pair_initial_false_known_rate": metric_mean(
            "sentinel_pair_initial", "false_known_rate"
        ),
        "pair_initial_known_acceptance_rate": metric_mean(
            "sentinel_pair_initial", "known_acceptance_rate"
        ),
        "asymmetric_veto_false_known_rate": metric_mean(
            "sentinel_asymmetric_repeat_veto", "false_known_rate"
        ),
        "asymmetric_veto_known_acceptance_rate": metric_mean(
            "sentinel_asymmetric_repeat_veto", "known_acceptance_rate"
        ),
        "asymmetric_veto_utility_change": mean(
            float(report["repeat_diagnostic"]["asymmetric_veto_utility_change"])  # type: ignore[index]
            for report in fold_reports
        ),
        "memory_top1_change": mean(
            float(report["memory_diagnostic"]["difference_memory_minus_classifier"])  # type: ignore[index]
            for report in fold_reports
        ),
    }
    maximum_loss = float(hypothesis_lock["H11"]["maximum_known_acceptance_loss"])  # type: ignore[index]
    hypothesis_results = score_hypotheses(
        aggregate,
        required_known_acceptance=required_acceptance,
        maximum_known_acceptance_loss=maximum_loss,
    )
    summary = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "locked_final_test_completed",
        "software_commit": head_commit,
        "dataset": {
            "name": "SmellNet-Base",
            "recording_count": len(recordings),
            "window_size": window_size,
            "fold_count": len(folds),
        },
        "aggregate_final_evidence": aggregate,
        "hypothesis_results": hypothesis_results,
        "folds": fold_reports,
        "claim_boundary": lock["claim_boundary"],
    }
    return summary, pd.DataFrame(evidence)


def _write_final_artifacts(
    output_dir: Path,
    summary: dict[str, object],
    evidence: pd.DataFrame,
) -> dict[str, str]:
    summary_path = output_dir / "noi_v0.5_locked_final.json"
    evidence_path = output_dir / "noi_v0.5_locked_final_evidence.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence.to_csv(evidence_path, index=False)
    digests = {
        str(summary_path): sha256_file(summary_path),
        str(evidence_path): sha256_file(evidence_path),
    }
    digest_path = output_dir / "noi_v0.5_locked_final.sha256"
    digest_path.write_text(
        "".join(f"{digest}  {path}\n" for path, digest in digests.items()),
        encoding="utf-8",
    )
    return digests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or execute the locked NOI v0.5 final test."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--lock-digest", type=Path, default=DEFAULT_LOCK_DIGEST)
    parser.add_argument("--execute-final", action="store_true")
    parser.add_argument("--confirm", default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    lock = load_and_verify_lock(args.lock, args.lock_digest)
    preflight = run_preflight(lock, output_dir=args.output_dir)
    if not args.execute_final:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        print("Preflight only: official final-test recordings were not scored.")
        return 0
    if args.confirm != CONFIRMATION:
        raise LockedFinalError(
            f"Final execution requires --confirm {CONFIRMATION}"
        )

    state_path = args.output_dir / "noi_v0.5_final_execution_state.json"
    started = {
        "schema": "noi-v0.5-final-execution-state-v1",
        "status": "started_irreversible",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "head_commit": preflight["head_commit"],
        "validation_lock_sha256": sha256_file(args.lock),
    }
    _write_exclusive_state(state_path, started)
    try:
        summary, evidence = run_locked_final(
            args.dataset_root,
            lock,
            head_commit=str(preflight["head_commit"]),
        )
        digests = _write_final_artifacts(args.output_dir, summary, evidence)
        completed = {
            **started,
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": digests,
        }
        state_path.chmod(0o644)
        state_path.write_text(
            json.dumps(completed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state_path.chmod(0o444)
    except BaseException as error:
        failed = {
            **started,
            "status": "failed_no_rerun_permitted",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        state_path.chmod(0o644)
        state_path.write_text(
            json.dumps(failed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state_path.chmod(0o444)
        raise

    print(json.dumps(summary["aggregate_final_evidence"], indent=2))
    print(json.dumps(summary["hypothesis_results"], indent=2))
    print(f"summary: {args.output_dir / 'noi_v0.5_locked_final.json'}")
    print(f"evidence: {args.output_dir / 'noi_v0.5_locked_final_evidence.csv'}")
    print(f"hashes: {args.output_dir / 'noi_v0.5_locked_final.sha256'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
