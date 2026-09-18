"""Run the first executable NOI v0.5 development pilot on SmellNet-Base.

This command deliberately uses official *training* recordings only. It
produces development evidence for choosing and locking the v0.5 operating
point; it never opens the official test partition for performance scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

import pandas as pd

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


SCHEMA = "noi-v0.5-smellnet-novel-odor-development-pilot-v1"


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


def _terminal_initial(
    model: OdorSentinel,
    windows: Iterable[TemporalSensorWindow],
) -> tuple[SentinelDecision, ...]:
    return tuple(model.decide(item, allow_repeat=False) for item in windows)


def _adaptive_repeat(
    model: OdorSentinel,
    windows: Iterable[TemporalSensorWindow],
) -> tuple[SentinelDecision, ...]:
    pairs = create_disjoint_repeat_sensing_pairs(windows)
    decisions: list[SentinelDecision] = []
    for pair in pairs:
        initial = model.decide(pair.initial, allow_repeat=True)
        decisions.append(
            model.decide_pair(pair)
            if initial.action == "repeat_sense"
            else initial
        )
    return tuple(decisions)


def _asymmetric_repeat_veto(
    model: OdorSentinel,
    windows: Iterable[TemporalSensorWindow],
) -> tuple[SentinelDecision, ...]:
    """Apply the safety-veto controller to every disjoint sensing pair."""

    return tuple(
        model.decide_pair_asymmetric_veto(pair)
        for pair in create_disjoint_repeat_sensing_pairs(windows)
    )


def _forced_closed_set(
    decisions: Iterable[SentinelDecision],
) -> tuple[SentinelDecision, ...]:
    return tuple(replace(value, action="recognize") for value in decisions)


def _confidence_threshold_baseline(
    decisions: Iterable[SentinelDecision],
    *,
    unknown_family: str,
    required_known_acceptance: float,
) -> tuple[tuple[SentinelDecision, ...], float]:
    values = tuple(decisions)
    known_scores = sorted(
        value.classifier_confidence
        for value in values
        if value.true_family != unknown_family
    )
    if not known_scores:
        raise ValueError("Confidence calibration needs known validation data.")
    rejection_quantile = max(0.0, 1.0 - required_known_acceptance)
    threshold_index = min(
        len(known_scores) - 1,
        int(rejection_quantile * len(known_scores)),
    )
    threshold = float(known_scores[threshold_index])
    baseline = tuple(
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
        for value in values
    )
    return baseline, threshold


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
                "unknown_family": fold.validation_unknown_family,
                "is_unknown": (
                    decision.true_family == fold.validation_unknown_family
                ),
            }
        )
        rows.append(row)
    return rows


def run_development_pilot(
    dataset_root: str | Path,
    *,
    window_size: int = 128,
    required_known_acceptance: float = 0.80,
    repeat_margin: float = 0.06,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Train and evaluate all five development-only family rotations."""

    recordings = discover_smellnet_base(dataset_root)
    folds = build_locked_novel_odor_folds(recordings)
    by_id = {value.recording_id: value for value in recordings}
    fold_reports: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []

    for fold_index, fold in enumerate(folds):
        train_windows = _windows_for_ids(
            fold.model_train_ids,
            by_id=by_id,
            window_size=window_size,
        )
        known_windows = _windows_for_ids(
            fold.validation_known_ids,
            by_id=by_id,
            window_size=window_size,
        )
        unknown_windows = _windows_for_ids(
            fold.validation_unknown_ids,
            by_id=by_id,
            window_size=window_size,
        )
        validation_windows = known_windows + unknown_windows

        model = OdorSentinel(random_state=1729 + fold_index).fit(
            train_windows
        )
        calibration = model.calibrate(
            known_windows,
            unknown_windows,
            required_known_acceptance=required_known_acceptance,
            repeat_margin=repeat_margin,
        )

        sentinel_initial = _terminal_initial(model, validation_windows)
        closed_set = _forced_closed_set(sentinel_initial)
        confidence, confidence_threshold = _confidence_threshold_baseline(
            sentinel_initial,
            unknown_family=fold.validation_unknown_family,
            required_known_acceptance=required_known_acceptance,
        )
        validation_pairs = create_disjoint_repeat_sensing_pairs(
            validation_windows
        )
        pair_initial = tuple(
            model.decide(pair.initial, allow_repeat=False)
            for pair in validation_pairs
        )
        repeat_decisions = _adaptive_repeat(model, validation_windows)
        veto_decisions = _asymmetric_repeat_veto(model, validation_windows)

        metrics = {
            "closed_set": evaluate_sentinel_decisions(
                closed_set,
                unknown_family=fold.validation_unknown_family,
            ).to_dict(),
            "confidence_threshold": evaluate_sentinel_decisions(
                confidence,
                unknown_family=fold.validation_unknown_family,
            ).to_dict(),
            "sentinel_initial": evaluate_sentinel_decisions(
                sentinel_initial,
                unknown_family=fold.validation_unknown_family,
            ).to_dict(),
            "sentinel_pair_initial": evaluate_sentinel_decisions(
                pair_initial,
                unknown_family=fold.validation_unknown_family,
            ).to_dict(),
            "sentinel_adaptive_repeat": evaluate_sentinel_decisions(
                repeat_decisions,
                unknown_family=fold.validation_unknown_family,
            ).to_dict(),
            "sentinel_asymmetric_repeat_veto": evaluate_sentinel_decisions(
                veto_decisions,
                unknown_family=fold.validation_unknown_family,
            ).to_dict(),
        }

        classifier_accuracy = mean(
            decision.predicted_family == decision.true_family
            for decision in sentinel_initial
            if decision.true_family != fold.validation_unknown_family
        )
        memory_accuracy = mean(
            model.predict_memory_family(window) == window.family
            for window in known_windows
        )
        requested_repeats = sum(
            model.decide(pair.initial, allow_repeat=True).action
            == "repeat_sense"
            for pair in validation_pairs
        )

        fold_reports.append(
            {
                "fold_id": fold.fold_id,
                "known_families": list(fold.known_families),
                "validation_unknown_family": (
                    fold.validation_unknown_family
                ),
                "final_unknown_family_not_accessed": (
                    fold.final_unknown_family
                ),
                "window_counts": {
                    "training": len(train_windows),
                    "validation_known": len(known_windows),
                    "validation_unknown": len(unknown_windows),
                },
                "calibration": asdict(calibration),
                "confidence_baseline_threshold": confidence_threshold,
                "metrics": metrics,
                "memory_diagnostic": {
                    "classifier_known_top1": classifier_accuracy,
                    "nearest_memory_known_top1": memory_accuracy,
                    "difference_memory_minus_classifier": (
                        memory_accuracy - classifier_accuracy
                    ),
                },
                "repeat_diagnostic": {
                    "requested_repeat_pairs": requested_repeats,
                    "available_pairs": len(validation_pairs),
                    "utility_change": (
                        metrics["sentinel_adaptive_repeat"][
                            "decision_utility"
                        ]
                        - metrics["sentinel_pair_initial"][
                            "decision_utility"
                        ]
                    ),
                    "asymmetric_veto_utility_change": (
                        metrics["sentinel_asymmetric_repeat_veto"][
                            "decision_utility"
                        ]
                        - metrics["sentinel_pair_initial"][
                            "decision_utility"
                        ]
                    ),
                },
            }
        )

        for name, values in (
            ("closed_set", closed_set),
            ("confidence_threshold", confidence),
            ("sentinel_initial", sentinel_initial),
            ("sentinel_pair_initial", pair_initial),
            ("sentinel_adaptive_repeat", repeat_decisions),
            ("sentinel_asymmetric_repeat_veto", veto_decisions),
        ):
            evidence_rows.extend(_evidence_rows(fold, name, values))

    def metric_mean(strategy: str, metric: str) -> float:
        return mean(
            report["metrics"][strategy][metric]  # type: ignore[index]
            for report in fold_reports
        )

    summary = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "development_only_official_test_not_scored",
        "dataset": {
            "recording_count": len(recordings),
            "window_size": window_size,
            "fold_count": len(folds),
        },
        "policy": {
            "required_known_acceptance": required_known_acceptance,
            "repeat_margin": repeat_margin,
            "actions": ["recognize", "repeat_sense", "abstain_unknown"],
        },
        "aggregate_development_evidence": {
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
            "adaptive_repeat_utility_change": mean(
                report["repeat_diagnostic"]["utility_change"]  # type: ignore[index]
                for report in fold_reports
            ),
            "asymmetric_veto_utility_change": mean(
                report["repeat_diagnostic"][
                    "asymmetric_veto_utility_change"
                ]  # type: ignore[index]
                for report in fold_reports
            ),
            "asymmetric_veto_false_known_rate": metric_mean(
                "sentinel_asymmetric_repeat_veto", "false_known_rate"
            ),
            "asymmetric_veto_known_acceptance_rate": metric_mean(
                "sentinel_asymmetric_repeat_veto", "known_acceptance_rate"
            ),
            "memory_top1_change": mean(
                report["memory_diagnostic"][
                    "difference_memory_minus_classifier"
                ]  # type: ignore[index]
                for report in fold_reports
            ),
        },
        "hypothesis_status": {
            "H10": "development_evidence_only",
            "H11": "development_evidence_only",
            "H12": "development_evidence_only",
        },
        "folds": fold_reports,
        "claim_boundary": (
            "These are validation-development results from physical sensor "
            "recordings. They do not constitute locked final-test evidence, "
            "chemical identification, safety certification, neural decoding, "
            "or deployment validation."
        ),
    }
    return summary, pd.DataFrame(evidence_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the executable NOI v0.5 SmellNet development pilot."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--required-known-acceptance", type=float, default=0.80)
    parser.add_argument("--repeat-margin", type=float, default=0.06)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary, evidence = run_development_pilot(
        args.dataset_root,
        window_size=args.window_size,
        required_known_acceptance=args.required_known_acceptance,
        repeat_margin=args.repeat_margin,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "noi_v0.5_development_pilot.json"
    evidence_path = args.output_dir / "noi_v0.5_development_evidence.csv"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence.to_csv(evidence_path, index=False)
    print(json.dumps(summary["aggregate_development_evidence"], indent=2))
    print(f"summary: {summary_path}")
    print(f"evidence: {evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
