"""Experiment runner for prespecified NOI retrieval baselines."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from src.baselines.retrieval_baselines import (
    BaselineEvaluation,
    BaselineKind,
    evaluate_baseline,
)
from src.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
)


class BaselineExperimentError(ValueError):
    """Raised when a baseline experiment is invalid."""


@dataclass(frozen=True)
class BaselineMetricSummary:
    """Aggregate retrieval metrics for one baseline and split."""

    baseline: BaselineKind
    split: SplitLabel
    event_count: int
    recall_at_1: float
    recall_at_10: float
    mean_reciprocal_rank: float
    ndcg_at_10: float

    def __post_init__(self) -> None:
        if self.event_count < 1:
            raise BaselineExperimentError(
                "event_count must be positive."
            )

        metrics = (
            self.recall_at_1,
            self.recall_at_10,
            self.mean_reciprocal_rank,
            self.ndcg_at_10,
        )

        if any(
            not isfinite(metric)
            or metric < 0.0
            or metric > 1.0
            for metric in metrics
        ):
            raise BaselineExperimentError(
                "Every retrieval metric must be finite "
                "and between 0 and 1."
            )

        if self.recall_at_10 < self.recall_at_1:
            raise BaselineExperimentError(
                "Recall@10 cannot be lower than Recall@1."
            )


@dataclass(frozen=True)
class BaselineExperimentResult:
    """Complete outputs from one baseline experiment."""

    summaries: tuple[BaselineMetricSummary, ...]
    evaluations: tuple[BaselineEvaluation, ...]
    top_k: int
    random_seed: int
    ridge_alpha: float

    def __post_init__(self) -> None:
        if not self.summaries:
            raise BaselineExperimentError(
                "At least one metric summary is required."
            )

        if len(self.summaries) != len(self.evaluations):
            raise BaselineExperimentError(
                "Every evaluation must have one metric summary."
            )

        summary_keys = {
            (
                summary.baseline,
                summary.split,
            )
            for summary in self.summaries
        }

        evaluation_keys = {
            (
                evaluation.baseline,
                evaluation.split,
            )
            for evaluation in self.evaluations
        }

        if summary_keys != evaluation_keys:
            raise BaselineExperimentError(
                "Summary and evaluation keys must match."
            )

        if len(summary_keys) != len(self.summaries):
            raise BaselineExperimentError(
                "Baseline and split combinations must be unique."
            )

    def get_summary(
        self,
        baseline: BaselineKind,
        split: SplitLabel,
    ) -> BaselineMetricSummary:
        """Return one summary by baseline and split."""

        for summary in self.summaries:
            if (
                summary.baseline is baseline
                and summary.split is split
            ):
                return summary

        raise BaselineExperimentError(
            "No summary exists for "
            f"{baseline.value} on {split.value}."
        )

    def strongest_baseline(
        self,
        split: SplitLabel,
    ) -> BaselineMetricSummary:
        """Return the baseline with the highest MRR on one split."""

        candidates = tuple(
            summary
            for summary in self.summaries
            if summary.split is split
        )

        if not candidates:
            raise BaselineExperimentError(
                f"No summaries exist for split {split.value}."
            )

        return sorted(
            candidates,
            key=lambda summary: (
                -summary.mean_reciprocal_rank,
                -summary.recall_at_10,
                summary.baseline.value,
            ),
        )[0]

    def to_records(
        self,
    ) -> tuple[dict[str, object], ...]:
        """Return deterministic dictionaries for tables and JSON."""

        return tuple(
            {
                "baseline": summary.baseline.value,
                "split": summary.split.value,
                "event_count": summary.event_count,
                "recall_at_1": summary.recall_at_1,
                "recall_at_10": summary.recall_at_10,
                "mean_reciprocal_rank": (
                    summary.mean_reciprocal_rank
                ),
                "ndcg_at_10": summary.ndcg_at_10,
            }
            for summary in self.summaries
        )


def run_baseline_experiment(
    dataset: SyntheticDataset,
    *,
    baselines: Iterable[BaselineKind] = tuple(BaselineKind),
    splits: Iterable[SplitLabel] = (
        SplitLabel.VALIDATION,
        SplitLabel.OOD_TEST,
    ),
    top_k: int = 10,
    random_seed: int = 2026,
    ridge_alpha: float = 1.0,
) -> BaselineExperimentResult:
    """Run every requested baseline on every requested split."""

    if not isinstance(dataset, SyntheticDataset):
        raise BaselineExperimentError(
            "dataset must be a SyntheticDataset."
        )

    selected_baselines = tuple(baselines)
    selected_splits = tuple(splits)

    if not selected_baselines:
        raise BaselineExperimentError(
            "At least one baseline must be selected."
        )

    if not selected_splits:
        raise BaselineExperimentError(
            "At least one split must be selected."
        )

    if any(
        not isinstance(baseline, BaselineKind)
        for baseline in selected_baselines
    ):
        raise BaselineExperimentError(
            "Every selected baseline must be a BaselineKind."
        )

    if any(
        not isinstance(split, SplitLabel)
        for split in selected_splits
    ):
        raise BaselineExperimentError(
            "Every selected split must be a SplitLabel."
        )

    if len(set(selected_baselines)) != len(selected_baselines):
        raise BaselineExperimentError(
            "Selected baselines cannot contain duplicates."
        )

    if len(set(selected_splits)) != len(selected_splits):
        raise BaselineExperimentError(
            "Selected splits cannot contain duplicates."
        )

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 10
    ):
        raise BaselineExperimentError(
            "top_k must be an integer of at least 10."
        )

    evaluations: list[BaselineEvaluation] = []
    summaries: list[BaselineMetricSummary] = []

    for split in selected_splits:
        for baseline in selected_baselines:
            evaluation = evaluate_baseline(
                dataset,
                baseline=baseline,
                split=split,
                top_k=top_k,
                random_seed=random_seed,
                ridge_alpha=ridge_alpha,
            )

            summary = summarize_evaluation(
                evaluation
            )

            evaluations.append(evaluation)
            summaries.append(summary)

    return BaselineExperimentResult(
        summaries=tuple(summaries),
        evaluations=tuple(evaluations),
        top_k=top_k,
        random_seed=random_seed,
        ridge_alpha=float(ridge_alpha),
    )


def summarize_evaluation(
    evaluation: BaselineEvaluation,
) -> BaselineMetricSummary:
    """Compute the locked aggregate metrics for one evaluation."""

    if not isinstance(evaluation, BaselineEvaluation):
        raise BaselineExperimentError(
            "evaluation must be a BaselineEvaluation."
        )

    return BaselineMetricSummary(
        baseline=evaluation.baseline,
        split=evaluation.split,
        event_count=len(evaluation.event_ids),
        recall_at_1=recall_at_k(
            evaluation.rankings,
            evaluation.relevant_items,
            k=1,
        ),
        recall_at_10=recall_at_k(
            evaluation.rankings,
            evaluation.relevant_items,
            k=10,
        ),
        mean_reciprocal_rank=mean_reciprocal_rank(
            evaluation.rankings,
            evaluation.relevant_items,
        ),
        ndcg_at_10=ndcg_at_k(
            evaluation.rankings,
            evaluation.relevant_items,
            k=10,
        ),
    )