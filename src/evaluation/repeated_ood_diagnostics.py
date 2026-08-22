"""Repeated OOD-oracle diagnostics for synthetic NOI evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np
from sklearn.linear_model import Ridge

from src.baselines.retrieval_baselines import (
    build_odor_library,
    mean_fuse_event,
)
from src.evaluation.retrieval_metrics import mean_reciprocal_rank
from src.evaluation.synthetic_records import SplitLabel, SyntheticDataset
from src.retrieval.cosine_retriever import CosineOdorRetriever


class RepeatedOODDiagnosticError(ValueError):
    """Raised when repeated OOD diagnostics are invalid."""


@dataclass(frozen=True)
class OODOracleRun:
    """Result from one disjoint OOD calibration/evaluation split."""

    seed: int
    calibration_event_ids: tuple[str, ...]
    evaluation_event_ids: tuple[str, ...]
    mrr: float

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise RepeatedOODDiagnosticError(
                "seed must be nonnegative."
            )

        if not self.calibration_event_ids:
            raise RepeatedOODDiagnosticError(
                "Calibration events cannot be empty."
            )

        if not self.evaluation_event_ids:
            raise RepeatedOODDiagnosticError(
                "Evaluation events cannot be empty."
            )

        if (
            set(self.calibration_event_ids)
            & set(self.evaluation_event_ids)
        ):
            raise RepeatedOODDiagnosticError(
                "Calibration and evaluation events must be disjoint."
            )

        if not isfinite(self.mrr) or not 0.0 <= self.mrr <= 1.0:
            raise RepeatedOODDiagnosticError(
                "MRR must be finite and in [0, 1]."
            )


@dataclass(frozen=True)
class RepeatedOODReport:
    """Aggregate results across repeated OOD-oracle splits."""

    runs: tuple[OODOracleRun, ...]
    mean_mrr: float
    standard_deviation_mrr: float
    minimum_mrr: float
    maximum_mrr: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    bootstrap_seed: int
    bootstrap_resamples: int

    def __post_init__(self) -> None:
        if len(self.runs) < 2:
            raise RepeatedOODDiagnosticError(
                "At least two oracle runs are required."
            )

        values = (
            self.mean_mrr,
            self.standard_deviation_mrr,
            self.minimum_mrr,
            self.maximum_mrr,
            self.bootstrap_ci_lower,
            self.bootstrap_ci_upper,
        )

        if any(not isfinite(value) for value in values):
            raise RepeatedOODDiagnosticError(
                "Aggregate statistics must be finite."
            )

        if not (
            0.0
            <= self.bootstrap_ci_lower
            <= self.mean_mrr
            <= self.bootstrap_ci_upper
            <= 1.0
        ):
            raise RepeatedOODDiagnosticError(
                "Bootstrap interval must contain the mean."
            )


def run_repeated_ood_diagnostics(
    dataset: SyntheticDataset,
    *,
    split_seeds: Iterable[int] = range(10),
    oracle_alpha: float = 1.0,
    calibration_fraction: float = 0.5,
    bootstrap_seed: int = 4242,
    bootstrap_resamples: int = 10_000,
) -> RepeatedOODReport:
    """Repeat the labeled OOD oracle across deterministic disjoint splits."""

    if not isinstance(dataset, SyntheticDataset):
        raise RepeatedOODDiagnosticError(
            "dataset must be a SyntheticDataset."
        )

    seeds = tuple(split_seeds)

    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise RepeatedOODDiagnosticError(
            "At least two unique split seeds are required."
        )

    if any(
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        for seed in seeds
    ):
        raise RepeatedOODDiagnosticError(
            "Split seeds must be nonnegative integers."
        )

    if (
        not isfinite(float(oracle_alpha))
        or oracle_alpha < 0.0
    ):
        raise RepeatedOODDiagnosticError(
            "oracle_alpha must be finite and nonnegative."
        )

    if not 0.0 < calibration_fraction < 1.0:
        raise RepeatedOODDiagnosticError(
            "calibration_fraction must be between 0 and 1."
        )

    if bootstrap_resamples < 100:
        raise RepeatedOODDiagnosticError(
            "bootstrap_resamples must be at least 100."
        )

    events = tuple(
        sorted(
            (
                event
                for event in dataset.events
                if event.split is SplitLabel.OOD_TEST
            ),
            key=lambda event: event.event_id,
        )
    )

    if len(events) < 4:
        raise RepeatedOODDiagnosticError(
            "At least four OOD events are required."
        )

    calibration_count = round(
        len(events) * calibration_fraction
    )

    if not 1 <= calibration_count < len(events):
        raise RepeatedOODDiagnosticError(
            "Calibration split size is invalid."
        )

    target_map = {
        target.item_id: np.asarray(
            target.odor_vector,
            dtype=np.float64,
        )
        for target in dataset.odor_targets
    }

    retriever = CosineOdorRetriever(
        build_odor_library(dataset)
    )

    runs: list[OODOracleRun] = []

    for seed in seeds:
        generator = np.random.default_rng(seed)
        indices = generator.permutation(len(events))

        calibration_events = tuple(
            events[index]
            for index in indices[:calibration_count]
        )
        evaluation_events = tuple(
            events[index]
            for index in indices[calibration_count:]
        )

        features = np.stack(
            [mean_fuse_event(event) for event in calibration_events]
        )
        targets = np.stack(
            [target_map[event.target_item_id] for event in calibration_events]
        )

        oracle = Ridge(
            alpha=float(oracle_alpha),
            fit_intercept=True,
        )
        oracle.fit(features, targets)

        rankings = []
        relevant_items = []

        for event in evaluation_events:
            prediction = oracle.predict(
                mean_fuse_event(event).reshape(1, -1)
            )[0]

            candidates = retriever.retrieve(
                prediction,
                top_k=10,
            )

            rankings.append(
                tuple(candidate.item_id for candidate in candidates)
            )
            relevant_items.append(
                frozenset((event.target_item_id,))
            )

        runs.append(
            OODOracleRun(
                seed=seed,
                calibration_event_ids=tuple(
                    event.event_id for event in calibration_events
                ),
                evaluation_event_ids=tuple(
                    event.event_id for event in evaluation_events
                ),
                mrr=mean_reciprocal_rank(
                    rankings,
                    relevant_items,
                ),
            )
        )

    mrr_values = np.asarray(
        [run.mrr for run in runs],
        dtype=np.float64,
    )

    bootstrap_generator = np.random.default_rng(
        bootstrap_seed
    )

    bootstrap_means = np.mean(
        bootstrap_generator.choice(
            mrr_values,
            size=(bootstrap_resamples, len(mrr_values)),
            replace=True,
        ),
        axis=1,
    )

    lower, upper = np.quantile(
        bootstrap_means,
        (0.025, 0.975),
    )

    return RepeatedOODReport(
        runs=tuple(runs),
        mean_mrr=float(np.mean(mrr_values)),
        standard_deviation_mrr=float(
            np.std(mrr_values, ddof=1)
        ),
        minimum_mrr=float(np.min(mrr_values)),
        maximum_mrr=float(np.max(mrr_values)),
        bootstrap_ci_lower=float(lower),
        bootstrap_ci_upper=float(upper),
        bootstrap_seed=bootstrap_seed,
        bootstrap_resamples=bootstrap_resamples,
    )