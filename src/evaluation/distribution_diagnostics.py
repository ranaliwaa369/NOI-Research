"""Diagnostics for distribution shift in synthetic NOI evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from src.baselines.retrieval_baselines import (
    build_odor_library,
    mean_fuse_event,
)
from src.evaluation.retrieval_metrics import mean_reciprocal_rank
from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
)
from src.retrieval.cosine_retriever import CosineOdorRetriever


FloatArray = NDArray[np.float64]


class DistributionDiagnosticError(ValueError):
    """Raised when distribution diagnostics cannot be computed."""


@dataclass(frozen=True)
class DistributionShiftReport:
    """Summary of development-to-OOD distribution diagnostics."""

    validation_count: int
    ood_count: int
    feature_dimension: int
    mean_shift_l2: float
    covariance_shift_frobenius: float
    centroid_cosine_similarity: float
    rbf_mmd_squared: float
    ood_oracle_calibration_count: int
    ood_oracle_evaluation_count: int
    ood_oracle_mrr: float

    def __post_init__(self) -> None:
        if min(
            self.validation_count,
            self.ood_count,
            self.feature_dimension,
            self.ood_oracle_calibration_count,
            self.ood_oracle_evaluation_count,
        ) < 1:
            raise DistributionDiagnosticError(
                "Diagnostic counts and dimensions must be positive."
            )

        values = (
            self.mean_shift_l2,
            self.covariance_shift_frobenius,
            self.centroid_cosine_similarity,
            self.rbf_mmd_squared,
            self.ood_oracle_mrr,
        )

        if any(not isfinite(value) for value in values):
            raise DistributionDiagnosticError(
                "Diagnostic values must be finite."
            )

        if not -1.0 <= self.centroid_cosine_similarity <= 1.0:
            raise DistributionDiagnosticError(
                "Centroid cosine similarity must be in [-1, 1]."
            )

        if self.mean_shift_l2 < 0.0:
            raise DistributionDiagnosticError(
                "Mean shift cannot be negative."
            )

        if self.covariance_shift_frobenius < 0.0:
            raise DistributionDiagnosticError(
                "Covariance shift cannot be negative."
            )

        if self.rbf_mmd_squared < 0.0:
            raise DistributionDiagnosticError(
                "MMD squared cannot be negative."
            )

        if not 0.0 <= self.ood_oracle_mrr <= 1.0:
            raise DistributionDiagnosticError(
                "Oracle MRR must be in [0, 1]."
            )


def analyze_distribution_shift(
    dataset: SyntheticDataset,
    *,
    oracle_alpha: float = 1.0,
) -> DistributionShiftReport:
    """Compare validation and OOD features and run an OOD-only oracle.

    The oracle uses half of the OOD examples for diagnostic calibration and
    evaluates on the remaining half. It is not a deployable baseline and
    must not be interpreted as evidence of generalization.
    """

    if not isinstance(dataset, SyntheticDataset):
        raise DistributionDiagnosticError(
            "dataset must be a SyntheticDataset."
        )

    if (
        isinstance(oracle_alpha, bool)
        or not isinstance(oracle_alpha, (int, float))
        or not isfinite(float(oracle_alpha))
        or oracle_alpha < 0.0
    ):
        raise DistributionDiagnosticError(
            "oracle_alpha must be finite and nonnegative."
        )

    validation_events = tuple(
        sorted(
            (
                event for event in dataset.events
                if event.split is SplitLabel.VALIDATION
            ),
            key=lambda event: event.event_id,
        )
    )

    ood_events = tuple(
        sorted(
            (
                event for event in dataset.events
                if event.split is SplitLabel.OOD_TEST
            ),
            key=lambda event: event.event_id,
        )
    )

    if len(validation_events) < 2 or len(ood_events) < 4:
        raise DistributionDiagnosticError(
            "At least two validation and four OOD events are required."
        )

    validation_matrix = np.stack(
        [mean_fuse_event(event) for event in validation_events]
    )
    ood_matrix = np.stack(
        [mean_fuse_event(event) for event in ood_events]
    )

    if validation_matrix.shape[1] != ood_matrix.shape[1]:
        raise DistributionDiagnosticError(
            "Validation and OOD dimensions must match."
        )

    validation_mean = validation_matrix.mean(axis=0)
    ood_mean = ood_matrix.mean(axis=0)

    mean_shift = float(
        np.linalg.norm(validation_mean - ood_mean)
    )

    validation_covariance = np.cov(
        validation_matrix,
        rowvar=False,
    )
    ood_covariance = np.cov(
        ood_matrix,
        rowvar=False,
    )

    covariance_shift = float(
        np.linalg.norm(
            validation_covariance - ood_covariance,
            ord="fro",
        )
    )

    centroid_cosine = _cosine(
        validation_mean,
        ood_mean,
    )

    mmd_squared = _rbf_mmd_squared(
        validation_matrix,
        ood_matrix,
    )

    calibration_events = ood_events[::2]
    evaluation_events = ood_events[1::2]

    target_map = {
        target.item_id: np.asarray(
            target.odor_vector,
            dtype=np.float64,
        )
        for target in dataset.odor_targets
    }

    calibration_features = np.stack(
        [mean_fuse_event(event) for event in calibration_events]
    )
    calibration_targets = np.stack(
        [target_map[event.target_item_id] for event in calibration_events]
    )

    oracle = Ridge(
        alpha=float(oracle_alpha),
        fit_intercept=True,
    )
    oracle.fit(
        calibration_features,
        calibration_targets,
    )

    retriever = CosineOdorRetriever(
        build_odor_library(dataset)
    )

    rankings: list[tuple[str, ...]] = []
    relevant_items: list[frozenset[str]] = []

    for event in evaluation_events:
        query = mean_fuse_event(event)
        prediction = oracle.predict(
            query.reshape(1, -1)
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

    oracle_mrr = mean_reciprocal_rank(
        rankings,
        relevant_items,
    )

    return DistributionShiftReport(
        validation_count=len(validation_events),
        ood_count=len(ood_events),
        feature_dimension=validation_matrix.shape[1],
        mean_shift_l2=mean_shift,
        covariance_shift_frobenius=covariance_shift,
        centroid_cosine_similarity=centroid_cosine,
        rbf_mmd_squared=mmd_squared,
        ood_oracle_calibration_count=len(calibration_events),
        ood_oracle_evaluation_count=len(evaluation_events),
        ood_oracle_mrr=oracle_mrr,
    )


def _cosine(
    left: FloatArray,
    right: FloatArray,
) -> float:
    """Return cosine similarity with explicit zero-norm protection."""

    denominator = float(
        np.linalg.norm(left) * np.linalg.norm(right)
    )

    if denominator <= np.finfo(np.float64).eps:
        return 0.0

    return float(
        np.clip(
            np.dot(left, right) / denominator,
            -1.0,
            1.0,
        )
    )


def _rbf_mmd_squared(
    left: FloatArray,
    right: FloatArray,
) -> float:
    """Return biased RBF maximum mean discrepancy squared."""

    combined = np.vstack((left, right))
    differences = combined[:, None, :] - combined[None, :, :]
    squared_distances = np.sum(
        differences * differences,
        axis=2,
    )

    positive_distances = squared_distances[
        squared_distances > 0.0
    ]

    bandwidth_squared = (
        float(np.median(positive_distances))
        if positive_distances.size
        else 1.0
    )

    bandwidth_squared = max(
        bandwidth_squared,
        np.finfo(np.float64).eps,
    )

    left_kernel = np.exp(
        -np.sum(
            (left[:, None, :] - left[None, :, :]) ** 2,
            axis=2,
        )
        / (2.0 * bandwidth_squared)
    )

    right_kernel = np.exp(
        -np.sum(
            (right[:, None, :] - right[None, :, :]) ** 2,
            axis=2,
        )
        / (2.0 * bandwidth_squared)
    )

    cross_kernel = np.exp(
        -np.sum(
            (left[:, None, :] - right[None, :, :]) ** 2,
            axis=2,
        )
        / (2.0 * bandwidth_squared)
    )

    value = float(
        left_kernel.mean()
        + right_kernel.mean()
        - 2.0 * cross_kernel.mean()
    )

    return max(0.0, value)

