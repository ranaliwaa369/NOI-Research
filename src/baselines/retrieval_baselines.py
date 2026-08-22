"""Prespecified retrieval baselines for synthetic NOI evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from src.evaluation.synthetic_records import (
    SplitLabel,
    SyntheticDataset,
    SyntheticEvent,
)
from src.retrieval.cosine_retriever import (
    CosineOdorRetriever,
    OdorLibraryItem,
)


FloatArray = NDArray[np.float64]
Ranking = tuple[str, ...]
RelevantItems = frozenset[str]


class BaselineError(ValueError):
    """Raised when a retrieval baseline cannot be evaluated fairly."""


class BaselineKind(str, Enum):
    """Prespecified baseline systems."""

    RANDOM = "random"
    TEXT_ONLY_COSINE = "text_only_cosine"
    MEAN_FUSION_COSINE = "mean_fusion_cosine"
    RIDGE_FUSION = "ridge_fusion"


@dataclass(frozen=True)
class BaselineEvaluation:
    """Rankings and ground truth for one baseline and split."""

    baseline: BaselineKind
    split: SplitLabel
    event_ids: tuple[str, ...]
    rankings: tuple[Ranking, ...]
    relevant_items: tuple[RelevantItems, ...]
    top_k: int
    training_event_count: int

    def __post_init__(self) -> None:
        count = len(self.event_ids)

        if len(self.rankings) != count:
            raise BaselineError(
                "Every event must have exactly one ranking."
            )

        if len(self.relevant_items) != count:
            raise BaselineError(
                "Every event must have exactly one relevance set."
            )

        if len(set(self.event_ids)) != count:
            raise BaselineError(
                "Evaluation event identifiers must be unique."
            )

        if self.top_k < 1:
            raise BaselineError(
                "top_k must be at least 1."
            )

        if any(
            len(ranking) > self.top_k
            for ranking in self.rankings
        ):
            raise BaselineError(
                "A ranking cannot exceed top_k."
            )

        if any(
            not relevant
            for relevant in self.relevant_items
        ):
            raise BaselineError(
                "Every relevance set must contain at least one item."
            )


class RidgeFusionRetriever:
    """Strong learned linear baseline fitted on training events only."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
    ) -> None:
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not isfinite(float(alpha))
            or float(alpha) < 0.0
        ):
            raise BaselineError(
                "alpha must be finite and nonnegative."
            )

        self._alpha = float(alpha)
        self._model: Ridge | None = None
        self._retriever: CosineOdorRetriever | None = None
        self._input_dimension: int | None = None
        self._training_event_count = 0

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def training_event_count(self) -> int:
        return self._training_event_count

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self,
        dataset: SyntheticDataset,
    ) -> "RidgeFusionRetriever":
        """Fit the linear mapping using training events only."""

        if not isinstance(dataset, SyntheticDataset):
            raise BaselineError(
                "dataset must be a SyntheticDataset."
            )

        training_events = tuple(
            sorted(
                (
                    event
                    for event in dataset.events
                    if event.split is SplitLabel.TRAIN
                ),
                key=lambda event: event.event_id,
            )
        )

        if not training_events:
            raise BaselineError(
                "At least one training event is required."
            )

        target_map = {
            target.item_id: np.asarray(
                target.odor_vector,
                dtype=np.float64,
            )
            for target in dataset.odor_targets
        }

        missing_targets = tuple(
            sorted(
                {
                    event.target_item_id
                    for event in training_events
                    if event.target_item_id not in target_map
                }
            )
        )

        if missing_targets:
            raise BaselineError(
                "Training targets are absent from the odor library: "
                f"{list(missing_targets)}"
            )

        input_vectors = np.stack(
            [
                mean_fuse_event(event)
                for event in training_events
            ],
            axis=0,
        )

        target_vectors = np.stack(
            [
                target_map[event.target_item_id]
                for event in training_events
            ],
            axis=0,
        )

        if target_vectors.ndim != 2:
            raise BaselineError(
                "Target odor vectors must form a two-dimensional matrix."
            )

        if not np.all(np.isfinite(target_vectors)):
            raise BaselineError(
                "Target odor vectors must contain only finite values."
            )

        model = Ridge(
            alpha=self._alpha,
            fit_intercept=True,
        )
        model.fit(input_vectors, target_vectors)

        self._model = model
        self._retriever = CosineOdorRetriever(
            build_odor_library(dataset)
        )
        self._input_dimension = input_vectors.shape[1]
        self._training_event_count = len(training_events)

        return self

    def retrieve(
        self,
        event: SyntheticEvent,
        *,
        top_k: int = 10,
    ) -> Ranking:
        """Predict an odor vector and rank the fixed odor library."""

        if self._model is None or self._retriever is None:
            raise BaselineError(
                "RidgeFusionRetriever must be fitted before retrieval."
            )

        query = mean_fuse_event(event)

        if query.shape[0] != self._input_dimension:
            raise BaselineError(
                "Event dimension differs from the fitted dimension."
            )

        predicted = np.asarray(
            self._model.predict(
                query.reshape(1, -1)
            )[0],
            dtype=np.float64,
        )

        predicted = _normalize_vector(
            predicted,
            label="predicted odor vector",
        )

        candidates = self._retriever.retrieve(
            predicted,
            top_k=top_k,
        )

        return tuple(
            candidate.item_id
            for candidate in candidates
        )


def build_odor_library(
    dataset: SyntheticDataset,
) -> tuple[OdorLibraryItem, ...]:
    """Convert synthetic targets into the fixed retrieval library."""

    if not isinstance(dataset, SyntheticDataset):
        raise BaselineError(
            "dataset must be a SyntheticDataset."
        )

    if not dataset.odor_targets:
        raise BaselineError(
            "The odor library cannot be empty."
        )

    return tuple(
        OdorLibraryItem(
            item_id=target.item_id,
            odor_vector=target.odor_vector,
            descriptors=("synthetic-reference-only",),
            cartridge_id=None,
            source_reference="synthetic-pilot",
            odor_family=str(target.family_id),
        )
        for target in sorted(
            dataset.odor_targets,
            key=lambda target: target.item_id,
        )
    )


def mean_fuse_event(
    event: SyntheticEvent,
) -> FloatArray:
    """Average text, image, and audio vectors and L2-normalize."""

    if not isinstance(event, SyntheticEvent):
        raise BaselineError(
            "event must be a SyntheticEvent."
        )

    vectors = (
        np.asarray(
            event.text_vector,
            dtype=np.float64,
        ),
        np.asarray(
            event.image_vector,
            dtype=np.float64,
        ),
        np.asarray(
            event.audio_vector,
            dtype=np.float64,
        ),
    )

    shapes = {
        vector.shape
        for vector in vectors
    }

    if len(shapes) != 1:
        raise BaselineError(
            "All event modalities must have identical dimensions."
        )

    if any(
        vector.ndim != 1
        for vector in vectors
    ):
        raise BaselineError(
            "Every event modality must be one-dimensional."
        )

    if any(
        not np.all(np.isfinite(vector))
        for vector in vectors
    ):
        raise BaselineError(
            "Event modalities must contain only finite values."
        )

    fused = np.mean(
        np.stack(vectors, axis=0),
        axis=0,
    )

    return _normalize_vector(
        fused,
        label="mean-fused context vector",
    )


def evaluate_baseline(
    dataset: SyntheticDataset,
    *,
    baseline: BaselineKind,
    split: SplitLabel,
    top_k: int = 10,
    random_seed: int = 2026,
    ridge_alpha: float = 1.0,
) -> BaselineEvaluation:
    """Evaluate one prespecified baseline on one locked split."""

    if not isinstance(dataset, SyntheticDataset):
        raise BaselineError(
            "dataset must be a SyntheticDataset."
        )

    if not isinstance(baseline, BaselineKind):
        raise BaselineError(
            "baseline must be a BaselineKind."
        )

    if not isinstance(split, SplitLabel):
        raise BaselineError(
            "split must be a SplitLabel."
        )

    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
    ):
        raise BaselineError(
            "top_k must be a positive integer."
        )

    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise BaselineError(
            "random_seed must be a nonnegative integer."
        )

    library = build_odor_library(dataset)

    if top_k > len(library):
        raise BaselineError(
            "top_k cannot exceed the odor-library size."
        )

    evaluation_events = tuple(
        sorted(
            (
                event
                for event in dataset.events
                if event.split is split
            ),
            key=lambda event: event.event_id,
        )
    )

    if not evaluation_events:
        raise BaselineError(
            f"No events are available for split {split.value}."
        )

    cosine_retriever = CosineOdorRetriever(
        library
    )

    ridge_retriever: RidgeFusionRetriever | None = None

    if baseline is BaselineKind.RIDGE_FUSION:
        ridge_retriever = RidgeFusionRetriever(
            alpha=ridge_alpha
        ).fit(dataset)

    rankings: list[Ranking] = []
    relevant_items: list[RelevantItems] = []

    for event in evaluation_events:
        if baseline is BaselineKind.RANDOM:
            ranking = _random_ranking(
                library,
                event_id=event.event_id,
                top_k=top_k,
                random_seed=random_seed,
            )

        elif baseline is BaselineKind.TEXT_ONLY_COSINE:
            candidates = cosine_retriever.retrieve(
                event.text_vector,
                top_k=top_k,
            )

            ranking = tuple(
                candidate.item_id
                for candidate in candidates
            )

        elif baseline is BaselineKind.MEAN_FUSION_COSINE:
            candidates = cosine_retriever.retrieve(
                mean_fuse_event(event),
                top_k=top_k,
            )

            ranking = tuple(
                candidate.item_id
                for candidate in candidates
            )

        elif baseline is BaselineKind.RIDGE_FUSION:
            if ridge_retriever is None:
                raise BaselineError(
                    "The ridge baseline was not initialized."
                )

            ranking = ridge_retriever.retrieve(
                event,
                top_k=top_k,
            )

        else:
            raise BaselineError(
                f"Unsupported baseline: {baseline}"
            )

        rankings.append(ranking)
        relevant_items.append(
            frozenset(
                (event.target_item_id,)
            )
        )

    training_event_count = sum(
        event.split is SplitLabel.TRAIN
        for event in dataset.events
    )

    return BaselineEvaluation(
        baseline=baseline,
        split=split,
        event_ids=tuple(
            event.event_id
            for event in evaluation_events
        ),
        rankings=tuple(rankings),
        relevant_items=tuple(relevant_items),
        top_k=top_k,
        training_event_count=training_event_count,
    )


def _random_ranking(
    library: tuple[OdorLibraryItem, ...],
    *,
    event_id: str,
    top_k: int,
    random_seed: int,
) -> Ranking:
    """Create an event-stable random ranking."""

    event_digest = sha256(
        event_id.encode("utf-8")
    ).digest()

    event_seed = int.from_bytes(
        event_digest[:8],
        byteorder="big",
        signed=False,
    )

    combined_seed = (
        event_seed ^ random_seed
    ) % (2**63 - 1)

    generator = np.random.default_rng(
        combined_seed
    )

    item_ids = np.asarray(
        [
            item.item_id
            for item in library
        ],
        dtype=object,
    )

    permutation = generator.permutation(
        item_ids
    )

    return tuple(
        str(item_id)
        for item_id in permutation[:top_k]
    )


def _normalize_vector(
    vector: FloatArray,
    *,
    label: str,
) -> FloatArray:
    """Return a finite unit vector."""

    if vector.ndim != 1:
        raise BaselineError(
            f"{label} must be one-dimensional."
        )

    if not np.all(np.isfinite(vector)):
        raise BaselineError(
            f"{label} must contain only finite values."
        )

    norm = float(
        np.linalg.norm(vector)
    )

    if (
        not np.isfinite(norm)
        or norm <= np.finfo(np.float64).eps
    ):
        raise BaselineError(
            f"{label} must have a nonzero finite norm."
        )

    return vector / norm