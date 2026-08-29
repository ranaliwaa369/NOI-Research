"""Seedwise validation-lock derivation for NOI v0.3."""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.evaluation.evidence_conflict import (
    ConflictCalibrationObservation,
    EvidenceConflictDetector,
    ReliabilityCalibrationObservation,
    calibrate_conflict_threshold,
    calibrate_reliability_threshold,
)
from src.evaluation.multisensory_conditions import (
    ConditionGenerationConfig,
    generate_multisensory_condition_views,
)
from src.evaluation.multisensory_records import (
    ConditionLabel,
    LatentMultisensoryEvent,
    MultisensorySplit,
    MultisensoryTarget,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationResult,
)
from src.evaluation.support_gate import (
    SupportGate,
    SupportMethod,
)


class ValidationLockError(ValueError):
    """Raised when seedwise lock isolation is violated."""


@dataclass(frozen=True, slots=True)
class SeedValidationInput:
    """Training and validation records exposed to one seed lock."""

    seed: int
    training_events: tuple[LatentMultisensoryEvent, ...]
    validation_events: tuple[LatentMultisensoryEvent, ...]
    permitted_targets: tuple[MultisensoryTarget, ...]
    condition_config: ConditionGenerationConfig
    final_test_event_count_exposed: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValidationLockError(
                "seed must be a nonnegative integer."
            )

        if not self.training_events or any(
            event.split is not MultisensorySplit.TRAIN
            for event in self.training_events
        ):
            raise ValidationLockError(
                "training_events must contain training data only."
            )

        if not self.validation_events or any(
            event.split is not MultisensorySplit.VALIDATION
            for event in self.validation_events
        ):
            raise ValidationLockError(
                "validation_events must contain validation data only."
            )

        if (
            not self.permitted_targets
            or self.final_test_event_count_exposed != 0
        ):
            raise ValidationLockError(
                "Final-test records or targets cannot be exposed "
                "to validation locking."
            )

        permitted_ids = {
            target.item_id
            for target in self.permitted_targets
        }
        required_ids = {
            event.target_item_id
            for event in (
                self.training_events
                + self.validation_events
            )
        }

        if permitted_ids != required_ids:
            raise ValidationLockError(
                "permitted_targets must exactly match train and "
                "validation event targets."
            )

        if self.condition_config.seed != self.seed:
            raise ValidationLockError(
                "Condition and generation seeds must match."
            )


@dataclass(frozen=True, slots=True)
class SeedValidationLock:
    """Five locked values and their validation-only audit."""

    seed: int
    support_threshold: float
    support_uncertainty_lower: float
    support_uncertainty_upper: float
    reliability_threshold: float
    conflict_threshold: float
    validation_false_known_rate: float
    validation_false_conflict_rate: float
    support_validation_event_count: int
    reliability_validation_observation_count: int
    conflict_validation_observation_count: int
    odor_noise_scale: float
    tactile_noise_scale: float
    temporal_offset_steps: int
    final_test_events_used: int
    final_test_labels_used: bool
    condition_metadata_used_as_model_input: bool
    target_labels_used_as_inference_input: bool
    quality_metadata_used_as_model_input: bool


def build_seed_validation_input(
    *,
    generated: NOIV03GenerationResult,
    condition_config: ConditionGenerationConfig,
) -> SeedValidationInput:
    """Expose only one seed's training and validation records."""

    if not isinstance(generated, NOIV03GenerationResult):
        raise ValidationLockError(
            "generated must be an NOIV03GenerationResult."
        )

    if not isinstance(
        condition_config,
        ConditionGenerationConfig,
    ):
        raise ValidationLockError(
            "condition_config must be a "
            "ConditionGenerationConfig."
        )

    training_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.TRAIN
    )
    validation_events = tuple(
        event
        for event in generated.latent_events
        if event.split is MultisensorySplit.VALIDATION
    )

    development_target_ids = {
        event.target_item_id
        for event in training_events + validation_events
    }
    permitted_targets = tuple(
        target
        for target in generated.targets
        if target.item_id in development_target_ids
    )

    seed = generated.provenance.seed

    return SeedValidationInput(
        seed=seed,
        training_events=training_events,
        validation_events=validation_events,
        permitted_targets=permitted_targets,
        condition_config=condition_config,
        final_test_event_count_exposed=0,
    )


def _predicted_family(
    family_ids: tuple[int, ...],
    distribution: tuple[float, ...],
) -> int:
    """Return deterministic maximum-probability family."""

    if (
        not family_ids
        or len(family_ids) != len(distribution)
    ):
        raise ValidationLockError(
            "Family distribution is incomplete."
        )

    index = max(
        range(len(distribution)),
        key=lambda item: (
            distribution[item],
            -family_ids[item],
        ),
    )

    return family_ids[index]


def derive_seed_validation_lock(
    *,
    lock_input: SeedValidationInput,
    support_bootstrap_seed: int,
    support_bootstrap_resamples: int,
    confidence_level: float,
    maximum_false_known_rate: float,
    maximum_false_conflict_rate: float,
) -> SeedValidationLock:
    """Derive all five policy values without final-test access."""

    if not isinstance(lock_input, SeedValidationInput):
        raise ValidationLockError(
            "lock_input must be a SeedValidationInput."
        )

    support_gate = SupportGate(
        method=SupportMethod.MAHALANOBIS,
    )
    support_gate.fit(lock_input.training_events)
    support_report = support_gate.calibrate_for_lock(
        lock_input.validation_events,
        maximum_false_known_rate=(
            maximum_false_known_rate
        ),
        bootstrap_seed=support_bootstrap_seed,
        bootstrap_resamples=support_bootstrap_resamples,
        confidence_level=confidence_level,
    )

    detector = EvidenceConflictDetector()
    detector.fit(lock_input.training_events)

    condition_result = generate_multisensory_condition_views(
        latent_events=lock_input.validation_events,
        targets=lock_input.permitted_targets,
        config=lock_input.condition_config,
    )

    validation_lookup = {
        event.latent_event_id: event
        for event in lock_input.validation_events
    }

    reliability_observations: list[
        ReliabilityCalibrationObservation
    ] = []
    conflict_observations: list[
        ConflictCalibrationObservation
    ] = []

    for view in condition_result.views:
        event = validation_lookup.get(
            view.latent_event_id
        )

        if event is None:
            raise ValidationLockError(
                "A condition view is not linked to validation."
            )

        evidence = detector.assess(
            olfactory_vector=view.olfactory_vector,
            tactile_vector=view.tactile_vector,
        )

        if evidence.odor_available:
            predicted = _predicted_family(
                evidence.family_ids,
                evidence.odor_family_distribution,
            )
            reliability_observations.append(
                ReliabilityCalibrationObservation(
                    source_split=MultisensorySplit.VALIDATION,
                    reliability=evidence.odor_reliability,
                    prediction_correct=(
                        predicted
                        == event.target_family_id
                    ),
                )
            )

        if evidence.touch_available:
            predicted = _predicted_family(
                evidence.family_ids,
                evidence.touch_family_distribution,
            )
            reliability_observations.append(
                ReliabilityCalibrationObservation(
                    source_split=MultisensorySplit.VALIDATION,
                    reliability=evidence.touch_reliability,
                    prediction_correct=(
                        predicted
                        == event.target_family_id
                    ),
                )
            )

        if evidence.conflict_available:
            conflict_observations.append(
                ConflictCalibrationObservation(
                    source_split=MultisensorySplit.VALIDATION,
                    conflict_score=evidence.conflict_score,
                    conflict_present=(
                        view.condition
                        is ConditionLabel.CONTRADICTORY_MODALITIES
                    ),
                )
            )

    reliability_report = (
        calibrate_reliability_threshold(
            tuple(reliability_observations)
        )
    )
    conflict_report = calibrate_conflict_threshold(
        tuple(conflict_observations),
        maximum_false_conflict_rate=(
            maximum_false_conflict_rate
        ),
    )

    values = (
        support_report.threshold,
        support_report.uncertainty_lower,
        support_report.uncertainty_upper,
        reliability_report.threshold,
        conflict_report.threshold,
    )

    if not all(math.isfinite(value) for value in values):
        raise ValidationLockError(
            "Every locked value must be finite."
        )

    return SeedValidationLock(
        seed=lock_input.seed,
        support_threshold=support_report.threshold,
        support_uncertainty_lower=(
            support_report.uncertainty_lower
        ),
        support_uncertainty_upper=(
            support_report.uncertainty_upper
        ),
        reliability_threshold=(
            reliability_report.threshold
        ),
        conflict_threshold=conflict_report.threshold,
        validation_false_known_rate=(
            support_report.validation_false_known_rate
        ),
        validation_false_conflict_rate=(
            conflict_report.validation_false_conflict_rate
        ),
        support_validation_event_count=(
            support_report.validation_event_count
        ),
        reliability_validation_observation_count=(
            reliability_report.validation_observation_count
        ),
        conflict_validation_observation_count=(
            conflict_report.validation_observation_count
        ),
        odor_noise_scale=(
            lock_input.condition_config.odor_noise_scale
        ),
        tactile_noise_scale=(
            lock_input.condition_config.tactile_noise_scale
        ),
        temporal_offset_steps=(
            lock_input.condition_config
            .locked_temporal_offset_steps
        ),
        final_test_events_used=0,
        final_test_labels_used=False,
        condition_metadata_used_as_model_input=False,
        target_labels_used_as_inference_input=False,
        quality_metadata_used_as_model_input=False,
    )
