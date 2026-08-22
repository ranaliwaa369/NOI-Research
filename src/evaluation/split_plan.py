"""Deterministic group-held-out split planning for NOI evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SplitPlanError(ValueError):
    """Raised when a leakage-resistant split cannot be constructed."""


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Auditable group assignments for train, validation, and OOD test."""

    training_families: tuple[int, ...]
    validation_families: tuple[int, ...]
    ood_test_families: tuple[int, ...]
    training_templates: tuple[int, ...]
    validation_templates: tuple[int, ...]
    ood_test_templates: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        family_development = set(self.training_families)
        family_validation = set(self.validation_families)
        family_ood = set(self.ood_test_families)

        if family_development != family_validation:
            raise SplitPlanError(
                "Training and validation must use the same development "
                "family pool for this prespecified design."
            )

        if family_development & family_ood:
            raise SplitPlanError(
                "OOD odor families must be disjoint from development."
            )

        template_sets = (
            set(self.training_templates),
            set(self.validation_templates),
            set(self.ood_test_templates),
        )

        if any(not values for values in template_sets):
            raise SplitPlanError(
                "Every split must contain at least one context template."
            )

        if (
            template_sets[0] & template_sets[1]
            or template_sets[0] & template_sets[2]
            or template_sets[1] & template_sets[2]
        ):
            raise SplitPlanError(
                "Context-template pools must be mutually disjoint."
            )


def create_split_plan(
    *,
    odor_families: int,
    context_templates: int,
    odor_family_held_out_fraction: float,
    context_template_held_out_fraction: float,
    validation_fraction: float,
    seed: int,
) -> SplitPlan:
    """Create deterministic group-level assignments without row splitting."""

    _validate_count("odor_families", odor_families, minimum=2)
    _validate_count("context_templates", context_templates, minimum=3)

    _validate_fraction(
        "odor_family_held_out_fraction",
        odor_family_held_out_fraction,
    )
    _validate_fraction(
        "context_template_held_out_fraction",
        context_template_held_out_fraction,
    )
    _validate_fraction(
        "validation_fraction",
        validation_fraction,
    )

    if (
        context_template_held_out_fraction + validation_fraction
        >= 1.0
    ):
        raise SplitPlanError(
            "Validation and OOD template fractions must leave "
            "templates for training."
        )

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitPlanError("seed must be an integer.")

    rng = np.random.default_rng(seed)

    family_ids = rng.permutation(odor_families)
    template_ids = rng.permutation(context_templates)

    ood_family_count = _fraction_to_count(
        odor_families,
        odor_family_held_out_fraction,
    )
    ood_template_count = _fraction_to_count(
        context_templates,
        context_template_held_out_fraction,
    )
    validation_template_count = _fraction_to_count(
        context_templates,
        validation_fraction,
    )

    ood_families = tuple(
        sorted(int(value) for value in family_ids[:ood_family_count])
    )
    development_families = tuple(
        sorted(int(value) for value in family_ids[ood_family_count:])
    )

    ood_templates = tuple(
        sorted(int(value) for value in template_ids[:ood_template_count])
    )

    validation_start = ood_template_count
    validation_end = validation_start + validation_template_count

    validation_templates = tuple(
        sorted(
            int(value)
            for value in template_ids[
                validation_start:validation_end
            ]
        )
    )

    training_templates = tuple(
        sorted(
            int(value)
            for value in template_ids[validation_end:]
        )
    )

    return SplitPlan(
        training_families=development_families,
        validation_families=development_families,
        ood_test_families=ood_families,
        training_templates=training_templates,
        validation_templates=validation_templates,
        ood_test_templates=ood_templates,
        seed=seed,
    )


def _validate_count(
    name: str,
    value: int,
    *,
    minimum: int,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise SplitPlanError(
            f"{name} must be an integer of at least {minimum}."
        )


def _validate_fraction(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 < float(value) < 1.0
    ):
        raise SplitPlanError(
            f"{name} must be strictly between 0 and 1."
        )


def _fraction_to_count(total: int, fraction: float) -> int:
    count = int(round(total * float(fraction)))

    if count < 1 or count >= total:
        raise SplitPlanError(
            "The requested fraction creates an empty group."
        )

    return count