"""Tests for deterministic leakage-resistant split planning."""

import pytest

from src.evaluation.split_plan import (
    SplitPlanError,
    create_split_plan,
)


def make_plan(seed: int = 1001):
    return create_split_plan(
        odor_families=20,
        context_templates=100,
        odor_family_held_out_fraction=0.30,
        context_template_held_out_fraction=0.20,
        validation_fraction=0.10,
        seed=seed,
    )


def test_prespecified_group_counts() -> None:
    plan = make_plan()

    assert len(plan.training_families) == 14
    assert len(plan.validation_families) == 14
    assert len(plan.ood_test_families) == 6
    assert len(plan.training_templates) == 70
    assert len(plan.validation_templates) == 10
    assert len(plan.ood_test_templates) == 20


def test_ood_families_are_disjoint() -> None:
    plan = make_plan()

    assert not (
        set(plan.training_families)
        & set(plan.ood_test_families)
    )


def test_template_pools_are_mutually_disjoint() -> None:
    plan = make_plan()

    training = set(plan.training_templates)
    validation = set(plan.validation_templates)
    ood = set(plan.ood_test_templates)

    assert not training & validation
    assert not training & ood
    assert not validation & ood


def test_all_identifiers_are_assigned_once() -> None:
    plan = make_plan()

    family_union = (
        set(plan.training_families)
        | set(plan.ood_test_families)
    )
    template_union = (
        set(plan.training_templates)
        | set(plan.validation_templates)
        | set(plan.ood_test_templates)
    )

    assert family_union == set(range(20))
    assert template_union == set(range(100))


def test_same_seed_produces_same_plan() -> None:
    first = make_plan(seed=1001)
    second = make_plan(seed=1001)

    assert first == second


def test_different_seed_changes_plan() -> None:
    first = make_plan(seed=1001)
    second = make_plan(seed=1002)

    assert first != second


def test_invalid_fraction_is_rejected() -> None:
    with pytest.raises(
        SplitPlanError,
        match="strictly between 0 and 1",
    ):
        create_split_plan(
            odor_families=20,
            context_templates=100,
            odor_family_held_out_fraction=0.0,
            context_template_held_out_fraction=0.20,
            validation_fraction=0.10,
            seed=1001,
        )


def test_validation_and_ood_must_leave_training_templates() -> None:
    with pytest.raises(
        SplitPlanError,
        match="leave templates for training",
    ):
        create_split_plan(
            odor_families=20,
            context_templates=100,
            odor_family_held_out_fraction=0.30,
            context_template_held_out_fraction=0.60,
            validation_fraction=0.40,
            seed=1001,
        )