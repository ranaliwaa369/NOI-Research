"""Execute the registered seedwise NOI v0.3 validation lock."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from src.evaluation.multisensory_conditions import (
    ConditionGenerationConfig,
)
from src.evaluation.noi_v0_3_generator import (
    NOIV03GenerationConfig,
    generate_noi_v0_3_events,
)
from src.evaluation.noi_v0_3_validation_lock import (
    build_seed_validation_input,
    derive_seed_validation_lock,
)


REGISTERED_SEEDS = tuple(range(1301, 1311))
BOOTSTRAP_SEED = 4242
BOOTSTRAP_RESAMPLES = 10000
CONFIDENCE_LEVEL = 0.95
MAXIMUM_FALSE_KNOWN_RATE = 0.05
MAXIMUM_FALSE_CONFLICT_RATE = 0.05


class ValidationLockExecutionError(ValueError):
    """Raised when full validation locking cannot be completed."""


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one versioned input."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    """Return the exact implementation commit."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _generation_config(seed: int) -> NOIV03GenerationConfig:
    """Return the registered full allocation for one seed."""

    return NOIV03GenerationConfig(
        seed=seed,
        train_event_count=7000,
        validation_event_count=1000,
        final_test_event_count=2000,
        validation_seen_item_count=400,
        validation_known_family_unseen_item_count=300,
        validation_unseen_family_count=300,
        final_seen_item_count=800,
        final_known_family_unseen_item_count=600,
        final_unseen_family_count=600,
        known_family_count=4,
        training_items_per_family=4,
        withheld_items_per_known_family=2,
        validation_unknown_family_count=2,
        final_unknown_family_count=2,
        items_per_unknown_family=3,
        generator_version="0.3.1-validation-lock",
        feasibility_only=False,
    )


def _condition_config(seed: int) -> ConditionGenerationConfig:
    """Return registered validation condition controls."""

    return ConditionGenerationConfig(
        seed=seed,
        odor_noise_scale=0.10,
        tactile_noise_scale=0.10,
        degraded_quality=0.40,
        locked_temporal_offset_steps=3,
        generator_version="0.3.1-validation-lock",
    )


def _base_payload() -> dict[str, Any]:
    """Return deterministic provenance before seed execution."""

    protocol_path = Path(
        "configs/noi_v0.3_protocol.yaml"
    )
    amendment_path = Path(
        "configs/protocol_amendment_v0.3.yaml"
    )

    return {
        "schema_version": "noi-v0.3-validation-lock-v1",
        "study_phase": "validation_lock",
        "confirmatory_evaluation_executed": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256_before_lock": _sha256(
            protocol_path
        ),
        "amendment_path": str(amendment_path),
        "amendment_sha256": _sha256(amendment_path),
        "implementation_commit": _git_commit(),
        "registered_seeds": list(REGISTERED_SEEDS),
        "completed_seeds": [],
        "complete": False,
        "controls": {
            "support_method": "mahalanobis",
            "support_bootstrap_seed": BOOTSTRAP_SEED,
            "support_bootstrap_resamples": (
                BOOTSTRAP_RESAMPLES
            ),
            "confidence_level": CONFIDENCE_LEVEL,
            "maximum_false_known_rate": (
                MAXIMUM_FALSE_KNOWN_RATE
            ),
            "maximum_false_conflict_rate": (
                MAXIMUM_FALSE_CONFLICT_RATE
            ),
            "threshold_storage": "values_by_seed",
            "cross_seed_pooling_used": False,
        },
        "allocation_per_seed": {
            "training_events": 7000,
            "validation_events": 1000,
            "final_test_events_generated_but_not_exposed": 2000,
            "validation_seen_item": 400,
            "validation_known_family_unseen_item": 300,
            "validation_unseen_family": 300,
        },
        "values_by_seed": {},
        "integrity": {
            "final_test_events_used": 0,
            "final_test_labels_used": False,
            "condition_metadata_used_as_model_input": False,
            "target_labels_used_as_inference_input": False,
            "quality_metadata_used_as_model_input": False,
            "final_test_threshold_feedback_used": False,
        },
    }


def _write_payload(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """Write canonical deterministic JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def run_validation_lock(
    output_path: Path,
) -> dict[str, Any]:
    """Derive and checkpoint all ten registered seed locks."""

    if not isinstance(output_path, Path):
        raise ValidationLockExecutionError(
            "output_path must be a Path."
        )

    payload = _base_payload()
    _write_payload(payload, output_path)

    for position, seed in enumerate(
        REGISTERED_SEEDS,
        start=1,
    ):
        print(
            f"[{position}/10] deriving validation lock "
            f"for seed {seed}",
            flush=True,
        )

        generated = generate_noi_v0_3_events(
            _generation_config(seed)
        )
        lock_input = build_seed_validation_input(
            generated=generated,
            condition_config=_condition_config(seed),
        )
        result = derive_seed_validation_lock(
            lock_input=lock_input,
            support_bootstrap_seed=BOOTSTRAP_SEED,
            support_bootstrap_resamples=(
                BOOTSTRAP_RESAMPLES
            ),
            confidence_level=CONFIDENCE_LEVEL,
            maximum_false_known_rate=(
                MAXIMUM_FALSE_KNOWN_RATE
            ),
            maximum_false_conflict_rate=(
                MAXIMUM_FALSE_CONFLICT_RATE
            ),
        )

        result_payload = asdict(result)

        if result_payload["final_test_events_used"] != 0:
            raise ValidationLockExecutionError(
                "Final-test events entered validation locking."
            )

        if result_payload["final_test_labels_used"]:
            raise ValidationLockExecutionError(
                "Final-test labels entered validation locking."
            )

        if (
            result_payload[
                "validation_false_known_rate"
            ]
            > MAXIMUM_FALSE_KNOWN_RATE
        ):
            raise ValidationLockExecutionError(
                "The support safety constraint was violated."
            )

        if (
            result_payload[
                "validation_false_conflict_rate"
            ]
            > MAXIMUM_FALSE_CONFLICT_RATE
        ):
            raise ValidationLockExecutionError(
                "The conflict safety constraint was violated."
            )

        payload["values_by_seed"][str(seed)] = (
            result_payload
        )
        payload["completed_seeds"].append(seed)
        _write_payload(payload, output_path)

        print(
            f"[{position}/10] seed {seed} locked",
            flush=True,
        )

    payload["complete"] = True
    _write_payload(payload, output_path)

    return payload


def main() -> None:
    """Run the registered validation lock from the CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Derive the registered NOI v0.3 seedwise "
            "validation locks."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/v0.3-validation-lock/"
            "validation_lock.json"
        ),
    )
    arguments = parser.parse_args()

    payload = run_validation_lock(arguments.output)

    print("VALIDATION LOCK: PASS")
    print("COMPLETED SEEDS:", payload["completed_seeds"])
    print("FINAL TEST EVENTS USED: 0")
    print("FINAL TEST LABELS USED: False")
    print("WROTE:", arguments.output)


if __name__ == "__main__":
    main()
