"""One-at-a-time execution for repeated NOI Track A runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.seen_item_final_experiment import (
    SeenItemFinalExperiment,
    run_seen_item_final_experiment,
)
from src.evaluation.seen_item_partition import (
    SeenItemPartitionConfig,
)
from src.evaluation.seen_item_repeated_config import (
    RepeatedSeedSpec,
    SeenItemRepeatedConfig,
    SeenItemRepeatedConfigError,
)
from src.evaluation.synthetic_generator import (
    generate_synthetic_pilot_with_seeds,
)


@dataclass(frozen=True, slots=True)
class RepeatedSeedRunResult:
    """One completed prespecified repeated-run result."""

    run_spec: RepeatedSeedSpec
    experiment: SeenItemFinalExperiment | Any

    @property
    def run_id(self) -> str:
        return self.run_spec.run_id

    @property
    def generator_seed(self) -> int:
        return self.run_spec.generator_seed

    @property
    def ood_seed(self) -> int:
        return self.run_spec.ood_seed

    @property
    def partition_seed(self) -> int:
        return self.run_spec.partition_seed


def run_one_repeated_seed(
    *,
    run_spec: RepeatedSeedSpec,
    repeated_config: SeenItemRepeatedConfig,
    base_partition_config: SeenItemPartitionConfig,
    system_configuration: Mapping[str, Any],
    policy_configuration: Mapping[str, Any],
    protocol_hash: str,
    trained_at_utc: datetime,
    synthetic_configuration_path: str | Path,
    research_protocol_path: str | Path,
) -> RepeatedSeedRunResult:
    """Run one exact seed combination from the locked schedule."""

    if not isinstance(run_spec, RepeatedSeedSpec):
        raise SeenItemRepeatedConfigError(
            "run_spec must be a RepeatedSeedSpec."
        )

    if not isinstance(
        repeated_config,
        SeenItemRepeatedConfig,
    ):
        raise SeenItemRepeatedConfigError(
            "repeated_config has an invalid type."
        )

    if run_spec not in repeated_config.runs:
        raise SeenItemRepeatedConfigError(
            "run_spec is not a prespecified repeated run."
        )

    if not isinstance(
        base_partition_config,
        SeenItemPartitionConfig,
    ):
        raise SeenItemRepeatedConfigError(
            "base_partition_config has an invalid type."
        )

    partition_config = replace(
        base_partition_config,
        total_event_count=repeated_config.event_count,
        partition_seed=run_spec.partition_seed,
    )

    dataset = generate_synthetic_pilot_with_seeds(
        synthetic_configuration_path,
        research_protocol_path,
        event_count=repeated_config.event_count,
        generator_seed=run_spec.generator_seed,
        ood_seed=run_spec.ood_seed,
    )

    experiment = run_seen_item_final_experiment(
        dataset=dataset,
        partition_config=partition_config,
        system_configuration=system_configuration,
        policy_configuration=policy_configuration,
        protocol_hash=protocol_hash,
        trained_at_utc=trained_at_utc,
    )

    return RepeatedSeedRunResult(
        run_spec=run_spec,
        experiment=experiment,
    )
