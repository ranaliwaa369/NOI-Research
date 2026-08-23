# Corrective Memory Pilot v0.1 Provenance

## Ownership

- Project: Neuro-Olfactive Intelligence
- Owner: GUARDIANX LLC
- Authors: Rana Al-Dahlake and Jalal Alazirji

## Release identity

- Release: `corrective-memory-pilot-v0.1`
- Status: Exploratory controlled synthetic mechanism evaluation
- Preregistration tag: `corrective-memory-v0.1-preimplementation`
- Preregistration commit: `483bd344611d200694b3fd5741487f7b1b6a30d4`
- Configuration validator commit: `0ab7ec5`
- Experiment implementation commit: `fda15d2`
- Reproducible workflow commit: `2e9d269`

The experimental design and all result-affecting settings were committed and
tagged before implementation or inspection of corrective-memory results.

## Locked eligibility

The source dataset contained:

- 140 training events
- 20 validation events
- 200 odor-library targets

Eligibility required a validation target to already be represented in the
training associative memory.

Observed prespecified eligibility counts:

- 15 eligible validation queries
- 14 eligible known targets
- 5 excluded validation targets not represented in training memory

All eligible targets were used. No target was selected or removed based on
retrieval performance.

## Controlled paired intervention

For every eligible target:

1. All training-memory records associated with that target were identified.
2. The stored target identity was changed to a deterministic decoy in both
   experimental arms.
3. The `no_update` arm retained the corrupted association.
4. The `corrected` arm restored every corrupted record to its true target.
5. Every restoration produced a correction audit record.
6. The same validation queries were evaluated in both paired arms.

The deterministic decoy was the next lexicographically sorted
training-represented target identifier, with cyclic wraparound.

## Locked retrieval condition

- Primary system: memory-only
- Hybrid alpha: 0.0
- Temporal decay: disabled
- Top-k: 10
- No OOD oracle
- No OOD tuning
- Paired analysis unit: `target_item_id`

Temporal decay was disabled to isolate the association correction from any
advantage caused by refreshing a memory timestamp.

## Statistical plan

- Primary contrast: corrected minus no-update MRR
- Paired unit: target item identifier
- Bootstrap seed: 4242
- Bootstrap resamples: 10,000
- Confidence level: 95%
- Minimum absolute MRR improvement: 0.05
- Bootstrap confidence interval required to exclude zero
- Maximum allowed mean old-memory degradation: 0.02

## Results

- Eligible targets: 14
- Eligible validation events: 15
- Mean MRR improvement: `0.9285714285714286`
- Bootstrap 95% CI: `(0.8214285714285714, 1.0)`
- Mean old-memory degradation: `0.0`
- Maximum old-memory degradation: `0.0`
- Correction success rule passed: `true`
- Old-memory degradation rule passed: `true`
- Oracle used: `false`
- OOD tuning used: `false`

## Interpretation

The controlled synthetic experiment supports the implementation claim that
the corrective-memory mechanism can restore deliberately corrupted,
previously stored target associations without degrading the other eligible
stored associations in this locked test.

The large effect size must be interpreted in light of the controlled design:
the experiment deliberately introduced known incorrect associations and then
provided the true target identities to the correction mechanism.

This experiment did not test:

- automatic error detection;
- autonomous discovery of the correct target;
- learning a previously unseen odor target;
- target-held-out generalization;
- human olfactory perception;
- clinical or diagnostic performance;
- physical odor generation or emission;
- chemical exposure safety;
- physical-device performance;
- deployment readiness.

Accordingly, this result validates a controlled software correction mechanism,
not autonomous olfactory intelligence or real-world corrective learning.

## Relationship to the negative OOD ablation

The earlier `noi-ablation-pilot-v0.1` found no incremental associative-memory
benefit under target-held-out OOD conditions. The present result does not
reverse or invalidate that finding.

Together, the two pilots establish a narrower boundary:

- memory cannot retrieve a target identity it has never stored;
- memory can restore a known stored association when an explicit correction
  supplies the correct target identity.

These are distinct engineering questions and must not be combined into a
single generalized success claim.

## Output files and SHA-256

### `corrective_memory_results.json`

`a1b4171303042bcff95241b59feec064a3861e59c07b232a48713ce84d94b7e5`

### `corrective_memory_summary.json`

`6cc0864369aac2bb7a667707d8171c151e4f635215dda5117b2d216417d21dfa`

### `run_manifest.json`

`9160b665385fb6c2b5549810e65deec32c8246bc18b0ba12da742e7f26ee85a6`

## Verification

- Full test suite before release: 560 passed
- Configuration checksum validation: passed
- Paired arms: passed
- Deterministic decoy selection: passed
- Restoration audit completeness: passed
- Export verification: passed
- Output hashes independently reproduced with `shasum -a 256`
- Working tree was clean before pilot execution

## Scope limitations

This release is exploratory, synthetic, computational, and limited to a
controlled implementation-level correction test.

It is not confirmatory, perceptual, clinical, chemical, physical-device,
emission-safety, autonomous-learning, or deployment-readiness evidence.

## Reproduction

From the repository root run:

    python -m pytest -q
    python -m experiments.run_corrective_memory_pilot

For an intentional deterministic rerun after output files already exist:

    from experiments.run_corrective_memory_pilot import (
        run_corrective_memory_pilot,
    )

    run_corrective_memory_pilot(overwrite=True)
