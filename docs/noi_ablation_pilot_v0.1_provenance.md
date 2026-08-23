# NOI Ablation Pilot v0.1 Provenance

## Ownership

- Project: Neuro-Olfactive Intelligence
- Owner: GUARDIANX LLC
- Authors: Rana Al-Dahlake and Jalal Alazirji

## Release identity

- Release: `noi-ablation-pilot-v0.1`
- Status: Exploratory synthetic computational pilot
- Ablation preregistration tag: `noi-ablation-v0.1-preimplementation`
- Preregistration commit: `3df8c31998ab83c071778daf285777226e8af687`
- Integrated pipeline commit: `e798532`
- Ablation implementation commit: `7b04aa3`
- Reproducible workflow commit: `23c3308`

The ablation settings were committed and tagged before implementation or
inspection of the NOI ablation results. The integrated pipeline itself had
already been implemented when the ablation-specific evaluation settings were
clarified. This timing distinction is recorded explicitly.

## Locked design

The experiment used:

- 140 training events for ridge fitting and associative-memory construction
- 20 validation events for hybrid-alpha selection
- 40 paired latent OOD events
- 3 paired OOD views per latent event: mild, moderate, and severe
- 120 total observed graded-OOD rows
- 200 odor-library targets
- Top-k: 10
- Paired analysis unit: `latent_event_id`
- Temporal displacement days: 0, 1, 7, 30, and 90
- No OOD oracle
- No OOD model fitting
- No OOD alpha tuning
- No policy-gate modification of retrieval rankings

## Prespecified systems

1. `ridge_only`
   - alpha: 1.0
   - temporal decay: disabled

2. `memory_only`
   - alpha: 0.0
   - temporal decay: enabled

3. `hybrid_without_temporal_decay`
   - alpha: selected using validation only
   - temporal decay: disabled

4. `full_hybrid`
   - alpha: selected using validation only
   - temporal decay: enabled

## Validation selection

The validation split selected:

- Hybrid alpha: `1.0`

Under the locked hybrid formula, alpha 1.0 assigns zero weight to the
associative-memory component. Consequently, the selected full hybrid reduces
to the ridge/library component for this pilot.

## Principal observed result

Across all 15 combinations of:

- 3 OOD severity tiers; and
- 5 temporal displacement conditions;

the `full_hybrid` rankings were exactly identical to the `ridge_only`
rankings.

Observed summary:

- Full hybrid equals ridge across all conditions: `true`
- Memory-only produced any nonzero MRR: `false`
- Incremental associative-memory benefit detected: `false`
- Negative result reported: `true`

## Interpretation

This exploratory pilot did not detect an incremental retrieval benefit from
the associative-memory component under the locked target-held-out OOD design.

The memory-only system achieved zero MRR because associative memory was built
only from training associations, while the evaluated OOD events contained
held-out odor families and target items. A memory restricted to previously
stored target identities cannot retrieve a correct target identity that it
has never stored.

This is an informative architectural limitation rather than evidence that
all contextual or associative-memory approaches are invalid. It shows that
the current closed-item episodic memory mechanism does not solve
target-generalization by itself.

The ridge projection retained strong synthetic performance under mild shift,
weaker performance under moderate shift, and collapsed under severe shift.
These observations support continued investigation of representation and
generalization mechanisms, but they do not establish a successful
associative-memory contribution.

No settings were changed after observing this negative result.

## Output files and SHA-256

### `noi_ablation_results.json`

`363f25e09172f6cb280dcf69b6c4633f06c5c3e759a7fa9ca315964a552691cd`

### `noi_ablation_summary.json`

`8847a62972690908e5ce85953dfe7ba8c43f7a2be0659dd14097ecd7576c9e36`

### `run_manifest.json`

`7b4e8f385a3822cc0a96d17bb3ed4ef29c3a6df369643e7e96b224130ef76eb1`

## Verification

- Full test suite before release: 482 passed
- Severe-reference replay: passed
- Paired ground truth: passed
- Export verification: passed
- Output hashes independently reproduced with `shasum -a 256`
- Working tree was clean before pilot execution

## Scope limitations

This release is:

- exploratory;
- synthetic;
- computational;
- an implementation and architecture evaluation.

It is not:

- final confirmatory evidence;
- human olfactory or perceptual validation;
- clinical or diagnostic evidence;
- chemical exposure evidence;
- physical odor-emission safety evidence;
- physical-device validation;
- deployment-readiness evidence.

## Reproduction

Run the test suite and workflow from the repository root:

    python -m pytest -q
    python -m experiments.run_noi_ablation_pilot

For an intentional deterministic rerun after output files already exist:

    from experiments.run_noi_ablation_pilot import (
        run_noi_ablation_pilot,
    )

    run_noi_ablation_pilot(overwrite=True)
