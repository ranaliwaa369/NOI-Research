# NOI v0.3 Feasibility Pilot Provenance

## Status

This is a **Feasibility** pilot for the NOI v0.3 multisensory research
pipeline. It is not a confirmatory experiment.

H6, H7, and H8 are **not tested** by this pilot. No feasibility metric,
threshold, action count, or diagnostic result may be described as evidence
supporting or rejecting those hypotheses.

## Permitted purpose

The pilot checks only:

- deterministic execution;
- record and artifact schemas;
- train/validation/final-test leakage controls;
- metric behavior;
- computational practicality;
- validation-only threshold-derivation mechanics;
- paired condition generation;
- reliability and conflict-gated fusion mechanics.

## Data and evidence boundary

All records are synthetic computational evidence. They do not establish
biological olfactory equivalence, physical tactile sensing, chemical
identification, clinical validity, safety performance, or deployment
readiness.

## Allocation

The feasibility seed is 1301. The scaled event allocation is:

- training: 70 latent events;
- validation: 10 latent events;
- final test: 20 latent events;
- final seen-item support: 8 events;
- final known-family/unseen-item support: 6 events;
- final unseen-family support: 6 events.

This scaled allocation is used only for software and methodological checks.

## Threshold integrity

Support models are fitted on training records only. Threshold derivation uses
validation records only. Final-test labels are not used for fitting,
calibration, threshold selection, or policy modification.

The feasibility pilot must not change confirmatory thresholds after inspecting
final-test results.

## Paired conditions

Every final-test latent event receives the seven prespecified condition views:
clean, degraded odor, degraded touch, missing touch, missing odor,
contradictory modalities, and temporal misalignment.

Ground truth and support regime remain fixed across the seven views.

## Interpretation restriction

The pilot may reveal implementation defects or impractical computation. It
cannot support H6, H7, or H8. Confirmatory claims remain prohibited until the
validation lock is completed and the prespecified independent seeds are run
without test-driven modification.

## Observed feasibility artifact

The pilot was executed from the repository root using the
versioned module runner:

`python -m experiments.run_noi_v0_3_feasibility --output results/v0.3-feasibility/feasibility_seed_1301.json`

The canonical JSON artifact for seed 1301 had SHA-256:

`8f2b4fa0ee836886254f7ec349abba8f64ad4f9054bf68b4d52cdae88f4e0930`

The `results/` directory is intentionally excluded from Git.
The artifact is reproducible from the versioned runner,
configuration, and seed. This hash records feasibility execution
only and is not confirmatory evidence.
