# NOI v0.3 Preimplementation Research Protocol

Protocol ID: NOI-PROTOCOL-0.3
Version: 0.3.0-preimplementation
Date: 2026-08-29
Project: Neuro-Olfactive Intelligence (NOI)
Owner: GuardianX LLC
Authors: Rana Al-Dahlake and Jalal Alazirji
Status: Preimplementation; not yet locked for confirmatory evaluation

## 1. Purpose

Version 0.3 evaluates whether NOI can identify the limits of its
knowledge, request simulated tactile evidence when olfactory evidence is
insufficient, and abstain when available modalities are missing,
degraded, or contradictory.

The primary contribution is not the unconditional addition of touch.
The contribution under evaluation is reliability-gated multisensory
decision making under open-set distribution shift.

This protocol extends v0.2 without modifying or reinterpreting any
v0.1 or v0.2 result.

## 2. Motivation from v0.2

Version 0.2 established three relevant findings:

1. Associative memory provided a repeatable contribution for previously
   represented items.
2. Direct identity retrieval for unseen odor families remained
   structurally unresolved.
3. Calibrated abstention was strong, while the prespecified robustness
   superiority hypothesis was not supported in any of the 36
   condition-tier tests.

Version 0.3 therefore tests support recognition before identity
retrieval and introduces touch only as conditional complementary
evidence.

## 3. Primary Research Question

Can a support-aware NOI system determine whether a query is represented
by its learned knowledge, acquire simulated tactile evidence when
olfactory evidence is uncertain, and reduce false confident retrieval
under novelty or modality conflict?

## 4. Scope and Claim Boundaries

This release is limited to deterministic synthetic computational
simulation.

The simulated tactile modality does not represent a validated physical
tactile sensor. Results must not be described as evidence of:

- human or animal sensory equivalence;
- clinical validity;
- real-world odor or touch sensing;
- chemical identification or chemical safety;
- robotic deployment readiness;
- comprehensive safety.

Touch cannot create the identity of an unseen class that is absent from
the learned label space. It may improve discrimination among supported
items or improve detection and rejection of unsupported queries.

## 5. Support Regimes

Every final evaluation event must belong to exactly one regime:

1. Seen item:
   the exact target item is represented in training memory.
2. Known family, unseen item:
   the target family is represented during training, but the exact item
   is absent from training memory.
3. Unseen family:
   neither the target item nor its family is represented during
   training.

Metrics from these regimes must never be pooled without a stratified
breakdown.

## 6. Simulated Tactile Representation

The tactile generator will represent complementary physical properties:

- surface roughness;
- stiffness or compliance;
- friction;
- contact geometry;
- pressure-response dynamics.

Tactile features must be generated from latent object properties and
must not encode target labels, split membership, or evaluation outcomes.

Olfactory and tactile noise must be generated independently except in
explicitly prespecified shared-environment conditions.

Every event must include:

- tactile availability;
- tactile quality;
- olfactory availability;
- olfactory quality;
- modality-conflict status;
- support regime;
- latent event identifier.

## 7. Prespecified Hypotheses

### H6: Support-Aware Open-Set Gate

A support-aware gate will reduce false-known decisions for unseen-family
queries relative to the strongest prespecified deployable baseline,
while preserving seen-item retrieval within the non-inferiority
tolerance.

H6 is the primary hypothesis.

H6 support requires all of the following:

- at least 0.05 absolute reduction in false-known rate;
- a paired 95% confidence interval for the reduction that excludes zero;
- no more than 0.02 absolute loss in seen-item mean reciprocal rank;
- threshold selection performed on validation data only.

### H7: Conditional Tactile Utility

Reliability-gated olfactory-tactile fusion will outperform the strongest
single-modality system and fixed-fusion baseline in prespecified
olfactory-ambiguous or olfactory-degraded conditions.

H7 support requires:

- at least 0.05 absolute or 10% relative improvement in mean reciprocal
  rank over the strongest applicable baseline;
- a paired 95% confidence interval for the improvement that excludes
  zero;
- no use of final-test labels for gating, weighting, or calibration.

H7 does not require touch to improve clean olfactory conditions.

### H8: Conflict-Aware Safe Fusion

When modalities are missing, degraded, or contradictory, the
reliability-gated system will reduce false confident decisions relative
to naive and fixed-weight fusion without materially degrading clean
multimodal retrieval.

H8 support requires:

- at least 0.05 absolute reduction in false confident decision rate
  under the prespecified conflict conditions;
- a paired 95% confidence interval for the reduction that excludes zero;
- no more than 0.02 absolute loss in clean-condition mean reciprocal
  rank.

## 8. Prespecified Systems

The following deployable systems will be compared:

1. odor-only ridge retrieval;
2. odor-only cosine retrieval;
3. touch-only ridge retrieval;
4. touch-only cosine retrieval;
5. naive olfactory-tactile concatenation;
6. fixed-weight olfactory-tactile fusion;
7. support gate with odor-only retrieval;
8. reliability-gated olfactory-tactile fusion;
9. support gate plus reliability-gated fusion and abstention.

Oracle systems may be used only as clearly labeled diagnostics.
They must not compete in confirmatory deployable comparisons.

## 9. Prespecified Conditions

Each applicable support regime will be evaluated under:

1. clean odor and clean touch;
2. degraded odor and clean touch;
3. clean odor and degraded touch;
4. missing touch;
5. missing odor;
6. contradictory odor and touch;
7. temporal misalignment between odor and touch.

The same latent event must be used across paired condition views.
Condition views of one latent event are not independent samples.

## 10. Decision Policy

The system must follow this order:

1. estimate olfactory support and uncertainty;
2. retrieve directly only when support and confidence satisfy
   validation-locked thresholds;
3. request touch when olfactory evidence is insufficient and touch is
   available;
4. estimate modality reliability and conflict;
5. fuse only reliable and sufficiently compatible evidence;
6. abstain when support remains absent or evidence remains conflicting.

The final-test set cannot change any threshold or decision rule.

## 11. Primary Metrics

Retrieval:

- Recall at 1;
- Recall at 10;
- mean reciprocal rank;
- nDCG at 10.

Open-set and support detection:

- AUROC;
- AUPR;
- FPR at 95% true-positive rate;
- false-known rate;
- unknown-detection rate.

Selective prediction and calibration:

- coverage;
- selective error;
- risk-coverage curve;
- expected calibration error;
- Brier score;
- false confident decision rate.

Tactile contribution:

- fused performance minus the maximum of odor-only and touch-only
  performance;
- touch-request rate;
- useful-touch rate;
- harmful-fusion rate.

## 12. Statistical Controls

- Use at least 10 prespecified independent seeds.
- Use identical paired events across systems.
- Report every seed and retain negative and null results.
- Use paired bootstrap confidence intervals with latent event ID as the
  resampling unit.
- Use a 95% confidence level and at least 10,000 bootstrap resamples.
- Apply Holm correction to confirmatory secondary comparisons.
- Report effect sizes with confidence intervals.
- Never tune on the final-test set.
- Never remove an unfavorable seed or condition after inspection.

## 13. Development and Locking Stages

### Stage A: Feasibility

Implementation tests may verify data schemas, deterministic generation,
metric correctness, and absence of leakage.

No confirmatory claim may be made from feasibility tests.

### Stage B: Pilot

A separately identified pilot may be used to detect implementation
errors and choose validation procedures.

Pilot events, seeds, and outputs must not be reused as final
confirmatory evidence.

### Stage C: Protocol Lock

Before final evaluation:

- configuration values must be finalized;
- seeds must be listed;
- thresholds must be validation-derived and frozen;
- configuration and protocol files must be hashed;
- a preimplementation tag must be created.

### Stage D: Confirmatory Evaluation

The locked evaluation must be executed once without final-test tuning.
All positive, null, and negative findings must be retained.

## 14. Leakage and Integrity Controls

- Target labels cannot be embedded directly in tactile features.
- Split membership cannot influence feature generation.
- Exact event duplicates cannot cross splits.
- Templates and latent event IDs must be audited.
- Support labels may train the support gate only on training data.
- Validation data may select thresholds but cannot fit final retrieval
  representations.
- Test labels cannot train, calibrate, repair, or select any system.
- Every configuration, manifest, result, and aggregate must be hashed.
- Oracle information must remain diagnostic and isolated.

## 15. Interpretation Rules

H6, H7, and H8 must be reported independently.

Failure of a hypothesis is a scientific result, not an implementation
failure, provided all integrity checks pass.

A multimodal result must not be described as synergy unless it exceeds
the strongest applicable unimodal baseline under the prespecified
criterion.

High abstention alone must not be described as successful recognition.
Coverage and selective error must always be reported together.

## 16. Expected v0.3 Contribution

Version 0.3 is designed to determine whether NOI can:

- recognize unsupported queries before forced identity retrieval;
- request touch only when additional evidence is needed;
- use complementary tactile information under olfactory ambiguity;
- avoid forced fusion when modalities conflict;
- preserve transparent and reproducible failure reporting.

The intended outcome is a bounded evaluation of support-aware,
reliability-gated multisensory retrieval, not a claim of general
olfactory intelligence.
