# NOI v0.3 Implementation Plan

Plan ID: NOI-IMPLEMENTATION-PLAN-0.3
Version: 0.3.0-preimplementation
Date: 2026-08-29
Parent protocol: NOI-PROTOCOL-0.3
Parent tag: noi-v0.3-preimplementation
Status: Development plan; no confirmatory results inspected

## 1. Objective

Implement the support-aware, reliability-gated olfactory-tactile
evaluation defined by the preregistered v0.3 protocol while preserving
all v0.1 and v0.2 behavior.

Implementation must proceed test-first and must not use final-test
labels, metrics, or outcomes for model selection.

## 2. Backward-Compatibility Decision

The following existing modules remain behaviorally unchanged:

- `src/evaluation/synthetic_records.py`
- `src/evaluation/synthetic_generator.py`
- `src/models.py`
- all existing v0.1 and v0.2 experiment modules

The existing `SyntheticEvent` requires equal dimensions for text,
image, and audio. The v0.3 tactile vector has eight dimensions and must
not be added to that record.

Version 0.3 therefore uses separate records and a separate generator.

## 3. v0.3 Data Architecture

### 3.1 Multisensory target

Each target contains:

- opaque item identifier;
- family identifier;
- 16-dimensional olfactory prototype;
- 8-dimensional simulated tactile prototype.

Tactile prototypes are generated from latent physical properties.
Target labels and split labels cannot be used as tactile feature values.

### 3.2 Latent event

Each latent event contains:

- immutable latent event identifier;
- split;
- target item and family;
- support regime;
- clean olfactory observation;
- clean tactile observation;
- template identifier;
- generation provenance.

One latent event is the statistical unit.

### 3.3 Condition view

Each latent event produces the applicable paired views:

1. clean;
2. degraded odor;
3. degraded touch;
4. missing touch;
5. missing odor;
6. contradictory modalities;
7. temporal misalignment.

A condition view stores availability, quality, conflict, and temporal
metadata. It does not duplicate or redefine ground truth.

Condition views from one latent event cannot be counted as independent
samples.

## 4. Planned Modules

### Milestone 1: Records and invariants

Create:

- `src/evaluation/multisensory_records.py`
- `tests/test_multisensory_records.py`

Responsibilities:

- immutable typed records;
- finite-vector validation;
- exact olfactory and tactile dimensions;
- unique target, latent-event, and view identifiers;
- target-family consistency;
- valid support regimes and condition labels;
- missing-modality consistency;
- conflict metadata consistency;
- prevention of duplicate condition views.

No vectors are generated in this milestone.

### Milestone 2: Deterministic tactile generation

Create:

- `src/evaluation/tactile_generator.py`
- `tests/test_tactile_generator.py`

Responsibilities:

- generate eight-dimensional tactile prototypes;
- preserve deterministic output for each seed;
- model roughness, stiffness, friction, geometry, and pressure response;
- separate family-level structure from item-level residuals;
- prohibit direct encoding of item IDs, family IDs, split labels, or
  support labels;
- retain finite normalized or bounded features;
- expose explicit provenance.

This milestone may use feasibility-only seeds. Confirmatory seeds remain
uninspected.

### Milestone 3: v0.3 split and event generation

Create:

- `src/evaluation/noi_v0_3_generator.py`
- `tests/test_noi_v0_3_generator.py`

Responsibilities:

- produce training, validation, and final-test latent events;
- enforce 7000/1000/2000 event allocation per seed;
- enforce final-test support allocation of 800/600/600;
- keep validation-unknown families distinct from final-test families;
- preserve exact seen-item reachability;
- create known-family unseen items without exact-item leakage;
- create unseen-family events without family leakage;
- generate unique deterministic identifiers;
- export reachability and support metadata.

A reduced-size feasibility mode may be used for unit tests. It must
preserve the same allocation rules or an explicitly documented scaled
allocation.

### Milestone 4: Paired stress views

Create:

- `src/evaluation/multisensory_conditions.py`
- `tests/test_multisensory_conditions.py`

Responsibilities:

- generate all seven paired condition views;
- apply odor noise only to odor;
- apply tactile noise only to touch;
- encode genuinely missing vectors as absent;
- create deterministic cross-family conflicts;
- apply the locked temporal offset;
- preserve target ground truth across all views;
- prevent condition views from changing support regimes.

### Milestone 5: Support gate

Create:

- `src/evaluation/support_gate.py`
- `tests/test_support_gate.py`

Prespecified methods:

- L2-normalized class-conditional Mahalanobis distance;
- cosine-margin comparison;
- nearest-prototype-distance comparison.

Responsibilities:

- fit only on permitted training data;
- select thresholds using validation only;
- output support score, support decision, and uncertainty status;
- request touch only inside the validation-defined uncertainty policy;
- never use final-test labels.

### Milestone 6: Reliability and conflict-gated fusion

Create:

- `src/evaluation/reliability_fusion.py`
- `tests/test_reliability_fusion.py`

Responsibilities:

- estimate per-modality reliability from available evidence;
- detect prespecified modality conflict;
- select odor-only, touch-only, fused, or abstain action;
- include naive concatenation and fixed 0.5/0.5 fusion baselines;
- retain an auditable decision trace;
- prevent unavailable modalities from receiving nonzero weight.

### Milestone 7: Metrics and paired analysis

Create:

- `src/evaluation/noi_v0_3_metrics.py`
- `src/evaluation/noi_v0_3_analysis.py`
- corresponding test modules

Responsibilities:

- retrieval metrics;
- open-set metrics;
- calibration metrics;
- risk-coverage analysis;
- touch synergy;
- touch-request and useful-touch rates;
- harmful-fusion and false-confident rates;
- paired bootstrap by latent event;
- Holm correction for secondary comparisons.

### Milestone 8: Feasibility pilot

Create:

- `experiments/run_noi_v0_3_feasibility.py`
- `docs/noi_v0.3_feasibility_provenance.md`

The feasibility pilot may verify:

- determinism;
- schemas;
- leakage controls;
- metric behavior;
- computational practicality;
- threshold derivation mechanics.

It cannot support H6, H7, or H8.

### Milestone 9: Validation lock

Before confirmatory evaluation:

- derive all null thresholds from validation data;
- replace null threshold values in the locked configuration;
- change protocol status through a documented amendment;
- create SHA-256 files;
- run the complete test suite;
- create a separate protocol-lock commit and tag;
- prohibit further threshold changes.

### Milestone 10: Confirmatory execution

Run each prespecified seed exactly once under the locked protocol.

Export:

- per-seed results;
- condition-level results;
- support-regime results;
- paired aggregate statistics;
- environment manifest;
- configuration and result hashes;
- final positive, null, and negative findings.

## 5. Test-First Rule

For each milestone:

1. write record or behavior tests;
2. confirm the new tests fail for the expected missing implementation;
3. implement the minimum required behavior;
4. run the milestone tests;
5. run the complete legacy and current suite;
6. commit only when every test passes.

Tests must not be weakened to make implementation pass.

## 6. Leakage Audit Requirements

The audit must explicitly verify:

- no exact event overlap;
- no latent event overlap across splits;
- no template leakage;
- no unseen-family leakage;
- no exact-item leakage into known-family unseen-item evaluation;
- no target label encoded in tactile values;
- no split or support label used to generate tactile values;
- validation-unknown and final-test-unknown families are disjoint;
- conflicting touch comes from a different item and family;
- final-test labels never enter threshold fitting.

## 7. Stop Conditions

Implementation must pause if:

- a proposed feature requires final-test information;
- deterministic regeneration fails;
- support allocations cannot be constructed without leakage;
- tactile features reveal labels through construction;
- a condition changes target ground truth;
- paired views are mistakenly treated as independent observations;
- legacy tests fail;
- protocol changes are needed after confirmatory results are inspected.

## 8. Scientific Interpretation

Touch is considered useful only when gated fusion exceeds the strongest
applicable unimodal baseline under the preregistered criterion.

Touch is not assumed to improve clean conditions.

Unsupported identity retrieval must be reported as unsupported rather
than converted into a forced nearest-class answer.

A failed hypothesis remains a valid result when implementation,
integrity, and reporting checks pass.
