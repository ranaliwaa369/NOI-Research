# NOI v0.3 Final Confirmatory Evaluation Results

## Document status

This document closes the confirmatory computational evaluation package
for NOI v0.3.

The evaluation was conducted as a synthetic, reproducible engineering
study. It does not establish human or animal olfactory equivalence,
human perceptual validity, clinical effectiveness, chemical safety,
physical-sensor performance, or deployment-safety certification.

Confirmatory execution occurred from the locked source-control commit
`f10ecc235b22273c07d1ab2ae00b39c10acb83c6`, tagged
`noi-v0.3-confirmatory-execution-lock`.

## Executive summary

NOI v0.3 produced one supported primary result and two unsupported
secondary results. The three hypotheses must be interpreted separately.

1. H6 was supported. The validation-locked support gate eliminated
   false-known decisions for unseen-family queries in this synthetic
   design while preserving clean seen-item retrieval.
2. H7 was not supported. Reliability-gated olfactory-tactile fusion was
   slightly worse than the strongest locked comparator in degraded-odor
   and missing-odor conditions.
3. H8 was not supported. The support-aware conflict policy produced a
   small, statistically detectable reduction in false-confident
   decisions, but the effect was far below the prespecified practical
   threshold.

The primary contribution is therefore support-aware rejection of
unsupported queries. The evaluation does not support a claim of tactile
synergy or a practically sufficient conflict-aware safety improvement.

## Evaluation scope

The locked confirmatory package used:

- `10` prespecified independent seeds: `1301` through `1310`;
- `7,000` training events per seed;
- `1,000` validation events per seed;
- `2,000` final-test latent events per seed;
- `7` paired condition views per final-test event;
- `9` registered systems;
- `20,000` total final-test latent events;
- `140,000` total paired condition views;
- `1,260,000` total system evaluations;
- `10,000` paired bootstrap resamples;
- bootstrap seed `4242`;
- latent event ID as the resampling unit;
- 95% confidence intervals;
- Holm correction for the three secondary comparisons.

The same latent event was retained across paired condition views.
Condition views were not treated as independent samples.

## Registered systems

The nine prespecified deployable systems were:

1. odor-only ridge retrieval;
2. odor-only cosine retrieval;
3. touch-only ridge retrieval;
4. touch-only cosine retrieval;
5. naive olfactory-tactile concatenation;
6. fixed-weight olfactory-tactile fusion;
7. support gate with odor-only retrieval;
8. reliability-gated olfactory-tactile fusion;
9. support gate plus reliability-gated fusion and abstention.

No oracle system competed in the confirmatory comparisons.

## H6: support-aware open-set gate

### Prespecified success rule

H6 required:

- at least `0.05` absolute reduction in unseen-family false-known rate;
- a paired 95% confidence interval excluding zero;
- no more than `0.02` absolute loss in clean seen-item MRR;
- validation-only threshold selection.

### Comparator selection

The clean seen-item MRR values for eligible comparators were:

| Comparator | Clean seen-item MRR |
|---|---:|
| Fixed-weight fusion | `1.000000` |
| Naive concatenation | `1.000000` |
| Odor-only cosine | `1.000000` |
| Odor-only ridge | `1.000000` |

The locked deterministic tie rule selected
`fixed_weight_fusion`.

### Result

Observed H6 result:

- False-known reduction: `1.000000`
- Mean paired difference: `1.000000`
- 95% paired bootstrap interval: `[1.000000, 1.000000]`
- Two-sided bootstrap p-value: `0.0001999800`
- Unseen-family latent events: `6,000`
- Paired view observations: `42,000`
- Clean seen-item MRR loss: `0.000000`
- H6 status: `supported`

The support gate abstained for unsupported unseen-family queries in this
locked simulation, whereas the selected non-gated comparator returned a
known candidate. Clean seen-item MRR was preserved.

The perfect synthetic separation must not be generalized to physical
sensor data or uncontrolled open-world queries. It may partly reflect
the separability of the registered generator and validation-derived
thresholds.

## H7: conditional tactile utility

### Prespecified success rule

H7 required reliability-gated olfactory-tactile fusion to improve MRR by
at least `0.05` absolute or `10%` relative over the strongest applicable
baseline, with a paired 95% confidence interval excluding zero.

Eligible conditions were `degraded_odor` and `missing_odor`.

### Comparator selection

| Candidate | Pooled MRR |
|---|---:|
| Fixed-weight fusion | `0.400000000` |
| Touch-only cosine | `0.400000000` |
| Touch-only ridge | `0.400000000` |
| Odor-only cosine | `0.199570833` |
| Odor-only ridge | `0.199545833` |

The locked deterministic tie rule selected
`fixed_weight_fusion`.

### Result

Observed H7 result:

- Proposed reliability-gated fusion MRR: `0.397787500`
- Selected comparator MRR: `0.400000000`
- Absolute MRR difference: `-0.002212500`
- Relative MRR difference: `-0.005531250`
- 95% paired bootstrap interval:
  `[-0.002662500, -0.001750000]`
- Raw two-sided bootstrap p-value: `0.0001999800`
- Holm-adjusted p-value: `0.0005999400`
- Latent events: `20,000`
- Paired observations: `40,000`
- H7 status: `not_supported`

The confidence interval excludes zero, but it lies entirely below zero.
The statistically detectable result is therefore in the direction
opposite to the registered hypothesis. The observed decrease was small,
but reliability-gated tactile fusion did not outperform the strongest
comparator.

The result must not be described as tactile synergy.

## H8: conflict-aware safe fusion

### Prespecified success rule

H8 required:

- at least `0.05` absolute reduction in false-confident decisions;
- a paired 95% confidence interval excluding zero;
- no more than `0.02` clean-condition MRR loss;
- successful comparison against both naive concatenation and
  fixed-weight fusion.

Eligible non-clean conditions were degraded odor, degraded touch,
missing touch, missing odor, contradictory modalities, and temporal
misalignment.

### Result

The result was identical against both required comparators:

| Comparator | False-confident reduction | 95% interval | Clean MRR loss | Status |
|---|---:|---:|---:|---|
| Fixed-weight fusion | `0.004008333` | `[0.002608333, 0.005441667]` | `0.000000` | Not supported |
| Naive concatenation | `0.004008333` | `[0.002608333, 0.005441667]` | `0.000000` | Not supported |

Additional statistical record:

- Raw two-sided bootstrap p-value: `0.0001999800`
- Holm-adjusted p-value: `0.0005999400`
- Latent events: `20,000`
- Paired observations per comparison: `120,000`
- H8 status: `not_supported`

The false-confident reduction was statistically detectable but only
about `0.0040`, or `0.40` percentage points. This was far below the
prespecified `0.05`, or 5 percentage-point, practical threshold.
Statistical significance therefore did not establish confirmatory
support.

## Multiple-comparison control

Holm correction covered:

1. H7;
2. H8 versus naive concatenation;
3. H8 versus fixed-weight fusion.

All three raw p-values were `0.0001999800`, and all three adjusted
p-values were `0.0005999400`.

Rejection of a zero-difference null did not automatically support the
directional and practical hypotheses. H7 was statistically different in
the unfavorable direction. H8 was favorable but materially too small.

## Integrated interpretation

NOI v0.3 supports a bounded conclusion:

- validation-locked support-aware gating prevented forced known-item
  retrieval for unsupported unseen-family events in the registered
  synthetic design;
- this protection did not reduce clean seen-item MRR;
- reliability-gated tactile fusion did not improve retrieval under the
  registered olfactory-degraded conditions;
- conflict-aware fusion produced a small reduction in false-confident
  decisions, but not the minimum practical effect required by H8.

The supported H6 result advances the support-aware routing direction
motivated by v0.2. The unsupported H7 and H8 results show that adding
simulated touch and reliability/conflict logic does not by itself
establish useful multimodal synergy or adequate safety improvement.

## Governance and leakage controls

The final package verified that:

- all ten registered seeds were retained;
- no silent seed or condition removal occurred;
- final-test labels were used for scoring only;
- final-test labels were not used for training or calibration;
- thresholds were not changed from final-test feedback;
- target labels were not inference inputs;
- condition metadata was not a model input;
- quality metadata was not a model input;
- paired condition views were not treated as independent samples;
- every seed artifact and final aggregate artifact was hashed;
- positive, null, and negative findings were retained.

## Independent raw-record audit

After finalization, an independent read-only audit recomputed the point
estimates directly from all ten raw seed files.

The audit reproduced:

- H6 false-known reduction: `1.000000`
- H6 clean seen-item MRR loss: `0.000000`
- H7 absolute MRR difference: `-0.002212500`
- H8 reduction versus fixed-weight fusion: `0.004008333`
- H8 reduction versus naive concatenation: `0.004008333`

All recomputed values matched the locked aggregate to an absolute
tolerance of `1e-12`.

## Reproducibility record

Final execution environment:

- Python: `3.11.14`
- Python implementation: `CPython`
- NumPy: `2.4.6`
- PyYAML: `6.0.3`
- Platform: `macOS-26.6.2-arm64-arm-64bit`
- Machine: `arm64`
- Execution-lock commit:
  `f10ecc235b22273c07d1ab2ae00b39c10acb83c6`
- Execution-lock tag:
  `noi-v0.3-confirmatory-execution-lock`
- Verification suite before execution: `1295 passed`
- Failed tests: `0`
- Dependency-lock SHA-256:
  `c14f5d344875a9afe640418589a1fe13e5412efb7bd21b963a4ed3d50116ce5e`

Locked configuration SHA-256 values:

- Protocol:
  `7d92d7d516399f1e9fb7cd8ed9bff5854caa87d8cca44c88923357c1b8ece79d`
- Validation amendment:
  `ced2b79c3aa4965af80f08fff0ed3dcd9d837dc30ba4dfec52440d96dcd95f44`
- Validation lock:
  `f887853f04ebe57c8c7486d6c5b3279736a1aeaae5f38479a25a086f8b50201f`
- Execution specification:
  `983b6e8e9565fdfd3e03d2813c96f911cdb91b344b66bdbe50f22ba81116d760`

Final artifact SHA-256 values:

- Aggregate:
  `4c427a7d1fe1a537e8efb209d7e632ff956fbdbf2b0c7ef8a1485720ba784320`
- Environment manifest:
  `3aeec90b198ccd52ef72a8d159f2f2c8e3293ba20b3ddd17cd5c61e3c2d9fa4d`
- Findings:
  `8d09ff18355ba13b583c24ea21f86e7547f9f3cbb0dd08c8bff09e2b6921a072`

## Principal reproducibility artifacts

Tracked release artifacts:

- `configs/noi_v0.3_protocol.yaml`
- `configs/protocol_amendment_v0.3.yaml`
- `configs/noi_v0.3_validation_lock.yaml`
- `configs/noi_v0.3_execution_spec.yaml`
- `artifacts/noi_v0.3_confirmatory/aggregate.json`
- `artifacts/noi_v0.3_confirmatory/environment_manifest.json`
- `artifacts/noi_v0.3_confirmatory/findings.md`
- `artifacts/noi_v0.3_confirmatory/seed_manifest.json`
- adjacent SHA-256 files for each artifact.

The ten raw seed JSON files and adjacent hashes are retained in the
local execution record. Their combined size is approximately `1.9 GB`,
so they are not committed directly to the ordinary Git repository.

## Limitations

The evidence remains limited by:

- fully synthetic data;
- simulated 16-dimensional olfactory vectors;
- simulated 8-dimensional tactile vectors;
- fixed generator assumptions;
- a small synthetic target topology;
- no physical odor or tactile sensors;
- no human or animal participants;
- no perceptual validation;
- no chemical generation or emission;
- no clinical evaluation;
- no deployment environment;
- no external laboratory replication;
- perfect H6 separation that may not persist under real distribution
  shift;
- no claim that statistical significance implies practical utility.

## Version 0.3 closure statement

The required NOI v0.3 confirmatory computational evaluation is complete.

All ten locked seeds were executed once. Validation-derived thresholds
remained frozen. Raw and aggregate artifacts were hashed. H6, H7, and H8
were evaluated independently. Supported, unsupported, and negative
results were retained.

NOI v0.3 can therefore be closed as a complete reproducible synthetic
research package after publication metadata and release packaging are
finalized.

## Future work

A future version may investigate:

- less separable and externally sourced open-set distributions;
- physical sensor-derived representations;
- learned reliability estimation under a new training protocol;
- calibrated uncertainty bands with non-degenerate width;
- touch representations that encode independent complementary evidence;
- conflict policies with larger practical effect sizes;
- per-condition error analysis under a newly prespecified protocol;
- external replication.

Any architectural or threshold change must be evaluated under a new
prespecification. It must not retroactively alter the v0.3 results.
