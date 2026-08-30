# Changelog

All notable public-release changes to the NOI Research Framework are
documented in this file.

## [0.3.0] - 2026-08-30

### Added

- Locked support-aware and reliability-gated multisensory architecture.
- Simulated 8-dimensional tactile representations paired with
  16-dimensional olfactory representations.
- Nine prespecified deployable systems and seven paired conditions.
- Ten-seed confirmatory execution with 1,260,000 system evaluations.
- Latent-event paired bootstrap analysis with 10,000 resamples.
- Holm correction for the three secondary comparisons.
- Deterministic finalization, environment manifest, aggregate findings,
  seed manifest, and SHA-256 sidecars.
- Final v0.3 results report and research manuscript.
- Post-hoc exploratory per-seed and hierarchical seed-event sensitivity
  analysis.
- Synthetic support-score separability diagnostic with explicit H6 scope
  limitations.
- Read-only ten-seed trace audit linking raw hashes, validation locks,
  inference decisions, scoring, paired views, and aggregate results.
- Deliberate rejection tests for modified hashes, integrity declarations,
  and locked values.

### Research findings

- H6 was supported: unseen-family false-known reduction was `1.000000`
  with a 95% interval of `[1.000000, 1.000000]`, and clean seen-item
  MRR loss was `0.000000`.
- H7 was not supported: the proposed fusion system had an absolute MRR
  difference of `-0.002212500` relative to fixed-weight fusion.
- H8 was not supported: false-confident reduction was `0.004008333`
  against both required comparators, below the registered `0.05`
  practical threshold.
- All ten registered seeds and all negative and null findings were
  retained.
- An independent read-only audit reproduced the aggregate point
  estimates from all raw records.

### Governance and limitations

- Protocol, amendment, validation thresholds, and execution mechanics
  were locked and hashed before confirmatory execution.
- No final-test training, calibration, tuning, or seed removal was used.
- Final-test labels were used for scoring only.
- The evidence is fully synthetic and does not establish physical
  sensing, biological equivalence, clinical validity, chemical safety,
  tactile synergy, or deployment readiness.

## [0.2.0] - 2026-08-28

### Added

- Ten-run repeated Track A seen-item evaluation.
- Ten-run Track B unseen-family evaluation.
- Validation-calibrated memory-support scoring and selective abstention.
- Seven-condition missing-modality robustness evaluation.
- Five-condition temporal-displacement robustness evaluation.
- Ten-run paired bootstrap aggregate analyses.
- Deterministic per-seed, aggregate, and environment-provenance exports.
- Vectorized associative-memory scoring with unchanged test behavior.
- Final v0.2 evaluation report and reproducibility record.

### Research findings

- Track A found a repeatable memory-only minus ridge-only MRR advantage of
  `0.009066818`, with a 95% paired bootstrap interval of
  `[0.004399445, 0.014384734]`.
- Track B direct unseen-family identity retrieval remained unresolved.
- Track B mean abstention was `0.96145`, `0.99420`, and `0.99915` for
  mild, moderate, and severe OOD tiers.
- The synthetic Track B calibration-and-abstention criterion was supported.
- Ridge-only was the strongest baseline in all 36 final robustness
  condition-tier tests.
- H4 was not supported: the full hybrid exceeded the strongest baseline in
  `0` of `36` prespecified condition-tier tests.
- All positive, null, and negative findings were retained.

### Governance and limitations

- No OOD oracle was used.
- No OOD model fitting or OOD alpha tuning was used.
- No final-test tuning was used.
- Seen-item and unseen-family findings remain separate.
- The evidence is synthetic computational evidence and does not establish
  human or animal olfactory equivalence, clinical effectiveness, physical
  sensor performance, chemical safety, or deployment-safety certification.

## [0.1.0] - 2026-08-22

### Added

- Reproducible baseline retrieval pilot.
- Graded out-of-distribution evaluation across mild, moderate, and severe tiers.
- NOI component-ablation evaluation.
- Controlled corrective-memory evaluation.
- Deterministic policy-conformance evaluation.
- Machine-readable run manifests, results, and summary exports.
- Public verification functions and automated tests.
- Citation and Zenodo archival metadata.
- Proprietary public-inspection license notice.

### Research findings

- Retrieval performance declined as out-of-distribution severity increased.
- The tested hybrid configuration did not improve on its ridge component.
- The tested memory-only configuration did not demonstrate independent retrieval utility.
- Explicit controlled corrections improved the selected correction queries without measured degradation on the locked preservation set.
- Policy conformance was exact on the locked simulated cases.

### Limitations

- Version 0.1.0 is exploratory rather than confirmatory.
- The experiments use controlled simulations and synthetic representations.
- The release does not establish real-world odor sensing, clinical validity,
  deployment readiness, or comprehensive safety.
