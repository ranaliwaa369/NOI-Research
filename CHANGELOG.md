# Changelog

All notable public-release changes to the NOI Research Framework are
documented in this file.

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
