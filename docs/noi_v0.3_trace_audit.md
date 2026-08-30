# NOI v0.3 Post-Confirmatory Trace Audit

## Status

This document reports a post-hoc, read-only integrity trace of the completed NOI v0.3 confirmatory artifacts.

- Classification: `post_hoc_read_only_integrity_audit`
- Confirmatory results modified: `false`
- Confirmatory hypothesis statuses modified: `false`
- Seeds audited: `10`
- Raw system-evaluation records audited: `1,260,000`

The audit found no violation of the registered and testable leakage boundaries. This is evidence of conformance to those boundaries, not a mathematical proof that no conceivable leakage channel exists.

## Audited chain

- `verified_raw_seed_artifact`
- `seedwise_validation_lock`
- `locked_support_application`
- `system_inference_export`
- `label_based_scoring`
- `aggregate_point_estimate_reproduction`

The trace connected each verified raw seed artifact to its seedwise validation lock, reproduced the locked support decision, reproduced label-based scoring, verified paired conditions and system alignment, and linked the raw records to the published aggregate point estimates.

## Verified boundaries

| Boundary | Verified value |
|---|:---:|
| `condition_metadata_used_as_model_input` | `false` |
| `final_test_labels_used_for_scoring_only` | `true` |
| `paired_views_treated_as_independent` | `false` |
| `quality_metadata_used_as_model_input` | `false` |
| `target_labels_used_as_inference_input` | `false` |
| `thresholds_changed_from_final_test` | `false` |
| `training_only_model_fitting` | `true` |
| `validation_only_threshold_derivation` | `true` |

## Per-seed trace summary

| Seed | Latent events | Condition views | System evaluations | Raw hash | Locks | Decisions | Scoring | Pairing |
|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| `1301` | `2,000` | `14,000` | `126,000` | `true` | `true` | `true` | `true` | `true` |
| `1302` | `2,000` | `14,000` | `126,000` | `true` | `true` | `true` | `true` | `true` |
| `1303` | `2,000` | `14,000` | `126,000` | `true` | `true` | `true` | `true` | `true` |
| `1304` | `2,000` | `14,000` | `126,000` | `true` | `true` | `true` | `true` | `true` |
| `1305` | `2,000` | `14,000` | `126,000` | `true` | `true` | `true` | `true` | `true` |
| `1306` | `2,000` | `14,000` | `126,000` | `true` | `true` | `true` | `true` | `true` |
| `1307` | `2,000` | `14,000` | `126,000` | `true` | `true` | `true` | `true` | `true` |
| `1308` | `2,000` | `14,000` | `126,000` | `true` | `true` | `true` | `true` | `true` |
| `1309` | `2,000` | `14,000` | `126,000` | `true` | `true` | `true` | `true` | `true` |
| `1310` | `2,000` | `14,000` | `126,000` | `true` | `true` | `true` | `true` | `true` |

All ten seed artifacts passed every registered trace check.

## Representative clean-condition trace

The following deterministic examples are from seed `1301` and the `support_gate_odor_only` system.

| Support regime | Support score | Locked threshold | Gate decision | Abstained | Correct | MRR |
|---|---:|---:|:---:|:---:|:---:|---:|
| `seen_item` | `-1.693846911157` | `-22.136865951940` | `true` | `false` | `true` | `1.000000` |
| `known_family_unseen_item` | `-6.552651472097` | `-22.136865951940` | `true` | `false` | `false` | `0.000000` |
| `unseen_family` | `-34.181674718005` | `-22.136865951940` | `false` | `true` | `false` | `0.000000` |

The seen item was supported and retrieved correctly. The unseen-family query was classified as unsupported and produced abstention. The known-family unseen item was classified as supported but was not retrieved correctly.

This distinction is important: H6 tested `unseen_family` queries. It did not establish rejection or correct identity retrieval for `known_family_unseen_item` queries.

## Deliberate rejection tests

The automated suite includes tests that intentionally introduce or attempt prohibited behavior, including:

- final-test events passed to support-gate fitting or calibration;
- final-test events passed to reliability or conflict calibration;
- exact-item leakage for known-family withheld items;
- training-family overlap for unseen-family events;
- target-identifier leakage in synthetic configuration;
- attempts to disable leakage checks;
- post-lock threshold changes;
- false integrity declarations;
- modified locked values;
- incorrect or missing SHA-256 sidecars.

These tests are located principally in `tests/test_noi_v0_3_generator.py`, `tests/test_noi_v0_3_protocol.py`, `tests/test_support_gate.py`, `tests/test_evidence_threshold_calibration.py`, and `tests/test_noi_v0_3_confirmatory_trace_audit.py`.

## Aggregate linkage

- Aggregate SHA-256: `4c427a7d1fe1a537e8efb209d7e632ff956fbdbf2b0c7ef8a1485720ba784320`
- Post-hoc sensitivity SHA-256: `b13cfca0d130d4f396d9a524cb31d44686114b2e18529247acf1ecc70710baa0`
- Aggregate point estimates reproduced: `true`
- Total linked system evaluations: `1,260,000`

## Interpretation boundary

The audit supports the statement that no violation was detected across the registered, implemented, and testable leakage boundaries. It must not be described as proof that leakage is impossible.

The audit creates no new confirmatory claim and does not change H6, H7, or H8.

## Reproducibility artifact

- `artifacts/noi_v0.3_confirmatory/trace_audit.json`
- `artifacts/noi_v0.3_confirmatory/trace_audit.json.sha256`
- SHA-256: `1329be76c392e82787dab14f3f3059bbe5f81f2253528f59f7309abbf345ca10`
