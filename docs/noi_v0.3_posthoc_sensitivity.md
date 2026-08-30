# NOI v0.3 Post-Hoc Sensitivity Analysis

## Analysis status

This document reports a post-hoc exploratory sensitivity analysis of the completed NOI v0.3 confirmatory evaluation.

It does not modify the locked protocol, registered hypotheses, comparators, thresholds, confirmatory point estimates, confidence intervals, p-values, or hypothesis statuses.

- Analysis classification: `post_hoc_exploratory`
- Confirmatory results modified: `false`
- Confirmatory hypothesis statuses modified: `false`
- Registered seeds: `10`
- Hierarchical bootstrap resamples: `10,000`
- Resampling hierarchy: seed, then latent event

## Purpose

The confirmatory paired bootstrap treated latent event ID as the resampling unit. This supplementary analysis evaluates sensitivity to variation at two levels by resampling the registered seed first and latent events within the selected seed second.

It also reports every seed-level effect and examines whether the perfect H6 result may reflect strong separability in the registered synthetic generator.

## Confirmatory point-estimate reproduction

| Quantity | Confirmatory | Recomputed | Matched |
|---|---:|---:|:---:|
| H6 false-known reduction | `1.000000000000` | `1.000000000000` | `true` |
| H7 absolute MRR difference | `-0.002212500000` | `-0.002212500000` | `true` |
| H8 reduction vs fixed-weight fusion | `0.004008333333` | `0.004008333333` | `true` |
| H8 reduction vs naive concatenation | `0.004008333333` | `0.004008333333` | `true` |

Every point estimate matched the locked aggregate to absolute tolerance `1e-12`.

## Seed-level effects

| Seed | H6 false-known reduction | H7 MRR difference | H8 vs fixed | H8 vs naive |
|---:|---:|---:|---:|---:|
| `1301` | `1.000000000` | `0.000000000` | `-0.002250000` | `-0.002250000` |
| `1302` | `1.000000000` | `-0.015750000` | `0.008166667` | `0.008166667` |
| `1303` | `1.000000000` | `-0.006250000` | `-0.001916667` | `-0.001916667` |
| `1304` | `1.000000000` | `0.000000000` | `0.015333333` | `0.015333333` |
| `1305` | `1.000000000` | `0.000000000` | `-0.005833333` | `-0.005833333` |
| `1306` | `1.000000000` | `0.000000000` | `0.018416667` | `0.018416667` |
| `1307` | `1.000000000` | `0.000000000` | `0.002916667` | `0.002916667` |
| `1308` | `1.000000000` | `0.000000000` | `-0.014583333` | `-0.014583333` |
| `1309` | `1.000000000` | `-0.000125000` | `0.019583333` | `0.019583333` |
| `1310` | `1.000000000` | `0.000000000` | `0.000250000` | `0.000250000` |

H6 was identical across all ten registered seeds. H7 was zero or negative for every seed. H8 varied in direction across seeds, with both favorable and unfavorable seed-level effects.

## Hierarchical seed-event bootstrap

| Quantity | Observed effect | 95% interval | Two-sided p-value |
|---|---:|---:|---:|
| H6 false-known reduction | `1.000000000000` | `[1.000000000000, 1.000000000000]` | `0.0001999800` |
| H7 absolute MRR difference | `-0.002212500000` | `[-0.005725000000, 0.000000000000]` | `0.0949905009` |
| H8 reduction vs fixed-weight fusion | `0.004008333333` | `[-0.002700416667, 0.010600000000]` | `0.2511748825` |
| H8 reduction vs naive concatenation | `0.004008333333` | `[-0.002591666667, 0.010741875000]` | `0.2401759824` |

These exploratory intervals do not replace the registered confirmatory analysis. They quantify additional uncertainty from variation among the ten synthetic generator realizations.

The hierarchical H7 and H8 intervals include zero. This reinforces the existing conclusions that H7 and H8 were not supported. It does not alter the confirmatory hypothesis decisions.

## H6 synthetic separability diagnostic

| Clean-condition support regime | Minimum score | Maximum score |
|---|---:|---:|
| `seen_item` | `-1.806007443673` | `-1.479071501593` |
| `known_family_unseen_item` | `-10.647001424384` | `-3.554940348134` |
| `unseen_family` | `-41.382860140296` | `-27.215016547852` |

- Seen and unseen-family ranges disjoint: `true`
- Seen minimum minus unseen-family maximum: `25.409009104179`

The clean seen-item and unseen-family support-score ranges were fully disjoint. The observed gap was large. The perfect H6 result may therefore reflect the registered synthetic family geometry and should not be generalized to naturally overlapping physical odor distributions.

## Conditionwise H6 diagnostic

| Condition | Pooled score ranges disjoint | Closest seen margin | Closest unseen margin | Unseen gate-supported rate | Unseen final predicted-supported rate |
|---|:---:|---:|---:|---:|---:|
| `clean` | `true` | `14.592782902` | `-6.028514748` | `0.000000` | `0.000000` |
| `degraded_odor` | `true` | `9.316631684` | `-3.041524097` | `0.000000` | `0.000000` |
| `degraded_touch` | `true` | `14.592782902` | `-6.028514748` | `0.000000` | `0.000000` |
| `missing_touch` | `true` | `14.592782902` | `-6.028514748` | `0.000000` | `0.000000` |
| `missing_odor` | `false` | `0.000000000` | `0.000000000` | `1.000000` | `0.000000` |
| `contradictory_modalities` | `true` | `14.592782902` | `-6.028514748` | `0.000000` | `0.000000` |
| `temporal_misalignment` | `true` | `14.592782902` | `-6.028514748` | `0.000000` | `0.000000` |

For the six conditions with olfactory input, the closest unseen-family support margin remained below the seedwise locked threshold and the support gate classified the unseen-family records as unsupported.

Under `missing_odor`, the implementation assigned the locked threshold as the support score. The support decision was therefore uncertain and gate-supported at equality, but the odor-only policy abstained because no olfactory vector was available. Consequently, the H6 false-known reduction in this condition arose from missing-input abstention rather than seen-versus-unseen score separation.

## Scope of H6

H6 evaluated unsupported `unseen_family` events. It did not test whether the gate rejects `known_family_unseen_item` events. The latter share a family with training data but contain an identity that was not stored.

Accordingly, H6 supports validation-locked abstention for the registered synthetic unseen-family queries only. It does not establish general open-world recognition or rejection of every unrepresented item.

## Interpretation

1. H6 remains supported under the registered confirmatory rule and is numerically stable across all ten seeds.
2. The perfect H6 result must be interpreted alongside the strong synthetic seen-versus-unseen separation.
3. H7 remains not supported; no positive seed-level MRR effect was observed.
4. H8 remains not supported; its small pooled effect varied in direction across seeds and remained far below the registered practical threshold of `0.05`.
5. No new confirmatory claim is created by this analysis.

## Limitations

- This analysis was designed after inspection of the confirmatory results.
- It reuses the existing confirmatory raw records.
- It does not test more difficult or overlapping synthetic family geometries.
- It does not establish performance with physical odor sensors, tactile hardware, human participants, or biological olfaction.
- The exploratory p-values are descriptive sensitivity measures and do not replace the registered confirmatory tests.

## Reproducibility artifact

- `artifacts/noi_v0.3_confirmatory/posthoc_sensitivity.json`
- `artifacts/noi_v0.3_confirmatory/posthoc_sensitivity.json.sha256`
- Post-hoc artifact SHA-256: `b13cfca0d130d4f396d9a524cb31d44686114b2e18529247acf1ecc70710baa0`
