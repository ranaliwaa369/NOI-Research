# NOI Graded-OOD Pilot v0.2 Provenance

Copyright © 2026 GUARDIANX LLC. All rights reserved.

## Status

Exploratory synthetic implementation evaluation.

This pilot is not the final confirmatory evaluation and does not establish human perceptual validity, clinical effectiveness, physical odor-emission safety, or deployment readiness.

## Repository State

- Workflow commit before provenance: `cc5cfa3498b1bb835f6233f4f9db129decae58e8`
- Protocol tag: `protocol-v0.1-preimplementation`
- Amendment tag: `protocol-amendment-v0.2-preimplementation`
- Generation-definition tag: `graded-ood-generation-v1-preimplementation`
- Test suite before export: `412 passed`

## Governing Configuration Hashes

- `graded_ood_generation.yaml`: `45d247dfb18152d64701b3f088be950aeece07ff3592eba82e702021b2ea3cb3`
- `protocol_amendment_v0.2.yaml`: `5292e0208576a79342923fdf72855de39bffe63331721fff01b94ae35aa39f7b`
- `research_protocol.yaml`: `e885f537b3209d7052d1517efc5be75324df39689cb4e54b38a241c95f22e512`
- `synthetic_data.yaml`: `16cc1ddfb752f5f4955931fd11bb5544af919153fca5650fe805b5a1f83ad817`

## Dataset and Verification

- Original synthetic events: 200
- Odor targets: 200
- Independent latent OOD events: 40
- Observed graded-OOD rows: 120
- Tier rows: 40 mild, 40 moderate, 40 severe
- Severe-reference replay: PASSED
- Paired-ground-truth verification: PASSED
- Analysis unit: `latent_event_id`
- OOD oracle used: No

## Baseline Results

| Tier | Baseline | R@1 | R@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|
| mild | random | 0.0250 | 0.0500 | 0.0300 | 0.0347 |
| mild | text_only_cosine | 0.0000 | 0.0500 | 0.0063 | 0.0159 |
| mild | mean_fusion_cosine | 0.0250 | 0.1500 | 0.0667 | 0.0869 |
| mild | ridge_fusion | 0.9000 | 0.9750 | 0.9292 | 0.9408 |
| moderate | random | 0.0250 | 0.0500 | 0.0300 | 0.0347 |
| moderate | text_only_cosine | 0.0000 | 0.0250 | 0.0036 | 0.0083 |
| moderate | mean_fusion_cosine | 0.0250 | 0.1500 | 0.0534 | 0.0754 |
| moderate | ridge_fusion | 0.2500 | 0.8500 | 0.4406 | 0.5395 |
| severe | random | 0.0250 | 0.0500 | 0.0300 | 0.0347 |
| severe | text_only_cosine | 0.0000 | 0.0250 | 0.0025 | 0.0072 |
| severe | mean_fusion_cosine | 0.0000 | 0.0500 | 0.0056 | 0.0151 |
| severe | ridge_fusion | 0.0000 | 0.0250 | 0.0050 | 0.0097 |

## Paired MRR Contrasts

Contrasts are lower-severity minus higher-severity MRR. Bootstrap resampling used the 40 latent events as the paired analysis units.

| Baseline | Contrast | ΔMRR | 95% bootstrap CI | Better/Tied/Worse |
|---|---|---:|---:|---:|
| random | mild_minus_moderate | 0.0000 | [0.0000, 0.0000] | 0/40/0 |
| random | moderate_minus_severe | 0.0000 | [0.0000, 0.0000] | 0/40/0 |
| random | mild_minus_severe | 0.0000 | [0.0000, 0.0000] | 0/40/0 |
| text_only_cosine | mild_minus_moderate | 0.0028 | [0.0000, 0.0083] | 1/39/0 |
| text_only_cosine | moderate_minus_severe | 0.0011 | [-0.0075, 0.0107] | 1/38/1 |
| text_only_cosine | mild_minus_severe | 0.0038 | [-0.0050, 0.0146] | 2/37/1 |
| mean_fusion_cosine | mild_minus_moderate | 0.0133 | [-0.0258, 0.0537] | 5/33/2 |
| mean_fusion_cosine | moderate_minus_severe | 0.0478 | [0.0021, 0.1114] | 6/32/2 |
| mean_fusion_cosine | mild_minus_severe | 0.0610 | [0.0083, 0.1288] | 6/32/2 |
| ridge_fusion | mild_minus_moderate | 0.4885 | [0.3785, 0.5956] | 29/11/0 |
| ridge_fusion | moderate_minus_severe | 0.4356 | [0.3292, 0.5477] | 34/6/0 |
| ridge_fusion | mild_minus_severe | 0.9242 | [0.8492, 0.9833] | 39/1/0 |

## Statistical Settings

- Bootstrap seed: `4242`
- Bootstrap resamples: `10000`
- Confidence level: `0.95`
- These intervals quantify uncertainty within the exploratory 40-event paired pilot; they do not represent replication across independent real-world datasets.

## Export Hashes

- `graded_ood_results.json`: `867556aed35c487ba4aaf8b9e27ca30e35067812732511edd5284c88a3174cd2`
- `paired_statistics.json`: `16b8519a51c4899033050a972569ef31b7bbd17de6ed001b77b13b881c00ab2a`
- `run_manifest.json`: `39feb9415ee6ac36c41a72ff1210d849f063717ea59eb31cac68b78a160e4610`

## Reproduction

```bash
python -m pytest -q
python -m experiments.run_graded_ood_pilot --overwrite
```

## Interpretation Boundary

The pilot demonstrates that the registered synthetic shift construction produces a controlled performance gradient for the ridge baseline: strong performance under mild shift, reduced performance under moderate shift, and failure under the retained severe stress test. This is evidence about the software evaluation environment, not evidence of human-like olfaction or real-world odor perception.
