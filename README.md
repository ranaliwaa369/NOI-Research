# Neuro-Olfactive Intelligence (NOI)

[![v0.3 archive DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22180610.svg)](https://doi.org/10.5281/zenodo.22180610)

## Overview

Neuro-Olfactive Intelligence (NOI) is a software-first artificial
intelligence research framework for contextual odor association,
temporally structured retrieval, corrective memory updating, graded
out-of-distribution evaluation, and policy-constrained output.

The current implementation operates on synthetic multimodal records and
simulated odor-library or cartridge identifiers. It does not physically
sense, generate, or emit odor.

## Research Objective

This project evaluates multimodal and associative-memory components
against non-associative and reduced-component baselines under controlled
in-distribution, out-of-distribution, noisy-input, missing-modality,
temporal-displacement, correction, and policy-conformance conditions.

## Release Status

The current release candidate is version 0.3.0.

Version 0.3 adds a locked confirmatory evaluation of support-aware,
reliability-gated synthetic olfactory-tactile retrieval. It includes ten
prespecified seeds, seven paired conditions, nine systems, 1,260,000
system evaluations, paired bootstrap analysis, Holm correction,
machine-readable aggregate artifacts, SHA-256 verification, and a
complete research manuscript.

The evidence is synthetic computational evidence. It is not evidence of
real-world odor sensing, human or animal olfactory equivalence, clinical
effectiveness, chemical safety, or deployment readiness.

## Implemented Evaluations

- Baseline retrieval pilot
- Graded out-of-distribution pilot
- NOI component-ablation pilot
- Controlled corrective-memory pilot
- Deterministic policy-conformance pilot
- Repeated Track A seen-item evaluation
- Track B unseen-family evaluation
- Selective memory-support abstention evaluation
- Missing-modality robustness evaluation
- Temporal-displacement robustness evaluation
- Support-aware open-set gating evaluation
- Conditional simulated-tactile utility evaluation
- Conflict-aware reliability-gated fusion evaluation
- Ten-seed NOI v0.3 confirmatory analysis
- Independent raw-record result audit

Versioned configurations are in `configs/`.
Experiment implementations are in `experiments/`.
Provenance records are in `docs/`.
Machine-readable outputs are in `results/`.

## Version 0.3 Confirmatory Findings

The locked v0.3 confirmatory evaluation completed all ten registered
seeds, covering 20,000 final-test latent events, 140,000 paired condition
views, nine systems, and 1,260,000 system evaluations.

The primary hypothesis H6 was supported. The validation-locked support
gate reduced unseen-family false-known decisions by `1.000000`, with a
95% paired bootstrap interval of `[1.000000, 1.000000]`, while producing
`0.000000` clean seen-item MRR loss.

The secondary hypothesis H7 was not supported. Reliability-gated
olfactory-tactile fusion was slightly worse than fixed-weight fusion in
the eligible degraded-odor and missing-odor conditions. The absolute MRR
difference was `-0.002212500`, with a 95% interval of
`[-0.002662500, -0.001750000]`.

The secondary hypothesis H8 was not supported. The false-confident
reduction was `0.004008333` against both fixed-weight fusion and naive
concatenation. Although statistically detectable, this effect remained
below the prespecified practical requirement of `0.05`.

These findings support validation-locked rejection of registered
synthetic unseen-family queries. They do not establish
tactile synergy, physical-sensor performance, biological equivalence,
clinical validity, chemical safety, or deployment readiness.

See `docs/noi_v0.3_final_results.md` for the complete results,
`docs/noi_v0.3_manuscript.md` for the research manuscript, and
`docs/noi_v0.3_posthoc_sensitivity.md` for the separately labeled
exploratory seed-level and hierarchical sensitivity analysis, and
`docs/noi_v0.3_trace_audit.md` for the read-only leakage and integrity
trace.

## Version 0.2 Findings

The repeated Track A evaluation found a positive associative-memory
contribution for previously represented target items. The mean paired
memory-only minus ridge-only MRR advantage was `0.009066818`, with a 95%
paired bootstrap interval of `[0.004399445, 0.014384734]`.

Track B found that direct unseen-family target identification remained
unresolved. The validation-calibrated support mechanism nevertheless
abstained at high rates for unsupported unseen-family events, without
using an OOD oracle or final-test tuning.

The final robustness hypothesis H4 was not supported. Ridge-only was the
strongest eligible baseline in all 36 missing-modality and
temporal-displacement condition-tier tests, and the full hybrid exceeded
it in 0 of 36 tests.

Together, these results show that the current associative memory helps
with previously represented items but does not solve unseen-family
identity retrieval. The negative robustness result is retained as an
architectural limitation and motivates future support-gated routing.

See `docs/noi_v0.2_final_results.md` for the complete interpretation and
reproducibility record.

## Version 0.1 Exploratory Findings

### Graded out-of-distribution behavior

| OOD tier | Mean reciprocal rank |
|---|---:|
| Mild | 0.9292 |
| Moderate | 0.4406 |
| Severe | 0.0050 |

Performance declined as synthetic distribution shift increased. These
values should not be generalized to physical odor sensing or uncontrolled
environments.

### NOI component ablation

The selected validation value was alpha = 1.0.

Across the evaluated conditions, the tested full hybrid configuration
matched the ridge configuration. The memory-only configuration produced
a mean reciprocal rank of zero.

Version 0.1.0 therefore does not claim an incremental retrieval benefit
from the tested associative-memory component. This negative result is
retained in the reproducible record.

### Controlled corrective memory

The pilot evaluated 14 correction targets across 15 queries. The observed
mean MRR improvement was approximately 0.9286, with a reported 95%
interval from approximately 0.8214 to 1.0000. Measured degradation on the
locked preservation set was 0.0.

This supports explicit correction under the controlled protocol only. It
does not demonstrate automatic error detection or autonomous correction.

### Policy conformance

The policy-conformance pilot produced:

- 26 of 26 exact outcomes
- 0 false allows
- 0 false blocks
- 1.0 evaluated rule coverage

These results establish conformance only to the locked simulated policy
cases. Not comprehensive safety evidence.

## Explicit Limitations

This release does not establish:

- Human-like olfactory memory
- Physical odor sensing or scent emission
- Human perceptual validity
- Disease diagnosis or clinical effectiveness
- General real-world performance
- Safe deployment in uncontrolled environments
- Comprehensive safety
- Completion of human-subject experiments

Future human-subject, clinical, environmental, industrial, or
physical-emission studies require separate protocols, appropriate ethical
review, domain expertise, hardware validation, and relevant safety tests.

## Reproducibility

The repository preserves fixed seeds, versioned and hashed configurations,
locked evaluation cases, documented synthetic-data generation, baseline
and ablation comparisons, negative results, run manifests, dependency
records, and automated tests.

The v0.3 protocol is `configs/noi_v0.3_protocol.yaml`, with its locked amendment, validation lock, and execution specification retained in `configs/`.

## Environment and Testing

Version 0.3.0 uses Python 3.11.

Create an environment and install the locked dependencies:

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements-lock.txt

Run the complete test suite:

    python -m pytest -q

Recorded release-candidate verification:

- Command: `python -m pytest -q`
- Git commit: `b56427f`
- Execution date: August 30, 2026
- Result: `1309 passed in 50.24s`
- Exit code: `0`

This is a historical execution record tied to the stated commit, not a
continuously updating test count.

Experiment entry points:

- `experiments/run_baseline_pilot.py`
- `experiments/run_graded_ood_pilot.py`
- `experiments/run_noi_ablation_pilot.py`
- `experiments/run_corrective_memory_pilot.py`
- `experiments/run_policy_conformance_pilot.py`
- `experiments/run_noi_v0_3_confirmatory.py`
- `experiments/analyze_noi_v0_3_confirmatory.py`
- `experiments/finalize_noi_v0_3_confirmatory.py`
- `experiments/analyze_noi_v0_3_posthoc_sensitivity.py`
- `experiments/audit_noi_v0_3_confirmatory_trace.py`

Consult the corresponding configuration and provenance record before
rerunning an experiment or interpreting its outputs.

## Citation

Citation metadata are provided in `CITATION.cff`.

Version 0.3 archival DOI:

https://doi.org/10.5281/zenodo.22180610

Version 0.2 archival DOI:

https://doi.org/10.5281/zenodo.22139127

## Authors

Rana Al-Dahlake and Jalal Alazirji
GuardianX LLC
Tukwila, Washington, United States

Corresponding author: Rana Al-Dahlake
ORCID: 0009-0001-8919-8177
Email: info@researchguardianx.com

## Intellectual Property and Access

Copyright © 2026 GuardianX LLC. All Rights Reserved.

This repository is proprietary and is not open-source software. Public
availability is provided for scientific inspection, verification,
citation, and archival purposes only.

No permission is granted to copy, modify, distribute, publish,
commercialize, deploy, reverse engineer, sublicense, or create derivative
works without prior written authorization from GuardianX LLC.

See `LICENSE` and `COPYRIGHT.md`.

## Repository

Official source repository:

https://github.com/ranaliwaa369/NOI-Research
