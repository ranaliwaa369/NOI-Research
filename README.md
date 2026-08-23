# Neuro-Olfactive Intelligence (NOI)

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

The current public research-software release is version 0.1.0.

This is an exploratory release containing implemented experiments,
machine-readable results, provenance records, locked configurations,
verification functions, and automated tests.

It is not confirmatory evidence of real-world system performance.

## Implemented Evaluations

- Baseline retrieval pilot
- Graded out-of-distribution pilot
- NOI component-ablation pilot
- Controlled corrective-memory pilot
- Deterministic policy-conformance pilot

Versioned configurations are in `configs/`.
Experiment implementations are in `experiments/`.
Provenance records are in `docs/`.
Machine-readable outputs are in `results/`.

## Main Exploratory Findings

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

The principal protocol is `configs/research_protocol.yaml`.

## Environment and Testing

Version 0.1.0 uses Python 3.11.

Create an environment and install the locked dependencies:

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements-lock.txt

Run the complete test suite:

    python -m pytest -q

Experiment entry points:

- `experiments/run_baseline_pilot.py`
- `experiments/run_graded_ood_pilot.py`
- `experiments/run_noi_ablation_pilot.py`
- `experiments/run_corrective_memory_pilot.py`
- `experiments/run_policy_conformance_pilot.py`

Consult the corresponding configuration and provenance record before
rerunning an experiment or interpreting its outputs.

## Citation

Citation metadata are provided in `CITATION.cff`.

The permanent Zenodo DOI will be added after archival deposition of the
version 0.1.0 GitHub release.

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
