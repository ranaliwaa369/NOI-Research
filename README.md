# Neuro-Olfactive Interface (NOI)

## Overview

The Neuro-Olfactive Interface (NOI) is a software-first artificial-intelligence research project for contextual odor association, temporally structured retrieval, corrective memory updating, and policy-constrained output.

The initial system retrieves an odor-library item or simulated cartridge identifier from multimodal contextual input. Every candidate output is evaluated by a deterministic and auditable policy gate before an output command can be authorized.

## Research Objective

This project investigates whether a multimodal associative-memory architecture can improve context-to-odor retrieval over strong non-associative and reduced-component baselines under in-distribution, out-of-distribution, noisy-input, missing-modality, and temporal-displacement conditions.

## Initial Use Case

The first use case is limited to selecting an item from a fixed odor library or a simulated cartridge inventory based on multimodal contextual information.

The current implementation does not physically emit scent.

## Proposed Architecture

NOI will contain the following principal components:

1. Multimodal context encoder
2. Odor-representation module
3. Temporally structured associative-memory store
4. Corrective memory-update mechanism
5. Deterministic policy gate
6. Auditable event and decision log

## Research Scope

The current research stage covers:

- Software architecture
- Synthetic in-distribution evaluation
- Independently generated synthetic out-of-distribution evaluation
- External benchmark evaluation where suitable data are available
- Baseline and ablation comparisons
- Reproducible statistical analysis
- Policy-conformance testing

## Explicit Limitations

This project does not currently claim:

- Human-like olfactory memory
- Human perceptual validity of retrieved odor representations
- Disease diagnosis or clinical effectiveness
- Safe physical scent emission
- Comprehensive real-world safety
- Completion of human-subject experiments

Any future human-subject, clinical, or physical-emission study will require a separately defined protocol, appropriate ethical review, domain expertise, and relevant safety validation.

## Prespecified Hypotheses

The primary hypotheses address:

- Contextual retrieval performance
- Corrective memory updating
- Policy conformance
- Robustness under noise, missing modalities, temporal displacement, and held-out data families

The prespecified thresholds, evaluation rules, splits, metrics, and statistical procedures are recorded in:

`configs/research_protocol.yaml`

## Reproducibility

The project is intended to preserve:

- Fixed and documented random seeds
- Versioned configuration files
- Locked final test sets
- Documented data-generation procedures
- Complete baseline comparisons
- All experimental runs, including negative results
- Environment and dependency versions

## Status

Pre-implementation research protocol and software scaffold.

No experimental performance results are reported at this stage.

## Authors

Rana Al-Dahlake and Jalal Alazirji  
GuardianX LLC  
Tukwila, Washington, United States

## License

Licensing terms will be finalized before the first public release.

## Copyright and License

Copyright © 2026 GUARDIANX LLC. All Rights Reserved.

This project is proprietary and is not licensed as open-source software.
No permission is granted to copy, modify, distribute, publish,
commercialize, reverse engineer, sublicense, or create derivative works
without prior written authorization from GUARDIANX LLC.

See `COPYRIGHT.md` for the complete proprietary-rights notice.
