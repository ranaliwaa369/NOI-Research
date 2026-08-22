# NOI Synthetic Pilot v0.1 — Provenance Record

## Project

Neuro-Olfactive Interface (NOI)

## Ownership

Copyright © 2026 GUARDIANX LLC. All Rights Reserved.

This provenance record and the associated software and generated artifacts
are proprietary materials of GUARDIANX LLC. No open-source license is
granted.

## Pilot Status

- Pilot identifier: `pilot-v0.1`
- Generation date: August 22, 2026
- Dataset type: Synthetic implementation evaluation
- Software commit: `d3d6810`
- Generator version: `0.1.0`
- Generator seed: `1001`
- Independent OOD seed: `9001`
- Research protocol SHA-256:
  `e885f537b3209d7052d1517efc5be75324df39689cb4e54b38a241c95f22e512`

## Dataset Composition

- Odor targets: 200
- Total contextual events: 200
- Training events: 140
- Validation events: 20
- OOD test events: 40

## Exported Artifact Hashes

### odor_targets.jsonl

SHA-256:

`171d69859f8133313cc0ed906f4a739bdee3340769d58be71be530bc3575e748`

### events.jsonl

SHA-256:

`c5a5556b46a036c4b1baa2e553c1d1c9e0593c27c1c2663b6da3e88a8d385a82`

### dataset_manifest.json

SHA-256:

`4187e4e0fd5fe9be40750a79f49c36f9b70a0706c0ad046aeb7d6b289078eb69`

## Validation Results

- Automated tests at pilot generation: 143 passed
- Leakage audit: PASSED
- Export SHA-256 verification: PASSED
- Duplicate event identifiers detected: None
- Cross-split feature duplicates detected: None
- OOD odor-family overlap detected: None
- Context-template overlap detected: None
- Target-family inconsistencies detected: None
- Required splits missing: None

## Reproduction Command

From the repository root with the project environment activated:

```bash
python -m experiments.export_synthetic_pilot
```

If the documented output directory already exists and intentional replacement
is required:

```bash
python -m experiments.export_synthetic_pilot --overwrite
```

## Scientific Scope and Limitations

This pilot validates the implemented synthetic-data generation, partitioning,
leakage-audit, deterministic export, and integrity-verification workflow.

It does not establish:

- human olfactory or perceptual validity;
- clinical or diagnostic performance;
- performance of a physical odor-emission device;
- comprehensive chemical, environmental, or human safety;
- generalization to real-world odorants, mixtures, concentrations, people,
  environments, or hardware.

The pilot must therefore not be described as evidence that NOI possesses
human-like olfactory memory. It is evidence only that the prespecified
software workflow executed reproducibly on the documented synthetic pilot.