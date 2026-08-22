# NOI Baseline Pilot v0.1 — Provenance and Diagnostic Record

## Project

Neuro-Olfactive Interface (NOI)

## Ownership

Copyright © 2026 GUARDIANX LLC. All Rights Reserved.

This record, its associated software, and generated experimental artifacts
are proprietary materials of GUARDIANX LLC. No open-source license is
granted.

## Experiment Status

- Experiment identifier: `baseline-pilot-v0.1`
- Execution date: August 22, 2026
- Dataset type: Synthetic implementation evaluation
- Pilot dataset identifier: `pilot-v0.1`
- Baseline workflow commit: `a802ffd`
- Baseline experiment runner commit: `7578043`
- Baseline implementation commit: `274dd32`
- Event count: 200
- Odor-target count: 200
- Training events: 140
- Validation events: 20
- OOD test events: 40
- Generator seed: `1001`
- Independent OOD seed: `9001`
- Baseline random seed: `2026`
- Ridge alpha: `1.0`
- Retrieval depth: `10`
- Automated tests at execution: 187 passed
- Leakage audit: PASSED

## Results Artifact

File:

`results/baseline-pilot-v0.1/baseline_results.json`

SHA-256:

`4987f02fa6629226fe23cf6fe349c94a60a24f67c2be1b908f6ac291234754b9`

The SHA-256 value produced by the experiment workflow matched the independent
macOS `shasum -a 256` result.

## Aggregate Pilot Results

| Split | Baseline | Recall@1 | Recall@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|
| Validation | Random | 0.0000 | 0.0500 | 0.0071 | 0.0167 |
| Validation | Text-only cosine | 0.0000 | 0.0500 | 0.0100 | 0.0193 |
| Validation | Mean-fusion cosine | 0.0000 | 0.2000 | 0.0597 | 0.0931 |
| Validation | Ridge fusion | 0.9000 | 1.0000 | 0.9500 | 0.9631 |
| OOD test | Random | 0.0000 | 0.0750 | 0.0175 | 0.0311 |
| OOD test | Text-only cosine | 0.0000 | 0.0250 | 0.0025 | 0.0072 |
| OOD test | Mean-fusion cosine | 0.0000 | 0.0500 | 0.0056 | 0.0151 |
| OOD test | Ridge fusion | 0.0000 | 0.0250 | 0.0050 | 0.0097 |

## Diagnostic Interpretation

The ridge-fusion baseline achieved high retrieval performance on the
in-distribution validation split, including an MRR of 0.9500 and Recall@10
of 1.0000. This demonstrates that the training and validation data contain
a learnable linear relationship between the fused contextual features and
the synthetic odor vectors.

However, the ridge-fusion baseline did not preserve this performance on the
OOD split. Its OOD MRR was 0.0050 and its OOD Recall@10 was 0.0250. The
unlearned cosine baselines also performed near the random range on OOD data.

These results do not establish that ridge fusion, NOI, or contextual
olfactory memory succeeds or fails in general. The complete NOI model was
not evaluated in this experiment.

The joint collapse of the evaluated systems on the OOD split indicates that
the current independent OOD transformation may represent a severe or
potentially unlearnable distribution shift. It is also possible that the
learned baseline fitted generator-specific in-distribution structure that
does not transfer to the independent OOD generator.

## Required Follow-up Before Confirmatory Evaluation

Before running the planned 10,000-event experiment, the project must:

1. quantify the distribution shift between development and OOD features;
2. implement a clearly labeled diagnostic oracle upper bound;
3. determine whether recoverable target information exists within OOD
   contextual features;
4. distinguish mild, moderate, and severe OOD conditions;
5. verify that the OOD task is difficult but not structurally impossible;
6. preserve the original pilot and protocol without silent modification;
7. document any amended generator or protocol as a new version;
8. repeat the final comparison across the prespecified independent seeds;
9. calculate confidence intervals, effect sizes, and corrected statistical
   tests before evaluating the hypotheses.

An oracle trained using OOD labels may be used only as a diagnostic upper
bound. It must not be presented as a fair deployable baseline or as evidence
of generalization.

## Scientific Scope and Limitations

This experiment is an exploratory synthetic pilot and implementation
diagnostic.

It does not establish:

- human olfactory or perceptual validity;
- clinical or diagnostic capability;
- safe operation of a physical odor-emission device;
- comprehensive chemical or environmental safety;
- successful real-world generalization;
- superiority of the complete NOI architecture;
- confirmation or rejection of the preregistered NOI hypotheses.

The observed validation performance must not be reported without the
corresponding OOD collapse. Positive and negative pilot findings must be
reported together.

## Reproduction Command

From the repository root with the project environment activated:

```bash
python -m experiments.run_baseline_pilot
```

If intentional replacement of an existing exploratory result is required:

```bash
python -m experiments.run_baseline_pilot --overwrite
```