# NOI v0.4 Sensor-Grounded Neuro-Olfactive Intelligence Plan

## Status

This document opens the development track after the published NOI v0.3
synthetic confirmatory study. It is an implementation plan, not a result.

## Problem

NOI v0.3 showed that a validation-locked support gate could reject the
registered synthetic unseen-family queries without reducing clean seen-item
MRR. The study did not use physical sensors. Its tactile-synergy hypothesis
and practical conflict-safety hypothesis were not supported.

The next unresolved question is whether a support-aware, memory- and
reliability-aware decision policy remains useful when its olfactory
representation comes from physical gas-sensor time series rather than a
separable synthetic generator.

The practical problem is larger than classification accuracy. A physical AI
system must determine whether an observed odor is represented, degraded,
drifting, contradictory, or genuinely novel. It must then choose whether to
recognize, collect another nonoverlapping sensor window, request another form
of evidence, or abstain. Current NOI evidence does not yet establish this
capability.

## Why this step matters now

Public machine-olfaction resources now provide physical sensor time series,
but real-world and unseen-mixture generalization remain difficult. At the
same time, embodied AI is increasingly expected to act from uncertain sensor
evidence. A high-confidence known answer to an unsupported chemical signal
can be more consequential than an explicit unknown decision.

The SmellNet benchmark provides a documented bridge from physical signals to
an auditable NOI decision layer. This makes a sensor-grounded test possible
without overstating that NOI has already built or validated a robot nose.

## Research gap

SmellNet primarily benchmarks substance classification and mixture-ratio
prediction. NOI v0.4 will evaluate a different but complementary question:
whether a physical-sensor AI system can jointly use temporal representation,
associative evidence memory, support estimation, and reliability assessment
to avoid a forced known identity.

The targeted gap is therefore a **sensor-to-context epistemic bridge**:

1. convert physical odor time series into a temporal representation;
2. compare current evidence with represented odor memories;
3. distinguish insufficient support from ordinary classification ambiguity;
4. estimate degradation, drift, and conflict;
5. choose among recognition, repeat sensing, and abstention.

The same decision pattern can later generalize to other embodied AI systems
that operate on noisy, drifting, incomplete, or previously unseen sensor
inputs.

## Proposed contribution

NOI v0.4 is intended to contribute a reproducible architecture and evaluation
protocol for **bounded machine smell**. Its novelty is not a new gas sensor.
Its contribution is the reliability layer between physical sensing and AI
action.

If the registered hypotheses succeed, the defensible claim will be:

> Physical odor-sensor streams can be connected to a neural-inspired,
> support-aware memory and decision architecture that recognizes represented
> evidence, detects unsupported inputs, requests additional temporal evidence
> when useful, and abstains when reliable identification is not justified.

This would be evidence for a computational pathway toward neuro-olfactive
intelligence. It would not be evidence of direct stimulation or decoding of
the olfactory nerve or brain.

## Draft research hypotheses

### H9 — Sensor-grounded open-set support

A validation-locked support gate will reduce false-known decisions for
held-out physical-sensor odor families without materially reducing retrieval
for represented substances.

### H10 — Repeat-sensing utility

For ambiguous queries, a second nonoverlapping temporal window will improve
the bounded decision relative to forced prediction from a single window,
without increasing false-known decisions.

### H11 — Reliability-aware associative memory

Reliability-aware retrieval from previously represented sensor recordings
will improve robustness under session shift relative to the strongest
non-memory baseline without unacceptable clean-condition loss.

All effect thresholds, split rules, model-selection rules, and exclusion
criteria must be fixed after development pilots and before final-test
inspection.

## Evidence ladder

1. **Completed — synthetic decision evidence.** NOI v0.3 tested support,
   reliability, conflict, missingness, and abstention with simulated vectors.
2. **Current — sensor-grounded ingestion.** Read, validate, hash, and audit
   SmellNet-Base physical-sensor recordings without using labels as features.
3. **Next — locked open-set protocol.** Create recording-grouped development,
   validation, and final-test partitions with disjoint unknown families and,
   where metadata permit, disjoint recording sessions.
4. **Then — sensor-grounded confirmatory evaluation.** Compare strong
   closed-set, open-set, temporal, and non-memory baselines with the complete
   support-aware memory policy.
5. **Future — embodied validation.** Connect a validated model to physical
   hardware and test sampling, airflow, localization, latency, and safety.

## Stage 0 deliverable

The current milestone adds:

- a six-channel SmellNet-Base CSV adapter;
- deterministic discovery of the official training/testing folders;
- source-file SHA-256 provenance;
- label/feature separation;
- rejection of missing, nonnumeric, nonfinite, or unknown-schema data;
- duplicate-ID and cross-split content checks;
- automated unit tests with small local fixtures.

No SmellNet raw data are committed to this repository.

## Claim boundary

Completion of Stage 0 will show only that the NOI codebase can ingest and
audit real physical-sensor records. It will not prove recognition accuracy,
embodied robotic smell, odor-source localization, direct neural interfacing,
real-world robustness, human-like olfaction, clinical validity, chemical
safety, or deployment readiness.

## Source

SmellNet: Dewei Feng, Wei Dai, Carol Li, Alistair Pernigo, Yunge Wen, and
Paul Pu Liang, *SmellNet: A Large-Scale Dataset for Real-World Smell
Recognition*, ICLR 2026. Paper: <https://arxiv.org/abs/2506.00239>. Code:
<https://github.com/MIT-MI/SmellNet>. Dataset:
<https://huggingface.co/datasets/DeweiFeng/SmellNet>.
