# NOI v0.5 Novel-Odor Intelligence Plan

## Status

This is a development plan, not a result. NOI v0.4 established auditable
ingestion of 300 SmellNet-Base physical-sensor recordings: 250 official
training recordings and 50 official testing recordings across 50 ingredients
and five families, with no duplicate identities or byte-identical files across
the official split boundary.

NOI v0.5 asks the next scientific question: can a physical-sensor AI system
handle a genuinely unrepresented odor family without converting uncertainty
into a confident known label?

## Problem and urgency

Closed-set odor classifiers must choose a represented identity. That design is
misaligned with embodied AI, where a gas-sensor observation can be novel,
degraded, drifting, contradictory, or simply too short. A wrong confident
label may cause an automated system to act on evidence it does not support.

The required capability is therefore larger than classification accuracy. A
bounded machine-smell system must decide whether to recognize, collect another
independent temporal window, or abstain as unknown or insufficient evidence.

## Research gap

Temporal odor classification, open-set recognition, drift compensation,
probabilistic detection, and early rejection each have prior literature. The
targeted gap is their reproducible integration with associative evidence
memory and an adaptive repeat-sensing action under one family-level novelty
protocol.

The proposed contribution is not the claim that open-set recognition or
reject options are new. It is a validation-locked architecture and comparison
that connects:

1. physical multichannel odor time series;
2. temporal representations;
3. training-only associative evidence memory;
4. family-level support and reliability estimates;
5. recognize, repeat-sense, and abstain actions.

## Locked hypothesis structure

### H10 — Novel-family false-known reduction (primary)

The complete locked policy must reduce false-known decisions on a held-out
odor family relative to the strongest eligible baseline while preserving
supported-odor performance within a prespecified margin.

### H11 — Repeat-sensing decision utility (secondary)

For queries defined as ambiguous using validation evidence only, a second
nonoverlapping sensor window must improve bounded decision utility over a
forced decision from the first window without increasing false-known risk.

### H12 — Reliability-aware memory under shift (secondary)

Training-only associative memory, gated by reliability, must improve
session-shift robustness over the strongest non-memory baseline without an
unacceptable clean-condition loss.

No effect size, margin, threshold, ambiguity rule, window length, or model
selection rule is confirmatory until it is locked after development pilots
and before any outer final-test result is viewed.

## Five nested family rotations

The locked family order is fruits, herbs, nuts, spices, and vegetables. Each
outer fold uses one family as final unknown, the next family in the cycle as
validation unknown, and the remaining three as known. This produces five
outer evaluations and prevents selection of an easy unknown family after
viewing its result.

Within each fold:

- known-family official training recordings supply model training and
  validation-known evidence;
- validation-unknown official training recordings select novelty thresholds;
- the final-unknown family is absent from development;
- official testing recordings are reserved for final known and final unknown
  evaluation;
- no recording identity or temporal window may cross roles.

## Current implementation milestone

The initial v0.5 code provides:

- deterministic five-fold family roles;
- recording-grouped training, validation, final, and unused partitions;
- fixed-size nonoverlapping sensor windows;
- disjoint initial/repeat-sensing pairs;
- automated leakage and boundary tests.

It does not yet train a representation, memory, or decision policy and does
not contain an H10-H12 result.

## Defensible future claim

If all locked hypotheses succeed, NOI v0.5 may claim sensor-grounded evidence
for a neural-inspired computational pathway that handles novel odors through
support-aware memory, additional evidence acquisition, and bounded decisions.
It must not claim direct neural interfacing, biological equivalence, clinical
validity, chemical safety, or deployment readiness.
