# NOI v0.2 Memory Evaluation Plan

## Status

Prespecified development plan for version 0.2.

This document does not modify or reinterpret the locked v0.1.0 results.

## v0.1 Reachability Audit

The deterministic v0.1 data generator produced:

- Train: 84 unique target items
- Validation: 19 unique target items
- OOD test: 27 unique target items

Train-to-validation overlap:

- 14 shared targets
- 5 targets unseen by memory
- Memory-reachable fraction: 0.7368

Train-to-OOD-test overlap:

- 0 shared targets
- 27 targets unseen by memory
- Memory-reachable fraction: 0.0

Therefore, the v0.1 memory-only system had no possible route to the
correct target for any OOD-test event. Its zero OOD MRR is a structural
reachability result under the locked odor-family-held-out design. It
must not be represented as an isolated test of episodic-memory quality.

## Evaluation Tracks

### Track A: Seen-item episodic retrieval

Purpose: test whether associative memory provides retrieval utility when
the correct odor item is represented in memory.

Requirements:

- Every evaluation target must have at least one training memory.
- Evaluation events, contexts, and templates remain held out.
- No duplicate events or template leakage across splits.
- Report performance under noise, missing modalities, and temporal
  displacement.
- Compare memory-only, ridge-only, and hybrid systems.

### Track B: Unseen-family generalization

Purpose: test generalization to odor families absent from memory.

Requirements:

- Preserve strict odor-family separation.
- Do not treat memory-only exact-item retrieval as the primary metric
  when the correct item is unreachable.
- Evaluate ridge and other representation-based baselines.
- Evaluate whether memory-aware systems abstain or defer when memory
  support is absent.
- Report target reachability and coverage with every result.

## Prespecified Systems

1. Ridge retrieval
2. Cosine or nearest-neighbor retrieval
3. Memory-only nearest-event retrieval
4. Per-item memory-prototype retrieval
5. Recency-weighted associative memory
6. Calibrated ridge-memory hybrid
7. Hybrid with reachability-aware abstention

## Repeated Runs

- Use at least 10 independent prespecified seeds.
- Preserve an identical evaluation protocol across systems.
- Retain every run, including negative results.
- Prohibit tuning on the locked final test sets.

## Primary Metrics

- Recall at 1
- Recall at 10
- Mean reciprocal rank
- nDCG at 10
- Reachable-target fraction
- Abstention coverage
- Selective error rate

## Statistical Reporting

- Report per-seed results.
- Report means and dispersion.
- Report paired confidence intervals for system differences.
- Use paired comparisons because systems evaluate identical events.
- Correct for multiple comparisons where applicable.
- Report effect sizes as well as significance tests.

## Primary Memory Question

Under the seen-item evaluation, does associative memory improve retrieval
over ridge and nearest-neighbor baselines without using evaluation-label
information or leaking held-out events?

## Generalization Question

Under the unseen-family evaluation, can the representation-based system
generalize while the memory-aware system identifies unsupported queries
and abstains or defers safely?

## Non-Negotiable Controls

- No evaluation target may be silently treated as memory-reachable.
- Every exported result must include a reachability indicator.
- Seen-item and unseen-family metrics must never be pooled without a
  stratified breakdown.
- v0.1 results remain immutable.
- All v0.2 configuration files, manifests, seeds, and outputs must be
  versioned and hashed.
