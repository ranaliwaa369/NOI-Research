# NOI Research Framework v0.1.0

This is the first archived exploratory release of the Neuro-Olfactive
Intelligence (NOI) reproducible research framework.

## Included evaluations

- Baseline retrieval pilot.
- Graded out-of-distribution pilot.
- NOI component-ablation pilot.
- Controlled corrective-memory pilot.
- Deterministic policy-conformance pilot.
- Reproducible JSON results, summaries, and run manifests.
- Automated verification and test suite.

## Main observations

The graded out-of-distribution evaluation produced mean reciprocal rank
values of approximately 0.9292 for mild shift, 0.4406 for moderate shift,
and 0.0050 for severe shift.

In the tested ablation, the full hybrid configuration matched the ridge
configuration across the evaluated tier/day conditions. The memory-only
configuration produced a mean reciprocal rank of zero. Therefore, this
release does not claim an incremental retrieval benefit from the tested
memory component.

In the controlled corrective-memory pilot, explicit corrections improved
the selected correction queries, with a mean MRR improvement of
approximately 0.9286 and no measured degradation on the locked preservation
set. This does not demonstrate automatic error detection.

The policy-conformance pilot produced exact outcomes on all 26 locked
simulated cases, with no false allows or false blocks. Not comprehensive
safety evidence.

## Scope statement

This release provides exploratory evidence from controlled simulations.
It does not demonstrate physical odor sensing, clinical validity,
general real-world performance, deployment readiness, or comprehensive
safety.

## Intellectual property

Copyright 2026 GuardianX LLC. All Rights Reserved.

The repository is publicly accessible for scientific inspection,
verification, citation, and archival purposes. Public access does not
grant permission to copy, modify, redistribute, deploy, commercialize,
or create derivative works.
