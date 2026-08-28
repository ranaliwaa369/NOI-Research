# NOI v0.2.0 Release Notes

DOI: https://doi.org/10.5281/zenodo.22139127

## Overview

NOI v0.2.0 closes the second reproducible synthetic evaluation package
for Neuro-Olfactive Intelligence.

The release adds repeated seen-item evaluation, unseen-family
generalization, selective memory-support abstention, missing-modality
robustness, temporal-displacement robustness, paired multi-seed
bootstrap analysis, deterministic exports, and computational
environment provenance.

## Principal findings

### Track A: seen-item associative memory

Across ten independent runs, memory-only minus ridge-only produced:

- Mean paired MRR advantage: `0.009066818`
- 95% paired bootstrap interval: `[0.004399445, 0.014384734]`
- Wins, ties, and losses: `9, 0, and 1`

This supports a repeatable associative-memory contribution for
previously represented items under the locked synthetic design.

### Track B: unseen-family generalization

Direct unseen-family identity retrieval remained unresolved.

The selective support mechanism produced mean abstention rates of:

- Mild: `0.96145`
- Moderate: `0.99420`
- Severe: `0.99915`

The locked synthetic calibration-and-abstention criterion was supported
without OOD oracle use or final-test tuning.

### Final robustness and H4

The final evaluation included 10 independent runs, 7 missing-modality
conditions, 5 temporal-displacement conditions, 3 severity tiers, and
4 systems.

Ridge-only was the strongest eligible baseline in all 36
condition-tier tests.

H4 was not supported:

- Supported condition-tier tests: `0`
- Unsupported condition-tier tests: `36`
- Full-hybrid minus strongest-baseline intervals were below zero in
  every condition-tier test.

This negative result is retained as an architectural limitation. It
does not invalidate the positive Track A finding or the Track B
abstention result.

## Reproducibility

- Complete automated suite: `778 passed`
- Raw seed artifacts: 10 Track A runs, 10 Track B runs, and 10 final
  robustness runs
- Deterministic aggregate JSON exports
- SHA-256 verification files
- Locked configuration files
- Final environment-provenance record
- No OOD oracle
- No OOD model fitting or OOD alpha tuning
- No final-test tuning

## Interpretation limits

This release provides synthetic computational evidence only. It does
not establish human or animal olfactory equivalence, human perceptual
validity, clinical effectiveness, physical sensor performance,
chemical safety, real-world odor sensing, or deployment-safety
certification.

The repository is publicly inspectable under its proprietary license.
See `LICENSE`, `CITATION.cff`, and
`docs/noi_v0.2_final_results.md`.
