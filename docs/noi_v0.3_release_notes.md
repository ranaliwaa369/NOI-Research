# NOI v0.3.0 Release Notes

## Release status

NOI v0.3.0 closes the locked confirmatory synthetic evaluation of
support-aware and reliability-gated olfactory-tactile retrieval.

## Evaluation scale

- Registered seeds: 10 (`1301` through `1310`)
- Training events: 70,000
- Validation events: 10,000
- Final-test latent events: 20,000
- Paired condition views: 140,000
- Registered systems: 9
- System evaluations: 1,260,000
- Bootstrap resamples: 10,000
- Failed integrity checks: 0
- Final-test tuning: not used

## Confirmatory findings

### H6: supported

The validation-locked support gate reduced unseen-family false-known
decisions by `1.000000`, with a 95% paired bootstrap interval of
`[1.000000, 1.000000]`. Clean seen-item MRR loss was `0.000000`.

This result supports rejection of registered unseen-family queries only
within the registered synthetic generator.

### H7: not supported

Reliability-gated olfactory-tactile fusion produced an absolute MRR
difference of `-0.002212500` relative to fixed-weight fusion, with a 95%
interval of `[-0.002662500, -0.001750000]`.

The result does not support a claim of tactile synergy.

### H8: not supported

False-confident reduction was `0.004008333` against both fixed-weight
fusion and naive concatenation, with a 95% interval of
`[0.002608333, 0.005441667]`.

The effect was below the prespecified practical threshold of `0.05`.

## Reproducibility

The release retains:

- locked and hashed protocol artifacts;
- the confirmatory execution-lock tag;
- aggregate results and SHA-256 digest;
- environment manifest and SHA-256 digest;
- findings summary and SHA-256 digest;
- ten-seed manifest and verified raw-artifact hashes;
- complete final results documentation;
- research manuscript;
- automated verification tests.

The raw per-seed JSON files total approximately 1.9 GB and remain
outside ordinary Git history. Their verified hashes are included in the
public seed manifest.

## Claim boundaries

All evidence is synthetic computational evidence. This release does not
establish physical odor sensing, biological or human olfactory
equivalence, human perceptual validity, tactile hardware performance,
clinical effectiveness, chemical safety, or deployment readiness.

## Principal documents

- `docs/noi_v0.3_research_protocol.md`
- `docs/noi_v0.3_final_results.md`
- `docs/noi_v0.3_manuscript.md`
- `docs/noi_v0.3_posthoc_sensitivity.md`
- `docs/noi_v0.3_trace_audit.md`
- `artifacts/noi_v0.3_confirmatory/aggregate.json`
- `artifacts/noi_v0.3_confirmatory/seed_manifest.json`
- `artifacts/noi_v0.3_confirmatory/posthoc_sensitivity.json`
- `artifacts/noi_v0.3_confirmatory/trace_audit.json`

## DOI

The v0.3 archival DOI will be added before final publication. The prior
v0.2 archival record remains available at:

https://doi.org/10.5281/zenodo.22139127
