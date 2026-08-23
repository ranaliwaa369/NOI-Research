# Policy Conformance Pilot v0.1 Provenance

## Ownership

- Project: Neuro-Olfactive Intelligence
- Owner: GUARDIANX LLC
- Authors: Rana Al-Dahlake and Jalal Alazirji

## Release identity

- Release: `policy-conformance-pilot-v0.1`
- Status: Exploratory locked simulated computational policy evaluation
- Preregistration tag: `policy-conformance-v0.1-preimplementation`
- Preregistration commit: `7354230`
- Configuration validator commit: `873a018`
- Experiment implementation commit: `c49c9e9`
- Reproducible workflow commit: `27535ae`

The 26 test cases, expected outcomes, metrics, and success thresholds were
committed and tagged before implementation or inspection of the
policy-conformance results.

## Locked test suite

The suite contained 26 deterministic simulated requests:

- 4 expected `ALLOW` cases
- 15 expected `BLOCK` cases
- 7 expected `REQUIRE_MISSING_INFORMATION` cases

The suite included:

- nominal conforming requests;
- exact threshold-boundary requests;
- absent consent;
- unknown items;
- disabled items;
- concentration-limit violations;
- duration-limit violations;
- minimum-environment-volume violations;
- minimum-ventilation violations;
- missing required fields;
- multiple missing fields;
- multiple simultaneous violations;
- deterministic rule-precedence cases.

Every request identifier was unique, and request order was locked.

## Locked metrics and targets

- False-allow target: 0
- Maximum false-block rate: 0.01
- Exact-conformance target: 1.0
- Policy-coverage target: 1.0
- Default policy action: `BLOCK`
- Physical emission performed: `false`

False allow was defined as predicting `ALLOW` when the expected outcome was
not `ALLOW`.

False block was defined as predicting `BLOCK` when the expected outcome was
`ALLOW`.

## Results

- Total cases: 26
- Exact matches: 26
- False allows: 0
- False-allow rate: 0.0
- False blocks: 0
- False-block rate: 0.0
- Exact-conformance rate: 1.0
- Policy coverage: 1.0
- H3 locked-suite success rule passed: `true`
- Physical emission performed: `false`

## Interpretation

The deterministic policy gate exactly matched the preregistered expected
outcome for every case in the locked simulated suite.

This supports only the claim that the current implementation conforms to
these 26 specified computational cases.

It does not demonstrate:

- comprehensive safety;
- chemical exposure safety;
- clinical or medical safety;
- physical-device safety;
- odor-emission safety;
- adversarial robustness;
- exhaustive policy coverage;
- compliance with laws or regulations;
- safe real-world deployment;
- deployment readiness.

The numerical thresholds in `policy_rules.yaml` are simulated
policy-conformance fixtures. They are not chemical exposure limits, medical
recommendations, physical operating instructions, or evidence of safe
concentrations or durations.

A zero false-allow count in 26 prespecified synthetic cases must not be
reported as a universal zero-risk or comprehensive-safety result.

## Output files and SHA-256

### `policy_conformance_results.json`

`c84f2b823abdb3ee52ad4f04dfb92f9eedc14050b8c154feb62dee66a85dbf9a`

### `policy_conformance_summary.json`

`6128191ae40efdca6d3430ab3d4a0d286d8647fef68e586d8c249fc3f89f110f`

### `run_manifest.json`

`5b8a42c334da8d26027b5f057a0fa1aeb8d3ab9c8e6e946e6d7c9a473ce8f7ee`

## Verification

- Full test suite before release: 631 passed
- Configuration checksum validation: passed
- Unique request identifiers: passed
- Locked case order: passed
- Decision audit retention: passed
- Export verification: passed
- Output hashes independently reproduced with `shasum -a 256`
- Working tree was clean before pilot execution

## Relationship to other NOI pilots

The present result is independent of retrieval accuracy.

It does not change the earlier findings that:

- the associative-memory component provided no incremental advantage in the
  target-held-out graded-OOD ablation; and
- explicit corrective updates restored deliberately corrupted known
  associations in the controlled H2 mechanism test.

The retrieval, correction, and policy-conformance results address distinct
engineering questions and must not be combined into a general claim of
olfactory intelligence or comprehensive safety.

## Scope limitations

This release is exploratory, synthetic, computational, and limited to a
locked policy-conformance test suite.

It is not confirmatory, perceptual, clinical, chemical, physical-device,
emission-safety, legal-compliance, adversarial-security, or
deployment-readiness evidence.

## Reproduction

From the repository root run:

    python -m pytest -q
    python -m experiments.run_policy_conformance_pilot

For an intentional deterministic rerun:

    from experiments.run_policy_conformance_pilot import (
        run_policy_conformance_pilot,
    )

    run_policy_conformance_pilot(overwrite=True)
