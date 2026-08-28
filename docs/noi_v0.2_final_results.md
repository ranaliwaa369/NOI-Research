# NOI v0.2 Final Evaluation Results

## Document status

This document closes the computational evaluation package for NOI v0.2.

The evaluation was conducted as a synthetic, reproducible engineering study. It does not establish human or animal olfactory equivalence, human perceptual validity, clinical effectiveness, chemical safety, physical-sensor performance, or deployment-safety certification.

The final source-control commit represented by the environment record is `c07f428716edfcafdc5d2c38080eb914bfa8bf71`.

## Executive summary

NOI v0.2 produced three distinct findings that must be interpreted separately.

1. Track A found repeatable evidence that associative memory improved retrieval for previously seen target items under the locked synthetic design.
2. Track B found that the memory-support mechanism could identify unsupported unseen-family events and abstain at high rates without using an OOD oracle or final-test tuning.
3. The final robustness evaluation did not support hypothesis H4. The full hybrid system did not outperform the strongest eligible baseline under any of the 36 prespecified missing-modality and temporal-displacement condition-tier tests.

The negative H4 result does not invalidate Track A or Track B. It identifies a specific architectural limitation: the current closed-item associative-memory component does not improve identity retrieval for target items and odor families that were never stored.

## Evaluation scope

The final v0.2 package covers:

- repeated seen-item retrieval evaluation;
- unseen-family generalization;
- mild, moderate, and severe graded OOD views;
- missing-modality robustness;
- temporal-displacement robustness;
- paired multi-seed analysis;
- deterministic JSON exports;
- SHA-256 artifact verification;
- explicit oracle and tuning governance.

The two final robustness axes were analyzed separately. A post hoc fully factorial missing-modality by temporal-displacement interaction analysis was prohibited by the locked protocol.

## Track A: repeated seen-item retrieval

Track A evaluated retrieval where the correct target identity was represented in the stored training memory.

Across ten independent runs, the primary paired comparison was memory-only minus ridge-only on mean reciprocal rank.

Observed result:

- Mean paired MRR advantage: `0.009066818`
- 95% paired bootstrap confidence interval: `[0.004399445, 0.014384734]`
- Wins, ties, and losses: `9, 0, and 1`
- Independent runs: `10`
- Oracle information used: `false`
- Final-test tuning used: `false`

The interval remained above zero. Under the locked synthetic Track A design, this supports a repeatable retrieval contribution from associative memory for previously represented target items.

This finding is limited to the tested synthetic setting. It is not evidence of biological memory, human olfactory memory, or general open-world recognition.

Track A aggregate SHA-256:

`6a210574087f17efeaf92a751985895716a40dd529ee96844c87c092b7158da0`

## Track B: unseen-family generalization and selective safety

Track B evaluated events whose target odor families and target identities were held out from training.

Reachability controls verified that:

- the evaluated OOD target identities were unreachable from the stored training memory;
- training and OOD odor families were strictly separated;
- target identifiers were not used as support features;
- family identifiers were not used as support features;
- OOD events were not used to calibrate the support threshold;
- final-test tuning was not used.

Direct unseen-family identity retrieval was largely unsuccessful. This is an important negative generalization result and was retained.

The selective support mechanism nevertheless produced high mean abstention rates:

| Severity tier | Mean abstention | 95% paired bootstrap interval | All runs met the 0.80 criterion |
|---|---:|---:|---:|
| Mild | `0.96145` | `[0.95295, 0.96985]` | `true` |
| Moderate | `0.99420` | `[0.99170, 0.99645]` | `true` |
| Severe | `0.99915` | `[0.99855, 0.99965]` | `true` |

The Track B confirmatory synthetic engineering criterion was supported because the calibration-coverage condition and abstention condition were both met.

This means that the system generally recognized insufficient memory support and abstained. It does not mean that the system learned to identify unseen odor families, and it is not a deployment-safety certification.

Track B aggregate SHA-256:

`e64bc99ac00f800f2a53e742d329463f69a7561ba6ad2e4691852aaa9077a32a`

## Final robustness evaluation

### Prespecified design

The final robustness evaluation used:

- `10` independent runs;
- `10,000` synthetic events per run;
- `2,000` latent OOD events per run;
- `3` graded OOD severity tiers;
- `4` systems;
- `7` missing-modality conditions;
- `5` temporal-displacement conditions;
- `144` system-condition-tier evaluations per run;
- `10,000` paired bootstrap resamples;
- bootstrap seed `4245`;
- independent run seed as the statistical analysis unit.

The systems were:

1. `ridge_only`
2. `memory_only`
3. `hybrid_without_temporal_decay`
4. `full_hybrid`

The eligible baselines were the first three systems. For each axis, condition, and severity tier, the strongest baseline was selected using the highest across-run mean MRR. The locked tie order was ridge-only, memory-only, and hybrid-without-temporal-decay.

### H4 success rule

H4 required the full-hybrid minus strongest-baseline paired MRR advantage to satisfy both conditions for every prespecified condition and tier:

1. the across-run mean advantage had to be greater than zero; and
2. the lower bound of the 95% paired bootstrap confidence interval had to be greater than zero.

### H4 result

The strongest baseline was `ridge_only` in all `36` condition-tier tests.

Observed summary:

- Total condition-tier tests: `36`
- Supported tests: `0`
- Unsupported tests: `36`
- H4 supported: `false`
- Oracle used: `false`
- OOD tuning used: `false`
- Final-test tuning used: `false`

Every observed full-hybrid minus ridge-only mean MRR difference was negative. Every corresponding 95% bootstrap interval was entirely below zero.

Across the condition-tier tests:

- the most negative mean advantage was approximately `-0.895021250`;
- the mean advantage closest to zero was approximately `-0.011150972`;
- no condition produced partial confirmatory support under the locked rule.

Therefore, H4 is rejected under the locked synthetic v0.2 design.

Final robustness aggregate SHA-256:

`f9c5b0bfebbff8d10182f9709a7faa6e2854f2c6506263f0399589b792c01760`

## Architectural interpretation

The results show an important boundary between closed-item memory retrieval and open-set target generalization.

Track A shows that associative memory can help when the correct identity has already been represented in memory.

Track B and the final robustness evaluation show that the same closed-item memory cannot supply a correct identity that it has never stored. When memory scores are mixed into retrieval for an unseen target family, the stored associations can favor known but incorrect target identities.

The observed pattern is consistent with memory interference under an unreachable-target design. This is an architectural interpretation of the computational evidence, not a claim about biological memory.

The support-calibration result suggests a practical direction: when memory support is insufficient, a future system should avoid unconditional hybrid mixing. It could abstain, use a non-memory retrieval path, or route the event to an explicitly open-set component.

Because this routing change was not part of the locked v0.2 final evaluation, it was not introduced after inspecting the final results.

## Integrated conclusion

NOI v0.2 does not support a claim that one fixed hybrid architecture dominates the strongest baseline across seen-item, unseen-family, missing-modality, and temporal-displacement conditions.

Instead, it supports a more specific and scientifically useful conclusion:

- associative memory provided repeatable value for previously represented items;
- unseen-family identity retrieval remained unresolved;
- calibrated abstention successfully recognized unsupported unseen-family events;
- unconditional hybrid memory mixing was not robust under the locked final design;
- a support-aware routing architecture is required for future work.

This mixed result is more informative than a single positive or negative label. It separates where the current memory mechanism helps, where it fails, and where abstention provides an engineering safeguard.

## Governance and leakage controls

The final package retains the following governance statements:

- Oracle information was not used.
- OOD events were not used for threshold calibration.
- OOD model fitting was prohibited.
- OOD alpha tuning was prohibited.
- Final-test tuning was prohibited.
- Seen-item and unseen-family results remained separate.
- Missing-modality and temporal-displacement axes remained separate.
- All negative and null results were retained.
- Every raw run artifact was exported with a SHA-256 digest.
- Aggregate artifacts were exported with SHA-256 digests.

## Reproducibility record

Final verification:

- Python: `3.11.14`
- NumPy: `2.4.6`
- scikit-learn: `1.9.0`
- PyYAML: `6.0.3`
- pytest: `8.4.2`
- macOS: `26.6.2`
- Git commit: `c07f428716edfcafdc5d2c38080eb914bfa8bf71`
- Git branch: `development-v0.2`
- Dependency-lock SHA-256: `c14f5d344875a9afe640418589a1fe13e5412efb7bd21b963a4ed3d50116ce5e`
- Verification suite: `778 passed`
- Failed tests: `0`

Final environment provenance SHA-256:

`b664892bee8eea81e21a3ec035dbec1645368fd63efe189a450a82c43f751052`

## Principal reproducibility artifacts

- `configs/track_b_unseen_family_evaluation_v0.2.2.yaml`
- `configs/robustness_evaluation_v0.2.3.yaml`
- `results/v0.2.1/repeated_track_a_aggregate.json`
- `results/v0.2.2/track_b_aggregate.json`
- `results/v0.2.3/final_robustness_aggregate.json`
- `results/v0.2.3/noi_v0.2_final_environment.json`
- Ten Track B seed artifacts and their SHA-256 files
- Ten final robustness seed artifacts and their SHA-256 files

## Limitations

The evidence remains limited by:

- fully synthetic data;
- synthetic modality representations;
- a fixed odor-target library;
- no physical odor sensors;
- no human or animal participants;
- no perceptual validation;
- no chemical generation or emission;
- no clinical evaluation;
- no deployment environment;
- no external laboratory replication;
- no claim of family-wise error control for condition-specific intervals.

The large negative robustness differences should not be generalized beyond the locked simulation. They identify a limitation of this implementation and evaluation design.

## Version 2 closure statement

The required NOI v0.2 computational work is complete.

Track A, Track B, missing-modality robustness, and temporal-displacement robustness have all been executed across their locked independent runs. Raw and aggregate artifacts are retained, hashes are verified, governance constraints are documented, and positive, null, and negative results are reported.

NOI v0.2 can therefore be closed as a complete reproducible synthetic research package.

## Future work

A future NOI version may investigate:

- support-gated routing between memory and non-memory retrieval;
- explicit open-set recognition;
- compositional target representations;
- family-level representation learning without identifier leakage;
- learned modality reliability weighting;
- missing-modality-aware fusion;
- external datasets and sensor-derived representations;
- prospective validation under a newly locked protocol.

Any future architectural change must be evaluated under a new prespecification and must not retroactively alter the v0.2 results.
