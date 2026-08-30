# Support-Aware and Reliability-Gated Multisensory Retrieval for Neuro-Olfactive Intelligence: A Reproducible Synthetic Evaluation

Rana Al-Dahlake¹,* and Jalal Alazirji¹

¹ GuardianX LLC, Tukwila, Washington, United States

**Corresponding author**

Rana Al-Dahlake
ORCID: 0009-0001-8919-8177
Email: info@researchguardianx.com

**Document Type**

Conceptual AI Systems Paper and Reproducible Evaluation Protocol

**Public Release**

Version 1.0

**Publication Date**

August 2026

**Digital Object Identifier**

[ZENODO DOI TO BE RESERVED BEFORE PUBLICATION]

## Abstract

Artificial olfaction systems are commonly evaluated as closed-set
classifiers, even though real queries may be unsupported, incomplete,
or accompanied by conflicting evidence. We present Neuro-Olfactive
Intelligence (NOI) v0.3, a synthetic and reproducible evaluation of
support-aware, reliability-gated olfactory-tactile retrieval under
open-set shift. The study used validation-locked thresholds, ten
prespecified seeds, 70,000 training events, 10,000 validation events,
20,000 final-test latent events, seven paired condition views, nine
systems, and 1,260,000 system evaluations. Paired bootstrap intervals
used latent event ID as the resampling unit, with 10,000 resamples and
Holm correction for secondary comparisons. The primary hypothesis H6
was supported: a support-aware gate reduced unseen-family false-known
decisions by 1.000000 relative to the strongest locked deployable
baseline, with a 95% bootstrap interval of [1.000000, 1.000000], while
clean seen-item mean reciprocal rank loss was 0.000000. The secondary
hypothesis H7 was not supported: reliability-gated
olfactory-tactile fusion produced an MRR difference of -0.0022125
relative to fixed-weight fusion in degraded-odor and missing-odor
conditions, with a 95% interval of [-0.0026625, -0.0017500].
H8 was also not supported: the false-confident reduction was
0.0040083 against both naive and fixed-weight fusion, which was
statistically detectable but below the prespecified practical
threshold of 0.05. These results support validation-locked rejection
of unsupported synthetic queries, but do not support tactile synergy
or a practically sufficient conflict-aware safety improvement. The
study uses simulated vectors only and does not establish physical
sensor performance, biological equivalence, clinical validity,
chemical safety, or deployment readiness.

**Keywords**

Artificial olfaction; neuro-olfactive artificial intelligence;
contextual olfactory memory; multimodal learning; associative memory;
odor representation; open-set recognition; selective prediction;
safety-aware AI; reproducible simulation

## 1. Introduction

Artificial olfaction includes sensor arrays, pattern-recognition
systems, and computational representations intended to distinguish
odor-related patterns. Early model-nose work established the idea that
distributed sensor responses could support discrimination without a
single sensor being specific to each odor [1]. Electronic-nose research
later expanded this principle into sensor-array and pattern-recognition
applications [2].

NOI is a software-first research framework. It does not claim to be a
physical electronic nose. The current implementation uses synthetic
vectors to study retrieval, contextual association, open-set support,
multimodal evidence, abstention, and reproducible evaluation.

Closed-set evaluation assumes that every query belongs to a known
class. Open-set recognition instead addresses situations in which a
query may not belong to any represented class [3]. This distinction is
important for memory-based retrieval because a system can produce a
high-scoring known answer even when the correct identity was never
represented during training.

NOI v0.2 identified this limitation directly. Associative memory helped
retrieval for represented items, but unsupported unseen-family identity
retrieval remained unresolved. A validation-calibrated support mechanism
could abstain, while unconditional hybrid mixing failed to dominate the
strongest baseline under missing-modality and temporal-displacement
conditions.

Version 0.3 therefore tested a support-aware routing architecture before
forced identity retrieval. It also tested whether simulated tactile
evidence could provide useful complementary information when olfactory
evidence was degraded, absent, or conflicting.

The study was designed before confirmatory execution. Hypotheses,
systems, seeds, conditions, practical effect thresholds, validation
procedures, confidence mapping, comparator selection, and statistical
controls were locked and hashed before final-test metrics were
inspected.

## 2. Related work

### 2.1 Artificial olfaction

Artificial olfaction has historically combined broadly responsive
sensors with pattern-recognition methods. Persaud and Dodd demonstrated
a model-nose discrimination principle based on distributed responses
[1]. Wilson and Bai later reviewed electronic-nose technologies and
their applications, while also documenting limitations involving
sensors, sampling, drift, selectivity, and validation [2].

NOI differs from physical electronic-nose work because it evaluates a
computational representation and decision policy rather than a gas
sensor array. Terms such as olfactory vector and tactile vector describe
simulated numerical features, not measured chemical or physical
signals.

### 2.2 Open-set recognition and abstention

Open-set recognition formalizes the problem of limiting risk from
unknown classes while retaining useful recognition of represented
classes [3]. Selective classification extends this idea by allowing a
model to abstain when its confidence or support is insufficient [4].
This introduces a tradeoff between coverage and error rather than
forcing a prediction for every query.

NOI v0.3 applies this principle to retrieval. The support gate is not
credited with identifying an unseen target. It is evaluated on whether
it prevents a false-known answer when the target is unsupported.

### 2.3 Multimodal learning

Multimodal machine learning studies representation, alignment, fusion,
translation, and co-learning across information sources [5]. Additional
modalities can help when they contain complementary information, but
fusion can also propagate noise or conflict. For this reason, a
multimodal system should be compared with strong unimodal and simple
fusion baselines rather than assumed to be beneficial.

NOI v0.3 evaluates simulated olfactory and tactile evidence under clean,
degraded, missing, contradictory, and temporally misaligned conditions.
The paper uses the term multimodal only for this computational
construction. It does not claim that the tactile vectors reproduce
biological touch.

### 2.4 Confidence and practical significance

A confidence score is useful only if its interpretation is fixed and
evaluated honestly. Modern predictive systems can be miscalibrated even
when their accuracy is high [6]. NOI v0.3 fixed its confidence mapping
before confirmatory execution and prohibited final-test recalibration.

Statistical significance is also distinct from practical significance.
The study therefore required minimum effect sizes in addition to
confidence intervals and p-values. Holm correction controlled the
secondary comparison family [7].

## 3. Research questions and hypotheses

The primary research question was:

> Can a support-aware, reliability-gated olfactory-tactile retrieval
> policy reduce unsupported and false-confident decisions without
> materially harming supported retrieval?

Three hypotheses were registered.

### 3.1 H6: support-aware open-set gate

H6 predicted that a validation-locked support gate would reduce
false-known decisions for unseen-family queries by at least 0.05,
produce a paired 95% interval excluding zero, and preserve clean
seen-item MRR within a maximum loss of 0.02.

H6 was the primary hypothesis.

### 3.2 H7: conditional tactile utility

H7 predicted that reliability-gated olfactory-tactile fusion would
outperform the strongest applicable baseline in degraded-odor and
missing-odor conditions by at least 0.05 absolute MRR or 10% relative
MRR, with a paired 95% interval excluding zero.

### 3.3 H8: conflict-aware safe fusion

H8 predicted that support-aware reliability fusion with abstention would
reduce false-confident decisions by at least 0.05 relative to both naive
concatenation and fixed-weight fusion, with a paired 95% interval
excluding zero and no more than 0.02 clean-condition MRR loss.

## 4. Methods

### 4.1 Study type and claim boundary

This was a confirmatory synthetic computational evaluation. It included
no human participants, animal participants, biological samples,
chemical generation, chemical emission, or physical sensors.

The study tests implementation behavior under a registered generator.
It does not test human perception, medical diagnosis, environmental
monitoring, or field deployment.

### 4.2 Synthetic representations

Each event included:

- a 16-dimensional simulated olfactory vector;
- an 8-dimensional simulated tactile vector when touch was available;
- a latent target item;
- a latent target family;
- a support regime;
- a paired condition view;
- a latent event identifier.

Target, family, support, split, condition, and quality metadata were
excluded from model inference inputs. Ground-truth fields were retained
only for final scoring and integrity auditing.

### 4.3 Data allocation and support regimes

Each of ten seeds generated:

- 7,000 training events;
- 1,000 validation events;
- 2,000 final-test latent events.

The final-test allocation per seed was:

- 800 seen-item events;
- 600 known-family unseen-item events;
- 600 unseen-family events.

Across ten seeds, the study contained 70,000 training events, 10,000
validation events, and 20,000 final-test latent events.

Validation-unknown and final-test-unknown families were disjoint.
Training targets, held-out known-family items, validation-unknown
families, and final-test-unknown families were generated under the
locked topology.

### 4.4 Conditions

Each final-test latent event generated seven paired views:

1. clean olfactory and tactile evidence;
2. degraded odor;
3. degraded touch;
4. missing touch;
5. missing odor;
6. contradictory modalities;
7. temporal misalignment.

The condition generator used locked olfactory noise scale 0.10, tactile
noise scale 0.10, degraded-quality metadata 0.40, and temporal offset
three steps. Quality metadata was not a model input.

The same latent event was retained across conditions. Paired views were
not treated as independent samples.

### 4.5 Systems

Nine systems were evaluated:

1. odor-only ridge retrieval;
2. odor-only cosine retrieval;
3. touch-only ridge retrieval;
4. touch-only cosine retrieval;
5. naive olfactory-tactile concatenation;
6. fixed-weight olfactory-tactile fusion;
7. support-gated odor-only retrieval;
8. reliability-gated olfactory-tactile fusion;
9. support-gated reliability fusion with abstention.

Candidate libraries and ridge models used training events only. The
locked retrieval order used descending similarity followed by ascending
item identifier as the deterministic tie break.

### 4.6 Validation-only threshold locking

Five values were derived separately for each registered seed:

- support threshold;
- lower support-uncertainty boundary;
- upper support-uncertainty boundary;
- modality reliability threshold;
- modality conflict threshold.

Only training and validation data were permitted before locking.
Cross-seed threshold pooling was prohibited. Final-test events and
labels did not change any threshold.

The validation lock was committed and tagged before confirmatory
implementation was completed. The execution specification and code were
then committed and tagged before final-test execution.

### 4.7 Confidence policy

The confidence policy was fixed before execution:

\[
c = \operatorname{clip}\left(
rac{s_{\max}+1}{2},0,1
ight),
\]

where \(s_{\max}\) is the top-ranked weighted cosine score. Abstention
confidence was 0.0. A false-confident decision used the locked confidence
threshold 0.80.

This mapping was evaluated but not refit on final-test data.

### 4.8 Metrics

The confirmatory comparisons used:

- mean reciprocal rank;
- false-known decision rate;
- false-confident decision rate;
- clean-condition MRR loss;
- absolute and relative effect sizes;
- paired bootstrap confidence intervals.

The result package also retained system, condition, and support-regime
summaries for reproducibility.

### 4.9 Statistical analysis

The analysis used 10,000 paired bootstrap resamples, bootstrap seed 4242,
and a 95% confidence level. Latent event ID was the resampling unit.

Comparator selection was locked before execution. Ties were resolved
deterministically by ascending system name. Holm correction was applied
to H7 and the two H8 comparisons.

A hypothesis required both statistical and registered practical
criteria. A small p-value alone did not establish support.

### 4.10 Leakage and integrity controls

The implementation verified that:

- no exact or latent event overlap crossed splits;
- unknown families were separated as registered;
- target identifiers were not encoded in tactile values;
- condition and quality metadata were not model inputs;
- target labels were not inference inputs;
- final-test labels were used for scoring only;
- final-test labels were not used for fitting or calibration;
- validation values were locked independently by seed;
- thresholds did not change after final-test inspection;
- all ten seeds and all registered conditions were retained;
- raw and aggregate outputs were hashed.

## 5. Results

### 5.1 Execution summary

The confirmatory execution produced:

| Quantity | Value |
|---|---:|
| Registered seeds | 10 |
| Final-test latent events | 20,000 |
| Paired condition views | 140,000 |
| Registered systems | 9 |
| System evaluations | 1,260,000 |
| Bootstrap resamples | 10,000 |
| Failed integrity checks | 0 |
| Final-test tuning used | False |

All ten seeds completed exactly once under the execution lock.

### 5.2 H6 result

All four eligible H6 comparators achieved clean seen-item MRR 1.000000.
The deterministic tie rule selected fixed-weight fusion.

| H6 quantity | Result |
|---|---:|
| False-known reduction | 1.000000 |
| 95% bootstrap interval | [1.000000, 1.000000] |
| Bootstrap p-value | 0.0001999800 |
| Clean seen-item MRR loss | 0.000000 |
| Required reduction | 0.050000 |
| Maximum permitted clean loss | 0.020000 |
| Status | Supported |

The support-gated system abstained on unsupported unseen-family queries
in the registered simulation while the non-gated comparator returned a
known candidate. The result met every H6 criterion.

### 5.3 H7 result

The eligible comparator values were:

| System | Pooled MRR |
|---|---:|
| Fixed-weight fusion | 0.400000000 |
| Touch-only cosine | 0.400000000 |
| Touch-only ridge | 0.400000000 |
| Odor-only cosine | 0.199570833 |
| Odor-only ridge | 0.199545833 |

The deterministic tie rule selected fixed-weight fusion.

| H7 quantity | Result |
|---|---:|
| Proposed fusion MRR | 0.397787500 |
| Comparator MRR | 0.400000000 |
| Absolute difference | -0.002212500 |
| Relative difference | -0.005531250 |
| 95% bootstrap interval | [-0.002662500, -0.001750000] |
| Raw p-value | 0.0001999800 |
| Holm-adjusted p-value | 0.0005999400 |
| Status | Not supported |

The confidence interval was entirely below zero. Reliability-gated
fusion was slightly worse, not better, than the selected comparator.
H7 was not supported.

### 5.4 H8 result

| Comparator | False-confident reduction | 95% interval | Clean MRR loss | Status |
|---|---:|---:|---:|---|
| Fixed-weight fusion | 0.004008333 | [0.002608333, 0.005441667] | 0.000000 | Not supported |
| Naive concatenation | 0.004008333 | [0.002608333, 0.005441667] | 0.000000 | Not supported |

The raw p-value for each comparison was 0.0001999800 and the
Holm-adjusted p-value was 0.0005999400.

The effect was favorable but approximately 0.40 percentage points,
which was far below the registered 5 percentage-point requirement.
H8 was not supported.

### 5.5 Independent raw-record audit

A post-execution read-only audit recomputed the point estimates directly
from all 1,260,000 raw system-evaluation records.

It reproduced:

- H6 false-known reduction 1.000000;
- H6 clean seen-item MRR loss 0.000000;
- H7 absolute MRR difference -0.002212500;
- H8 reduction versus fixed-weight fusion 0.004008333;
- H8 reduction versus naive concatenation 0.004008333.

All values matched the locked aggregate to absolute tolerance
\(10^{-12}\). No parameter or result was changed after this audit.

## 6. Discussion

### 6.1 Support-aware routing was the supported contribution

The primary result shows that a validation-locked support decision can
prevent forced retrieval when the target family is unsupported. This
directly addresses the open-set failure observed in v0.2.

The result should be interpreted as rejection of an unsupported query,
not recognition of the unseen target. The gate did not identify a new
family or retrieve an unreachable identity.

The effect was perfect in this synthetic design. This is useful for
implementation verification but also signals a limitation. The
registered generator may separate supported and unsupported events more
cleanly than physical data would. External and sensor-derived tests are
required before estimating real-world performance.

### 6.2 Simulated touch did not establish synergy

H7 tested whether tactile evidence improved retrieval specifically when
olfactory evidence was degraded or absent. It did not.

The fixed-weight and touch-only comparators reached pooled MRR 0.400000,
while reliability-gated fusion reached 0.3977875. The difference was
small but consistently negative.

This finding means the registered tactile representation and fusion
policy did not add complementary retrieval information beyond the
strongest baseline. It does not imply that physical tactile sensing can
never help artificial olfaction. It identifies a limitation of this
representation and policy.

### 6.3 Conflict logic produced a small but insufficient effect

H8 separated statistical detectability from practical utility. The
combined policy reduced false-confident decisions by approximately
0.0040 and preserved clean MRR. The interval excluded zero, but the
effect was less than one tenth of the required 0.05 reduction.

It would be incorrect to label this result a confirmed safety
improvement. The appropriate conclusion is that the registered policy
produced a small computational effect that did not meet the practical
criterion.

### 6.4 Relationship to v0.2

Version 0.2 concluded that support-aware routing was needed because
unconditional hybrid memory mixing could return incorrect represented
items for unreachable targets. Version 0.3 prospectively implemented and
tested that direction.

The H6 result supports the routing component. The H7 and H8 results show
that routing alone does not validate the proposed multisensory fusion
mechanism. This separates a supported open-set contribution from
unsupported multimodal claims.

### 6.5 Statistical and practical interpretation

All three secondary tests had small Holm-adjusted p-values. Their
interpretation differed:

- H7 was statistically different in the unfavorable direction.
- H8 was statistically favorable but practically too small.
- Neither result met its registered support rule.

This illustrates why confirmatory evaluation should report effect size,
direction, uncertainty, and practical thresholds rather than p-values
alone.

## 7. Limitations

The study has important limitations.

First, all data were synthetic. The olfactory and tactile vectors were
not measured from physical sensors.

Second, the target topology and generators were fixed and relatively
small. The perfect H6 separation may reflect generator structure and may
not persist under more complex distribution shift.

Third, tactile evidence was simulated. Its failure to improve retrieval
cannot be generalized to real material, pressure, temperature, texture,
or proprioceptive sensing.

Fourth, confidence was derived from a fixed transformation of the
top-ranked score. The study evaluated this mapping but did not establish
general calibration.

Fifth, the analysis tested the registered aggregate comparisons. It did
not authorize post hoc selection of favorable seeds, systems,
conditions, or thresholds.

Sixth, there was no external replication, physical validation, perceptual
study, chemical validation, clinical study, or deployment evaluation.

## 8. Reproducibility and governance

The repository preserves:

- the preimplementation protocol;
- the validation amendment;
- seedwise validation locks;
- the confirmatory execution specification;
- source-control lock tags;
- ten raw seed hashes;
- aggregate results;
- an environment manifest;
- final findings;
- automated tests.

The execution-lock commit was
`f10ecc235b22273c07d1ab2ae00b39c10acb83c6`.

The confirmatory aggregate SHA-256 was:

`4c427a7d1fe1a537e8efb209d7e632ff956fbdbf2b0c7ef8a1485720ba784320`

The seed manifest SHA-256 was:

`f5da31b6ce63b2a024576a99eb3d8b08fb6a9434a665e32e49f401950edcfd33`

The complete verification suite contained 1,295 passing tests before
confirmatory execution.

## 9. Ethical and safety statement

This computational study did not include human or animal subjects and
did not collect personal, clinical, biometric, or chemical-exposure
data.

The terms safety-aware and false-confident refer to registered software
metrics. They do not indicate safety certification. Any physical,
clinical, environmental, or human-subject extension requires separate
governance, domain expertise, risk assessment, and applicable ethical
and regulatory review.

## 10. Conclusion

NOI v0.3 supports validation-locked, support-aware rejection of
unsupported queries in a synthetic open-set retrieval setting. It does
not support the registered claims of tactile retrieval synergy or a
practically sufficient conflict-aware reduction in false-confident
decisions.

The strongest defensible conclusion is narrow: support should be checked
before forced identity retrieval. Additional modality fusion should be
treated as an empirical question rather than an assumed benefit.

The mixed findings are retained as the final v0.3 result. Future
architectural changes require a newly prespecified evaluation and cannot
retroactively alter this record.

## Data and code availability

Source code, locked configurations, aggregate artifacts, environment
metadata, seed hashes, and documentation are available at:

https://github.com/ranaliwaa369/NOI-Research

The raw per-seed JSON artifacts total approximately 1.9 GB and are
retained separately from the ordinary Git history. Their SHA-256 values
are listed in the public seed manifest.

A versioned archival DOI will be added before publication.

## Author contributions

Rana Al-Dahlake led conceptualization, protocol development, software
implementation, confirmatory analysis, reproducibility documentation,
and manuscript drafting.

Jalal Alazirji contributed project review, conceptual discussion, and
manuscript review.

Both authors approved the public version.

## Competing interests

The authors are affiliated with GuardianX LLC, which develops and
maintains the NOI research framework. This affiliation is disclosed for
transparency. No clinical, physical-sensor, or deployment product claim
is made from this study.

## References

1. Persaud K, Dodd G. Analysis of discrimination mechanisms in the
   mammalian olfactory system using a model nose. *Nature*.
   1982;299:352-355.
   https://doi.org/10.1038/299352a0

2. Wilson AD, Baietto M. Applications and advances in electronic-nose
   technologies. *Sensors*. 2009;9(7):5099-5148.
   https://doi.org/10.3390/s90705099

3. Scheirer WJ, Rocha A, Sapkota A, Boult TE. Toward open set
   recognition. *IEEE Transactions on Pattern Analysis and Machine
   Intelligence*. 2013;35(7):1757-1772.
   https://doi.org/10.1109/TPAMI.2012.256

4. Geifman Y, El-Yaniv R. Selective classification for deep neural
   networks. *Advances in Neural Information Processing Systems*.
   2017;30:4878-4887.
   [NeurIPS paper](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)

5. Baltrušaitis T, Ahuja C, Morency LP. Multimodal machine learning:
   a survey and taxonomy. *IEEE Transactions on Pattern Analysis and
   Machine Intelligence*. 2019;41(2):423-443.
   https://doi.org/10.1109/TPAMI.2018.2798607

6. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern
   neural networks. *Proceedings of Machine Learning Research*.
   2017;70:1321-1330.
   https://proceedings.mlr.press/v70/guo17a.html

7. Holm S. A simple sequentially rejective multiple test procedure.
   *Scandinavian Journal of Statistics*. 1979;6(2):65-70.
   https://doi.org/10.2307/4615733

8. Efron B, Tibshirani RJ. *An Introduction to the Bootstrap*.
   Chapman and Hall/CRC; 1993.
   https://doi.org/10.1201/9780429246593
