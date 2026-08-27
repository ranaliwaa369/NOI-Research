# NOI v0.2.1 Repeated Track A Results

## Status

Final confirmatory report for the repeated Track A
seen-item episodic-retrieval evaluation.

This report extends the NOI v0.2 development work. It does
not modify or reinterpret the locked NOI v0.1.0 results.

Track A is complete. This report does not close the entire
v0.2 research plan because the unseen-family Track B and
additional robustness evaluations remain separate work.

## Research Question

When every evaluation target is represented in training
memory, does associative memory improve retrieval over the
prespecified ridge baseline?

## Evaluation Design

The evaluation used 10 independent, prespecified synthetic
runs.

Each run used separate seeds for:

- Synthetic-data generation
- OOD generation
- Seen-item template partitioning

Calibration and final-test templates were disjoint.

The final-test set was not used to select the hybrid mixing
parameter.

Every final-test target was represented in the training
memory. Therefore, the memory-reachable fraction was 1.0 in
all 10 runs.

## Evaluated Systems

1. Memory-only associative retrieval
2. Ridge-only retrieval
3. Calibrated ridge-memory hybrid

The memory-only versus ridge-only MRR comparison was the
prespecified primary confirmatory comparison.

Hybrid comparisons were treated as secondary descriptive
analyses.

## Reproducibility Controls

- Independent runs: 10
- All per-run artifacts were exported as deterministic JSON
- Every per-run JSON artifact has a verified SHA-256 digest
- The aggregate result has a verified SHA-256 digest
- Oracle information was not used
- Final-test tuning was not used
- All final-test targets were memory-reachable
- The full repository test suite passed: 690 tests
- Dependency versions and the execution environment were
  recorded in a hashed provenance artifact

## Aggregate Performance

| System | Recall@1 mean | Recall@10 mean | MRR mean | nDCG@10 mean |
|---|---:|---:|---:|---:|
| Memory-only | 0.983009104 | 1.000000000 | 0.991036729 | 0.993359188 |
| Ridge-only | 0.967990521 | 0.999795501 | 0.981969911 | 0.986511280 |
| Hybrid | 0.989211780 | 1.000000000 | 0.994437082 | 0.995885777 |

## Primary Confirmatory Result

Primary direction:

`memory-only MRR minus ridge-only MRR`

Results:

- Mean paired difference: 0.009066818
- Median paired difference: 0.007442361
- Standard deviation: 0.008414131
- Paired bootstrap 95% confidence interval:
  [0.004399445, 0.014384734]
- Run-level wins, ties, and losses: 9, 0, and 1

The confidence interval excludes zero in the prespecified
positive direction.

Under this synthetic, fully memory-reachable seen-item
evaluation, the result supports the hypothesis that
associative memory improves MRR over the ridge-only
baseline.

The estimated mean improvement is approximately 0.0091 MRR
points. The result should be interpreted together with the
high absolute performance of both systems and the synthetic
nature of the evaluation.

## Secondary Hybrid Results

The hybrid system had the highest mean MRR and Recall@1.

Hybrid minus memory-only MRR:

- Mean difference: 0.003400353
- Wins, ties, and losses: 9, 1, and 0

Hybrid minus ridge-only MRR:

- Mean difference: 0.012467171
- Wins, ties, and losses: 10, 0, and 0

These hybrid comparisons are descriptive secondary results.
They are not presented as additional confirmatory
hypothesis tests.

## Hybrid Calibration

Selected hybrid-alpha counts across the 10 runs:

| Alpha | Number of runs |
|---:|---:|
| 0.00 | 1 |
| 0.25 | 7 |
| 0.50 | 1 |
| 0.75 | 1 |

The alpha was selected using calibration data only.

The most frequently selected value was 0.25, indicating that
most repeated runs favored a memory-dominant mixture with a
smaller ridge contribution.

## Interpretation

The earlier NOI memory-only OOD result could not isolate
episodic-memory quality because the correct OOD targets were
absent from memory.

Track A corrected that structural limitation by evaluating
only targets that were explicitly reachable from training
memory while keeping evaluation events and templates held
out.

Across 10 independent synthetic runs, memory-only retrieval
outperformed ridge-only retrieval on the primary MRR
comparison in 9 runs.

This provides reproducible computational evidence that the
associative-memory component contributes retrieval utility
when the requested odor item is represented in memory.

## What This Result Does Not Establish

This evaluation does not establish:

- Human-like olfactory perception
- Biological equivalence to human or animal memory
- Performance with physical olfactory sensors
- Generalization to odor families absent from memory
- Robustness under every noise or missing-modality condition
- Real-world safety or deployment readiness
- Clinical, diagnostic, or medical validity

The result concerns the tested computational retrieval
mechanism under a controlled synthetic protocol.

## Remaining v0.2 Work

The following work remains outside the completed Track A
package:

1. Track B unseen-family generalization
2. Reachability-aware abstention or deferral
3. Selective coverage and selective-error reporting
4. Prespecified noise evaluation
5. Missing-modality evaluation
6. Temporal-displacement evaluation
7. Final integration of Track A and Track B into the broader
   NOI v0.2 manuscript

Seen-item and unseen-family results must remain stratified
and must not be pooled into one performance estimate.

## Artifacts

Repeated-run protocol:

- `configs/seen_item_repeated_evaluation_v0.2.1.yaml`
- `configs/seen_item_repeated_evaluation_v0.2.1.sha256`

Per-run results:

- `results/v0.2.1/repeated_track_a/seed-01.json`
  through `seed-10.json`
- A corresponding SHA-256 file accompanies every run

Aggregate result:

- `results/v0.2.1/repeated_track_a_aggregate.json`
- SHA-256:
  `6a210574087f17efeaf92a751985895716a40dd529ee96844c87c092b7158da0`

Environment provenance:

- `results/v0.2.1/repeated_track_a_environment.json`
- SHA-256:
  `4e3d03cf03e7f1f87bebef99a2275fa0b323a199b3b81d34f7e8ec6adb469e72`

Source commit recorded for the execution environment:

`8f1e7400f8360b159ee952b8354609844d2ba085`

## Conclusion

The repeated Track A evaluation supports a limited and
specific conclusion:

When evaluation targets are present in memory, the NOI
associative-memory mechanism provides a reproducible MRR
advantage over the tested ridge-only baseline in this
synthetic evaluation.

Track A is therefore complete. Claims about unseen-family
generalization, physical sensing, biological olfaction, or
real-world deployment require separate evidence.
