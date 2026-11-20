# Initial implementation roadmap

The first implementation phase focused on an end-to-end learning loop that could be read, tested and run on one machine. Later research uses this implementation as its reference.

| Milestone | Result available in the repository |
| --- | --- |
| Establish a baseline | Separate measurements for rules, MCTS, self-play and SGD. The profiling command allows comparisons on the same hardware. |
| Improve the rules engine | Incremental hashing, cached move analysis and differential comparison with independent earlier engines. |
| Batch search within a tree | Collect pending leaves with virtual loss, deduplicate evaluations and compensate every path during backup. |
| Batch search across games | Drive multiple independent self-play slots and combine inference requests by network. |
| Make training resumable | Save both models, optimizer state, replay, random streams and iteration history at iteration boundaries. Support replay warm starts and optional mixed precision. |
| Measure strength | Fixed anchors, head-to-head evaluation, SGF export, a GTP frontend and external opponents. |
| Check fidelity | Regression tests for the sign of value backup, training policy versus move temperature, first-play urgency, resignation and superko. |

Performance acceptance used legal-move and search equivalence alongside timing data. Strength acceptance used independent opponents and uncertainty intervals. The historical steps above describe implementation milestones. Measured playing strength depends on weights, search and game rules. [Paper mapping](../ARTICLE_TO_CODE_MAP.md) · [Performance](performance.md) · [Scaling proposals](roadmap_v3.md)
