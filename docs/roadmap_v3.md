# Scaling proposals

This historical roadmap considered how to increase useful self-play per unit of compute. Several proposals now have implementations that can be tested independently.

| Original direction | Current implementation or remaining question |
| --- | --- |
| Native rules engine | A faster Python implementation and frozen reference engines are available. A future native backend would need equivalent move ordering, history, superko and area scoring. |
| Parallel or distributed self-play | A local worker pool replicates the best network per process. A shared inference service across machines remains a separate proposal. |
| Randomized playout caps | Implemented for PUCT. The replay policy mask separates full searches from cheap moves. Monitor target quality alongside games per second. |
| Gumbel AlphaZero | Learner-side self-play supports Gumbel-Top-k and Sequential Halving. Cold and mature networks responded differently in recorded matches. |
| Auxiliary targets | Ownership and normalized score heads train on scored terminal games. Warmup controls how strongly fresh heads update a pretrained trunk. |
| Distillation and variable komi | Distillation can train a single student against an ensemble. A komi-conditioned network would also need a change to the input representation and compatible checkpoints. |
| Observability | Structured iteration metrics, SGF exports, timing, external-engine matches and scoring probes support investigation of an actual trained checkpoint. |

The library exposes these methods as separate settings so an experiment can isolate the effects of changing search, data or optimizer behavior. [Experiments](rnd_experiments.md) · [Research roadmap](roadmap_v4.md)
