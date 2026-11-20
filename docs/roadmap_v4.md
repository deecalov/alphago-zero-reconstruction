# Research roadmap

The research program uses independent matches to decide whether a new network should replace a frozen incumbent. Faster search, better training loss or a higher internal promotion rate each measures a different property.

## Search with fixed weights

Start with a checkpoint, its saved search settings and a same-weights control. Compare c_puct, first-play urgency, ensemble membership, tree reuse or time allocation one change at a time. Re-evaluate the most promising setting with an independent seed and more games to account for grid selection bias. External GTP opponents provide another fixed point on the scale. [Strength evaluation](strength.md)

## Improve training targets

Explore low-budget Gumbel search, auxiliary ownership and score targets, forced root visits and policy pruning. Use actual leaf batch width, game endings, full-search policy weight and resignation behavior to interpret the target distribution. A large score against a branch control warrants a direct match against the incumbent. The recorded pruning confirmation interval included 50%. [Experiments](rnd_experiments.md)

## Measure generalization

Keep fresh holdout games separate from SGD, monitor policy and value independently and compare them with buffer estimates at the same weights. A checkpoint-averaging experiment also needs recalibrated BatchNorm statistics and a match against individual checkpoints. Continue to evaluate the restored or averaged model under the same search settings as its baseline.

## Scaling questions

The local worker pool can remove some Python contention but duplicates network memory. A shared inference service, distributed game generation or a native rules backend would change the performance model and reproducibility requirements. Measure slot fill and end-to-end generation speed before selecting among them. New network architectures or komi-conditioned inputs require new compatibility and strength measurements. [Performance](performance.md) · [Checkpoint compatibility](checkpoints.md)
