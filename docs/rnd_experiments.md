# Research experiments

The optional methods below serve different purposes. To compare one change, preserve the starting weights, replay data, game rules and evaluation budget, then test the resulting checkpoint against the same frozen opponent. The figures here describe historical experiments with specific networks and search budgets.

| Experiment | Implementation and interpretation |
| --- | --- |
| Auxiliary ownership and score | `network.aux_heads` adds dense terminal supervision. Resigned games lack terminal labels and are masked. `aux_warmup_steps` can protect a mature trunk from new heads' initial gradients. |
| Randomized playout caps | Mix deep policy-producing moves with cheaper moves. Normalize the policy loss by active policy weights. Value can learn from both kinds of move. |
| Forced playouts and target pruning | A PUCT root quota helps inspect high-prior moves. Pruning reduces low-mass entries in the learning policy. These controls are independent, and only the quota needs PUCT. |
| Distillation | `examples.distill_merge` labels replay positions with mean teacher probabilities and values, then fits one student. Report holdout prediction error and play a match against each teacher. |
| Ensembles | `EnsembleNet` combines compatible networks within one search, increasing inference cost. Use a per-side budget to compare under a common resource limit. |
| Frozen league | Play a fraction of games against past networks. Search settings and value calibration belong to each network. Only the learner can resign on its turn. |
| Gumbel AlphaZero | Sequential Halving at the root and improved-policy targets support small budgets. `examples.probe_gumbel_batch` measures agreement with sequential search under identical noise. |
| Checkpoint averaging | `examples.average_checkpoints` averages tensors from one trajectory, accumulating in float64. Supply replay with `--buffer` to recompute BatchNorm running statistics. The newest checkpoint supplies metadata. |
| Gate and holdout | `--eval-gate none` trains with the latest network every iteration. `--holdout-fraction` reserves entire games and logs train/holdout policy CE and value MSE for the same network. |

## What the matches showed

- **Gumbel from scratch.** In an eight-hour cold-start comparison, the model trained at Gumbel/32 lost 1–119 to the PUCT/160 model. A warm start from mature weights was a materially different experiment. On a weak 5×5 network at 32 simulations, leaf batches of 2, 4, 8 and 16 changed the sequential move about 5%, 8%, 18% and 25% of the time.
- **League and auxiliary heads.** A warm-start Gumbel/32 learner with a 25% league of deeper-search opponents scored 37–83 and 51–69 against two frozen champions. A separate auxiliary-head run with warmup scored 58–62 and 63–57 against those anchors. The combined 121–119 is compatible with parity.
- **Policy pruning.** A pruning branch beat its matched control 146–94 in 240 games. The confirmation against the incumbent finished 128–112, a score of 53.3% with a 95% interval of 47.0–59.5%. After 12 more hours of training with pruning at 0.1, the continued run scored 46–74 and crossed its prespecified stop threshold. During that segment value loss fell from 0.58 to 0.38 and target entropy fell from 0.372 to 0.216. Neither trend predicted the loss of match strength.
- **Short gates.** A five-of-eight gate promoted 37.1% of candidates in the recorded runs. A fair coin passes the same gate about 36.3% of the time. Use a fixed independent strength match to distinguish selection noise from lasting progress.

## Replay fit and generalization

On November 13, 2026, a checkpoint-averaging experiment compared three networks on 8,192 replay positions after recalibrating BatchNorm. The most recent candidate gave policy CE 1.748 and value MSE 0.427. Averaging ten checkpoints gave 1.895 and 0.755. Averaging the most recent five gave 1.828 and 0.739. Both averages missed the predeclared loss criterion, so the experiment stopped before playing a match. The newest candidate had trained on more of those replay positions. That difference limits what the screen reveals about strength on fresh games.

One checkpoint's value MSE was 0.345 on familiar replay positions and approximately 0.76–0.79 on comparably recent unseen positions. A lower training loss can therefore reflect memorization. Reserve entire games so neighboring positions do not appear on both sides of the split, and compare against the same network's training-buffer estimate. Identical opening positions can still occur in independent games.

A separate 5×5 experiment compared 400 updates with a learning-rate-zero BatchNorm control and then played the promoted weights against initialization. The strength and value-prediction criteria reached different conclusions. [Learning verification](gpu_run.md#learning-verification) · [Match methodology](strength.md)
