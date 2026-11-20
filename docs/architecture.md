# Architecture

One iteration has three model roles. The **best** network generates games. The **candidate** retains its optimizer state while learning from replay. A **frozen anchor** measures progress against a fixed opponent. A failed promotion leaves the best network in place and the candidate continues training.

| Stage | Inputs and outputs |
| --- | --- |
| Play | `game.py` maintains the board, turn and repetition history. `network.py` predicts policy logits and a value. `mcts.py` searches legal actions using those predictions. |
| Record | `selfplay.py` stores the pre-move state and policy. After the game it adds +1, 0 or −1 from the player's perspective. `parallel_selfplay.py` shares leaf inference across independent games. |
| Sample | `replay_buffer.py` evicts the oldest positions and samples minibatches. `augmentations.py` applies the same symmetry to the input planes, policy and any ownership map. |
| Update | `train.py` computes policy cross-entropy, value MSE and optional auxiliary terms. SGD updates the candidate. |
| Evaluate | `evaluate.py` alternates colors in a candidate–best match. `pipeline.py` applies the selected gate, records metrics and writes checkpoints. |
| Benchmark | `benchmark.py`, `players.py` and `elo.py` measure the best network against fixed opponents with a separate RNG. |

## Search and training interfaces

Search accepts a `PolicyValueModel` with `predict_single`, `predict_batch` and `eval`. A single `AlphaGoZeroNet` and an `EnsembleNet` both implement this interface. The ensemble averages probabilities after each member's softmax. Networks may differ in depth or width but must agree on input planes, board, rules and action space.

The move policy, the learning target and the outcome have separate lifetimes. Temperature changes the action actually played. The PUCT learning target retains the distribution of root visits. A Gumbel search supplies an improved policy. An outcome becomes available only when the game ends. [Search](mcts.md)

## Parallelism and state

`parallel_games` sets the number of active game slots per worker. `selfplay_workers` adds processes with separate model replicas. The parent assigns games, sends current weights, collects results in worker order and owns the replay buffers, optimizer and resignation controller. The worker threshold stays fixed for its part of an iteration. The parent applies the collected calibration observations.

Holdout assignment operates on whole games before replay insertion. Only training-buffer entries enter SGD. Holdout monitoring and optional benchmarks use dedicated random streams. [Performance](performance.md) · [Reproducibility](reproducibility.md)
