# From the paper to the implementation

Use this map alongside Silver et al., *Mastering the game of Go without human knowledge*, Nature 550 (2017), 354–359. The demo and paper-like presets call the same functions with different budgets. [Paper](https://doi.org/10.1038/nature24270) · [Configuration](docs/configuration.md)

| Paper concept | What to follow in this repository |
| --- | --- |
| Figure 1a, self-play | `play_self_play_game` saves a position and the search policy before each move. At game end it attaches the final outcome in that position's perspective. [Self-play](alphago_zero/selfplay.py) |
| Figure 1b, learning | `ReplayBuffer` samples earlier examples. `train_for_steps` fits the policy and value heads with momentum SGD. [Replay](alphago_zero/replay_buffer.py) · [Training](alphago_zero/train.py) |
| Figure 2, selection | `_select_action` chooses the move maximizing Q plus a prior-weighted exploration bonus. [Tree search](alphago_zero/mcts.py) |
| Figure 2, expansion and evaluation | `_expand` evaluates the root. `_evaluate_and_backup` batches leaves. Legal-action softmax, inverse symmetry and exact terminal values enter here. [Search guide](docs/mcts.md) |
| Figure 2, backup | `_backup` negates the leaf value before adding it to the parent edge, repeating at every ply. [Sign regression tests](tests/test_mcts.py) |
| Figure 2, play | `SearchOutput.policy_target` exposes the learning target. The separate move policy uses temperature and can become deterministic while the target still covers several moves. [Tree search](alphago_zero/mcts.py) |
| Equation 1 | Value MSE plus cross-entropy against the search distribution. `create_optimizer` applies L2 regularization through weight decay. [Training](alphago_zero/train.py) |
| Methods, positions and symmetries | `GoState.encode` interleaves eight historical boards for both sides and appends a color plane. Rotations and reflections transform board moves and leave pass fixed. [Rules](alphago_zero/game.py) · [Symmetries](alphago_zero/augmentations.py) |
| Methods, architecture | `AlphaGoZeroNet` shares one residual tower. The policy head emits a logit per point plus pass. The value head emits a scalar bounded by tanh. [Network](alphago_zero/network.py) |
| Methods, search exploration | Self-play mixes Dirichlet noise into root priors and samples early moves. Evaluation disables root noise and resignation. [Search](docs/mcts.md) · [Evaluation](alphago_zero/evaluate.py) |
| Methods, resignation | Protected games continue past potential resignations to calibrate the threshold against actual outcomes. [Self-play](alphago_zero/selfplay.py) |
| Evaluation and replacement | The training loop promotes the candidate according to the chosen gate. The paper-like preset uses 400 games and a 0.55 threshold. [Pipeline](alphago_zero/pipeline.py) |
| Extended Data Table 3 | `paper_learning_rate` implements the optimizer-step schedule. [Configuration](alphago_zero/config.py) |

The project uses a bounded buffer of **positions** and saves the optimizer, buffers and RNG states for a resumable single-machine run. The original distributed training infrastructure and its compute scale are outside this implementation. Gumbel search, ownership targets, external-engine matches and weight averaging have their own experimental settings. [Architecture](docs/architecture.md) · [Experiments](docs/rnd_experiments.md)
