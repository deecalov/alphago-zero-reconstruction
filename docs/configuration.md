# Configuring a run

Choose a preset with the board and workload closest to the experiment. `AlphaGoZeroConfig()`, the two factory functions and `examples.gpu_train` each have their own defaults. Save the resolved `config.json` with the results when comparing runs.

| Setting | `make_demo_config()` | `make_paper_like_config()` |
| --- | ---: | ---: |
| Board / komi | 5×5 / 2.5 | 19×19 / 7.5 |
| Channels / residual blocks | 64 / 4 | 256 / 19 |
| Search simulations / move | 32 | 1,600 |
| Dirichlet alpha / mixing fraction | 0.30 / 0.25 | 0.03 / 0.25 |
| Opening moves sampled at temperature one | 8 | 30 |
| Replay capacity in positions | 10,000 | 500,000 |
| SGD batch size / steps per iteration | 64 / 16 | 2,048 / 1,000 |
| Self-play games per iteration | 4 | 25,000 |
| Evaluation games / promotion threshold | 4 / 0.55 | 400 / 0.55 |

```python
from alphago_zero import make_demo_config, run_training_pipeline

config = make_demo_config("runs/experiment")
config.training.num_iterations = 2
config.training.seed = 73
config.validate()
summary = run_training_pipeline(config)
print(summary["final_best_path"])
```

If changing board size programmatically, set both `config.game.board_size` and `config.network.board_size`. Weights depend on the board dimensions. Keep a trained network's komi fixed when comparing playing strength.

## Trade-offs in search and training

| Control | Operational effect |
| --- | --- |
| `num_simulations`, `eval_batch_size`, `virtual_loss` | Increase simulations to search more of each tree. Increasing the leaf batch reduces inference calls but changes which pending paths are explored. Virtual loss takes effect when multiple paths are pending. |
| `parallel_games`, `selfplay_workers` | The first batches several games per process. The second replicates the network across processes. Each worker needs enough assigned games to occupy its slots. |
| `fpu_reduction` | `None` assigns zero Q to an unvisited edge. A reduction of zero starts it at the parent value. |
| `search_algo`, `gumbel_max_considered` | Gumbel controls learner-side self-play and requires at least two root candidates in the configuration. Large leaf batches can distort its elimination schedule. |
| `playout_cap_prob`, `playout_cap_cheap_sims` | A probability strictly between zero and one mixes full-budget and cheap PUCT moves. Cheap targets have zero policy weight. Their positions can still teach value. |
| `forced_playout_fraction`, `policy_target_prune_fraction` | Optional PUCT root quota and post-search target pruning. Gumbel rejects forced root visits and randomized playout caps. |
| `network.aux_heads`, `aux_warmup_steps` | Enable ownership and score heads. A linear warmup protects existing trunk features when adding randomly initialized heads. |
| `disable_resign_prob` | Fraction of games played without resignation. Use 1.0 to obtain scored endings for all games, including ownership targets. |
| `league_prob`, `league_checkpoints` | The parallel driver can select a frozen opponent and assign it a color. Only the learner's resignation decisions use its calibrated threshold. Configure `parallel_games > 1` for league play. |
| `holdout_game_fraction`, `holdout_buffer_size` | Reserve entire games, with a separate position capacity for monitoring. Zero fraction disables holdout. |
| `lr_schedule`, `weight_decay_exclude_norm_bias` | The constant schedule uses `learning_rate`. The paper schedule uses 0.01 before step 400,000, 0.001 before 600,000 and 0.0001 thereafter. Excluding BatchNorm and bias from decay changes optimizer groups and must agree on resume. |

The **fixed** gate checks the win threshold. **Binomial** also requires sufficient wins under an exact one-sided binomial test. **None** skips the match and promotes each candidate. A small match can be adequate to test code paths while giving an unstable strength estimate. The evaluator can stop when remaining games cannot change its decision. [Strength evaluation](strength.md)

Run `python -m examples.gpu_train --help` for the full command-line interface. Configuration validation checks numeric ranges, finite values, board compatibility, incompatible search modes and mutually exclusive resume choices. [Checkpoints](checkpoints.md)
