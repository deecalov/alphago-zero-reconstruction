# Documentation

| Guide | Question it answers |
| --- | --- |
| [Architecture](architecture.md) | Which model plays, which model learns, and when do they change? |
| [Go rules](go_rules.md) | How are liberties, repetitions, draws and final scores handled? |
| [Search](mcts.md) | Whose value is stored on an edge, and what changes when leaves are batched? |
| [Configuration](configuration.md) | Which presets, budgets and controls affect an experiment? |
| [Checkpoints](checkpoints.md) | Which artifact can play, resume or seed another experiment? |
| [Reproducibility](reproducibility.md) | What does a seed control, and what else must match between runs? |
| [GPU training](gpu_run.md) | How do I run on CUDA and test whether a network learned? |
| [Testing](testing.md) | How do I run CPU, CUDA and GNU Go checks? |
| [Performance](performance.md) | What limits throughput, and how many game slots actually fill? |
| [Strength evaluation](strength.md) | How do I compare checkpoints, use GTP and interpret uncertainty? |
| [Research experiments](rnd_experiments.md) | Which optional methods have been implemented and what have measurements shown? |
| [Game improvement audit](game_improvement_audit.md) | Which rule and play interfaces have independent controls, and where do scorers disagree? |
| [Initial roadmap](roadmap.md) · [Scaling proposals](roadmap_v3.md) · [Research roadmap](roadmap_v4.md) | How did the implementation evolve and which questions are still open? |

For a first run, start with the [repository README](../README.md). To read the source in the paper's order, use the [paper-to-code map](../ARTICLE_TO_CODE_MAP.md).
