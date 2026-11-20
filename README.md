# AlphaGo Zero Reconstruction

Train a residual network to play Go from its own games. This independent PyTorch implementation follows the self-play, search, training and evaluation loop in Silver et al., *Mastering the game of Go without human knowledge*. It includes a CPU-sized experiment, the Go engine, a GTP interface and tools for measuring playing strength. [Original paper](https://doi.org/10.1038/nature24270) · [Paper-to-code map](ARTICLE_TO_CODE_MAP.md)

## Run your first game and training loop

Use Python 3.11 or newer. From the repository root, create and activate a virtual environment. On Windows PowerShell use `.\.venv\Scripts\Activate.ps1`. On Linux or macOS use `source .venv/bin/activate`.

```sh
python -m venv .venv
# Activate the environment, then install the package and its test tools.
python -m pip install -e ".[dev]"
python examples/run_tiny_demo.py
```

The demo runs two iterations on a 5×5 board. Each iteration generates two games, performs four optimizer steps and evaluates the candidate. The script prints a JSON summary and writes weights, metrics and resume state to `runs/tiny_demo/`. Its short matches exercise the training loop. For an experiment that measures a change in strength, see the [learning verification protocol](docs/gpu_run.md#learning-verification).

## Play the trained network

```sh
python -m examples.play_gtp --checkpoint runs/tiny_demo/final_best.pt --device cpu
```

Enter these commands one line at a time. The engine plays Black's first move, prints the board and exits. A GTP-compatible Go GUI can launch the same command.

```text
boardsize 5
komi 2.5
genmove b
showboard
quit
```

The checkpoint fixes the board size and komi. Training a 9×9 network produces a checkpoint suitable for a 9×9 game.

## How a position becomes a training example

1. The network reads 17 input planes and predicts move logits and an outcome for the player to move.
2. Search expands legal moves using those priors and evaluates leaves with the same network. Self-play saves the position and improved policy before playing a move.
3. When the game ends, each saved position receives an outcome from its own player's perspective. Training samples these examples from replay and updates the network with momentum SGD.
4. The candidate plays the current best with alternating colors. The configured gate decides which network generates the next set of games.

`make_demo_config()` runs locally. `make_paper_like_config()` selects a 19×19 board, a 256-channel tower and 1,600 simulations per move. Its training workload is substantially larger. Both use the same pipeline. Gumbel search, auxiliary ownership and score heads, leagues and checkpoint averaging are optional experiments.

## Where to go next

| Task | Guide |
| --- | --- |
| Follow one iteration through the code | [Architecture](docs/architecture.md) |
| Understand legal moves, scoring or the search value sign | [Go rules](docs/go_rules.md) · [Search](docs/mcts.md) |
| Configure a run, resume training or recover a snapshot | [Configuration](docs/configuration.md) · [Checkpoints](docs/checkpoints.md) |
| Train on CUDA or reproduce the learning diagnostic | [GPU training](docs/gpu_run.md) |
| Compare checkpoints or play an external engine | [Strength evaluation](docs/strength.md) |
| Run tests or investigate throughput | [Testing](docs/testing.md) · [Performance](docs/performance.md) |
| Review research results and open questions | [Experiments](docs/rnd_experiments.md) · [Documentation index](docs/README.md) |

Licensed under the [MIT License](LICENSE).

Copyright © 2026 [Pavel Deecalov](https://github.com/deecalov) and the AlphaGo Zero Reconstruction contributors.
