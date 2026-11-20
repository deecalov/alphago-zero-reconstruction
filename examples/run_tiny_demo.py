from __future__ import annotations

"""Run two small iterations to exercise self-play, training, evaluation and promotion on CPU."""

import json
from pathlib import Path
import sys

import torch

# Allow direct execution from an unpacked source tree.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphago_zero import make_demo_config, run_training_pipeline


def main() -> None:
    # Use one CPU thread for predictable performance on the tiny network.
    torch.set_num_threads(1)

    cfg = make_demo_config(work_dir=PROJECT_ROOT / "runs" / "tiny_demo")

    # Reduce the demo workload further so a CPU run finishes quickly.
    cfg.training.num_iterations = 2
    cfg.training.self_play_games_per_iteration = 2
    cfg.training.gradient_steps_per_iteration = 4
    cfg.training.evaluation_games = 2
    cfg.training.batch_size = 16

    cfg.search.num_simulations = 8
    cfg.search.temperature_moves = 6

    summary = run_training_pipeline(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
