from __future__ import annotations

"""Short end-to-end smoke test for the reconstruction."""

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphago_zero import (
    AlphaGoZeroMCTS,
    AlphaGoZeroNet,
    ReplayBuffer,
    ResignationController,
    create_optimizer,
    make_demo_config,
    play_self_play_game,
    train_for_steps,
)
from alphago_zero.game import GoState


def main() -> None:
    # One CPU thread avoids excess overhead on tiny networks.
    torch.set_num_threads(1)

    rng = np.random.default_rng(123)
    cfg = make_demo_config(work_dir=PROJECT_ROOT / "runs" / "smoke_test")

    # Use a small workload for a quick integration check.
    cfg.search.num_simulations = 6
    cfg.training.batch_size = 8
    cfg.training.gradient_steps_per_iteration = 2
    cfg.training.self_play_games_per_iteration = 1
    cfg.training.evaluation_games = 2

    state = GoState.new_game(
        board_size=cfg.game.board_size,
        komi=cfg.game.komi,
        max_moves=cfg.game.max_moves,
    )

    encoded = state.encode()
    assert encoded.shape == (17, cfg.game.board_size, cfg.game.board_size)

    model = AlphaGoZeroNet(cfg.network)
    policy_logits, value = model(torch.from_numpy(encoded).unsqueeze(0).float())
    assert policy_logits.shape == (1, cfg.action_size)
    assert value.shape == (1, 1)

    mcts = AlphaGoZeroMCTS(model=model, config=cfg.search, rng=rng)
    output = mcts.run(state, temperature=1.0, add_root_noise=True)
    assert output.policy.shape == (cfg.action_size,)
    assert np.isclose(output.policy.sum(), 1.0), output.policy.sum()

    resignation_controller = ResignationController(
        threshold=cfg.search.resign_threshold,
        target_false_positive_rate=cfg.search.target_false_positive_rate,
        adjustment=cfg.search.resignation_adjustment,
        window=cfg.search.resignation_window,
    )
    result = play_self_play_game(
        model=model,
        config=cfg,
        rng=rng,
        resignation_controller=resignation_controller,
    )
    assert len(result.examples) > 0

    buffer = ReplayBuffer(capacity=256, board_size=cfg.game.board_size, rng=rng)
    buffer.add_game(result.examples)
    assert len(buffer) == len(result.examples)

    optimizer = create_optimizer(model, cfg)
    metrics, new_global_step = train_for_steps(
        model=model,
        optimizer=optimizer,
        replay_buffer=buffer,
        config=cfg,
        rng=rng,
        global_step=0,
    )
    assert metrics.steps == cfg.training.gradient_steps_per_iteration
    assert new_global_step == cfg.training.gradient_steps_per_iteration

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
