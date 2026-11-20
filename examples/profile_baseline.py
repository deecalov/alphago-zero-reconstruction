from __future__ import annotations

"""Profile the Go engine, search, self-play and SGD separately."""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphago_zero import (
    AlphaGoZeroMCTS,
    AlphaGoZeroNet,
    GoState,
    NetworkConfig,
    SearchConfig,
    make_demo_config,
)
from alphago_zero.selfplay import ResignationController, play_self_play_game


def _random_game_states(board_size: int, num_moves: int, rng: np.random.Generator) -> list[GoState]:
    """Generate positions from a random game to benchmark more than the empty board."""
    state = GoState.new_game(board_size=board_size, komi=7.5)
    states = [state]
    for _ in range(num_moves):
        if state.is_terminal():
            break
        actions = state.legal_actions()
        state = state.play(int(rng.choice(actions)))
        states.append(state)
    return states


def bench_engine(board_size: int, rng: np.random.Generator, num_games: int = 3) -> dict[str, float]:
    """Measure random-game moves per second and mean legal-move enumeration time."""
    t0 = time.perf_counter()
    total_moves = 0
    for _ in range(num_games):
        state = GoState.new_game(board_size=board_size, komi=7.5)
        while not state.is_terminal():
            actions = state.legal_actions()
            state = state.play(int(rng.choice(actions)))
            total_moves += 1
    dt = time.perf_counter() - t0
    moves_per_sec = total_moves / dt if dt > 0 else float("inf")

    # Time legal_actions on midgame positions.
    states = _random_game_states(board_size, num_moves=board_size * board_size // 2, rng=rng)
    mid_states = states[len(states) // 2 :]
    t0 = time.perf_counter()
    calls = 0
    for s in mid_states:
        if s.is_terminal():
            continue
        s.legal_actions()
        calls += 1
    dt_legal = (time.perf_counter() - t0) / max(1, calls)

    return {
        "moves_per_sec": moves_per_sec,
        "legal_actions_ms": dt_legal * 1000.0,
        "total_moves": float(total_moves),
    }


def _tiny_model(board_size: int) -> AlphaGoZeroNet:
    torch.manual_seed(0)
    cfg = NetworkConfig(
        board_size=board_size,
        num_input_planes=17,
        channels=32,
        num_res_blocks=2,
        value_head_hidden=32,
    )
    model = AlphaGoZeroNet(cfg)
    model.eval()
    return model


def bench_mcts(board_size: int, num_simulations: int, rng: np.random.Generator) -> dict[str, float]:
    """Measure search simulations per second from an empty position."""
    model = _tiny_model(board_size)
    config = SearchConfig(num_simulations=num_simulations)
    search = AlphaGoZeroMCTS(model=model, config=config, rng=rng)
    state = GoState.new_game(board_size=board_size, komi=7.5)

    t0 = time.perf_counter()
    repeats = 3
    for _ in range(repeats):
        search.run(root_state=state, temperature=1.0, add_root_noise=True, root=None)
    dt = time.perf_counter() - t0
    sims_total = num_simulations * repeats
    return {"sims_per_sec": sims_total / dt if dt > 0 else float("inf")}


def _selfplay_once(board_size: int, num_simulations: int, seed: int) -> tuple[int, float]:
    """Play one self-play game and return move count and elapsed seconds."""
    rng = np.random.default_rng(seed)
    cfg = make_demo_config(work_dir=PROJECT_ROOT / "runs" / "profile_tmp")
    cfg.game.board_size = board_size
    cfg.network.board_size = board_size
    cfg.network.channels = 32
    cfg.network.num_res_blocks = 2
    cfg.network.value_head_hidden = 32
    cfg.search.num_simulations = num_simulations
    cfg.validate()

    model = _tiny_model(board_size)
    controller = ResignationController(
        threshold=-0.95,
        target_false_positive_rate=0.05,
        adjustment=0.02,
        window=100,
    )
    t0 = time.perf_counter()
    result = play_self_play_game(model=model, config=cfg, rng=rng, resignation_controller=controller)
    dt = time.perf_counter() - t0
    return result.num_moves, dt


def bench_selfplay(board_size: int, num_simulations: int) -> dict[str, float]:
    moves, dt = _selfplay_once(board_size, num_simulations, seed=7)
    return {
        "moves_per_sec": moves / dt if dt > 0 else float("inf"),
        "game_sec": dt,
        "moves": float(moves),
    }


def bench_parallel_selfplay(
    board_size: int,
    num_simulations: int,
    num_games: int,
    parallel_games: int,
) -> dict[str, float]:
    """Measure moves per second with inference batched across games."""
    from alphago_zero.parallel_selfplay import play_parallel_self_play_games

    rng = np.random.default_rng(7)
    cfg = make_demo_config(work_dir=PROJECT_ROOT / "runs" / "profile_tmp")
    cfg.game.board_size = board_size
    cfg.network.board_size = board_size
    cfg.network.channels = 32
    cfg.network.num_res_blocks = 2
    cfg.network.value_head_hidden = 32
    cfg.search.num_simulations = num_simulations
    cfg.training.parallel_games = parallel_games
    cfg.validate()

    model = _tiny_model(board_size)
    controller = ResignationController(
        threshold=-0.95,
        target_false_positive_rate=0.05,
        adjustment=0.02,
        window=100,
    )
    t0 = time.perf_counter()
    results = play_parallel_self_play_games(
        model=model,
        config=cfg,
        rng=rng,
        resignation_controller=controller,
        num_games=num_games,
    )
    dt = time.perf_counter() - t0
    total_moves = sum(r.num_moves for r in results)
    return {
        "moves_per_sec": total_moves / dt if dt > 0 else float("inf"),
        "total_sec": dt,
        "moves": float(total_moves),
    }


def bench_sgd(board_size: int, batch_size: int, device: str) -> dict[str, float]:
    """Measure SGD steps per second on a synthetic batch with the demo residual network."""
    torch.manual_seed(0)
    cfg = NetworkConfig(
        board_size=board_size,
        num_input_planes=17,
        channels=64,
        num_res_blocks=4,
        value_head_hidden=64,
    )
    model = AlphaGoZeroNet(cfg).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)

    action_size = board_size * board_size + 1
    states = torch.randn(batch_size, 17, board_size, board_size, device=device)
    target_p = torch.softmax(torch.randn(batch_size, action_size, device=device), dim=1)
    target_v = torch.rand(batch_size, device=device) * 2 - 1

    # Warm up backend kernels before starting the timer.
    steps_warmup, steps_measure = 3, 15
    for _ in range(steps_warmup):
        optimizer.zero_grad(set_to_none=True)
        logits, value = model(states)
        loss = torch.nn.functional.mse_loss(value.squeeze(1), target_v) - (
            target_p * torch.log_softmax(logits, dim=1)
        ).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps_measure):
        optimizer.zero_grad(set_to_none=True)
        logits, value = model(states)
        loss = torch.nn.functional.mse_loss(value.squeeze(1), target_v) - (
            target_p * torch.log_softmax(logits, dim=1)
        ).sum(dim=1).mean()
        loss.backward()
        optimizer.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"sgd_steps_per_sec": steps_measure / dt if dt > 0 else float("inf")}


def profile_selfplay(board_size: int, num_simulations: int, top: int = 15) -> str:
    """Return a self-play cProfile report sorted by cumulative time."""
    profiler = cProfile.Profile()
    profiler.enable()
    _selfplay_once(board_size, num_simulations, seed=11)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("cumulative").print_stats(top)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline profiling (roadmap phase 0)")
    parser.add_argument("--full", action="store_true", help="include slower benchmarks such as the 19x19 engine")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(1)
    rng = np.random.default_rng(42)

    print("ENV", f"python={sys.version.split()[0]}", f"torch={torch.__version__}", f"numpy={np.__version__}")
    print("DEVICE_FOR_SGD", args.device)
    print()

    sizes = [5, 9] + ([19] if args.full else [])
    for size in sizes:
        res = bench_engine(size, rng)
        print(
            f"ENGINE {size}x{size}: moves/sec={res['moves_per_sec']:.1f} "
            f"legal_actions={res['legal_actions_ms']:.2f} ms (moves={int(res['total_moves'])})"
        )
    print()

    for size in (5, 9):
        res = bench_mcts(size, num_simulations=64, rng=rng)
        print(f"MCTS {size}x{size} (sims=64, tiny net): sims/sec={res['sims_per_sec']:.1f}")
    print()

    res = bench_selfplay(9, num_simulations=32)
    print(
        f"SELF-PLAY 9x9 (sims=32, tiny net): moves/sec={res['moves_per_sec']:.2f} "
        f"game={res['game_sec']:.1f}s moves={int(res['moves'])}"
    )

    for parallel in (1, 4, 8):
        res = bench_parallel_selfplay(9, num_simulations=32, num_games=8, parallel_games=parallel)
        print(
            f"PARALLEL SELF-PLAY 9x9 (8 games, P={parallel}): "
            f"moves/sec={res['moves_per_sec']:.2f} total={res['total_sec']:.1f}s"
        )
    print()

    res = bench_sgd(9, batch_size=256, device=args.device)
    print(f"SGD 9x9 (batch=256, demo net 64ch/4blk): steps/sec={res['sgd_steps_per_sec']:.2f}")
    print()

    print("PROFILE self-play 9x9 (sims=16), top by cumulative:")
    print(profile_selfplay(9, num_simulations=16))


if __name__ == "__main__":
    main()
