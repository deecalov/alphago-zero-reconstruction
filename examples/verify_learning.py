"""Bounded learning experiment, distinct from CUDA/pipeline smoke tests.

Run from the source tree: python -m examples.verify_learning --output runs/learning
The protocol is written before training. Evaluation never selects checkpoints.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.game import BLACK, WHITE
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.parallel_selfplay import play_parallel_self_play_games
from alphago_zero.pipeline import (
    _environment_info, _save_checkpoint, _save_json, _set_global_seed,
    run_training_pipeline,
)
from alphago_zero.replay_buffer import ReplayBuffer, TrainingExample
from alphago_zero.selfplay import ResignationController
from alphago_zero.train import create_optimizer, train_for_steps
from examples.eval_match import load_checkpoint, play_game, wilson_interval


@torch.inference_mode()
def prediction_errors(model: AlphaGoZeroNet, examples: list[TrainingExample]) -> dict[str, float]:
    """Unaugmented, inference-mode errors on a fixed dataset; no BN updates."""
    model.eval()
    totals = np.zeros(3, dtype=np.float64)
    for offset in range(0, len(examples), 256):
        batch = examples[offset:offset + 256]
        states = torch.as_tensor(np.stack([x.state for x in batch]), device=model.device)
        policies = torch.as_tensor(np.stack([x.policy for x in batch]), device=model.device)
        values = torch.tensor([x.value for x in batch], device=model.device)
        logits, predicted = model(states.float())
        ce = -(policies * F.log_softmax(logits, dim=1)).sum(dim=1)
        entropy = -(policies * policies.clamp_min(1e-12).log()).sum(dim=1)
        mse = (predicted.squeeze(1) - values).square()
        totals += [ce.sum().item(), (ce - entropy).sum().item(), mse.sum().item()]
    ce_mean, kl_mean, mse_mean = (totals / len(examples)).tolist()
    return {"policy_ce": ce_mean, "policy_kl": kl_mean, "value_mse": mse_mean,
            "total_loss": ce_mean + mse_mean}


def parameter_distance(left: AlphaGoZeroNet, right: AlphaGoZeroNet) -> float:
    """Trainable weights only: BN running statistics are excluded."""
    return sum((a.detach() - b.detach()).square().sum().item()
               for a, b in zip(left.parameters(), right.parameters(), strict=True)) ** 0.5


def fixed_data_diagnostic(initial: AlphaGoZeroNet, cfg: AlphaGoZeroConfig) -> dict:
    # Independent RNG streams, with complete games assigned to a split before generation.
    split_examples = {}
    for split, games, seed_offset in (("train", 64, 101), ("holdout", 32, 202)):
        results = play_parallel_self_play_games(
            model=initial, config=cfg,
            rng=np.random.default_rng(cfg.training.seed + seed_offset),
            resignation_controller=ResignationController(-0.95, 0.05, 0.02, 100),
            num_games=games,
        )
        split_examples[split] = [example for game in results for example in game.examples]
        print(f"diagnostic {split}: {games} games, {len(split_examples[split])} positions", flush=True)
    buffer = ReplayBuffer(len(split_examples["train"]), cfg.game.board_size)
    buffer.add_many(split_examples["train"])
    diagnostic_cfg = copy.deepcopy(cfg)
    diagnostic_cfg.training.gradient_steps_per_iteration = 400
    report: dict[str, Any] = {
        "games": {"train": 64, "holdout": 32},
        "positions": {key: len(value) for key, value in split_examples.items()},
        "initial": {key: prediction_errors(initial, data) for key, data in split_examples.items()},
    }
    # lr=0 control sees the exact same batches and updates BN, but cannot learn weights.
    # This isolates SGD improvement from merely calibrating BatchNorm statistics.
    for name, learning_rate in (("trained", cfg.training.learning_rate), ("bn_only_control", 0.0)):
        model = copy.deepcopy(initial)
        diagnostic_cfg.training.learning_rate = learning_rate
        metrics, steps = train_for_steps(
            model, create_optimizer(model, diagnostic_cfg), buffer, diagnostic_cfg,
            np.random.default_rng(cfg.training.seed + 303), 0,
        )
        report[name] = {
            "steps": steps, "train_metrics": asdict(metrics),
            "parameter_l2_change": parameter_distance(model, initial),
            **{key: prediction_errors(model, data) for key, data in split_examples.items()},
        }
    trained = report["trained"]["holdout"]
    baseline = report["initial"]["holdout"]
    control = report["bn_only_control"]["holdout"]
    report["passed"] = bool(
        report["trained"]["parameter_l2_change"] > 0
        and report["bn_only_control"]["parameter_l2_change"] == 0
        and all(trained[key] < min(baseline[key], control[key])
                for key in ("policy_kl", "value_mse"))
    )
    return report


def match(final_path: Path, initial_path: Path, output: Path, seed: int, games: int) -> dict:
    final, search, game = load_checkpoint(final_path, torch.device("cuda"))
    initial, initial_search, initial_game = load_checkpoint(initial_path, torch.device("cuda"))
    assert search == initial_search and game == initial_game
    rng = np.random.default_rng(seed)
    wins = black_wins = white_wins = 0
    outcomes = []
    started = time.perf_counter()
    for index in range(games):
        color = BLACK if index % 2 == 0 else WHITE
        winner = play_game(
            black_model=final if color == BLACK else initial,
            white_model=initial if color == BLACK else final,
            black_search=search, white_search=search, game=game, rng=rng, opening_moves=6,
        )
        won = winner == color
        wins += int(won)
        black_wins += int(won and color == BLACK)
        white_wins += int(won and color == WHITE)
        outcomes.append({"final_color": color, "winner": winner})
        rate, lower, upper = wilson_interval(wins, index + 1)
        report = {
            "games_planned": games, "games_completed": index + 1,
            "wins": wins, "losses": index + 1 - wins, "win_rate": rate,
            "wilson_95": [lower, upper], "wins_as_black": black_wins,
            "wins_as_white": white_wins, "seed": seed, "opening_moves": 6,
            "search": asdict(search), "elapsed_sec": time.perf_counter() - started,
            "outcomes": outcomes,
            "passed": index + 1 == games and lower > 0.5,
        }
        _save_json(output, report)
        if (index + 1) % 10 == 0:
            print(f"match {index + 1}/{games}: {wins} wins ({rate:.1%})", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--match-games", type=int, default=120)
    args = parser.parse_args()
    if args.iterations < 1 or args.match_games < 2 or args.match_games % 2:
        parser.error("iterations must be positive; match-games must be positive and even")
    if not torch.cuda.is_available():
        raise RuntimeError("A real CUDA GPU is required; CPU fallback is forbidden for this experiment")
    # Refuse to overwrite a finished or partial experiment: negative results must remain visible.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    cfg = AlphaGoZeroConfig()
    cfg.network.channels = 32
    cfg.network.num_res_blocks = 2
    cfg.search.num_simulations = 32
    cfg.search.eval_batch_size = 8
    cfg.search.temperature_moves = 10
    cfg.search.disable_resign_prob = 1.0
    cfg.training.seed = args.seed
    cfg.training.parallel_games = 8
    cfg.training.self_play_games_per_iteration = 16
    cfg.training.gradient_steps_per_iteration = 64
    cfg.training.num_iterations = args.iterations
    cfg.training.replay_buffer_size = 16000
    cfg.training.evaluation_games = 32
    cfg.training.evaluation_opening_moves = 6
    cfg.training.save_every_iteration = False
    cfg.runtime.device = "cuda"
    cfg.runtime.use_amp = True
    cfg.runtime.save_resume_every = 5
    cfg.runtime.work_dir = args.output / "pipeline"
    cfg.validate()
    root = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("alphago_zero", "examples") for path in sorted((root / folder).glob("*.py"))
    }
    protocol = {
        "config": asdict(cfg), "environment": _environment_info(torch.device("cuda")),
        "gpu": torch.cuda.get_device_name(), "source_sha256": source_hashes,
        "diagnostic": {"train_games": 64, "holdout_games": 32, "steps": 400,
                       "data_seeds": [args.seed + 101, args.seed + 202],
                       "batch_seed": args.seed + 303,
                       "criterion": "holdout policy KL and value MSE lower than initial AND lr=0 BN control"},
        "primary_match": {"model": "pipeline/final_best.pt", "opponent": "initial.pt",
                          "games": args.match_games, "seed": args.seed + 10000,
                          "opening_moves": 6, "criterion": "Wilson 95% lower bound > 0.5",
                          "checkpoint_selection": "pipeline promotion gate only; no selection on this match"},
    }
    _save_json(args.output / "protocol.json", protocol)
    _set_global_seed(args.seed)
    initial = AlphaGoZeroNet(cfg.network).cuda()
    _save_checkpoint(args.output / "initial.pt", initial, None, {"config": asdict(cfg)})
    started = time.perf_counter()
    print("phase 1: held-out prediction diagnostic (separate model, not reused by pipeline)", flush=True)
    diagnostic = fixed_data_diagnostic(initial, cfg)
    _save_json(args.output / "diagnostic.json", diagnostic)
    print(json.dumps(diagnostic), flush=True)
    print("phase 2: complete self-play -> replay -> SGD -> promotion pipeline, from the same initial seed", flush=True)
    summary = run_training_pipeline(cfg)
    print(json.dumps(summary), flush=True)
    print("phase 3: independent final_best vs initial match, fixed sample size", flush=True)
    match_report = match(Path(summary["final_best_path"]), args.output / "initial.pt",
                         args.output / "match.json", args.seed + 10000, args.match_games)
    report = {
        "diagnostic_passed": diagnostic["passed"], "strength_passed": match_report["passed"],
        "passed": diagnostic["passed"] and match_report["passed"],
        "completed_iterations": summary["completed_iterations"],
        "global_step": summary["global_step"], "best_iteration": summary["best_iteration"],
        "elapsed_sec": time.perf_counter() - started,
        "scope": "one seed, 5x5, 32 MCTS simulations; not a claim about 9x9/19x19 strength",
    }
    _save_json(args.output / "result.json", report)
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
