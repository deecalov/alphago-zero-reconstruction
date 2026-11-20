from __future__ import annotations

"""Run the training pipeline on CUDA with configurable board, network and search settings."""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphago_zero import AlphaGoZeroNet, make_demo_config, run_training_pipeline
from alphago_zero.cli_types import (
    finite_float,
    fpu_value,
    non_negative_float,
    positive_float,
    positive_int,
)
from alphago_zero.config import SearchConfig

# Read the default virtual loss from the search configuration.
_DEFAULT_VIRTUAL_LOSS = SearchConfig().virtual_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU training run for AlphaGo Zero reconstruction")
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resign-min-visits", type=int, default=8)
    p.add_argument("--sims", type=int, default=32, help="num_simulations MCTS")
    p.add_argument("--self-play-games", type=int, default=4)
    p.add_argument("--grad-steps", type=int, default=150)
    p.add_argument("--eval-games", type=int, default=128)
    p.add_argument(
        "--eval-gate", choices=["fixed", "binomial", "none"], default="binomial",
        help="candidate promotion: binomial requires a win threshold and an exact test (default). "
             "fixed checks only the win threshold. With none, skip the match and promote "
             "the latest candidate on every iteration for self-play "
             "(--eval-games is unused). In historical runs, a five-of-eight gate "
             "promoted 37.1%% of candidates versus 36.3%% for a fair coin and used about 37%% of the "
             "iteration time. With the gate disabled, track fixed anchors and run independent matches. "
             "See docs/rnd_experiments.md",
    )
    p.add_argument("--eval-alpha", type=float, default=0.05)
    p.add_argument("--eval-opening-moves", type=int, default=6)
    p.add_argument("--board", type=int, default=9)
    p.add_argument(
        "--komi", type=non_negative_float, default=None,
        help="finite nonnegative komi (demo default 2.5, common 9x9 setting 7.5)",
    )
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--res-blocks", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer", type=int, default=50_000)
    p.add_argument(
        "--lr", type=non_negative_float, default=None,
        help="constant-schedule learning rate (demo default 1e-2). "
             "A smaller finite nonnegative value can be used when resuming a mature model",
    )
    p.add_argument(
        "--eval-batch", type=int, default=16,
        help="MCTS leaf inference batch size (search.eval_batch_size)",
    )
    p.add_argument(
        "--virtual-loss", type=non_negative_float, default=None,
        help="temporary virtual loss on selected edges (search.virtual_loss, default "
             f"{_DEFAULT_VIRTUAL_LOSS}). The penalty steers simultaneous simulations "
             "toward different paths. It affects batches larger than one in learner "
             "self-play, the learner side of league games, the gate and the benchmark "
             "candidate. Frozen league opponents and anchors retain their own "
             "settings. Use a finite nonnegative number, including zero. "
             "On one small 5x5 network, values in 0–0.5 reduced Gumbel schedule distortion "
             "with batches of 2–4. Measure again on larger boards. See docs/rnd_experiments.md",
    )
    p.add_argument(
        "--parallel-games", type=int, default=4,
        help="number of concurrent self-play games (training.parallel_games). Slots "
             "are limited by the number of games per worker per iteration (--self-play-games divided "
             "among --selfplay-workers). Excess slots trigger a warning. The header reports "
             "the effective count as SELFPLAY_SLOTS",
    )
    p.add_argument(
        "--selfplay-workers", type=positive_int, default=1,
        help="number of self-play worker processes (training.selfplay_workers, positive). "
             "One selects single-process mode. Additional processes split the games and "
             "each maintains a model replica and up to --parallel-games slots. "
             "Provide enough games per iteration to fill the slots. Each worker uses one CPU thread. "
             "Changing the worker count changes RNG streams. "
             "See docs/performance.md",
    )
    p.add_argument(
        "--holdout-fraction", type=non_negative_float, default=0.0,
        help="fraction of complete self-play games reserved for monitoring (training.holdout_game_fraction, "
             "in [0, 1)). Compare them with a training-buffer sample after each iteration using "
             "policy cross-entropy and value MSE in metrics.json "
             "(generalization). Zero disables the metric. An example fraction is 0.05. "
             "See docs/rnd_experiments.md",
    )
    p.add_argument(
        "--holdout-buffer", type=positive_int, default=8192,
        help="held-out buffer capacity in positions (training.holdout_buffer_size, "
             "positive). Used when --holdout-fraction > 0",
    )
    p.add_argument(
        "--benchmark-every", type=int, default=0,
        help="iterations between Elo benchmarks against anchors (zero disables)",
    )
    p.add_argument(
        "--benchmark-games", type=int, default=8,
        help="games played against each anchor per benchmark",
    )
    p.add_argument(
        "--sgf", action="store_true",
        help="save self-play games under work_dir/sgf",
    )
    p.add_argument(
        "--fpu", type=fpu_value, default=None,
        help="FPU reduction r >= 0 sets an unvisited Q to parent value minus r. "
             "Use none or omit the flag for the classic unvisited Q of zero",
    )
    p.add_argument(
        "--aux-heads", action="store_true",
        help="enable auxiliary ownership and score heads "
             "for dense territorial supervision alongside the value target",
    )
    p.add_argument(
        "--ownership-weight", type=non_negative_float, default=0.15,
        help="finite nonnegative ownership loss weight (with --aux-heads)",
    )
    p.add_argument(
        "--score-weight", type=non_negative_float, default=0.05,
        help="finite nonnegative score loss weight (with --aux-heads)",
    )
    p.add_argument(
        "--playout-cap-prob", type=float, default=0.0,
        help="probability of a full search when randomizing playout caps. "
             "Other moves use --playout-cap-cheap simulations without root noise and "
             "carry zero policy weight. Zero disables randomization",
    )
    p.add_argument(
        "--playout-cap-cheap", type=int, default=0,
        help="simulations for a cheap move with randomized playout caps",
    )
    p.add_argument(
        "--forced-playout-fraction", type=float, default=0.0,
        help="forced PUCT root visits: give each root child "
             "a quota of fraction * prior * root visits. "
             "Gumbel search uses Sequential Halving and rejects this option. "
             "Zero disables forced visits",
    )
    p.add_argument(
        "--policy-target-prune-fraction", type=float, default=0.0,
        help="policy target pruning: remove entries with mass "
             "below fraction * max(target) from the learning target, "
             "then renormalize. Move choice and visit counts are unaffected. Zero disables",
    )
    p.add_argument(
        "--search-algo", choices=["puct", "gumbel"], default="puct",
        help="learner self-play search: puct (default) or "
             "gumbel for Gumbel-Top-k and Sequential Halving with an improved-policy "
             "training target for small budgets such as 16–32 simulations "
             "(--sims). Gate, benchmark and frozen league opponents use PUCT. "
             "Gumbel is incompatible with randomized playout caps. Leaf batches of 2–4 "
             "limit schedule distortion. See docs/rnd_experiments.md",
    )
    p.add_argument(
        "--gumbel-max-considered", type=int, default=16,
        help="number of root actions considered by Sequential Halving "
             "(with --search-algo gumbel)",
    )
    p.add_argument(
        "--gumbel-scale", type=float, default=1.0,
        help="root Gumbel noise scale (1 samples from the prior, 0 disables noise)",
    )
    p.add_argument(
        "--gumbel-c-visit", type=float, default=50.0,
        help="c_visit in sigma(q) = (c_visit + max N) * c_scale * q (paper value 50)",
    )
    p.add_argument(
        "--gumbel-c-scale", type=float, default=0.1,
        help="c_scale in sigma(q) (paper value 0.1)",
    )
    p.add_argument(
        "--init-weights", type=str, default=None,
        help="initialize weights from matching checkpoint tensors. "
             "Auxiliary heads missing from the checkpoint retain their random "
             "initialization. Compatible with --warmstart-buffer and mutually exclusive with --resume",
    )
    p.add_argument(
        "--disable-resign-prob", type=float, default=None,
        help="fraction of self-play games played through without resignation (demo default 0.10). "
             "Set to 1 to disable resignation in every game and obtain terminal "
             "ownership and score labels for --aux-heads. "
             "A threshold inherited from a different network can have the wrong "
             "calibration for the new value head",
    )
    p.add_argument(
        "--aux-warmup", type=int, default=0,
        help="ramp auxiliary loss weights linearly over N optimizer steps "
             "(zero applies full weight immediately). Use during initialization of new "
             "auxiliary heads on a pretrained trunk. See docs/rnd_experiments.md",
    )
    p.add_argument(
        "--league-prob", type=float, default=0.0,
        help="fraction of games played against frozen league members "
             "to anchor the self-play data distribution (zero disables)",
    )
    p.add_argument(
        "--league-checkpoint", action="append", default=None,
        help="frozen league checkpoint (.pt), repeatable",
    )
    p.add_argument(
        "--exclude-league-from-calibration", action="store_true",
        help="exclude league outcomes from resignation threshold calibration. "
             "Calibrate on ordinary self-play only. Requires "
             "--league-prob > 0. By default both kinds of game contribute",
    )
    p.add_argument(
        "--superko", choices=["positional", "situational"], default="positional",
        help="superko rule variant",
    )
    p.add_argument(
        "--amp", action="store_true",
        help="enable bfloat16 autocast for training steps",
    )
    p.add_argument("--work-dir", type=str, default=str(PROJECT_ROOT / "runs" / "gpu_run"))
    p.add_argument("--no-save-every", action="store_true", help="disable checkpoint saves on every iteration")
    p.add_argument(
        "--checkpoint-every", type=int, default=0,
        help="iterations between candidate/best checkpoint saves (zero disables). "
             "Also works with --no-save-every",
    )
    p.add_argument(
        "--resume-every", type=int, default=1,
        help="iterations between full resume snapshots (zero disables)",
    )
    p.add_argument(
        "--max-hours", type=positive_float, default=None,
        help="positive finite run budget in hours. Stop at an iteration boundary "
             "and save final artifacts. --iterations remains an upper limit",
    )
    p.add_argument(
        "--resume", type=str, default=None,
        help="resume a full interrupted run from resume_state.pt",
    )
    p.add_argument(
        "--warmstart-buffer", type=str, default=None,
        help="initialize replay from a previous resume_state.pt while "
             "starting fresh weights at iteration one. The new --channels and"
             "--res-blocks may differ while using the existing self-play examples. "
             "Mutually exclusive with --resume",
    )
    p.add_argument(
        "--lr-schedule", choices=["constant", "paper"], default="constant",
        help="constant learning rate or the paper's step schedule "
             "(1e-2 before 400k steps, 1e-3 before 600k, then 1e-4)",
    )
    p.add_argument(
        "--wd-exclude-norm-bias", action="store_true",
        help="exclude BatchNorm parameters and biases from weight decay, applying it "
             "only to convolution and linear weights. The default decays all parameters. "
             "A full resume restores optimizer groups and requires the same setting. "
             "See docs/rnd_experiments.md",
    )
    p.add_argument(
        "--anchor-checkpoint", type=str, default=None,
        help="frozen checkpoint for the third Elo ladder anchor",
    )
    p.add_argument(
        "--anchor-rating", type=finite_float, default=0.0,
        help="externally measured finite Elo for the checkpoint anchor (see docs/strength.md). "
             "Positive and negative ratings are accepted",
    )
    return p.parse_args()


def print_table(metrics_path: Path) -> None:
    """Print one metrics.json summary row per iteration."""
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = data.get("iterations", [])
    header = (
        f"{'iter':>4} | {'step':>5} | {'total':>7} | {'policy':>7} | {'value':>7} | "
        f"{'examples':>8} | {'avg_len':>7} | {'resign':>6} | {'cand_wr':>7} | {'promoted':>8} | {'elo':>7}"
    )
    print("TABLE_BEGIN")
    print(header)
    print("-" * len(header))
    for r in rows:
        t = r["train"]
        e = r["evaluation"]
        benchmark = r.get("benchmark")
        elo = f"{benchmark['elo_estimate']:>7.1f}" if benchmark else f"{'-':>7}"
        print(
            f"{r['iteration']:>4} | {r['global_step']:>5} | "
            f"{t['total_loss']:>7.4f} | {t['policy_loss']:>7.4f} | {t['value_loss']:>7.4f} | "
            f"{r['examples_generated']:>8} | {r['avg_game_length']:>7.1f} | "
            f"{r['resignations']:>6} | {e['candidate_win_rate']:>7.3f} | {str(r['promoted']):>8} | {elo}"
        )
    print("TABLE_END")


def warn_if_virtual_loss_is_inert(requested: float | None, eval_batch_size: int) -> None:
    """Warn when a virtual loss override cannot affect strictly sequential search."""
    if requested is None or eval_batch_size > 1:
        return
    warnings.warn(
        f"--virtual-loss {requested} has no effect with --eval-batch 1 because the loss is restored "
        "before selecting another leaf. Set --eval-batch > 1 "
        "or omit the flag (see docs/mcts.md).",
        UserWarning,
        stacklevel=2,
    )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. The GPU training command requires CUDA")

    torch.manual_seed(0)

    cfg = make_demo_config(work_dir=args.work_dir)
    cfg.runtime.device = "cuda"

    cfg.game.board_size = args.board
    cfg.network.board_size = args.board
    if args.komi is not None:
        cfg.game.komi = args.komi
    cfg.network.channels = args.channels
    cfg.network.num_res_blocks = args.res_blocks
    cfg.network.value_head_hidden = args.channels

    cfg.search.num_simulations = args.sims
    cfg.search.resign_min_visits = args.resign_min_visits
    cfg.search.temperature_moves = 8
    cfg.search.eval_batch_size = args.eval_batch
    if args.virtual_loss is not None:  # Zero is a valid explicit override.
        cfg.search.virtual_loss = args.virtual_loss
    cfg.search.fpu_reduction = args.fpu
    cfg.search.playout_cap_prob = args.playout_cap_prob
    cfg.search.playout_cap_cheap_sims = args.playout_cap_cheap
    cfg.search.forced_playout_fraction = args.forced_playout_fraction
    cfg.search.policy_target_prune_fraction = args.policy_target_prune_fraction
    cfg.search.search_algo = args.search_algo
    cfg.search.gumbel_max_considered = args.gumbel_max_considered
    cfg.search.gumbel_scale = args.gumbel_scale
    cfg.search.gumbel_c_visit = args.gumbel_c_visit
    cfg.search.gumbel_c_scale = args.gumbel_c_scale
    if args.disable_resign_prob is not None:
        cfg.search.disable_resign_prob = args.disable_resign_prob

    cfg.network.aux_heads = args.aux_heads
    cfg.training.ownership_loss_weight = args.ownership_weight
    cfg.training.score_loss_weight = args.score_weight
    cfg.training.aux_warmup_steps = args.aux_warmup
    cfg.training.league_prob = args.league_prob
    cfg.training.league_checkpoints = list(args.league_checkpoint or [])
    cfg.training.exclude_league_from_resign_calibration = args.exclude_league_from_calibration

    cfg.game.superko_rule = args.superko

    cfg.training.parallel_games = args.parallel_games
    cfg.training.selfplay_workers = args.selfplay_workers
    cfg.training.holdout_game_fraction = args.holdout_fraction
    cfg.training.holdout_buffer_size = args.holdout_buffer
    cfg.training.seed = args.seed

    cfg.training.batch_size = args.batch_size
    cfg.training.gradient_steps_per_iteration = args.grad_steps
    cfg.training.self_play_games_per_iteration = args.self_play_games
    cfg.training.num_iterations = args.iterations
    cfg.training.evaluation_games = args.eval_games
    cfg.training.evaluation_gate = args.eval_gate
    cfg.training.evaluation_significance = args.eval_alpha
    cfg.training.evaluation_opening_moves = args.eval_opening_moves
    cfg.training.replay_buffer_size = args.buffer
    cfg.training.save_every_iteration = not args.no_save_every
    cfg.training.checkpoint_every = args.checkpoint_every
    cfg.training.benchmark_every = args.benchmark_every
    cfg.training.benchmark_games = args.benchmark_games
    cfg.training.benchmark_anchor_checkpoint = args.anchor_checkpoint
    cfg.training.benchmark_anchor_rating = args.anchor_rating
    cfg.training.lr_schedule = args.lr_schedule
    cfg.training.weight_decay_exclude_norm_bias = args.wd_exclude_norm_bias
    if args.lr is not None:
        cfg.training.learning_rate = args.lr

    cfg.runtime.save_selfplay_sgf = args.sgf
    cfg.runtime.use_amp = args.amp
    cfg.runtime.save_resume_every = args.resume_every
    cfg.runtime.resume_from = args.resume
    cfg.runtime.warmstart_buffer_from = args.warmstart_buffer
    cfg.runtime.init_weights_from = args.init_weights
    if args.max_hours is not None:
        cfg.runtime.time_budget_sec = args.max_hours * 3600.0

    cfg.validate()
    warn_if_virtual_loss_is_inert(args.virtual_loss, cfg.search.eval_batch_size)

    params = sum(p.numel() for p in AlphaGoZeroNet(cfg.network).parameters())
    print("DEVICE", cfg.runtime.device)
    print(
        "BOARD", cfg.game.board_size,
        "CHANNELS", cfg.network.channels,
        "RES_BLOCKS", cfg.network.num_res_blocks,
        "SIMS", cfg.search.num_simulations,
        "ITERS", cfg.training.num_iterations,
    )
    print(
        "LR_SCHEDULE", cfg.training.lr_schedule,
        "LR", cfg.training.learning_rate,
        "WD_EXCLUDE_NORM_BIAS", cfg.training.weight_decay_exclude_norm_bias,
        "RESUME", cfg.runtime.resume_from or "-",
        "ANCHOR", cfg.training.benchmark_anchor_checkpoint or "-",
        "ANCHOR_RATING", cfg.training.benchmark_anchor_rating,
    )
    print(
        "AUX_HEADS", cfg.network.aux_heads,
        "OWN_W", cfg.training.ownership_loss_weight,
        "SCORE_W", cfg.training.score_loss_weight,
        "AUX_WARMUP", cfg.training.aux_warmup_steps,
        "PCAP_PROB", cfg.search.playout_cap_prob,
        "PCAP_CHEAP", cfg.search.playout_cap_cheap_sims,
        "FORCED_PLAYOUT", cfg.search.forced_playout_fraction,
        "TARGET_PRUNE", cfg.search.policy_target_prune_fraction,
        "INIT_WEIGHTS", cfg.runtime.init_weights_from or "-",
    )
    print(
        "SEARCH_ALGO", cfg.search.search_algo,
        "GUMBEL_M", cfg.search.gumbel_max_considered,
        "GUMBEL_SCALE", cfg.search.gumbel_scale,
        "GUMBEL_C_VISIT", cfg.search.gumbel_c_visit,
        "GUMBEL_C_SCALE", cfg.search.gumbel_c_scale,
    )
    print("EVAL_BATCH", cfg.search.eval_batch_size, "VIRTUAL_LOSS", cfg.search.virtual_loss)
    print(
        "SELFPLAY_WORKERS", cfg.training.selfplay_workers,
        "PARALLEL_GAMES", cfg.training.parallel_games,
        "SELFPLAY_GAMES", cfg.training.self_play_games_per_iteration,
        # A process can use only as many slots as its share of games.
        "SELFPLAY_SLOTS", cfg.training.effective_parallel_games(),
    )
    print(
        "HOLDOUT_FRACTION", cfg.training.holdout_game_fraction,
        "HOLDOUT_BUFFER", cfg.training.holdout_buffer_size,
    )
    print(
        "EVAL_GATE", cfg.training.evaluation_gate,
        "EVAL_GAMES", cfg.training.evaluation_games,
        "EVAL_OPENING_MOVES", cfg.training.evaluation_opening_moves,
    )
    print(
        "LEAGUE_PROB", cfg.training.league_prob,
        "LEAGUE", ";".join(str(p) for p in cfg.training.league_checkpoints) or "-",
        "LEAGUE_EXCLUDED_FROM_CALIBRATION", cfg.training.exclude_league_from_resign_calibration,
    )
    print("NET_PARAMS", params)
    sys.stdout.flush()

    t0 = time.time()
    summary = run_training_pipeline(cfg)
    dt = time.time() - t0

    completed = int(summary.get("completed_iterations") or args.iterations)
    print("ELAPSED_SEC", round(dt, 1))
    print("COMPLETED_ITERS", completed)
    print("SEC_PER_ITER", round(dt / max(1, completed), 2))
    print("PEAK_GPU_MEM_MB", round(torch.cuda.max_memory_allocated() / 1e6, 1))

    print_table(Path(summary["metrics_log_path"]))

    print("SUMMARY_JSON_BEGIN")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("SUMMARY_JSON_END")


if __name__ == "__main__":
    main()
