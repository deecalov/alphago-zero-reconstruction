from __future__ import annotations

"""Train with self-play, replay sampling, gradient updates and candidate evaluation."""

from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import subprocess
import time
import warnings
from typing import Any

import numpy as np
import torch

from .benchmark import benchmark_model, play_anchor_series
from .config import AlphaGoZeroConfig, NetworkConfig, SearchConfig
from .checkpoint_io import atomic_output
from .elo import rating_from_win_rate
from .evaluate import evaluate_candidate
from .network import AlphaGoZeroNet
from .parallel_selfplay import ParallelSelfPlayStats, play_parallel_self_play_games
from .players import MCTSPlayer, Player, PolicyPlayer, RandomPlayer
from .replay_buffer import ReplayBuffer
from .selfplay import ResignationController, SelfPlayResult, play_self_play_game
from .selfplay_pool import SelfPlayPool
from .sgf import game_to_sgf, save_sgf
from .train import create_optimizer, evaluate_on_buffer, train_for_steps


# Independent RNG streams select holdout games and sample monitoring positions without consuming
# training randomness.
_HOLDOUT_STREAM = 1
_GENERALIZATION_STREAM = 2


def holdout_game_flags(config: AlphaGoZeroConfig, iteration: int, num_games: int) -> list[bool]:
    """Choose whole games for the holdout buffer with the configured probability."""
    fraction = config.training.holdout_game_fraction
    if fraction <= 0.0 or num_games <= 0:
        return [False] * max(0, num_games)
    lottery = np.random.default_rng([config.training.seed, iteration, _HOLDOUT_STREAM])
    return [bool(draw < fraction) for draw in lottery.random(num_games)]


def _to_jsonable(obj: Any) -> Any:
    """Convert nested run data to JSON-serializable values."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _ensure_work_dirs(work_dir: Path) -> dict[str, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = work_dir / "checkpoints"
    logs = work_dir / "logs"
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return {
        "work_dir": work_dir,
        "checkpoints": checkpoints,
        "logs": logs,
    }


def _save_json(path: Path, data: Any) -> None:
    with atomic_output(path) as f:
        f.write(json.dumps(_to_jsonable(data), indent=2, ensure_ascii=False).encode("utf-8"))


def _save_checkpoint(
    path: Path,
    model: AlphaGoZeroNet,
    optimizer: torch.optim.Optimizer | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model weights and optional optimizer state."""
    payload = {
        "model_state_dict": model.state_dict(),
        "model_summary": model.summary_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra is not None:
        payload.update(extra)
    with atomic_output(path) as f:
        torch.save(payload, f)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _environment_info(device: torch.device) -> dict[str, str]:
    """Record library versions, actual device and platform in the run artifacts."""
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device": str(device),
        "platform": platform.platform(),
    }


def _save_resume_state(
    path: Path,
    *,
    current_model: AlphaGoZeroNet,
    best_model: AlphaGoZeroNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    resignation_controller: ResignationController,
    rng: np.random.Generator,
    iteration: int,
    global_step: int,
    best_iteration: int,
    metrics_log: list[dict[str, Any]],
    config: AlphaGoZeroConfig,
    anchor_model: AlphaGoZeroNet | None = None,
    anchor_cache: dict[str, float] | None = None,
    holdout_buffer: ReplayBuffer | None = None,
) -> None:
    """Save the models, optimizer, replay, counters, calibration and RNG state required to continue the run."""
    payload: dict[str, Any] = {
        "kind": "resume_state",
        "schema_version": 2,
        "config": _to_jsonable(asdict(config)),
        "anchor_model_state_dict": anchor_model.state_dict() if anchor_model is not None else None,
        "anchor_cache": dict(anchor_cache or {}),
        "iteration": iteration,
        "global_step": global_step,
        "best_iteration": best_iteration,
        "current_model_state_dict": current_model.state_dict(),
        "best_model_state_dict": best_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_buffer": replay_buffer.state_dict(),
        # Optional buffer of held-out games.
        "holdout_buffer": holdout_buffer.state_dict() if holdout_buffer is not None else None,
        "resignation_controller": {
            "criterion_version": 2,
            "threshold": float(resignation_controller.threshold),
            "history": list(resignation_controller.history),
        },
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "python_random_state": random.getstate(),
        "metrics_log": metrics_log,
    }
    with atomic_output(path, keep_previous=True) as f:
        torch.save(payload, f)


def _load_resume_state(
    path: Path,
    *,
    device: torch.device,
    current_model: AlphaGoZeroNet,
    best_model: AlphaGoZeroNet,
    optimizer: torch.optim.Optimizer,
    replay_buffer: ReplayBuffer,
    resignation_controller: ResignationController,
    rng: np.random.Generator,
    config: AlphaGoZeroConfig,
    anchor_model: AlphaGoZeroNet | None = None,
    anchor_cache: dict[str, float] | None = None,
    holdout_buffer: ReplayBuffer | None = None,
) -> dict[str, Any]:
    """Restore full training state and return the snapshot payload and counters."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "resume_state":
        raise ValueError(f"{path} is not a resume snapshot (kind={payload.get('kind')!r})")
    version = payload.get("schema_version", 1)
    if version not in (1, 2):
        raise ValueError(f"unsupported resume schema_version: {version}")
    saved_config = payload.get("config")
    if saved_config is None and version == 1:
        sidecar = path.parent / "config.json"
        if not sidecar.exists():
            raise ValueError("legacy resume requires the original config.json beside the snapshot")
        saved_config = json.loads(sidecar.read_text(encoding="utf-8"))
        warnings.warn("Legacy resume checks compatibility using the adjacent config.json", UserWarning)
    if not isinstance(saved_config, dict):
        raise ValueError("resume snapshot has no config")
    for section in ("game", "network"):
        expected = _to_jsonable(asdict(getattr(config, section)))
        raw = saved_config.get(section, {})
        # Defaults fill fields absent in historical configs (e.g. aux_heads).
        saved = asdict(type(getattr(config, section))(**raw))
        if saved != expected:
            changed = [key for key in expected if saved.get(key) != expected[key]]
            raise ValueError(f"incompatible resume: {section}: {', '.join(changed)}")
    # Check optimizer parameter-group settings before restoring state. Supply historical defaults for
    # older snapshots.
    saved_training = saved_config.get("training", {})
    for key, historical_default in (
        ("momentum", None),
        ("weight_decay", None),
        ("weight_decay_exclude_norm_bias", False),
    ):
        if saved_training.get(key, historical_default) != getattr(config.training, key):
            raise ValueError(f"incompatible resume: training.{key}; the full optimizer state is restored")

    current_model.load_state_dict(payload["current_model_state_dict"])
    best_model.load_state_dict(payload["best_model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    replay_buffer.load_state_dict(payload["replay_buffer"])
    # Older snapshots lack holdout data. Start their monitoring buffer empty.
    saved_holdout = payload.get("holdout_buffer")
    if holdout_buffer is not None and saved_holdout is not None:
        holdout_buffer.load_state_dict(saved_holdout, keep_capacity=True)
    if anchor_model is not None and payload.get("anchor_model_state_dict") is not None:
        anchor_model.load_state_dict(payload["anchor_model_state_dict"])
    if anchor_cache is not None:
        anchor_cache.update(payload.get("anchor_cache", {}))

    controller_state = payload["resignation_controller"]
    resignation_controller.history.clear()
    saved_search = saved_config.get("search", {})
    old_min_visits = saved_search.get("resign_min_visits")
    # Old snapshots used PUCT. Changing the search algorithm or required visits changes the meaning of
    # resignation evidence.
    old_search_algo = saved_search.get("search_algo", "puct")
    if (
        controller_state.get("criterion_version") == 2
        and old_min_visits == config.search.resign_min_visits
        and old_search_algo == config.search.search_algo
    ):
        resignation_controller.threshold = float(controller_state["threshold"])
        resignation_controller.history.extend(bool(x) for x in controller_state["history"])
    else:
        warnings.warn(
            "Resignation criterion changed (resign_min_visits or search_algo). "
            "Resetting calibration to the configured threshold",
            UserWarning,
        )
        resignation_controller.threshold = config.search.resign_threshold

    # Restore every RNG stream so the next iteration follows the uninterrupted run.
    rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_states = payload.get("torch_cuda_rng_state")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])
    random.setstate(payload["python_random_state"])
    return payload


def _load_buffer_only(
    path: Path,
    *,
    device: torch.device,
    replay_buffer: ReplayBuffer,
    expected_board_size: int,
) -> int:
    """Initialize replay from a previous snapshot while starting with a fresh model and optimizer."""
    payload = torch.load(path, map_location=device, weights_only=False)
    buffer_state = payload.get("replay_buffer") if isinstance(payload, dict) else None
    if not isinstance(buffer_state, dict):
        raise ValueError(
            f"{path} has no replay_buffer for a warm start"
        )
    saved_board = int(buffer_state.get("board_size", -1))
    if saved_board != expected_board_size:
        raise ValueError(
            f"buffer board_size ({saved_board}) differs from configured size "
            f"({expected_board_size}). Replay warm start is unavailable"
        )
    replay_buffer.load_state_dict(buffer_state, keep_capacity=True)
    return len(replay_buffer)


def _gpu_stats(device: torch.device) -> dict[str, Any] | None:
    """Collect torch allocator statistics and optional device temperature and utilization for metrics.json."""
    if device.type != "cuda":
        return None
    stats: dict[str, Any] = {
        "torch_allocated_mb": round(torch.cuda.memory_allocated(device) / 1e6, 1),
        "torch_reserved_mb": round(torch.cuda.memory_reserved(device) / 1e6, 1),
        "torch_max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
    }
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        lines = query.stdout.strip().splitlines()
        if query.returncode == 0 and lines:
            index = device.index if device.index is not None else 0
            row = lines[index if index < len(lines) else 0]
            used, total, temperature, utilization = (s.strip() for s in row.split(","))
            stats.update(
                {
                    "smi_memory_used_mb": float(used),
                    "smi_memory_total_mb": float(total),
                    "temperature_c": float(temperature),
                    "utilization_pct": float(utilization),
                }
            )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        # Training can still record allocator statistics when nvidia-smi is unavailable.
        pass
    return stats


def _init_weights_partial(path: Path, model: AlphaGoZeroNet) -> dict[str, int]:
    """Copy tensors matching by name and shape. Leave unmatched parameters at their initial values."""
    payload = torch.load(path, map_location=model.device, weights_only=False)
    source: dict[str, torch.Tensor] = payload["model_state_dict"]
    target = model.state_dict()

    matched = {
        name: tensor
        for name, tensor in source.items()
        if name in target and target[name].shape == tensor.shape
    }
    model.load_state_dict(matched, strict=False)
    return {
        "loaded_tensors": len(matched),
        "model_tensors": len(target),
        "checkpoint_tensors": len(source),
    }


def _load_checkpoint_anchor(
    path: Path | str,
    device: torch.device,
) -> tuple[AlphaGoZeroNet, SearchConfig]:
    """Load a frozen Elo anchor with its own model and search configuration."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    raw = payload["config"]
    model = AlphaGoZeroNet(NetworkConfig(**raw["network"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, SearchConfig(**raw["search"])


def _run_benchmark(
    *,
    best_model: AlphaGoZeroNet,
    anchor_model: AlphaGoZeroNet,
    config: AlphaGoZeroConfig,
    iteration: int,
    anchor_cache: dict[str, float],
    checkpoint_anchor: tuple[AlphaGoZeroNet, SearchConfig] | None = None,
) -> dict[str, Any]:
    """Benchmark the best model against fixed opponents with an RNG derived separately from the training stream."""
    bench_rng = np.random.default_rng(config.training.seed * 1_000_003 + iteration)
    games = config.training.benchmark_games

    random_player = RandomPlayer(rng=bench_rng)
    policy_player = PolicyPlayer(model=anchor_model)

    if "policy_anchor_rating" not in anchor_cache:
        wins, losses, draws = play_anchor_series(policy_player, random_player, config, games)
        anchor_cache["policy_anchor_rating"] = rating_from_win_rate(
            (wins + 0.5 * draws) / max(1, wins + losses + draws), opponent_rating=0.0
        )

    anchors: dict[str, tuple[Player, float]] = {
        "random": (random_player, 0.0),
        "raw_policy_init": (policy_player, anchor_cache["policy_anchor_rating"]),
    }

    # A rated checkpoint extends the ladder when the weaker anchors lose nearly every game.
    if checkpoint_anchor is not None:
        frozen_model, frozen_search = checkpoint_anchor
        anchors["frozen_checkpoint"] = (
            MCTSPlayer(model=frozen_model, search_config=frozen_search, rng=bench_rng),
            config.training.benchmark_anchor_rating,
        )

    result = benchmark_model(
        model=best_model,
        config=config,
        rng=bench_rng,
        anchors=anchors,
        games_per_anchor=games,
    )
    result["policy_anchor_rating"] = round(anchor_cache["policy_anchor_rating"], 1)
    return result


def _save_selfplay_sgf(
    sgf_dir: Path,
    results: list[SelfPlayResult],
    config: AlphaGoZeroConfig,
    iteration: int,
) -> None:
    """Write each self-play game from this iteration to SGF for inspection."""
    for game_idx, result in enumerate(results):
        sgf_text = game_to_sgf(
            board_size=config.game.board_size,
            komi=config.game.komi,
            moves=result.moves,
            winner=result.winner,
            # Write a drawn SGF with the explicit result RE[0].
            result="0" if result.winner is None else None,
            resigned=result.resigned,
            score_margin=result.score_margin,
            black_name=f"selfplay-iter{iteration:03d}",
            white_name=f"selfplay-iter{iteration:03d}",
            comment=f"iteration={iteration} game={game_idx} moves={result.num_moves}",
        )
        save_sgf(sgf_dir / f"game_{game_idx:03d}.sgf", sgf_text)


def load_checkpoint_into_model(
    path: Path | str,
    model: AlphaGoZeroNet,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load weights into a matching model and return checkpoint metadata. Use trusted checkpoints."""
    payload = torch.load(path, map_location=model.device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    return payload


def run_training_pipeline(config: AlphaGoZeroConfig) -> dict[str, Any]:
    """Run the training loop and write artifacts to work_dir. Return a summary and release any worker pool
    on exit.
    """
    config.validate()
    pool = SelfPlayPool(config) if config.training.selfplay_workers > 1 else None
    try:
        return _run_training_pipeline(config, pool)
    finally:
        if pool is not None:
            pool.close()


def _run_training_pipeline(config: AlphaGoZeroConfig, pool: SelfPlayPool | None) -> dict[str, Any]:
    """Execute a run with the optional self-play worker pool."""
    config.validate()
    pipeline_started = time.perf_counter()
    _set_global_seed(config.training.seed)
    rng = np.random.default_rng(config.training.seed)

    work = _ensure_work_dirs(Path(config.runtime.work_dir))

    # Resolve the requested torch device.
    #
    # Recognize indexed CUDA devices and fall back to CPU when CUDA is unavailable.
    requested_device = str(config.runtime.device).lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        actual_device = torch.device("cpu")
    else:
        try:
            actual_device = torch.device(requested_device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"invalid runtime.device=\'{config.runtime.device}': {exc}"
            ) from exc

    # A single CPU thread reduces overhead on small boards and networks.
    if actual_device.type == "cpu":
        torch.set_num_threads(1)

    # Record the run environment in JSON and in each checkpoint.
    environment = _environment_info(actual_device)
    run_metadata: dict[str, Any] = {
        "environment": environment,
        "seed": config.training.seed,
        "config": _to_jsonable(asdict(config)),
    }

    # Optimize current_model during SGD.
    current_model = AlphaGoZeroNet(config.network).to(actual_device)

    # Apply a weight warm start before copying the best and anchor networks.
    init_weights_info: dict[str, int] | None = None
    if config.runtime.init_weights_from is not None:
        init_weights_info = _init_weights_partial(
            Path(config.runtime.init_weights_from), current_model
        )

    # The best model generates games and defends against promotion candidates.
    best_model = AlphaGoZeroNet(config.network).to(actual_device)
    best_model.load_state_dict(current_model.state_dict())

    # Freeze initial weights as a raw-policy benchmark anchor before restoring any full snapshot.
    anchor_model: AlphaGoZeroNet | None = None
    anchor_cache: dict[str, float] = {}
    checkpoint_anchor: tuple[AlphaGoZeroNet, SearchConfig] | None = None
    if config.training.benchmark_every > 0:
        anchor_model = AlphaGoZeroNet(config.network).to(actual_device)
        anchor_model.load_state_dict(current_model.state_dict())
        anchor_model.eval()
        if config.training.benchmark_anchor_checkpoint is not None:
            # Load the optional anchor before starting the expensive loop.
            checkpoint_anchor = _load_checkpoint_anchor(
                config.training.benchmark_anchor_checkpoint, actual_device
            )

    optimizer = create_optimizer(current_model, config)
    replay_buffer = ReplayBuffer(
        capacity=config.training.replay_buffer_size,
        board_size=config.game.board_size,
        rng=rng,
    )

    # Keep whole held-out games in a separate buffer inaccessible to training.
    holdout_buffer: ReplayBuffer | None = None
    if config.training.holdout_game_fraction > 0.0:
        holdout_buffer = ReplayBuffer(
            capacity=config.training.holdout_buffer_size,
            board_size=config.game.board_size,
            rng=rng,
        )

    # Load frozen league opponents once and leave them in evaluation mode.
    league: list[tuple[str, AlphaGoZeroNet, SearchConfig]] = []
    if config.training.league_prob > 0.0:
        for league_path in config.training.league_checkpoints:
            frozen_model, frozen_search = _load_checkpoint_anchor(league_path, actual_device)
            league.append((Path(league_path).stem, frozen_model, frozen_search))

    resignation_controller = ResignationController(
        threshold=config.search.resign_threshold,
        target_false_positive_rate=config.search.target_false_positive_rate,
        adjustment=config.search.resignation_adjustment,
        window=config.search.resignation_window,
    )

    global_step = 0
    best_iteration = 0
    start_iteration = 1
    metrics_log: list[dict[str, Any]] = []

    # Restore training state and continue at the next iteration.
    if config.runtime.resume_from is not None:
        resume_payload = _load_resume_state(
            Path(config.runtime.resume_from),
            device=actual_device,
            current_model=current_model,
            best_model=best_model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            resignation_controller=resignation_controller,
            rng=rng,
            config=config,
            anchor_model=anchor_model,
            anchor_cache=anchor_cache,
            holdout_buffer=holdout_buffer,
        )
        global_step = int(resume_payload["global_step"])
        best_iteration = int(resume_payload["best_iteration"])
        start_iteration = int(resume_payload["iteration"]) + 1
        metrics_log = list(resume_payload["metrics_log"])

    # Preserve the old sidecar when an incompatible resume fails validation.
    _save_json(work["work_dir"] / "config.json", asdict(config))
    # Seed the new run's replay buffer while allowing a different model architecture.
    warmstart_examples = 0
    if config.runtime.warmstart_buffer_from is not None:
        warmstart_examples = _load_buffer_only(
            Path(config.runtime.warmstart_buffer_from),
            device=actual_device,
            replay_buffer=replay_buffer,
            expected_board_size=config.game.board_size,
        )

    stopped_by_time_budget = False

    for iteration in range(start_iteration, config.training.num_iterations + 1):
        # Check elapsed time at iteration boundaries so the pipeline can finalize complete artifacts.
        if (
            config.runtime.time_budget_sec is not None
            and iteration > start_iteration
            and time.perf_counter() - pipeline_started >= config.runtime.time_budget_sec
        ):
            stopped_by_time_budget = True
            break

        # --------------------------------------------------------------
        # Generate games with the current best model.
        # --------------------------------------------------------------
        selfplay_started = time.perf_counter()
        self_play_results: list[SelfPlayResult] = []
        total_examples_generated = 0
        selfplay_worker_sec: list[float] = []
        # Parallel slot statistics are available only with the parallel driver.
        selfplay_stats: ParallelSelfPlayStats | None = None

        if pool is not None:
            # Split games across worker processes with independent replicas of the best model.
            pool_outcome = pool.play(best_model, resignation_controller.threshold, iteration)
            self_play_results = pool_outcome.results
            selfplay_worker_sec = pool_outcome.worker_seconds
            selfplay_stats = pool_outcome.stats
            for observed_false_positive in pool_outcome.observations:
                resignation_controller.observe_false_positive(observed_false_positive)
        elif config.training.parallel_games > 1:
            # Batch leaves from concurrent games for joint inference.
            selfplay_stats = ParallelSelfPlayStats()
            self_play_results = play_parallel_self_play_games(
                model=best_model,
                config=config,
                rng=rng,
                resignation_controller=resignation_controller,
                num_games=config.training.self_play_games_per_iteration,
                league=league or None,
                stats=selfplay_stats,
            )
        else:
            for _ in range(config.training.self_play_games_per_iteration):
                result = play_self_play_game(
                    model=best_model,
                    config=config,
                    rng=rng,
                    resignation_controller=resignation_controller,
                )
                self_play_results.append(result)

        # Route completed games to replay or holdout. Preserve completion order within each buffer.
        holdout_flags = holdout_game_flags(config, iteration, len(self_play_results))
        holdout_games = 0
        holdout_examples = 0
        for result, held_out in zip(self_play_results, holdout_flags):
            total_examples_generated += len(result.examples)
            if held_out and holdout_buffer is not None:
                holdout_buffer.add_game(result.examples)
                holdout_games += 1
                holdout_examples += len(result.examples)
            else:
                replay_buffer.add_game(result.examples)

        selfplay_sec = time.perf_counter() - selfplay_started
        total_selfplay_moves = int(sum(r.num_moves for r in self_play_results))

        if config.runtime.save_selfplay_sgf:
            _save_selfplay_sgf(
                work["work_dir"] / "sgf" / f"iter_{iteration:03d}",
                self_play_results,
                config,
                iteration,
            )

        avg_game_length = (
            float(np.mean([r.num_moves for r in self_play_results]))
            if self_play_results
            else 0.0
        )
        resign_count = int(sum(r.resigned for r in self_play_results))
        no_resign_games = int(sum(r.resignation_disabled for r in self_play_results))
        false_positive_count = int(sum(r.false_positive_resignation for r in self_play_results))

        # Summarize league results from the learner's perspective.
        league_games = [r for r in self_play_results if r.league_opponent is not None]
        league_wins = sum(
            1
            for r in league_games
            if r.league_color is not None
            and r.winner is not None
            and r.winner != r.league_color
        )

        # Record termination reasons so move-limit outcomes can be distinguished from naturally finished
        # games.
        game_endings = {
            reason: int(sum(1 for r in self_play_results if r.ended_by == reason))
            for reason in ("double_pass", "resign", "max_moves")
        }
        game_draws = int(sum(1 for r in self_play_results if r.winner is None))

        # --------------------------------------------------------------
        # Train the candidate on accumulated replay.
        # --------------------------------------------------------------
        train_started = time.perf_counter()
        train_metrics, global_step = train_for_steps(
            model=current_model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            config=config,
            rng=rng,
            global_step=global_step,
        )
        train_sec = time.perf_counter() - train_started

        # --------------------------------------------------------------
        # Evaluate the same model on training and held-out data with a dedicated sampling RNG.
        # --------------------------------------------------------------
        generalization: dict[str, Any] | None = None
        if holdout_buffer is not None:
            probe_rng = np.random.default_rng(
                [config.training.seed, iteration, _GENERALIZATION_STREAM]
            )
            positions = len(holdout_buffer)
            generalization = {
                "train": evaluate_on_buffer(current_model, replay_buffer, positions, probe_rng),
                "holdout": evaluate_on_buffer(current_model, holdout_buffer, positions, probe_rng),
                "holdout_games": holdout_games,
                "holdout_examples": holdout_examples,
                "holdout_buffer_size": len(holdout_buffer),
            }

        # --------------------------------------------------------------
        # 3) Evaluation: candidate vs best
        # --------------------------------------------------------------
        evaluation_started = time.perf_counter()
        evaluation = evaluate_candidate(
            candidate_model=current_model,
            best_model=best_model,
            config=config,
            rng=rng,
        )
        evaluation_sec = time.perf_counter() - evaluation_started
        promoted = evaluation.promoted

        if promoted:
            best_model.load_state_dict(current_model.state_dict())
            best_iteration = iteration

        # --------------------------------------------------------------
        # Benchmark against fixed Elo anchors when scheduled.
        # --------------------------------------------------------------
        benchmark_result: dict[str, Any] | None = None
        benchmark_sec = 0.0
        if (
            anchor_model is not None
            and config.training.benchmark_every > 0
            and iteration % config.training.benchmark_every == 0
        ):
            benchmark_started = time.perf_counter()
            benchmark_result = _run_benchmark(
                best_model=best_model,
                anchor_model=anchor_model,
                config=config,
                iteration=iteration,
                anchor_cache=anchor_cache,
                checkpoint_anchor=checkpoint_anchor,
            )
            benchmark_sec = time.perf_counter() - benchmark_started

        # --------------------------------------------------------------
        # Write metrics and checkpoints.
        # --------------------------------------------------------------
        iteration_log = {
            "iteration": iteration,
            "global_step": global_step,
            "device": str(actual_device),
            # Record CUDA memory, temperature and utilization when available.
            "gpu": _gpu_stats(actual_device),
            "replay_buffer_size": len(replay_buffer),
            "examples_generated": total_examples_generated,
            "avg_game_length": avg_game_length,
            "resignations": resign_count,
            "game_endings": game_endings,
            "game_draws": game_draws,
            "no_resign_games": no_resign_games,
            "false_positive_resignations": false_positive_count,
            "league_games": len(league_games),
            "league_wins": league_wins,
            "resignation_controller": resignation_controller.summary_dict(),
            # Record configured versus active slots, mean occupancy and realized batch size.
            "selfplay_slots": (
                None
                if selfplay_stats is None
                else {"parallel_games": config.training.parallel_games, **selfplay_stats.to_dict()}
            ),
            "train": {
                "total_loss": train_metrics.total_loss,
                "policy_loss": train_metrics.policy_loss,
                "value_loss": train_metrics.value_loss,
                "ownership_loss": train_metrics.ownership_loss,
                "score_loss": train_metrics.score_loss,
                "learning_rate": train_metrics.learning_rate,
                "steps": train_metrics.steps,
            },
            # Record policy cross-entropy and value MSE on each buffer when holdout is enabled.
            "generalization": generalization,
            "evaluation": {
                "candidate_wins": evaluation.candidate_wins,
                "best_wins": evaluation.best_wins,
                "draws": evaluation.draws,
                "games": evaluation.games,
                "candidate_win_rate": evaluation.candidate_win_rate,
                "stopped_early": evaluation.stopped_early,
                "games_planned": evaluation.games_planned,
                "required_wins": evaluation.required_wins,
                "gate": evaluation.gate,
                "promoted": evaluation.promoted,
            },
            "promoted": promoted,
            # Record the Elo estimate only on benchmark iterations.
            "benchmark": benchmark_result,
            # Record per-stage timings and effective data generation speed.
            "timings": {
                "selfplay_sec": round(selfplay_sec, 3),
                "train_sec": round(train_sec, 3),
                "evaluation_sec": round(evaluation_sec, 3),
                "benchmark_sec": round(benchmark_sec, 3),
                "selfplay_moves_per_sec": (
                    round(total_selfplay_moves / selfplay_sec, 2) if selfplay_sec > 0 else 0.0
                ),
                # Per-worker game times reveal uneven work distribution.
                "selfplay_worker_sec": [round(seconds, 3) for seconds in selfplay_worker_sec],
            },
        }
        metrics_log.append(iteration_log)

        # Save the environment and seed alongside iteration metrics.
        _save_json(
            work["logs"] / "metrics.json",
            {
                "environment": environment,
                "seed": config.training.seed,
                "iterations": metrics_log,
            },
        )

        save_snapshot = config.training.save_every_iteration or (
            config.training.checkpoint_every > 0
            and iteration % config.training.checkpoint_every == 0
        )
        if save_snapshot:
            _save_checkpoint(
                work["checkpoints"] / f"candidate_iter_{iteration:03d}.pt",
                model=current_model,
                optimizer=optimizer,
                extra={
                    **run_metadata,
                    "iteration": iteration,
                    "global_step": global_step,
                    "promoted": promoted,
                    "kind": "candidate",
                    # Store the calibrated threshold for use in live play.
                    "resign_threshold": float(resignation_controller.threshold),
                },
            )
            _save_checkpoint(
                work["checkpoints"] / "best.pt",
                model=best_model,
                optimizer=None,
                extra={
                    **run_metadata,
                    "iteration": iteration,
                    "global_step": global_step,
                    "best_iteration": best_iteration,
                    "kind": "best",
                    "resign_threshold": float(resignation_controller.threshold),
                },
            )

        # Replace the previous full resume snapshot atomically.
        if (
            config.runtime.save_resume_every > 0
            and iteration % config.runtime.save_resume_every == 0
        ):
            _save_resume_state(
                work["work_dir"] / "resume_state.pt",
                current_model=current_model,
                best_model=best_model,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                resignation_controller=resignation_controller,
                rng=rng,
                iteration=iteration,
                global_step=global_step,
                best_iteration=best_iteration,
                metrics_log=metrics_log,
                config=config,
                anchor_model=anchor_model,
                anchor_cache=anchor_cache,
                holdout_buffer=holdout_buffer,
            )

    # Save the final candidate and best model under predictable filenames.
    _save_checkpoint(
        work["work_dir"] / "final_candidate.pt",
        model=current_model,
        optimizer=optimizer,
        extra={
            **run_metadata,
            "global_step": global_step,
            "kind": "final_candidate",
            # Include the final calibrated resignation threshold.
            "resign_threshold": float(resignation_controller.threshold),
        },
    )
    _save_checkpoint(
        work["work_dir"] / "final_best.pt",
        model=best_model,
        optimizer=None,
        extra={
            **run_metadata,
            "global_step": global_step,
            "best_iteration": best_iteration,
            "kind": "final_best",
            "resign_threshold": float(resignation_controller.threshold),
        },
    )

    summary = {
        "work_dir": str(work["work_dir"]),
        "device": str(actual_device),
        "environment": environment,
        "seed": config.training.seed,
        "iterations": config.training.num_iterations,
        "completed_iterations": len(metrics_log),
        "stopped_by_time_budget": stopped_by_time_budget,
        "start_iteration": start_iteration,
        "resumed_from": str(config.runtime.resume_from) if config.runtime.resume_from else None,
        "warmstart_buffer_from": (
            str(config.runtime.warmstart_buffer_from)
            if config.runtime.warmstart_buffer_from
            else None
        ),
        "warmstart_buffer_examples": warmstart_examples,
        "init_weights_from": (
            str(config.runtime.init_weights_from)
            if config.runtime.init_weights_from
            else None
        ),
        "init_weights_info": init_weights_info,
        "global_step": global_step,
        "best_iteration": best_iteration,
        "replay_buffer_size": len(replay_buffer),
        "current_model": current_model.summary_dict(),
        "best_model": best_model.summary_dict(),
        "resignation_controller": resignation_controller.summary_dict(),
        "metrics_log_path": str(work["logs"] / "metrics.json"),
        "final_best_path": str(work["work_dir"] / "final_best.pt"),
        "final_candidate_path": str(work["work_dir"] / "final_candidate.pt"),
    }
    _save_json(work["work_dir"] / "summary.json", summary)
    return summary
