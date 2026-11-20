from __future__ import annotations

"""Parallelize self-play over persistent worker processes, each with its own model replica."""

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import AlphaGoZeroConfig, SearchConfig
from .network import AlphaGoZeroNet
from .parallel_selfplay import ParallelSelfPlayStats, play_parallel_self_play_games
from .selfplay import ResignationController, SelfPlayResult

# Size the local observation window so the worker's threshold stays fixed for its share of games.
_RECORDING_WINDOW = 1_000_000_000

# Polling interval for checking whether the worker is still alive.
_POLL_SECONDS = 1.0

# The connection endpoint type differs between POSIX and Windows.
Channel = Any


def resolve_device(requested: str) -> torch.device:
    """Resolve the worker device using the same CUDA fallback as the pipeline."""
    name = str(requested).lower()
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def split_games(num_games: int, num_workers: int) -> list[int]:
    """Split games as evenly as possible, assigning the remainder to the first workers."""
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if num_games < 0:
        raise ValueError(f"num_games must be nonnegative, got {num_games}")
    base, extra = divmod(num_games, num_workers)
    return [base + (1 if index < extra else 0) for index in range(num_workers)]


def worker_seed(seed: int, iteration: int, worker_index: int) -> list[int]:
    """Derive each worker stream from the run seed, iteration and worker index."""
    return [int(seed), int(iteration), int(worker_index)]


def recording_controller(threshold: float, target_false_positive_rate: float) -> ResignationController:
    """Collect resignation observations while keeping the shared iteration threshold fixed."""
    return ResignationController(
        threshold=float(threshold),
        target_false_positive_rate=target_false_positive_rate,
        adjustment=0.0,
        window=_RECORDING_WINDOW,
    )


def play_worker_share(
    model: AlphaGoZeroNet,
    config: AlphaGoZeroConfig,
    *,
    threshold: float,
    seed: list[int],
    num_games: int,
    league: list[tuple[str, AlphaGoZeroNet, SearchConfig]] | None = None,
    stats: ParallelSelfPlayStats | None = None,
) -> tuple[list[SelfPlayResult], list[bool]]:
    """Play one worker's games and return results and false-resignation observations."""
    controller = recording_controller(threshold, config.search.target_false_positive_rate)
    results = play_parallel_self_play_games(
        model=model,
        config=config,
        rng=np.random.default_rng(seed),
        resignation_controller=controller,
        num_games=num_games,
        league=league,
        stats=stats,
    )
    return results, list(controller.history)


def _load_league(
    config: AlphaGoZeroConfig, device: torch.device
) -> list[tuple[str, AlphaGoZeroNet, SearchConfig]]:
    """Load frozen league networks for the worker."""
    if config.training.league_prob <= 0.0:
        return []
    # Import lazily to avoid a circular dependency with the pipeline.
    from .pipeline import _load_checkpoint_anchor

    league: list[tuple[str, AlphaGoZeroNet, SearchConfig]] = []
    for path in config.training.league_checkpoints:
        frozen_model, frozen_search = _load_checkpoint_anchor(path, device)
        league.append((Path(path).stem, frozen_model, frozen_search))
    return league


def _worker_main(connection: Channel, config: AlphaGoZeroConfig, worker_index: int) -> None:
    """Receive jobs, play the assigned games and send results or errors to the parent."""
    try:
        # Use one CPU thread per worker to avoid oversubscribing BLAS.
        torch.set_num_threads(1)
        device = resolve_device(config.runtime.device)
        model = AlphaGoZeroNet(config.network).to(device)
        model.eval()
        league = _load_league(config, device)
        connection.send(("ready",))
        while True:
            message = connection.recv()
            if message[0] == "stop":
                break
            _kind, weights, threshold, iteration, num_games = message
            model.load_state_dict({name: torch.from_numpy(array) for name, array in weights.items()})
            started = time.perf_counter()
            stats = ParallelSelfPlayStats()
            results, observations = play_worker_share(
                model,
                config,
                threshold=threshold,
                seed=worker_seed(config.training.seed, iteration, worker_index),
                num_games=num_games,
                league=league or None,
                stats=stats,
            )
            connection.send(("ok", results, observations, time.perf_counter() - started, stats))
    except (EOFError, KeyboardInterrupt):
        # Exit when the parent closes its connection.
        pass
    except BaseException:  # noqa: BLE001 - report all worker failures to the parent
        try:
            connection.send(("error", traceback.format_exc()))
        except (OSError, ValueError):
            pass
    finally:
        connection.close()


@dataclass(slots=True)
class PoolPlayResult:
    """Aggregate results from one iteration of pooled self-play."""

    results: list[SelfPlayResult]
    # Resignation calibration observations in worker order.
    observations: list[bool]
    # Game time per worker excluding communication, for comparing load.
    worker_seconds: list[float]
    # Aggregate slot usage across worker drivers.
    stats: ParallelSelfPlayStats = field(default_factory=ParallelSelfPlayStats)


class SelfPlayPool:
    """Persistent process pool for one training run."""

    def __init__(self, config: AlphaGoZeroConfig, num_workers: int | None = None) -> None:
        self.config = config
        self.num_workers = int(config.training.selfplay_workers if num_workers is None else num_workers)
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        self._connections: list[Channel] = []
        self._processes: list[BaseProcess] = []
        context = mp.get_context("spawn")
        try:
            for index in range(self.num_workers):
                parent_end, child_end = context.Pipe()
                # Daemon workers exit with their parent.
                process = context.Process(
                    target=_worker_main,
                    args=(child_end, config, index),
                    name=f"selfplay-worker-{index}",
                    daemon=True,
                )
                process.start()
                child_end.close()
                self._connections.append(parent_end)
                self._processes.append(process)
            for index in range(self.num_workers):
                self._expect(index, "ready")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> SelfPlayPool:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def play(
        self,
        model: AlphaGoZeroNet,
        threshold: float,
        iteration: int,
        num_games: int | None = None,
    ) -> PoolPlayResult:
        """Dispatch an iteration and collect results in worker order."""
        total = self.config.training.self_play_games_per_iteration if num_games is None else num_games
        shares = split_games(total, self.num_workers)
        weights = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
        busy = [index for index, share in enumerate(shares) if share > 0]
        for index in busy:
            self._connections[index].send(("play", weights, float(threshold), int(iteration), shares[index]))

        outcome = PoolPlayResult(results=[], observations=[], worker_seconds=[0.0] * self.num_workers)
        for index in busy:
            _kind, results, observations, seconds, stats = self._expect(index, "ok")
            outcome.stats.add(stats)
            outcome.results.extend(results)
            outcome.observations.extend(bool(flag) for flag in observations)
            outcome.worker_seconds[index] = float(seconds)
        return outcome

    def close(self) -> None:
        """Stop the workers. Repeated calls are safe."""
        for connection in self._connections:
            try:
                connection.send(("stop",))
            except (OSError, ValueError):
                pass
        for process in self._processes:
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for connection in self._connections:
            connection.close()
        self._connections = []
        self._processes = []

    def _expect(self, index: int, kind: str) -> tuple[Any, ...]:
        """Wait for the expected response type and report worker failures as RuntimeError."""
        connection, process = self._connections[index], self._processes[index]
        while not connection.poll(_POLL_SECONDS):
            if not process.is_alive():
                raise RuntimeError(
                    f"self-play worker {index} exited without replying (code {process.exitcode})"
                )
        try:
            message: tuple[Any, ...] = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"self-play worker {index} closed the connection without replying") from error
        if message[0] == "error":
            raise RuntimeError(f"self-play worker {index} failed:\n{message[1]}")
        if message[0] != kind:
            raise RuntimeError(f"self-play worker {index}: expected {kind!r}, received {message[0]!r}")
        return message
