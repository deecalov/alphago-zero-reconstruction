from __future__ import annotations

"""Reusable fixtures for the project test suite."""

from dataclasses import asdict
from pathlib import Path
import copy
import os

import numpy as np
import pytest
import torch

from alphago_zero.config import (
    AlphaGoZeroConfig,
    GameConfig,
    NetworkConfig,
    RuntimeConfig,
    SearchConfig,
    TrainingConfig,
)
from alphago_zero.game import GoState
from alphago_zero.network import AlphaGoZeroNet


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--require-cuda", action="store_true", help="Fail instead of silently skipping GPU validation")
    parser.addoption(
        "--require-gnugo",
        action="store_true",
        help="Fail instead of silently skipping tests that need a real GNU Go (PATH or AGZ_GNUGO_COMMAND)",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    if session.config.getoption("--require-cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise pytest.UsageError("--require-cuda: CUDA is unavailable")


@pytest.fixture(autouse=True)
def _single_thread() -> None:
    """Use one CPU thread for predictable execution of small-network tests."""
    torch.set_num_threads(1)


@pytest.fixture
def seed() -> int:
    """Shared deterministic test seed."""
    return 1234


@pytest.fixture
def rng(seed: int) -> np.random.Generator:
    """Seeded NumPy generator."""
    return np.random.default_rng(seed)


@pytest.fixture
def identity_symmetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the identity symmetry to compare search paths in a fixed orientation."""
    monkeypatch.setattr("alphago_zero.mcts.random_dihedral_index", lambda rng: 0)


@pytest.fixture
def tiny_network_config() -> NetworkConfig:
    """Construct a minimal network configuration for fast tests."""
    return NetworkConfig(
        board_size=5,
        num_input_planes=17,
        channels=8,
        num_res_blocks=1,
        value_head_hidden=8,
    )


@pytest.fixture
def tiny_model(tiny_network_config: NetworkConfig) -> AlphaGoZeroNet:
    """Initialize a tiny network with a fixed torch seed."""
    torch.manual_seed(0)
    return AlphaGoZeroNet(tiny_network_config)


@pytest.fixture
def tiny_config(tmp_path: Path) -> AlphaGoZeroConfig:
    """Create a small end-to-end configuration that writes into pytest's temporary directory."""
    cfg = AlphaGoZeroConfig(
        game=GameConfig(board_size=5, komi=2.5, max_moves_multiplier=2),
        network=NetworkConfig(
            board_size=5,
            channels=8,
            num_res_blocks=1,
            value_head_hidden=8,
        ),
        search=SearchConfig(
            num_simulations=6,
            c_puct=1.5,
            root_dirichlet_alpha=0.3,
            root_dirichlet_epsilon=0.25,
            temperature_moves=3,
            deterministic_temperature=1e-6,
            reuse_tree=True,
            resign_threshold=-0.95,
            disable_resign_prob=0.1,
        ),
        training=TrainingConfig(
            replay_buffer_size=256,
            batch_size=8,
            gradient_steps_per_iteration=2,
            self_play_games_per_iteration=1,
            num_iterations=1,
            learning_rate=1e-2,
            momentum=0.9,
            weight_decay=1e-4,
            lr_schedule="constant",
            evaluation_games=2,
            evaluation_gate="fixed",
            replace_threshold=0.55,
            seed=7,
            save_every_iteration=True,
        ),
        runtime=RuntimeConfig(device="cpu", work_dir=tmp_path / "runs"),
    )
    cfg.validate()
    return cfg


@pytest.fixture
def new_state() -> GoState:
    """Empty 5x5 board with demo rules."""
    return GoState.new_game(board_size=5, komi=2.5)


def _write_checkpoint(path: Path, config: AlphaGoZeroConfig, net: AlphaGoZeroNet) -> Path:
    """Write a pipeline checkpoint with config and model_state_dict."""
    torch.save({"kind": "best", "config": asdict(config), "model_state_dict": net.state_dict()}, path)
    return path


@pytest.fixture
def checkpoints(
    tmp_path: Path, tiny_config: AlphaGoZeroConfig, tiny_network_config: NetworkConfig
) -> tuple[Path, Path]:
    """Create two checkpoints with different saved search settings for command-line match tests."""
    other = copy.deepcopy(tiny_config)
    other.search.num_simulations = 10
    other.search.c_puct = 1.25
    other.search.fpu_reduction = 0.2
    torch.manual_seed(1)
    net_a = AlphaGoZeroNet(tiny_network_config)
    torch.manual_seed(2)
    net_b = AlphaGoZeroNet(tiny_network_config)
    return (
        _write_checkpoint(tmp_path / "a.pt", tiny_config, net_a),
        _write_checkpoint(tmp_path / "b.pt", other, net_b),
    )
