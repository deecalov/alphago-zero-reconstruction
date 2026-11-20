"""Ensemble prediction mathematics, compatibility and CLI tests."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig, GameConfig, NetworkConfig, SearchConfig
from alphago_zero.ensemble import EnsembleNet, require_same_rules
from alphago_zero.game import GoState
from alphago_zero.gtp import GTPEngine
from alphago_zero.mcts import AlphaGoZeroMCTS
from alphago_zero.network import AlphaGoZeroNet, PolicyValueModel
from examples import eval_match, play_gtp

ACTION_SIZE = 26  # 5x5 + pass


def _make_net(config: NetworkConfig, seed: int) -> AlphaGoZeroNet:
    """Build an evaluation-mode network with deterministic initialization."""
    torch.manual_seed(seed)
    net = AlphaGoZeroNet(config)
    net.eval()
    return net


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Reference float64 softmax over the last dimension."""
    shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _random_states(batch: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((batch, 17, 5, 5)).astype(np.float32)


@pytest.fixture
def net_a(tiny_network_config: NetworkConfig) -> AlphaGoZeroNet:
    return _make_net(tiny_network_config, seed=1)


@pytest.fixture
def net_b(tiny_network_config: NetworkConfig) -> AlphaGoZeroNet:
    return _make_net(tiny_network_config, seed=2)


# ----------------------------------------------------------------------
# Ensemble mathematics
# ----------------------------------------------------------------------
def test_policy_is_arithmetic_mean_of_member_softmaxes(
    net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet
) -> None:
    states = _random_states(batch=4)
    logits, values = EnsembleNet([net_a, net_b]).predict_batch(states)
    logits_a, values_a = net_a.predict_batch(states)
    logits_b, values_b = net_b.predict_batch(states)

    expected = (_softmax(logits_a) + _softmax(logits_b)) / 2.0
    np.testing.assert_allclose(_softmax(logits), expected, atol=1e-6)
    np.testing.assert_allclose(values, (values_a + values_b) / 2.0, atol=1e-6)
    # The mean of two different predictors should differ from both individual predictions.
    assert not np.allclose(_softmax(logits), _softmax(logits_a), atol=1e-4)
    assert not np.allclose(_softmax(logits), _softmax(logits_b), atol=1e-4)


def test_output_contract_matches_single_network(
    net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet
) -> None:
    states = _random_states(batch=3)
    logits, values = EnsembleNet([net_a, net_b]).predict_batch(states)
    assert logits.shape == (3, ACTION_SIZE) and logits.dtype == np.float32
    assert values.shape == (3,) and values.dtype == np.float32
    # The returned log probabilities represent a normalized distribution.
    np.testing.assert_allclose(np.exp(logits.astype(np.float64)).sum(axis=-1), 1.0, atol=1e-5)


def test_predict_single_matches_the_same_row_of_predict_batch(
    net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet
) -> None:
    ensemble = EnsembleNet([net_a, net_b])
    states = _random_states(batch=3)
    batch_logits, batch_values = ensemble.predict_batch(states)
    for index in range(3):
        single_logits, single_value = ensemble.predict_single(states[index])
        assert single_logits.shape == (ACTION_SIZE,) and single_logits.dtype == np.float32
        assert isinstance(single_value, float)
        np.testing.assert_allclose(single_logits, batch_logits[index], atol=1e-5)
        assert single_value == pytest.approx(float(batch_values[index]), abs=1e-5)


def test_ensemble_of_identical_members_equals_the_single_network(net_a: AlphaGoZeroNet) -> None:
    states = _random_states(batch=4)
    logits, values = EnsembleNet([net_a, net_a]).predict_batch(states)
    single_logits, single_values = net_a.predict_batch(states)
    np.testing.assert_allclose(_softmax(logits), _softmax(single_logits), atol=1e-6)
    np.testing.assert_allclose(values, single_values, atol=1e-6)


def test_member_order_does_not_matter(net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet) -> None:
    states = _random_states(batch=4)
    forward_logits, forward_values = EnsembleNet([net_a, net_b]).predict_batch(states)
    reverse_logits, reverse_values = EnsembleNet([net_b, net_a]).predict_batch(states)
    np.testing.assert_allclose(forward_logits, reverse_logits, atol=1e-6)
    np.testing.assert_allclose(forward_values, reverse_values, atol=1e-6)


def test_members_may_have_different_architectures(tiny_network_config: NetworkConfig) -> None:
    """Combine networks of different widths and depths while keeping compatible inputs and outputs."""
    small = _make_net(tiny_network_config, seed=1)
    large = _make_net(
        NetworkConfig(board_size=5, channels=12, num_res_blocks=2, value_head_hidden=16), seed=2
    )
    logits, values = EnsembleNet([small, large]).predict_batch(_random_states(batch=2))
    assert logits.shape == (2, ACTION_SIZE)
    assert np.isfinite(logits).all() and np.isfinite(values).all()
    assert np.all(np.abs(values) <= 1.0)


def test_extreme_member_logits_stay_finite(
    net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use extreme logits to exercise both max subtraction and the probability floor."""
    confident_in_zero = np.zeros((1, ACTION_SIZE), dtype=np.float32)
    confident_in_zero[0, 0] = 5000.0
    confident_in_one = np.zeros((1, ACTION_SIZE), dtype=np.float32)
    confident_in_one[0, 1] = 5000.0
    for member, forced in ((net_a, confident_in_zero), (net_b, confident_in_one)):
        monkeypatch.setattr(member, "predict_batch", lambda x, f=forced: (f, np.zeros(1, np.float32)))
    logits, _values = EnsembleNet([net_a, net_b]).predict_batch(_random_states(batch=1))
    assert np.isfinite(logits).all()
    np.testing.assert_allclose(np.exp(logits[0, :2].astype(np.float64)), [0.5, 0.5], atol=1e-6)
    assert np.all(logits[0, 2:] < -100.0)


# ----------------------------------------------------------------------
# Validation and inference interface
# ----------------------------------------------------------------------
def test_empty_ensemble_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        EnsembleNet([])


def test_different_board_size_is_rejected(net_a: AlphaGoZeroNet) -> None:
    other = _make_net(NetworkConfig(board_size=7, channels=8, num_res_blocks=1, value_head_hidden=8), seed=3)
    with pytest.raises(ValueError, match="board_size"):
        EnsembleNet([net_a, other])


def test_different_input_planes_are_rejected(net_a: AlphaGoZeroNet) -> None:
    other = _make_net(
        NetworkConfig(board_size=5, num_input_planes=9, channels=8, num_res_blocks=1, value_head_hidden=8),
        seed=3,
    )
    with pytest.raises(ValueError, match="num_input_planes"):
        EnsembleNet([net_a, other])


def test_eval_switches_every_member_to_inference_mode(
    net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet
) -> None:
    net_a.train()
    net_b.train()
    ensemble = EnsembleNet([net_a, net_b])
    assert ensemble.eval() is ensemble
    assert not net_a.training and not net_b.training


def test_ensemble_exposes_board_geometry_and_satisfies_model_protocol(
    net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet
) -> None:
    ensemble = EnsembleNet([net_a, net_b])
    assert ensemble.board_size == 5 and ensemble.action_size == ACTION_SIZE
    assert isinstance(ensemble, PolicyValueModel)
    assert isinstance(net_a, PolicyValueModel)


def test_wrong_input_shape_is_reported_by_members(net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet) -> None:
    ensemble = EnsembleNet([net_a, net_b])
    with pytest.raises(ValueError, match="expected"):
        ensemble.predict_batch(np.zeros((17, 5, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="expected"):
        ensemble.predict_single(np.zeros((1, 17, 5, 5), dtype=np.float32))


# ----------------------------------------------------------------------
# Search and GTP integration
# ----------------------------------------------------------------------
@pytest.mark.parametrize("eval_batch_size", [1, 4])
def test_mcts_runs_with_an_ensemble(
    net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet, eval_batch_size: int
) -> None:
    """Batch size one uses predict_single. Larger batches use predict_batch."""
    config = SearchConfig(num_simulations=12, eval_batch_size=eval_batch_size)
    output = AlphaGoZeroMCTS(EnsembleNet([net_a, net_b]), config, np.random.default_rng(0)).run(
        GoState.new_game(board_size=5, komi=2.5), temperature=1.0, add_root_noise=False
    )
    assert output.visit_distribution.sum() == pytest.approx(1.0, abs=1e-5)
    assert output.policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert sum(edge.visit_count for edge in output.root.edges.values()) >= 1


def test_mcts_with_identical_members_matches_the_single_network(net_a: AlphaGoZeroNet) -> None:
    config = SearchConfig(num_simulations=16, eval_batch_size=4)
    state = GoState.new_game(board_size=5, komi=2.5)
    single = AlphaGoZeroMCTS(net_a, config, np.random.default_rng(7)).run(
        state, temperature=1.0, add_root_noise=False
    )
    doubled = AlphaGoZeroMCTS(EnsembleNet([net_a, net_a]), config, np.random.default_rng(7)).run(
        state, temperature=1.0, add_root_noise=False
    )
    np.testing.assert_allclose(doubled.visit_distribution, single.visit_distribution, atol=1e-9)
    assert doubled.root_q == pytest.approx(single.root_q, abs=1e-5)


def test_gtp_engine_plays_a_move_with_an_ensemble(
    tiny_config: AlphaGoZeroConfig, net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet
) -> None:
    engine = GTPEngine(
        model=EnsembleNet([net_a, net_b]),
        config=tiny_config,
        rng=np.random.default_rng(0),
        allow_resign=False,
    )
    response, quit_flag = engine.handle_line("genmove b")
    assert response is not None and response.startswith("=") and not quit_flag


# ----------------------------------------------------------------------
# Rule compatibility across ensemble members
# ----------------------------------------------------------------------
def test_require_same_rules_accepts_equal_rules() -> None:
    require_same_rules([GameConfig(board_size=9, komi=7.5), GameConfig(board_size=9, komi=7.5)], ["a.pt", "b.pt"])


def test_require_same_rules_rejects_different_komi() -> None:
    games = [GameConfig(board_size=9, komi=7.5), GameConfig(board_size=9, komi=6.5)]
    with pytest.raises(ValueError, match="b.pt.*komi"):
        require_same_rules(games, ["a.pt", "b.pt"])


def test_require_same_rules_rejects_different_board_size() -> None:
    games = [GameConfig(board_size=9, komi=7.5), GameConfig(board_size=13, komi=7.5)]
    with pytest.raises(ValueError, match="b.pt.*board"):
        require_same_rules(games, ["a.pt", "b.pt"])


# ----------------------------------------------------------------------
# Multiple checkpoints in eval_match and play_gtp
# ----------------------------------------------------------------------
def _write_checkpoint(path: Path, config: AlphaGoZeroConfig, net: AlphaGoZeroNet) -> Path:
    """Write a pipeline checkpoint with config and model_state_dict."""
    torch.save({"kind": "best", "config": asdict(config), "model_state_dict": net.state_dict()}, path)
    return path


@pytest.fixture
def checkpoints(
    tmp_path: Path, tiny_config: AlphaGoZeroConfig, net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet
) -> tuple[Path, Path]:
    second_config = copy.deepcopy(tiny_config)
    second_config.search.c_puct = 3.0  # Distinguish which member supplied the search configuration.
    return (
        _write_checkpoint(tmp_path / "a.pt", tiny_config, net_a),
        _write_checkpoint(tmp_path / "b.pt", second_config, net_b),
    )


def test_load_model_group_single_checkpoint_returns_the_bare_network(
    checkpoints: tuple[Path, Path],
) -> None:
    model, search, game = eval_match.load_model_group([checkpoints[0]], torch.device("cpu"))
    assert isinstance(model, AlphaGoZeroNet)
    assert search.num_simulations == 6 and game.board_size == 5


def test_load_model_group_builds_ensemble_with_first_search_config(
    checkpoints: tuple[Path, Path],
) -> None:
    model, search, _game = eval_match.load_model_group(
        list(checkpoints), torch.device("cpu"), sims_override=9
    )
    assert isinstance(model, EnsembleNet) and len(model.members) == 2
    assert search.c_puct == 1.5  # a.pt setting (b.pt uses 3.0).
    assert search.num_simulations == 9  # The explicit simulations override took effect.


def test_load_model_group_rejects_members_with_different_komi(
    tmp_path: Path, tiny_config: AlphaGoZeroConfig, net_a: AlphaGoZeroNet, net_b: AlphaGoZeroNet
) -> None:
    other = copy.deepcopy(tiny_config)
    other.game.komi = 7.5
    paths = [
        _write_checkpoint(tmp_path / "a.pt", tiny_config, net_a),
        _write_checkpoint(tmp_path / "b.pt", other, net_b),
    ]
    with pytest.raises(ValueError, match="komi"):
        eval_match.load_model_group(paths, torch.device("cpu"))


def test_load_model_group_requires_at_least_one_path() -> None:
    with pytest.raises(ValueError, match="at least one"):
        eval_match.load_model_group([], torch.device("cpu"))


def test_eval_match_cli_plays_an_ensemble_and_reports_members(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    out = tmp_path / "match.json"
    a, b = checkpoints
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_match.py", "--model", str(a), str(b), "--opponent", str(a),
            "--games", "2", "--simulations", "4", "--opening-moves", "2",
            "--device", "cpu", "--seed", "3", "--json-out", str(out),
        ],
    )
    eval_match.main()
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["games"] == 2 and result["wins"] + result["losses"] == 2
    assert result["model_members"] == [str(a), str(b)]
    assert result["opponent_members"] == [str(a)]
    assert result["model"] == f"{a} + {b}"
    assert result["opponent"] == str(a)  # Preserve the single-path field for an ordinary opponent.
    assert result["model_sims"] == 4 and result["opponent_sims"] == 4


def _launch_play_gtp(monkeypatch: pytest.MonkeyPatch, *cli_args: str) -> GTPEngine:
    """Run play_gtp.main without reading stdin and return the constructed engine."""
    captured: dict[str, GTPEngine] = {}
    monkeypatch.setattr(GTPEngine, "run", lambda self, *args, **kwargs: captured.setdefault("engine", self))
    monkeypatch.setattr(sys, "argv", ["play_gtp.py", *cli_args, "--device", "cpu"])
    play_gtp.main()
    return captured["engine"]


def test_play_gtp_cli_builds_an_ensemble_from_several_checkpoints(
    monkeypatch: pytest.MonkeyPatch, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    engine = _launch_play_gtp(monkeypatch, "--checkpoint", str(a), str(b), "--no-resign")
    assert isinstance(engine.model, EnsembleNet) and len(engine.model.members) == 2
    response, _quit = engine.handle_line("genmove b")
    assert response is not None and response.startswith("=")


def test_play_gtp_cli_with_one_checkpoint_keeps_the_plain_network(
    monkeypatch: pytest.MonkeyPatch, checkpoints: tuple[Path, Path]
) -> None:
    engine = _launch_play_gtp(monkeypatch, "--checkpoint", str(checkpoints[0]))
    assert isinstance(engine.model, AlphaGoZeroNet)


def test_play_gtp_cli_takes_the_search_config_from_the_first_checkpoint(
    monkeypatch: pytest.MonkeyPatch, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints  # a.pt uses c_puct 1.5 and b.pt uses 3.0.
    first_a = _launch_play_gtp(monkeypatch, "--checkpoint", str(a), str(b))
    first_b = _launch_play_gtp(monkeypatch, "--checkpoint", str(b), str(a))
    assert first_a.config.search.c_puct == 1.5
    assert first_b.config.search.c_puct == 3.0  # The first member follows command-line order.


def test_play_gtp_cli_applies_the_simulations_override_to_an_ensemble(
    monkeypatch: pytest.MonkeyPatch, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    engine = _launch_play_gtp(monkeypatch, "--checkpoint", str(a), str(b), "--simulations", "3")
    assert engine.config.search.num_simulations == 3  # The saved checkpoints use six simulations.


def test_play_gtp_cli_rejects_checkpoints_with_different_komi(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_config: AlphaGoZeroConfig,
    net_a: AlphaGoZeroNet,
    net_b: AlphaGoZeroNet,
) -> None:
    other = copy.deepcopy(tiny_config)
    other.game.komi = 7.5
    a = _write_checkpoint(tmp_path / "a.pt", tiny_config, net_a)
    b = _write_checkpoint(tmp_path / "b.pt", other, net_b)
    with pytest.raises(ValueError, match="komi"):
        _launch_play_gtp(monkeypatch, "--checkpoint", str(a), str(b))
