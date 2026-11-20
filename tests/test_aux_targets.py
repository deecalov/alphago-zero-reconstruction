from __future__ import annotations

"""Auxiliary ownership and score targets, randomized playout caps and their training masks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig, NetworkConfig
from alphago_zero.game import GoState
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.pipeline import _init_weights_partial
from alphago_zero.replay_buffer import ReplayBuffer, TrainingExample
from alphago_zero.selfplay import (
    ResignationController,
    _aux_targets_from_final_state,
    _draw_move_budget,
    play_self_play_game,
)
from alphago_zero.train import create_optimizer, train_for_steps


def _make_controller(config: AlphaGoZeroConfig) -> ResignationController:
    return ResignationController(
        threshold=config.search.resign_threshold,
        target_false_positive_rate=config.search.target_false_positive_rate,
        adjustment=config.search.resignation_adjustment,
        window=config.search.resignation_window,
    )


# ---------------------------------------------------------------------------
# game.ownership_map
# ---------------------------------------------------------------------------
def test_ownership_map_semantics() -> None:
    """Check stones, single-color regions and neutral regions."""
    state = GoState.new_game(board_size=5, komi=2.5)
    # Black and White walls border their own territory on opposite sides with a neutral strip between.
    for row in range(5):
        state = state.play(row * 5 + 1)  # BLACK
        state = state.play(row * 5 + 3)  # WHITE

    ownership = state.ownership_map()
    assert ownership.shape == (5, 5)
    assert set(np.unique(ownership)) <= {-1, 0, 1}
    for row in range(5):
        assert ownership[row, 0] == 1  # Black territory.
        assert ownership[row, 1] == 1  # Black stone.
        assert ownership[row, 2] == 0  # Neutral region.
        assert ownership[row, 3] == -1  # White stone.
        assert ownership[row, 4] == -1  # White territory.


def test_ownership_map_consistent_with_area_score() -> None:
    """Check that ownership totals match area scores before adding komi."""
    rng = np.random.default_rng(3)
    state = GoState.new_game(board_size=5, komi=2.5)
    # Play a random legal game to its terminal position.
    while not state.is_terminal():
        state = state.play(int(rng.choice(state.legal_actions())))

    ownership = state.ownership_map()
    black_score, white_score = state.area_score()
    assert float(np.sum(ownership == 1)) == pytest.approx(black_score)
    assert float(np.sum(ownership == -1)) == pytest.approx(white_score - state.komi)


def test_aux_targets_perspective() -> None:
    """White ownership and score targets reverse the Black perspective."""
    state = GoState.new_game(board_size=5, komi=2.5)
    for row in range(5):
        state = state.play(row * 5 + 1)
        state = state.play(row * 5 + 3)
    ownership_black, diff_black = _aux_targets_from_final_state(state)

    # Ten points each and komi 2.5 give (10 - 12.5) / 25 = -0.1.
    assert diff_black == pytest.approx(-0.1)
    sign_white = -1.0
    np.testing.assert_allclose(ownership_black * sign_white, -ownership_black)


# ---------------------------------------------------------------------------
# Replay auxiliary fields, augmentation and serialization
# ---------------------------------------------------------------------------
def _example_with_aux(board_size: int = 5, policy_weight: float = 1.0) -> TrainingExample:
    rng = np.random.default_rng(11)
    state = rng.random((17, board_size, board_size)).astype(np.float32)
    policy = np.full(board_size * board_size + 1, 1.0 / (board_size * board_size + 1), np.float32)
    ownership = rng.choice([-1.0, 0.0, 1.0], size=(board_size, board_size)).astype(np.float32)
    return TrainingExample(
        state=state,
        policy=policy,
        value=1.0,
        ownership=ownership,
        score=0.25,
        policy_weight=policy_weight,
    )


def test_sample_batch_full_shapes_and_masks() -> None:
    buf = ReplayBuffer(capacity=16, board_size=5, rng=np.random.default_rng(0))
    buf.add(_example_with_aux(policy_weight=1.0))
    # Example from a resignation with no auxiliary target and a cheap move with zero policy weight.
    legacy = _example_with_aux(policy_weight=0.0)
    legacy.ownership = None
    legacy.score = 0.0
    buf.add(legacy)

    batch = buf.sample_batch_full(batch_size=8, augment=True)
    assert batch.states.shape == (8, 17, 5, 5)
    assert batch.policies.shape == (8, 26)
    assert batch.ownerships.shape == (8, 25)
    assert batch.scores.shape == (8,)
    assert set(np.unique(batch.aux_mask)) <= {0.0, 1.0}
    assert set(np.unique(batch.policy_weights)) <= {0.0, 1.0}
    # Mask out missing ownership labels.
    for i in range(8):
        if batch.aux_mask[i] == 0.0:
            np.testing.assert_allclose(batch.ownerships[i], 0.0)


def test_sample_batch_full_dihedral_coherence() -> None:
    """Apply the same symmetry to state and ownership, using identical patterns to detect mismatches."""
    buf = ReplayBuffer(capacity=4, board_size=5, rng=np.random.default_rng(5))
    example = _example_with_aux()
    example.state[0] = example.ownership
    buf.add(example)

    batch = buf.sample_batch_full(batch_size=16, augment=True)
    for i in range(16):
        np.testing.assert_allclose(
            batch.states[i, 0].reshape(-1), batch.ownerships[i], atol=1e-6
        )


def test_buffer_state_dict_roundtrip_with_aux() -> None:
    src = ReplayBuffer(capacity=8, board_size=5, rng=np.random.default_rng(1))
    src.add(_example_with_aux(policy_weight=0.0))
    no_aux = _example_with_aux(policy_weight=1.0)
    no_aux.ownership = None
    src.add(no_aux)

    dst = ReplayBuffer(capacity=8, board_size=5, rng=np.random.default_rng(2))
    dst.load_state_dict(src.state_dict())

    assert len(dst) == 2
    first, second = dst._buffer[0], dst._buffer[1]
    assert first.ownership is not None
    np.testing.assert_allclose(first.ownership, src._buffer[0].ownership)
    assert first.score == pytest.approx(0.25)
    assert first.policy_weight == pytest.approx(0.0)
    assert second.ownership is None
    assert second.policy_weight == pytest.approx(1.0)


def test_buffer_loads_legacy_snapshot_without_aux_keys() -> None:
    """Restore old replay snapshots that predate auxiliary fields."""
    src = ReplayBuffer(capacity=8, board_size=5, rng=np.random.default_rng(1))
    src.add(_example_with_aux())
    payload = src.state_dict()
    for key in ("ownerships", "aux_mask", "scores", "policy_weights"):
        payload.pop(key)

    dst = ReplayBuffer(capacity=8, board_size=5, rng=np.random.default_rng(2))
    dst.load_state_dict(payload)
    assert len(dst) == 1
    assert dst._buffer[0].ownership is None
    assert dst._buffer[0].policy_weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Network heads and partial weight initialization
# ---------------------------------------------------------------------------
def test_forward_full_shapes_with_aux_heads() -> None:
    torch.manual_seed(0)
    net = AlphaGoZeroNet(
        NetworkConfig(board_size=5, channels=8, num_res_blocks=1, value_head_hidden=8, aux_heads=True)
    )
    x = torch.randn(3, 17, 5, 5)
    outputs = net.forward_full(x)
    assert outputs["policy_logits"].shape == (3, 26)
    assert outputs["value"].shape == (3, 1)
    assert outputs["ownership"].shape == (3, 25)
    assert outputs["score"].shape == (3, 1)
    assert torch.all(outputs["ownership"].abs() <= 1.0)
    assert torch.all(outputs["score"].abs() <= 1.0)
    # The ordinary forward path used by MCTS is still available.
    logits, value = net(x)
    assert logits.shape == (3, 26)
    assert value.shape == (3, 1)


def test_classic_network_has_no_aux_keys() -> None:
    torch.manual_seed(0)
    net = AlphaGoZeroNet(NetworkConfig(board_size=5, channels=8, num_res_blocks=1, value_head_hidden=8))
    assert net.ownership_head is None
    assert all(not k.startswith(("ownership_head", "score_head")) for k in net.state_dict())
    outputs = net.forward_full(torch.randn(1, 17, 5, 5))
    assert "ownership" not in outputs


def test_init_weights_partial_transfers_trunk(tmp_path: Path) -> None:
    """Copy matching tensors and leave new auxiliary heads at initialization."""
    torch.manual_seed(0)
    classic = AlphaGoZeroNet(
        NetworkConfig(board_size=5, channels=8, num_res_blocks=1, value_head_hidden=8)
    )
    ckpt = tmp_path / "classic.pt"
    torch.save({"model_state_dict": classic.state_dict()}, ckpt)

    torch.manual_seed(99)
    aux_net = AlphaGoZeroNet(
        NetworkConfig(board_size=5, channels=8, num_res_blocks=1, value_head_hidden=8, aux_heads=True)
    )
    info = _init_weights_partial(ckpt, aux_net)

    assert info["loaded_tensors"] == len(classic.state_dict())
    assert info["model_tensors"] > info["loaded_tensors"]  # Auxiliary parameters are absent from this checkpoint.
    torch.testing.assert_close(
        aux_net.state_dict()["policy_head.fc.weight"],
        classic.state_dict()["policy_head.fc.weight"],
    )


# ---------------------------------------------------------------------------
# Training with auxiliary losses
# ---------------------------------------------------------------------------
def test_train_for_steps_with_aux_heads(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.network.aux_heads = True
    torch.manual_seed(0)
    model = AlphaGoZeroNet(tiny_config.network)
    optimizer = create_optimizer(model, tiny_config)

    buf = ReplayBuffer(capacity=64, board_size=5, rng=np.random.default_rng(0))
    for i in range(16):
        ex = _example_with_aux(policy_weight=float(i % 2))
        if i % 4 == 0:
            ex.ownership = None  # Some examples lack auxiliary targets.
        buf.add(ex)

    metrics, step = train_for_steps(
        model=model,
        optimizer=optimizer,
        replay_buffer=buf,
        config=tiny_config,
        rng=np.random.default_rng(1),
        global_step=0,
    )
    assert step == tiny_config.training.gradient_steps_per_iteration
    assert np.isfinite(metrics.total_loss)
    assert metrics.ownership_loss > 0.0
    assert metrics.score_loss >= 0.0
    # The total includes weighted auxiliary losses.
    assert metrics.total_loss >= metrics.policy_loss + metrics.value_loss - 1e-6


def test_train_for_steps_classic_reports_zero_aux(tiny_config: AlphaGoZeroConfig) -> None:
    torch.manual_seed(0)
    model = AlphaGoZeroNet(tiny_config.network)
    optimizer = create_optimizer(model, tiny_config)
    buf = ReplayBuffer(capacity=64, board_size=5, rng=np.random.default_rng(0))
    for _ in range(8):
        buf.add(_example_with_aux())
    metrics, _ = train_for_steps(
        model=model,
        optimizer=optimizer,
        replay_buffer=buf,
        config=tiny_config,
        rng=np.random.default_rng(1),
        global_step=0,
    )
    assert metrics.ownership_loss == 0.0
    assert metrics.score_loss == 0.0


# ---------------------------------------------------------------------------
# Playout cap randomization (B2)
# ---------------------------------------------------------------------------
def test_draw_move_budget_disabled_and_enabled(tiny_config: AlphaGoZeroConfig) -> None:
    rng = np.random.default_rng(0)
    # When randomization is disabled every search is full.
    assert _draw_move_budget(tiny_config, rng) == (tiny_config.search.num_simulations, True)

    tiny_config.search.playout_cap_prob = 0.5
    tiny_config.search.playout_cap_cheap_sims = 2
    draws = [_draw_move_budget(tiny_config, rng) for _ in range(200)]
    full = [d for d in draws if d[1]]
    cheap = [d for d in draws if not d[1]]
    assert all(d[0] == tiny_config.search.num_simulations for d in full)
    assert all(d[0] == 2 for d in cheap)
    # Both categories should occur with probability 0.5.
    assert len(full) > 40 and len(cheap) > 40


def test_selfplay_with_playout_cap_mixes_policy_weights(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    tiny_config.search.playout_cap_prob = 0.5
    tiny_config.search.playout_cap_cheap_sims = 2
    tiny_config.search.disable_resign_prob = 1.0  # Play to a scored terminal position.
    controller = _make_controller(tiny_config)

    result = play_self_play_game(tiny_model, tiny_config, rng, controller)
    weights = {ex.policy_weight for ex in result.examples}
    assert weights <= {0.0, 1.0}
    assert 0.0 in weights and 1.0 in weights  # Both kinds of search occurred.

    # Every example in a scored game has terminal auxiliary labels.
    for ex in result.examples:
        assert ex.ownership is not None
        assert ex.ownership.shape == (5, 5)
        assert abs(ex.score) <= 1.0


def test_config_validation_rejects_bad_playout_cap(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.search.playout_cap_prob = 0.5
    tiny_config.search.playout_cap_cheap_sims = 0
    with pytest.raises(ValueError):
        tiny_config.validate()

    tiny_config.search.playout_cap_cheap_sims = tiny_config.search.num_simulations
    with pytest.raises(ValueError):
        tiny_config.validate()


# ---------------------------------------------------------------------------
# Regression checks for policy weights, auxiliary warmup and resignation calibration
# ---------------------------------------------------------------------------
def test_policy_loss_normalized_by_weight_sum() -> None:
    """Adding zero-weight cheap targets must preserve the full-search policy loss."""
    from alphago_zero.train import _compute_losses

    torch.manual_seed(0)
    logits = torch.randn(8, 26)
    values = torch.zeros(8, 1)
    target_policy = torch.full((8, 26), 1.0 / 26)
    target_value = torch.zeros(8)

    # Only the first example has a full-search policy target.
    weights = torch.zeros(8)
    weights[0] = 1.0
    _, policy_loss_masked, _ = _compute_losses(
        logits, values, target_policy, target_value, policy_weights=weights
    )
    _, policy_loss_single, _ = _compute_losses(
        logits[:1], values[:1], target_policy[:1], target_value[:1]
    )
    torch.testing.assert_close(policy_loss_masked, policy_loss_single)

    # A unit weight for every example matches the unweighted loss.
    _, policy_loss_ones, _ = _compute_losses(
        logits, values, target_policy, target_value, policy_weights=torch.ones(8)
    )
    _, policy_loss_none, _ = _compute_losses(logits, values, target_policy, target_value)
    torch.testing.assert_close(policy_loss_ones, policy_loss_none)


def test_aux_weight_warmup_scale() -> None:
    """Ramp auxiliary weights linearly from zero to one over warmup_steps."""
    from alphago_zero.train import _aux_weight_scale

    assert _aux_weight_scale(0, 0) == 1.0  # Warmup disabled.
    assert _aux_weight_scale(10_000_000, 0) == 1.0
    assert _aux_weight_scale(0, 1000) == 0.0
    assert _aux_weight_scale(500, 1000) == pytest.approx(0.5)
    assert _aux_weight_scale(1000, 1000) == 1.0
    assert _aux_weight_scale(5000, 1000) == 1.0


def test_warmstart_does_not_inherit_resign_threshold(tmp_path: Path) -> None:
    """A replay-only warm start must not import another network's resignation calibration."""
    import torch as _torch

    from alphago_zero.pipeline import _load_buffer_only

    src = ReplayBuffer(capacity=8, board_size=5, rng=np.random.default_rng(1))
    src.add(_example_with_aux())
    snapshot = {
        "replay_buffer": src.state_dict(),
        # Use a deliberately aggressive threshold in the source snapshot.
        "resignation_controller": {"threshold": -0.3, "history": [True, False]},
    }
    path = tmp_path / "foreign_resume.pt"
    _torch.save(snapshot, path)

    dst = ReplayBuffer(capacity=8, board_size=5, rng=np.random.default_rng(2))
    loaded = _load_buffer_only(
        path,
        device=_torch.device("cpu"),
        replay_buffer=dst,
        expected_board_size=5,
    )
    assert loaded == 1  # Only replay is transferred. The API does not accept a controller.
