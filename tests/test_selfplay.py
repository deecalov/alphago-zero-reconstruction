from __future__ import annotations

"""Self-play data generation, game endings and resignation calibration tests."""

import numpy as np
import pytest

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.selfplay import ResignationController, game_end_reason, play_self_play_game


def _make_controller(config: AlphaGoZeroConfig) -> ResignationController:
    return ResignationController(
        threshold=config.search.resign_threshold,
        target_false_positive_rate=config.search.target_false_positive_rate,
        adjustment=config.search.resignation_adjustment,
        window=config.search.resignation_window,
    )


def test_self_play_produces_valid_examples(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    controller = _make_controller(tiny_config)
    result = play_self_play_game(tiny_model, tiny_config, rng, controller)

    assert len(result.examples) > 0
    assert result.winner in (BLACK, WHITE)
    for example in result.examples:
        assert example.state.shape == (17, 5, 5)
        assert example.policy.shape == (tiny_config.action_size,)
        assert example.value in (1.0, -1.0)
        # Policy sums to one.
        assert float(example.policy.sum()) == pytest.approx(1.0, abs=1e-4)


def test_game_end_reason_classifies_all_endings() -> None:
    # Resignation takes precedence over other ending reasons.
    assert game_end_reason(None, resigned=True) == "resign"
    # Two passes take precedence when the same move also hits the limit.
    passed = GoState.new_game(board_size=5, komi=2.5, max_moves=2)
    passed = passed.play(passed.pass_action).play(passed.pass_action)
    assert passed.is_terminal()
    assert game_end_reason(passed, resigned=False) == "double_pass"
    # Use max_moves when the limit is reached without two passes.
    cut = GoState.new_game(board_size=5, komi=2.5, max_moves=1).play(0)
    assert cut.is_terminal()
    assert game_end_reason(cut, resigned=False) == "max_moves"


def test_self_play_result_reports_end_reason(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    controller = _make_controller(tiny_config)
    result = play_self_play_game(tiny_model, tiny_config, rng, controller)
    assert result.ended_by in ("double_pass", "resign", "max_moves")
    # The ending reason agrees with the other result fields.
    assert (result.ended_by == "resign") == result.resigned
    if result.resigned:
        assert result.score_margin is None
    else:
        assert result.score_margin is not None


def test_policy_target_is_visit_distribution_not_onehot(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    # Disable resignation and force late-game sampling to expose accidental one-hot policy targets.
    tiny_config.search.num_simulations = 16
    tiny_config.search.disable_resign_prob = 1.0
    controller = _make_controller(tiny_config)

    result = play_self_play_game(tiny_model, tiny_config, rng, controller)

    # Require at least one learning target with two or more nonzero probabilities.
    has_soft_target = any(int(np.count_nonzero(ex.policy)) >= 2 for ex in result.examples)
    assert has_soft_target


def test_resignation_controller_lowers_threshold_on_false_positives() -> None:
    controller = ResignationController(
        threshold=-0.9, target_false_positive_rate=0.05, adjustment=0.02, window=100
    )
    for _ in range(50):
        controller.observe_false_positive(True)
    # Excess false resignations should make the threshold more negative.
    assert controller.threshold < -0.9


def test_resignation_controller_raises_threshold_when_few_false_positives() -> None:
    controller = ResignationController(
        threshold=-0.9, target_false_positive_rate=0.05, adjustment=0.02, window=100
    )
    for _ in range(50):
        controller.observe_false_positive(False)
    # A low false-positive rate allows a less negative threshold.
    assert controller.threshold > -0.9


def test_resignation_controller_clips_threshold() -> None:
    controller = ResignationController(
        threshold=-0.9, target_false_positive_rate=0.05, adjustment=0.5, window=100
    )
    for _ in range(50):
        controller.observe_false_positive(True)
    # Respect the lower bound on the resignation threshold.
    assert controller.threshold >= -0.999


@pytest.mark.parametrize("window", [20, 250])
def test_resignation_controller_history_respects_window(window: int) -> None:
    # Check that the configurable window replaces the old fixed size of 100.
    controller = ResignationController(
        threshold=-0.9, target_false_positive_rate=0.05, adjustment=0.0, window=window
    )
    for i in range(window + 30):
        controller.observe_false_positive(i < 30)
    # The earliest 30 false positives have left the sliding window.
    assert len(controller.history) == window
    assert controller.false_positive_rate == 0.0


def test_resignation_controller_rejects_empty_window() -> None:
    with pytest.raises(ValueError):
        ResignationController(
            threshold=-0.9, target_false_positive_rate=0.05, adjustment=0.02, window=0
        )
