"""Gumbel configuration, search dispatch, both self-play drivers, league games and full resume."""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig, SearchConfig
from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.gumbel import GumbelMCTS
from alphago_zero.mcts import AlphaGoZeroMCTS
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.parallel_selfplay import play_parallel_self_play_games
from alphago_zero.pipeline import run_training_pipeline
from alphago_zero.selfplay import ResignationController, SelfPlayResult, play_self_play_game


def _gumbel(config: AlphaGoZeroConfig, simulations: int = 16, batch: int = 8) -> AlphaGoZeroConfig:
    config.search.search_algo = "gumbel"
    config.search.num_simulations = simulations
    config.search.eval_batch_size = batch
    with warnings.catch_warnings():
        # Use large batches deliberately to test driver behavior. Check warnings independently.
        warnings.simplefilter("ignore", UserWarning)
        config.validate()
    return config


def _controller(config: AlphaGoZeroConfig) -> ResignationController:
    return ResignationController(
        threshold=config.search.resign_threshold,
        target_false_positive_rate=config.search.target_false_positive_rate,
        adjustment=config.search.resignation_adjustment,
        window=config.search.resignation_window,
    )


def _assert_valid_games(results: list[SelfPlayResult], config: AlphaGoZeroConfig) -> None:
    """Check consistency of improved policies, outcome labels and legal moves in actual games."""
    action_size = config.game.board_size**2 + 1
    for result in results:
        assert result.winner in (BLACK, WHITE)
        # Save each example before the move. Resignation may add a final example without an action.
        assert len(result.examples) == result.num_moves == len(result.moves) + int(result.resigned)
        state = GoState.new_game(
            board_size=config.game.board_size, komi=config.game.komi, max_moves=config.game.max_moves
        )
        for action in result.moves:
            assert action in state.legal_actions(), "played an illegal move"
            state = state.play(action)
        for example in result.examples:
            assert example.policy.shape == (action_size,)
            assert np.isfinite(example.policy).all()
            assert (example.policy >= 0.0).all()
            assert example.policy.sum() == pytest.approx(1.0, abs=1e-5)
            assert example.value in (1.0, -1.0)
            assert example.policy_weight == 1.0


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def test_defaults_keep_puct_and_validate(tiny_config: AlphaGoZeroConfig) -> None:
    assert SearchConfig().search_algo == "puct"
    assert tiny_config.search.search_algo == "puct"
    tiny_config.search.search_algo = "gumbel"
    tiny_config.search.eval_batch_size = 4  # Larger batches trigger a validation warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tiny_config.validate()  # Default Gumbel parameters are valid.


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("search_algo", "alphabeta", "search_algo"),
        ("gumbel_max_considered", 1, "gumbel_max_considered"),
        ("gumbel_max_considered", 0, "gumbel_max_considered"),
        ("gumbel_scale", -0.1, "gumbel_scale"),
        ("gumbel_scale", float("nan"), "gumbel_scale"),
        ("gumbel_scale", float("inf"), "gumbel_scale"),
        ("gumbel_c_visit", -1.0, "gumbel_c_visit"),
        ("gumbel_c_visit", float("nan"), "gumbel_c_visit"),
        ("gumbel_c_visit", float("inf"), "gumbel_c_visit"),
        ("gumbel_c_scale", 0.0, "gumbel_c_scale"),
        ("gumbel_c_scale", float("nan"), "gumbel_c_scale"),
        ("gumbel_c_scale", float("inf"), "gumbel_c_scale"),
    ],
)
def test_invalid_gumbel_settings_are_rejected(
    tiny_config: AlphaGoZeroConfig, field: str, value: object, message: str
) -> None:
    setattr(tiny_config.search, field, value)
    with pytest.raises(ValueError, match=message):
        tiny_config.validate()


def test_gumbel_is_incompatible_with_playout_cap(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.search.search_algo = "gumbel"
    tiny_config.search.playout_cap_prob = 0.25
    tiny_config.search.playout_cap_cheap_sims = 2
    with pytest.raises(ValueError, match="playout cap"):
        tiny_config.validate()
    # PUCT still accepts randomized playout caps.
    tiny_config.search.search_algo = "puct"
    tiny_config.validate()


@pytest.mark.parametrize("batch", [5, 8, 16])
def test_large_eval_batch_with_gumbel_warns_but_stays_valid(
    tiny_config: AlphaGoZeroConfig, batch: int
) -> None:
    """Warn about schedule distortion while allowing the large leaf batch."""
    tiny_config.search.search_algo = "gumbel"
    tiny_config.search.eval_batch_size = batch
    with pytest.warns(UserWarning, match=rf"eval_batch_size={batch}\b") as caught:
        tiny_config.validate()
    assert caught[0].filename == __file__  # Attribute the warning to the caller of validate().


@pytest.mark.parametrize(("algo", "batch"), [("gumbel", 1), ("gumbel", 4), ("puct", 16), ("puct", 64)])
def test_no_batch_warning_for_small_gumbel_batches_or_puct(
    tiny_config: AlphaGoZeroConfig, algo: str, batch: int
) -> None:
    tiny_config.search.search_algo = algo  # type: ignore[assignment]
    tiny_config.search.eval_batch_size = batch
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tiny_config.validate()


def test_search_config_from_old_checkpoint_gets_gumbel_defaults() -> None:
    """Older checkpoints reconstruct SearchConfig with defaults for Gumbel fields."""
    raw = asdict(SearchConfig(num_simulations=160))
    for key in ("search_algo", "gumbel_max_considered", "gumbel_scale", "gumbel_c_visit", "gumbel_c_scale"):
        raw.pop(key)
    restored = SearchConfig(**raw)
    assert restored.num_simulations == 160
    assert restored.search_algo == "puct"
    assert (restored.gumbel_max_considered, restored.gumbel_scale) == (16, 1.0)


# ----------------------------------------------------------------------
# Sequential self-play
# ----------------------------------------------------------------------


def test_sequential_game_records_improved_policy_not_visit_counts(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    config = _gumbel(tiny_config, simulations=16)
    result = play_self_play_game(tiny_model, config, np.random.default_rng(3), _controller(config))
    _assert_valid_games([result], config)
    # An empty 5x5 board has 26 actions. Sixteen simulations cannot visit every action, but an improved
    # policy can assign them all mass.
    first_policy = result.examples[0].policy
    assert np.count_nonzero(first_policy) == 26


def test_sequential_game_is_deterministic_by_seed(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    config = _gumbel(tiny_config)
    first = play_self_play_game(tiny_model, config, np.random.default_rng(8), _controller(config))
    again = play_self_play_game(tiny_model, config, np.random.default_rng(8), _controller(config))
    assert first.moves == again.moves
    assert first.winner == again.winner
    for a, b in zip(first.examples, again.examples, strict=True):
        assert np.array_equal(a.policy, b.policy)
        assert np.array_equal(a.state, b.state)


def test_gumbel_noise_gives_diverse_games(tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet) -> None:
    """With temperature ignored, opening variation must come from Gumbel noise."""
    config = _gumbel(tiny_config)
    games = {
        tuple(play_self_play_game(tiny_model, config, np.random.default_rng(s), _controller(config)).moves)
        for s in range(6)
    }
    assert len(games) > 1


# ----------------------------------------------------------------------
# Parallel self-play and driver equivalence
# ----------------------------------------------------------------------


@pytest.mark.parametrize("search_algo", ["puct", "gumbel"])
@pytest.mark.filterwarnings("ignore:search_algo='gumbel':UserWarning")  # Use batch eight deliberately.
@pytest.mark.parametrize("batch", [1, 8])
def test_sequential_and_parallel_drivers_play_identical_games_with_one_slot(
    tiny_config: AlphaGoZeroConfig, search_algo: str, batch: int
) -> None:
    """At one game slot both drivers should consume RNG in the same order and play identical games."""
    tiny_config.search.search_algo = search_algo
    tiny_config.search.num_simulations = 16
    tiny_config.search.eval_batch_size = batch
    tiny_config.training.parallel_games = 1
    tiny_config.validate()
    for seed in range(4):
        torch.manual_seed(0)
        sequential = play_self_play_game(
            AlphaGoZeroNet(tiny_config.network), tiny_config, np.random.default_rng(seed), _controller(tiny_config)
        )
        torch.manual_seed(0)
        parallel = play_parallel_self_play_games(
            AlphaGoZeroNet(tiny_config.network),
            tiny_config,
            np.random.default_rng(seed),
            _controller(tiny_config),
            1,
        )[0]
        assert sequential.moves == parallel.moves
        assert (sequential.winner, sequential.resigned) == (parallel.winner, parallel.resigned)
        assert sequential.resignation_disabled == parallel.resignation_disabled
        assert len(sequential.examples) == len(parallel.examples)
        for a, b in zip(sequential.examples, parallel.examples, strict=True):
            assert np.array_equal(a.policy, b.policy)
            assert a.value == b.value


@pytest.mark.parametrize("slots", [1, 4])
def test_parallel_gumbel_games_are_valid_and_deterministic(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, slots: int
) -> None:
    config = _gumbel(tiny_config, simulations=16, batch=8)
    config.training.parallel_games = slots
    runs = [
        play_parallel_self_play_games(tiny_model, config, np.random.default_rng(21), _controller(config), 6)
        for _ in range(2)
    ]
    assert len(runs[0]) == 6
    _assert_valid_games(runs[0], config)
    assert [r.moves for r in runs[0]] == [r.moves for r in runs[1]]
    for a, b in zip(runs[0], runs[1], strict=True):
        for ea, eb in zip(a.examples, b.examples, strict=True):
            assert np.array_equal(ea.policy, eb.policy)


@pytest.mark.parametrize("driver", ["sequential", "parallel"])
def test_gumbel_games_carry_aux_targets_to_the_terminal(tiny_config: AlphaGoZeroConfig, driver: str) -> None:
    """Scored terminal games supply ownership and margin targets under either algorithm."""
    tiny_config.network.aux_heads = True
    tiny_config.search.disable_resign_prob = 1.0  # Play through to a scored terminal position.
    config = _gumbel(tiny_config, simulations=8, batch=4)
    config.training.parallel_games = 2
    torch.manual_seed(3)
    model = AlphaGoZeroNet(config.network)
    rng = np.random.default_rng(9)
    if driver == "sequential":
        results = [play_self_play_game(model, config, rng, _controller(config))]
    else:
        results = play_parallel_self_play_games(model, config, rng, _controller(config), 2)
    _assert_valid_games(results, config)
    board = config.game.board_size
    for result in results:
        assert not result.resigned
        assert result.examples
        for example in result.examples:
            assert example.ownership is not None
            assert example.ownership.shape == (board, board)
            assert np.abs(example.ownership).max() <= 1.0
            assert abs(example.score) <= 1.0


@pytest.mark.filterwarnings("ignore:search_algo='gumbel':UserWarning")  # Use batch eight deliberately.
def test_league_champions_stay_on_puct_while_the_learner_plays_gumbel(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    """Frozen league opponents use PUCT even if their checkpoint was trained with Gumbel."""
    config = _gumbel(tiny_config, simulations=8)
    config.training.parallel_games = 3
    config.training.league_prob = 1.0
    config.training.league_checkpoints = ["champion.pt"]
    config.validate()
    torch.manual_seed(5)
    champion = AlphaGoZeroNet(config.network)
    champion_search = copy.deepcopy(config.search)  # The saved search configuration contains search_algo="gumbel".
    champion_search.num_simulations = 6

    calls = {"puct": 0, "gumbel": 0}
    puct_summarize = AlphaGoZeroMCTS.summarize
    gumbel_summarize = GumbelMCTS.summarize

    def count_puct(self: AlphaGoZeroMCTS, root, temperature):  # noqa: ANN001, ANN202
        calls["puct"] += 1
        return puct_summarize(self, root, temperature)

    def count_gumbel(self: GumbelMCTS, root, temperature):  # noqa: ANN001, ANN202
        calls["gumbel"] += 1
        return gumbel_summarize(self, root, temperature)

    monkeypatch.setattr(AlphaGoZeroMCTS, "summarize", count_puct)
    monkeypatch.setattr(GumbelMCTS, "summarize", count_gumbel)
    results = play_parallel_self_play_games(
        tiny_model,
        config,
        np.random.default_rng(2),
        _controller(config),
        4,
        league=[("champion", champion, champion_search)],
    )
    assert len(results) == 4
    assert all(r.league_opponent == "champion" for r in results)
    assert calls["puct"] > 0, "champion moves must use PUCT search"
    assert calls["gumbel"] > 0, "learner moves must use Gumbel search"
    _assert_valid_games(results, config)


# ----------------------------------------------------------------------
# Pipeline and resume
# ----------------------------------------------------------------------


@pytest.mark.parametrize("slots", [1, 3])
def test_pipeline_runs_end_to_end_with_gumbel(tiny_config: AlphaGoZeroConfig, slots: int) -> None:
    config = _gumbel(tiny_config, simulations=16, batch=4)
    config.training.parallel_games = slots
    config.training.self_play_games_per_iteration = 3
    summary = run_training_pipeline(config)
    assert summary["completed_iterations"] == 1
    assert summary["replay_buffer_size"] > 0
    saved = Path(config.runtime.work_dir) / "config.json"
    assert '"search_algo": "gumbel"' in saved.read_text(encoding="utf-8")


def test_pipeline_trains_aux_heads_on_gumbel_games(tiny_config: AlphaGoZeroConfig) -> None:
    """Ensure Gumbel terminal targets reach replay and produce an auxiliary ownership loss."""
    tiny_config.network.aux_heads = True
    tiny_config.search.disable_resign_prob = 1.0
    config = _gumbel(tiny_config, simulations=8, batch=4)
    config.training.parallel_games = 2
    config.training.self_play_games_per_iteration = 2
    run_training_pipeline(config)
    metrics = json.loads((Path(config.runtime.work_dir) / "logs" / "metrics.json").read_text(encoding="utf-8"))
    train = metrics["iterations"][0]["train"]
    assert train["ownership_loss"] > 0.0
    assert train["score_loss"] > 0.0


def _resume_threshold(work: Path) -> float:
    payload = torch.load(work / "resume_state.pt", weights_only=False)
    return float(payload["resignation_controller"]["threshold"])


def _tamper_threshold(work: Path, value: float) -> None:
    path = work / "resume_state.pt"
    payload = torch.load(path, weights_only=False)
    payload["resignation_controller"]["threshold"] = value
    torch.save(payload, path)


def test_resume_resets_resign_calibration_when_search_algo_changes(tiny_config: AlphaGoZeroConfig) -> None:
    """Switching from PUCT to Gumbel requires resetting resignation calibration."""
    run_training_pipeline(tiny_config)
    work = Path(tiny_config.runtime.work_dir)
    _tamper_threshold(work, -0.5)  # Use a former run threshold distinct from the configured -0.95.

    resumed = copy.deepcopy(tiny_config)
    resumed.runtime.resume_from = work / "resume_state.pt"
    resumed.training.num_iterations = 2
    _gumbel(resumed, simulations=16, batch=4)
    with pytest.warns(UserWarning, match="search_algo"):
        run_training_pipeline(resumed)
    assert _resume_threshold(work) == pytest.approx(resumed.search.resign_threshold, abs=0.1)


def test_resume_keeps_resign_calibration_when_search_algo_is_unchanged(tiny_config: AlphaGoZeroConfig) -> None:
    run_training_pipeline(tiny_config)
    work = Path(tiny_config.runtime.work_dir)
    _tamper_threshold(work, -0.5)

    resumed = copy.deepcopy(tiny_config)
    resumed.runtime.resume_from = work / "resume_state.pt"
    resumed.training.num_iterations = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_training_pipeline(resumed)
    assert not [w for w in caught if "Resetting calibration" in str(w.message)]
    assert _resume_threshold(work) == pytest.approx(-0.5, abs=0.1)
