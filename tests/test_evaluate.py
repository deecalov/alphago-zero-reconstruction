from __future__ import annotations

"""Candidate evaluation invariants and promotion decisions."""

from dataclasses import replace
import itertools
import math

import numpy as np
import pytest

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero import evaluate as evaluate_module
from alphago_zero.evaluate import (
    evaluate_candidate,
    play_evaluation_game,
    required_wins_for_promotion,
)
from alphago_zero.game import BLACK, WHITE
from alphago_zero.network import AlphaGoZeroNet


def test_play_evaluation_game_returns_color(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    winner = play_evaluation_game(tiny_model, tiny_model, tiny_config, rng)
    assert winner in (BLACK, WHITE)


def test_evaluate_candidate_counts_are_consistent(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    summary = evaluate_candidate(tiny_model, tiny_model, tiny_config, rng)
    # Early stopping can shorten a match but cannot increase the game count.
    assert 1 <= summary.games <= tiny_config.training.evaluation_games
    assert summary.candidate_wins + summary.best_wins + summary.draws == summary.games
    assert 0.0 <= summary.candidate_win_rate <= 1.0


def test_evaluate_candidate_plays_full_match_without_early_stop(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    tiny_config.training = replace(tiny_config.training, evaluation_early_stop=False)
    summary = evaluate_candidate(tiny_model, tiny_model, tiny_config, rng)
    assert summary.games == tiny_config.training.evaluation_games
    assert not summary.stopped_early


# ----------------------------------------------------------------------
# Scripted outcomes for early stopping
# ----------------------------------------------------------------------
def _scripted_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    config: AlphaGoZeroConfig,
    candidate_outcomes: list[bool],
) -> object:
    """Run evaluation on a prescribed sequence of candidate results."""
    game_counter = itertools.count()

    def fake_game(black_model: object, white_model: object, config: object, rng: object) -> int:
        game_idx = next(game_counter)
        candidate_is_black = game_idx % 2 == 0
        candidate_won = candidate_outcomes[game_idx]
        if candidate_is_black:
            return BLACK if candidate_won else WHITE
        return WHITE if candidate_won else BLACK

    monkeypatch.setattr(evaluate_module, "play_evaluation_game", fake_game)
    return evaluate_candidate(
        candidate_model=None,  # type: ignore[arg-type]  # The stubbed game does not use models.
        best_model=None,  # type: ignore[arg-type]
        config=config,
        rng=np.random.default_rng(0),
    )


def _gate_config(tiny_config: AlphaGoZeroConfig, games: int, early_stop: bool) -> AlphaGoZeroConfig:
    tiny_config.training = replace(
        tiny_config.training,
        evaluation_games=games,
        evaluation_early_stop=early_stop,
        replace_threshold=0.55,
    )
    return tiny_config


def test_evaluate_candidate_counts_draws(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig
) -> None:
    # Drawn games count toward the total without giving either side a win.
    config = _gate_config(tiny_config, games=4, early_stop=False)
    monkeypatch.setattr(
        evaluate_module, "play_evaluation_game", lambda *args, **kwargs: None
    )
    summary = evaluate_candidate(
        candidate_model=None,  # type: ignore[arg-type]
        best_model=None,  # type: ignore[arg-type]
        config=config,
        rng=np.random.default_rng(0),
    )
    assert summary.games == 4
    assert summary.draws == 4
    assert summary.candidate_wins == 0 and summary.best_wins == 0
    assert not summary.promoted


def test_early_stop_when_promotion_secured(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig
) -> None:
    # Three wins in a four-game match pass a 0.55 threshold regardless of the fourth result.
    config = _gate_config(tiny_config, games=4, early_stop=True)
    summary = _scripted_evaluation(monkeypatch, config, [True, True, True, True])
    assert summary.stopped_early  # type: ignore[attr-defined]
    assert summary.games == 3  # type: ignore[attr-defined]
    assert summary.candidate_win_rate >= config.training.replace_threshold  # type: ignore[attr-defined]


def test_early_stop_when_promotion_impossible(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig
) -> None:
    # Two losses make three wins in four games impossible.
    config = _gate_config(tiny_config, games=4, early_stop=True)
    summary = _scripted_evaluation(monkeypatch, config, [False, False, True, True])
    assert summary.stopped_early  # type: ignore[attr-defined]
    assert summary.games == 2  # type: ignore[attr-defined]
    assert summary.candidate_win_rate < config.training.replace_threshold  # type: ignore[attr-defined]


@pytest.mark.parametrize("threshold", [0.5, 0.55, 0.6, 0.7, 0.75, 1.0 / 3.0, 0.0, 1.0])
def test_required_wins_matches_gate_comparison(threshold: float) -> None:
    """Match the exact floating-point threshold comparison across match sizes."""
    for games in range(1, 1001):
        expected = next(
            (w for w in range(games + 1) if w / games >= threshold), games + 1
        )
        assert required_wins_for_promotion(games, threshold) == expected, games


def test_required_wins_zero_games() -> None:
    assert required_wins_for_promotion(0, 0.55) == 0


@pytest.mark.parametrize("games", [8, 32, 128, 400])
def test_binomial_gate_controls_false_promotion_probability(games: int) -> None:
    critical = required_wins_for_promotion(games, 0.55, significance=0.05)
    tail = sum(math.comb(games, k) for k in range(critical, games + 1)) / 2**games
    assert tail <= 0.05
    previous_tail = sum(math.comb(games, k) for k in range(critical - 1, games + 1)) / 2**games
    assert previous_tail > 0.05 or (critical - 1) / games < 0.55


def test_binomial_gate_early_stop_matches_full_match_exhaustively(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig,
) -> None:
    cfg = _gate_config(tiny_config, games=8, early_stop=True)
    cfg.training.evaluation_gate = "binomial"
    for outcomes in itertools.product([False, True], repeat=8):
        summary = _scripted_evaluation(monkeypatch, cfg, list(outcomes))
        assert summary.required_wins == 7  # type: ignore[attr-defined]
        assert summary.promoted == (sum(outcomes) >= 7)  # type: ignore[attr-defined]
    # 5/8 used to be promoted; it supplies no significant evidence of improvement.
    summary = _scripted_evaluation(monkeypatch, cfg, [True] * 5 + [False] * 3)
    assert not summary.promoted  # type: ignore[attr-defined]


def test_no_games_never_promotes_even_at_zero_threshold(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig,
) -> None:
    tiny_config.training.evaluation_games = 0
    tiny_config.training.replace_threshold = 0.0
    summary = _scripted_evaluation(monkeypatch, tiny_config, [])
    assert not summary.promoted  # type: ignore[attr-defined]


# ----------------------------------------------------------------------
# Promotion with the gate disabled
# ----------------------------------------------------------------------
def test_gate_none_promotes_without_playing_or_consuming_rng(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig,
) -> None:
    """With no gate, promote immediately without calling the game runner or consuming RNG."""

    def forbidden(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("no games should be played with gate 'none'")

    monkeypatch.setattr(evaluate_module, "play_evaluation_game", forbidden)
    tiny_config.training.evaluation_gate = "none"
    tiny_config.training.evaluation_games = 8
    tiny_config.validate()

    generator = np.random.default_rng(5)
    state_before = generator.bit_generator.state
    summary = evaluate_candidate(
        candidate_model=None,  # type: ignore[arg-type]  # Neither network is called.
        best_model=None,  # type: ignore[arg-type]
        config=tiny_config,
        rng=generator,
    )
    assert generator.bit_generator.state == state_before
    assert summary.promoted
    assert summary.gate == "none"
    assert (summary.games, summary.games_planned, summary.required_wins) == (0, 0, 0)
    assert (summary.candidate_wins, summary.best_wins, summary.draws) == (0, 0, 0)
    assert not summary.stopped_early


def test_gate_none_differs_from_an_ordinary_gate_with_zero_games(
    tiny_config: AlphaGoZeroConfig,
) -> None:
    """A zero-game fixed gate does not promote. Only the explicit none mode bypasses the gate."""
    tiny_config.training.evaluation_games = 0
    for gate, promoted in (("fixed", False), ("none", True)):
        tiny_config.training.evaluation_gate = gate  # type: ignore[assignment]
        tiny_config.validate()
        summary = evaluate_candidate(
            candidate_model=None,  # type: ignore[arg-type]
            best_model=None,  # type: ignore[arg-type]
            config=tiny_config,
            rng=np.random.default_rng(0),
        )
        assert summary.promoted is promoted, gate


def test_gate_none_skips_the_binomial_sample_size_check(tiny_config: AlphaGoZeroConfig) -> None:
    """A disabled gate does not need a sample large enough for a significance test."""
    tiny_config.training.evaluation_gate = "none"
    tiny_config.training.evaluation_games = 4
    tiny_config.validate()


@pytest.mark.parametrize("gate", ["off", "None", "", "binominal"])
def test_unknown_gate_is_rejected(tiny_config: AlphaGoZeroConfig, gate: str) -> None:
    tiny_config.training.evaluation_gate = gate  # type: ignore[assignment]
    with pytest.raises(ValueError, match="evaluation_gate"):
        tiny_config.validate()


def test_binomial_gate_rejects_impossible_sample_size(tiny_config: AlphaGoZeroConfig) -> None:
    tiny_config.training.evaluation_gate = "binomial"
    tiny_config.training.evaluation_games = 4
    with pytest.raises(ValueError, match="evaluation_games"):
        tiny_config.validate()


def test_early_stop_does_not_reject_reachable_promotion_on_large_match(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig
) -> None:
    """In a 400-game match the 220th win can still secure promotion on the final game."""
    games = 400
    config = _gate_config(tiny_config, games=games, early_stop=True)
    outcomes = [True] * 219 + [False] * 180 + [True]
    summary = _scripted_evaluation(monkeypatch, config, outcomes)
    assert summary.games == games  # type: ignore[attr-defined]
    assert not summary.stopped_early  # type: ignore[attr-defined]
    assert summary.candidate_win_rate >= config.training.replace_threshold  # type: ignore[attr-defined]


@pytest.mark.parametrize("games", [20, 40, 100, 200, 400])
def test_early_stop_decision_matches_full_match_on_boundary_sequences(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig, games: int
) -> None:
    """Early stopping agrees with a full match at the promotion boundary."""
    threshold = 0.55
    required = required_wins_for_promotion(games, threshold)
    for total_wins in (required - 1, required, required + 1):
        if not 0 <= total_wins <= games:
            continue
        losses = games - total_wins
        sequences = [
            [True] * total_wins + [False] * losses,
            [False] * losses + [True] * total_wins,
        ]
        if total_wins >= 1:
            sequences.append([True] * (total_wins - 1) + [False] * losses + [True])
        for outcomes in sequences:
            config = _gate_config(tiny_config, games=games, early_stop=True)
            summary = _scripted_evaluation(monkeypatch, config, outcomes)
            early_decision = summary.candidate_win_rate >= threshold  # type: ignore[attr-defined]
            full_decision = total_wins / games >= threshold
            assert early_decision == full_decision, (games, total_wins, outcomes[:3])


def test_early_stop_decision_matches_full_match_exhaustively(
    monkeypatch: pytest.MonkeyPatch, tiny_config: AlphaGoZeroConfig
) -> None:
    """Enumerate short outcome sequences and compare early and full decisions."""
    threshold = 0.55
    games = 5
    for outcomes_bits in itertools.product([False, True], repeat=games):
        outcomes = list(outcomes_bits)

        config_early = _gate_config(tiny_config, games=games, early_stop=True)
        summary_early = _scripted_evaluation(monkeypatch, config_early, outcomes)

        full_win_rate = sum(outcomes) / games
        early_decision = summary_early.candidate_win_rate >= threshold  # type: ignore[attr-defined]
        full_decision = full_win_rate >= threshold
        assert early_decision == full_decision, (
            f"outcomes={outcomes}: early={early_decision}, full={full_decision}"
        )
        # An early-stopped match cannot be longer than the full match.
        assert summary_early.games <= games  # type: ignore[attr-defined]


# ----------------------------------------------------------------------
# Opening variation
# ----------------------------------------------------------------------
def test_opening_moves_sampling_plays_valid_game(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    tiny_config.training = replace(tiny_config.training, evaluation_opening_moves=3)
    winner = play_evaluation_game(tiny_model, tiny_model, tiny_config, rng)
    assert winner in (BLACK, WHITE)


def test_opening_moves_sampling_is_seed_deterministic(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    tiny_config.training = replace(tiny_config.training, evaluation_opening_moves=2)
    winners = [
        play_evaluation_game(
            tiny_model, tiny_model, tiny_config, np.random.default_rng(123)
        )
        for _ in range(2)
    ]
    assert winners[0] == winners[1]
