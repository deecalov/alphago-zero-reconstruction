from __future__ import annotations

"""Tests for players, Elo estimation and fixed-opponent benchmarking."""

import math

import numpy as np
import pytest

from alphago_zero.benchmark import benchmark_model, play_anchor_series
from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.elo import (
    MatchRecord,
    estimate_rating,
    expected_score,
    rating_from_win_rate,
)
from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.players import MCTSPlayer, PolicyPlayer, RandomPlayer, play_match


# ----------------------------------------------------------------------
# Players
# ----------------------------------------------------------------------
def test_random_player_plays_legal_moves(rng: np.random.Generator) -> None:
    player = RandomPlayer(rng=rng)
    state = GoState.new_game(board_size=5, komi=2.5)
    for _ in range(20):
        action = player.select_action(state)
        assert state.is_legal(action)
        state = state.play(action)
        if state.is_terminal():
            break


def test_policy_player_plays_legal_moves(tiny_model: AlphaGoZeroNet) -> None:
    player = PolicyPlayer(model=tiny_model)
    state = GoState.new_game(board_size=5, komi=2.5)
    for _ in range(10):
        action = player.select_action(state)
        assert state.is_legal(action)
        state = state.play(action)
        if state.is_terminal():
            break


def test_policy_player_is_deterministic(tiny_model: AlphaGoZeroNet) -> None:
    player = PolicyPlayer(model=tiny_model)
    state = GoState.new_game(board_size=5, komi=2.5)
    assert player.select_action(state) == player.select_action(state)


def test_mcts_player_deterministic_by_seed(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    actions = []
    for _ in range(2):
        player = MCTSPlayer(
            model=tiny_model,
            search_config=tiny_config.search,
            rng=np.random.default_rng(42),
        )
        actions.append(player.select_action(state))
    assert actions[0] == actions[1]
    assert state.is_legal(actions[0])


def test_play_match_returns_winner_color(rng: np.random.Generator) -> None:
    black = RandomPlayer(rng=rng)
    white = RandomPlayer(rng=rng)
    winner = play_match(black=black, white=white, board_size=5, komi=2.5)
    assert winner in (BLACK, WHITE)


# ----------------------------------------------------------------------
# Elo
# ----------------------------------------------------------------------
def test_expected_score_basics() -> None:
    assert expected_score(0.0, 0.0) == pytest.approx(0.5)
    # A 400-point Elo advantage gives expected score odds of ten to one.
    assert expected_score(400.0, 0.0) == pytest.approx(10.0 / 11.0)
    # The two players' expected scores sum to one.
    assert expected_score(120.0, -80.0) + expected_score(-80.0, 120.0) == pytest.approx(1.0)


def test_rating_from_win_rate_formula() -> None:
    assert rating_from_win_rate(0.5) == pytest.approx(0.0)
    # A 75% score gives an advantage of about 190.8 Elo.
    assert rating_from_win_rate(0.75) == pytest.approx(400.0 * math.log10(3.0), abs=1e-9)
    # Shifting the opponent's rating shifts the estimate equally.
    assert rating_from_win_rate(0.5, opponent_rating=300.0) == pytest.approx(300.0)
    # Extreme scores yield finite ratings.
    assert math.isfinite(rating_from_win_rate(0.0))
    assert math.isfinite(rating_from_win_rate(1.0))
    assert rating_from_win_rate(1.0) > rating_from_win_rate(0.9)


def test_estimate_rating_even_score_matches_anchor() -> None:
    # An even score against a 100 Elo opponent gives the same rating.
    rating = estimate_rating([MatchRecord(opponent_rating=100.0, wins=5, losses=5)])
    assert rating == pytest.approx(100.0, abs=1e-6)


def test_estimate_rating_monotonic_in_wins() -> None:
    ratings = [
        estimate_rating([MatchRecord(opponent_rating=0.0, wins=w, losses=10 - w)])
        for w in range(11)
    ]
    assert all(b > a for a, b in zip(ratings, ratings[1:]))
    # Smoothing keeps a ten-win sweep finite.
    assert all(math.isfinite(r) for r in ratings)


def test_estimate_rating_multiple_anchors_between_extremes() -> None:
    # Beating a zero-rated anchor and losing to a 400-rated anchor gives an intermediate estimate.
    rating = estimate_rating(
        [
            MatchRecord(opponent_rating=0.0, wins=9, losses=1),
            MatchRecord(opponent_rating=400.0, wins=1, losses=9),
        ]
    )
    assert 0.0 < rating < 400.0


def test_estimate_rating_draws_count_as_half_points() -> None:
    # An even score with draws matches the anchor. Each draw earns half a point.
    rating = estimate_rating(
        [MatchRecord(opponent_rating=100.0, wins=2, losses=2, draws=6)]
    )
    assert rating == pytest.approx(100.0, abs=1e-6)
    record = MatchRecord(opponent_rating=0.0, wins=1, losses=1, draws=2)
    assert record.games == 4
    assert record.points == 2.0


def test_estimate_rating_requires_games() -> None:
    with pytest.raises(ValueError):
        estimate_rating([])
    with pytest.raises(ValueError):
        estimate_rating([MatchRecord(opponent_rating=0.0, wins=0, losses=0)])


# ----------------------------------------------------------------------
# Benchmark
# ----------------------------------------------------------------------
def test_play_anchor_series_counts(
    tiny_config: AlphaGoZeroConfig, rng: np.random.Generator
) -> None:
    candidate = RandomPlayer(rng=rng)
    anchor = RandomPlayer(rng=rng)
    wins, losses, draws = play_anchor_series(candidate, anchor, tiny_config, games=4)
    assert wins + losses + draws == 4
    assert wins >= 0 and losses >= 0 and draws >= 0


def test_benchmark_model_structure(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet, rng: np.random.Generator
) -> None:
    result = benchmark_model(
        model=tiny_model,
        config=tiny_config,
        rng=rng,
        anchors={"random": (RandomPlayer(rng=rng), 0.0)},
        games_per_anchor=2,
    )
    assert set(result) == {"anchors", "elo_estimate", "games_per_anchor"}
    random_stats = result["anchors"]["random"]
    assert random_stats["wins"] + random_stats["losses"] + random_stats["draws"] == 2
    assert random_stats["games"] == 2
    assert 0.0 <= random_stats["win_rate"] <= 1.0
    assert math.isfinite(result["elo_estimate"])
