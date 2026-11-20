"""Sequential probability ratio test arithmetic, thresholds and stopping behavior."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alphago_zero.sprt import SprtConfig, SprtDecision, expected_games, sprt_decision, sprt_llr

# Single-game LLR increments for H0 at 0.50 and H1 at 0.55.
WIN_STEP = 0.09531018
LOSS_STEP = -0.10536052


# ----------------------------------------------------------------------
# Parameters and decision boundaries
# ----------------------------------------------------------------------
def test_default_config_matches_the_roadmap() -> None:
    config = SprtConfig()
    assert (config.wr0, config.wr1, config.alpha, config.beta) == (0.50, 0.55, 0.05, 0.05)


def test_bounds_for_symmetric_error_rates() -> None:
    config = SprtConfig()
    # ln(0.95 / 0.05) = ln 19
    assert config.upper_bound == pytest.approx(2.944439, abs=1e-6)
    assert config.lower_bound == pytest.approx(-2.944439, abs=1e-6)


def test_bounds_for_asymmetric_error_rates() -> None:
    config = SprtConfig(alpha=0.01, beta=0.10)
    assert config.upper_bound == pytest.approx(4.499810, abs=1e-6)  # ln(0.90 / 0.01)
    assert config.lower_bound == pytest.approx(-2.292535, abs=1e-6)  # ln(0.10 / 0.99)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"wr0": 0.55, "wr1": 0.50},  # H1 requires a higher win rate than H0.
        {"wr0": 0.50, "wr1": 0.50},
        {"wr0": 0.0, "wr1": 0.55},  # Probabilities of zero or one give degenerate likelihoods.
        {"wr0": 0.50, "wr1": 1.0},
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"beta": 0.0},
        {"beta": 1.0},
        {"alpha": 0.6, "beta": 0.5},  # When alpha + beta exceeds one the lower boundary exceeds the upper.
        {"alpha": 0.5, "beta": 0.5},  # When alpha + beta equals one the boundaries coincide at zero.
        {"wr0": float("nan")},
        {"wr1": float("nan")},
        {"alpha": float("nan")},
        {"beta": float("nan")},
        {"wr0": float("-inf")},
        {"wr1": float("inf")},
        {"alpha": float("inf")},
        {"beta": float("-inf")},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SprtConfig(**kwargs)


# ----------------------------------------------------------------------
# Log likelihood ratio
# ----------------------------------------------------------------------
def test_llr_matches_the_hand_computed_value() -> None:
    # Ten wins and five losses give 10 * ln(1.1) + 5 * ln(0.9).
    assert sprt_llr(10, 5, SprtConfig()) == pytest.approx(10 * WIN_STEP + 5 * LOSS_STEP, abs=1e-6)
    assert sprt_llr(10, 5, SprtConfig()) == pytest.approx(0.426299, abs=1e-6)


def test_llr_is_zero_before_any_game() -> None:
    assert sprt_llr(0, 0, SprtConfig()) == 0.0


def test_llr_of_symmetric_hypotheses_depends_only_on_the_win_difference() -> None:
    config = SprtConfig(wr0=0.40, wr1=0.60)
    assert sprt_llr(30, 30, config) == pytest.approx(0.0, abs=1e-12)
    assert sprt_llr(35, 30, config) == pytest.approx(5 * math.log(1.5), abs=1e-9)


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        sprt_llr(-1, 3, SprtConfig())
    with pytest.raises(ValueError):
        sprt_decision(3, -1, SprtConfig())


# ----------------------------------------------------------------------
# Stopping criterion
# ----------------------------------------------------------------------
def test_decision_values_are_the_json_contract() -> None:
    assert SprtDecision.ACCEPT_H1.value == "H1"
    assert SprtDecision.ACCEPT_H0.value == "H0"
    assert SprtDecision.CONTINUE.value == "continue"


def test_thirty_one_straight_wins_accept_h1() -> None:
    config = SprtConfig()
    assert sprt_decision(30, 0, config) is SprtDecision.CONTINUE  # 30 * 0.09531 = 2.859 < 2.944
    assert sprt_decision(31, 0, config) is SprtDecision.ACCEPT_H1  # 31 * 0.09531 = 2.955 >= 2.944


def test_twenty_eight_straight_losses_accept_h0() -> None:
    config = SprtConfig()
    assert sprt_decision(0, 27, config) is SprtDecision.CONTINUE  # -2.845 > -2.944
    assert sprt_decision(0, 28, config) is SprtDecision.ACCEPT_H0  # -2.950 <= -2.944


def test_balanced_results_keep_the_test_running() -> None:
    assert sprt_decision(50, 50, SprtConfig()) is SprtDecision.CONTINUE


def test_looser_error_rates_decide_earlier() -> None:
    loose = SprtConfig(alpha=0.2, beta=0.2)  # Boundaries at ±ln(4), about ±1.386.
    assert sprt_decision(14, 0, loose) is SprtDecision.CONTINUE  # 14 * 0.09531 = 1.334
    assert sprt_decision(15, 0, loose) is SprtDecision.ACCEPT_H1  # 15 * 0.09531 = 1.430
    assert sprt_decision(14, 0, SprtConfig()) is SprtDecision.CONTINUE


def test_wider_gap_between_hypotheses_decides_earlier() -> None:
    wide = SprtConfig(wr1=0.60)  # A win adds ln(1.2), about 0.18232.
    assert sprt_decision(16, 0, wide) is SprtDecision.CONTINUE  # 2.917 < 2.944
    assert sprt_decision(17, 0, wide) is SprtDecision.ACCEPT_H1  # 3.099 >= 2.944


# ----------------------------------------------------------------------
# Simulated test properties
# ----------------------------------------------------------------------
def _simulate(
    win_rate: float, config: SprtConfig, paths: int, max_games: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate independent matches and check the stopping boundaries after complete color pairs."""
    rng = np.random.default_rng(seed)
    llr = np.zeros(paths)
    active = np.ones(paths, dtype=bool)
    outcome = np.zeros(paths, dtype=np.int8)
    games = np.full(paths, max_games)
    for game in range(1, max_games + 1):
        llr += np.where(rng.random(paths) < win_rate, WIN_STEP, LOSS_STEP)
        if game % 2 == 0:
            up = active & (llr >= config.upper_bound)
            down = active & (llr <= config.lower_bound)
            outcome[up] = 1
            outcome[down] = -1
            games[up | down] = game
            active &= ~(up | down)
        if not active.any():
            break
    return outcome, games


def test_error_rates_match_alpha_and_beta() -> None:
    config = SprtConfig()
    # Under H0 the fraction of incorrect H1 decisions should be close to alpha.
    outcome, _ = _simulate(0.50, config, paths=4000, max_games=4000, seed=11)
    false_h1 = float(np.mean(outcome == 1))
    assert 0.030 < false_h1 < 0.065
    # Under H1 the fraction of incorrect H0 decisions should be close to beta.
    outcome, _ = _simulate(0.55, config, paths=4000, max_games=4000, seed=12)
    false_h0 = float(np.mean(outcome == -1))
    assert 0.030 < false_h0 < 0.065


def test_clear_advantage_is_decided_with_fewer_games_than_a_fixed_match() -> None:
    outcome, games = _simulate(0.70, SprtConfig(), paths=4000, max_games=120, seed=13)
    assert float(np.mean(outcome == 1)) > 0.85  # A cap of 120 games usually suffices
    assert float(np.mean(games)) < 95  # and the mean length is substantially lower.


# ----------------------------------------------------------------------
# Expected sample size
# ----------------------------------------------------------------------
def test_expected_games_at_the_default_hypotheses() -> None:
    under_h0, under_h1 = expected_games(SprtConfig())
    # Evaluate the expected boundary crossing under both hypotheses.
    assert under_h0 == pytest.approx(527.3, rel=0.005)
    assert under_h1 == pytest.approx(529.1, rel=0.005)


def test_expected_games_use_alpha_and_beta_on_the_right_hypotheses() -> None:
    # alpha = 0.01, beta = 0.10: B = ln 90 = 4.49981, A = ln(0.10 / 0.99) = -2.292535
    under_h0, under_h1 = expected_games(SprtConfig(alpha=0.01, beta=0.10))
    # Check the expected lengths with asymmetric error probabilities.
    assert under_h0 == pytest.approx(442.7, rel=0.005)
    assert under_h1 == pytest.approx(762.8, rel=0.005)


def test_expected_games_grow_when_the_hypotheses_are_closer() -> None:
    # Sample size scales approximately with the inverse square of the win-rate separation.
    close = expected_games(SprtConfig(wr1=0.53))
    default = expected_games(SprtConfig())
    assert close[0] == pytest.approx(1469.5, rel=0.01)
    assert close[1] == pytest.approx(1471.0, rel=0.01)
    assert close[0] > 2.5 * default[0] and close[1] > 2.5 * default[1]
