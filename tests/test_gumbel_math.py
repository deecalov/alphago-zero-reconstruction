"""Test Gumbel search mathematics independently of the game tree."""

from __future__ import annotations

import numpy as np
import pytest

from alphago_zero.gumbel import (
    _softmax,
    completed_q_scores,
    considered_visit_sequence,
    mixed_value,
    score_considered,
)

# ----------------------------------------------------------------------
# Sequential Halving visit schedule
# ----------------------------------------------------------------------


def _levels(*blocks: tuple[int, int]) -> tuple[int, ...]:
    """Expand (value, repetitions) pairs into an explicit schedule."""
    return tuple(value for value, repeat in blocks for _ in range(repeat))


@pytest.mark.parametrize(
    ("max_considered", "num_simulations", "expected"),
    [
        # The four survivors each receive four visits.
        (16, 32, _levels((0, 16), (1, 8), (2, 4), (3, 4))),
        # Three rounds reduce eight candidates to four and then two finalists.
        (8, 16, _levels((0, 8), (1, 4), (2, 2), (3, 2))),
        (4, 8, (0, 0, 0, 0, 1, 1, 2, 2)),
        # The simulation budget truncates the tail.
        (2, 5, (0, 0, 1, 1, 2)),
        # Six candidates need ceil(log2(6)) = 3 rounds.
        (6, 24, _levels((0, 6), (1, 3), (2, 3)) + (3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8)),
        # Three candidates need two rounds, with extra visits of two and three.
        (3, 12, (0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4)),
        # A long budget allows repeated rounds for the finalists.
        (
            8,
            64,
            _levels((0, 8), (1, 8), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4))
            + tuple(v for v in range(7, 17) for _ in range(2))
            + tuple(v for v in range(17, 21) for _ in range(2)),
        ),
    ],
)
def test_considered_visit_sequence_matches_reference_values(
    max_considered: int, num_simulations: int, expected: tuple[int, ...]
) -> None:
    assert considered_visit_sequence(max_considered, num_simulations) == expected


@pytest.mark.parametrize("num_simulations", [1, 2, 7, 32])
def test_single_candidate_is_visited_again_and_again(num_simulations: int) -> None:
    """With one candidate the eligible count increases by one every simulation."""
    assert considered_visit_sequence(1, num_simulations) == tuple(range(num_simulations))
    assert considered_visit_sequence(0, num_simulations) == tuple(range(num_simulations))


@pytest.mark.parametrize("max_considered", [2, 3, 4, 5, 8, 13, 16, 25, 82])
@pytest.mark.parametrize("num_simulations", [1, 3, 8, 16, 32, 64, 100, 400])
def test_schedule_is_always_executable(max_considered: int, num_simulations: int) -> None:
    """For any tested candidate count and budget, every scheduled visit must be realizable."""
    sequence = considered_visit_sequence(max_considered, num_simulations)
    assert len(sequence) == num_simulations
    assert all(a <= b for a, b in zip(sequence, sequence[1:], strict=False)), "visit schedule must be nondecreasing"

    visits = [0] * max_considered
    for required in sequence:
        candidates = [i for i, count in enumerate(visits) if count == required]
        assert candidates, f"no candidate with {required} visits (m={max_considered}, n={num_simulations})"
        visits[candidates[0]] += 1
    assert sum(visits) == num_simulations
    # The first round visits every candidate up to the available budget.
    assert sum(1 for count in visits if count > 0) == min(max_considered, num_simulations)


def test_final_visits_follow_halving_structure() -> None:
    """With 16 candidates and 32 simulations, expect four with four visits, four with two and eight with one."""
    visits = [0] * 16
    for required in considered_visit_sequence(16, 32):
        index = next(i for i, count in enumerate(visits) if count == required)
        visits[index] += 1
    assert sorted(visits, reverse=True) == [4] * 4 + [2] * 4 + [1] * 8


def test_considered_visit_sequence_is_cached_and_immutable() -> None:
    first = considered_visit_sequence(16, 32)
    assert isinstance(first, tuple)
    assert considered_visit_sequence(16, 32) is first


# ----------------------------------------------------------------------
# Mixed value and completed Q
# ----------------------------------------------------------------------


def test_mixed_value_matches_hand_computation() -> None:
    """Use raw value 0.2 and visited edges with counts two and one, Q 0.5 and -0.3, and priors 0.4 and 0.2."""
    value = mixed_value(
        raw_value=0.2,
        q_values=np.array([0.5, 0.0, -0.3, 0.0]),
        visit_counts=np.array([2.0, 0.0, 1.0, 0.0]),
        prior_probs=np.array([0.4, 0.3, 0.2, 0.1]),
    )
    assert value == pytest.approx(0.225, abs=1e-12)


def test_mixed_value_without_visits_is_raw_network_value() -> None:
    value = mixed_value(
        raw_value=-0.37,
        q_values=np.zeros(3),
        visit_counts=np.zeros(3),
        prior_probs=np.array([0.5, 0.3, 0.2]),
    )
    assert value == -0.37


def test_mixed_value_tolerates_zero_prior_on_visited_action() -> None:
    """A visited action with zero prior must give a finite mixed value."""
    value = mixed_value(
        raw_value=0.1,
        q_values=np.array([0.6, 0.0]),
        visit_counts=np.array([3.0, 0.0]),
        prior_probs=np.array([0.0, 1.0]),
    )
    assert np.isfinite(value)
    # The sole visited action supplies weighted Q, giving (0.1 + 3*0.6)/4.
    assert value == pytest.approx(0.475, abs=1e-9)


def test_completed_q_scores_match_hand_computation() -> None:
    """Check normalization of completed values [0.5, 0.225, -0.3, 0.225]."""
    scores = completed_q_scores(
        raw_value=0.2,
        q_values=np.array([0.5, 0.0, -0.3, 0.0]),
        visit_counts=np.array([2.0, 0.0, 1.0, 0.0]),
        prior_probs=np.array([0.4, 0.3, 0.2, 0.1]),
        c_visit=50.0,
        c_scale=0.1,
    )
    np.testing.assert_allclose(scores, [5.2, 3.4125, 0.0, 3.4125], atol=1e-9)


def test_completed_q_scores_are_zero_when_nothing_distinguishes_actions() -> None:
    """Without visits all completed values coincide and normalize to zero."""
    scores = completed_q_scores(
        raw_value=0.3,
        q_values=np.zeros(5),
        visit_counts=np.zeros(5),
        prior_probs=np.full(5, 0.2),
        c_visit=50.0,
        c_scale=0.1,
    )
    np.testing.assert_array_equal(scores, np.zeros(5))


def test_completed_q_scores_grow_with_visits_and_stay_within_scale() -> None:
    """The score range grows as (c_visit + max N) * c_scale."""
    q = np.array([0.9, -0.9, 0.0])
    probs = np.full(3, 1.0 / 3.0)
    shallow = completed_q_scores(0.0, q, np.array([1.0, 1.0, 1.0]), probs, 50.0, 0.1)
    deep = completed_q_scores(0.0, q, np.array([40.0, 40.0, 40.0]), probs, 50.0, 0.1)
    assert shallow.min() == 0.0 and shallow.max() == pytest.approx(5.1)
    assert deep.min() == 0.0 and deep.max() == pytest.approx(9.0)


def test_completed_q_scores_rank_actions_like_their_values() -> None:
    """Q transformation preserves the completed values' order."""
    q = np.array([0.1, 0.8, -0.4, 0.3])
    visits = np.array([1.0, 3.0, 2.0, 1.0])
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    scores = completed_q_scores(0.0, q, visits, probs, 50.0, 0.1)
    assert list(np.argsort(scores)) == list(np.argsort(q))


# ----------------------------------------------------------------------
# Root candidate scores
# ----------------------------------------------------------------------


def test_score_considered_matches_hand_computation_and_penalises_wrong_visit_counts() -> None:
    """Shift logits by their maximum, add noise and transformed Q, and exclude ineligible visit counts."""
    scores = score_considered(
        considered_visit=1,
        gumbel=np.array([0.2, 0.5, -0.1]),
        logits=np.array([-1.0, -0.5, -2.0]),
        sigma_q=np.array([1.0, 2.0, 3.0]),
        visit_counts=np.array([1.0, 0.0, 1.0]),
    )
    assert scores[0] == pytest.approx(0.7)  # 0.2 - 0.5 + 1
    assert scores[1] == -np.inf  # This action has zero visits and the round requires one.
    assert scores[2] == pytest.approx(1.4)  # -0.1 - 1.5 + 3
    assert int(np.argmax(scores)) == 2


def test_score_considered_clamps_hopeless_candidates_to_finite_low_logit() -> None:
    """A lone eligible candidate wins even with a very low finite score."""
    scores = score_considered(
        considered_visit=0,
        gumbel=np.array([-1e12, -1e12]),
        logits=np.array([0.0, -5.0]),
        sigma_q=np.zeros(2),
        visit_counts=np.array([3.0, 0.0]),
    )
    assert scores[0] == -np.inf
    assert scores[1] == -1e9
    assert int(np.argmax(scores)) == 1


def test_softmax_is_stable_for_huge_magnitudes_and_sums_to_one() -> None:
    probs = _softmax(np.array([1000.0, 1001.0, 990.0]))
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)
    assert probs[1] == pytest.approx(1.0 / (1.0 + np.exp(-1.0) + np.exp(-11.0)))
    # Also check log priors near the numerical floor.
    tiny = _softmax(np.array([-800.0, -801.0]))
    assert tiny.sum() == pytest.approx(1.0)


def test_score_considered_is_invariant_to_logit_shift() -> None:
    """A constant shift of all logits leaves the candidate comparison unchanged."""
    kwargs = dict(
        considered_visit=0,
        gumbel=np.array([0.3, -0.2, 0.9]),
        sigma_q=np.array([0.0, 1.5, 0.5]),
        visit_counts=np.zeros(3),
    )
    base = score_considered(logits=np.array([-1.0, -2.0, -0.5]), **kwargs)
    shifted = score_considered(logits=np.array([-1.0, -2.0, -0.5]) + 123.0, **kwargs)
    np.testing.assert_allclose(base, shifted, atol=1e-9)
