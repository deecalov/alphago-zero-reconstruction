"""Per-side search overrides, result reporting and sequential stopping in the match CLI."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alphago_zero.config import AlphaGoZeroConfig, SearchConfig
from alphago_zero.game import BLACK, WHITE
from alphago_zero.network import AlphaGoZeroNet
from examples import eval_match

# The checkpoints fixture supplies two small models with different search settings.


def _run_match_searches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: Path | list[Path],
    opponent: Path,
    extra_args: list[str],
    model_wins: Callable[[int], bool],
) -> tuple[dict[str, Any], list[tuple[SearchConfig, SearchConfig]]]:
    """Run eval_match.main with scripted model outcomes and capture the search configurations."""
    loaded: list[object] = []
    real_load = eval_match.load_model_group

    def spy(paths: Any, device: Any, sims_override: Any = None) -> Any:
        group = real_load(paths, device, sims_override)
        loaded.append(group[0])  # The evaluated model loads first.
        return group

    searches: list[tuple[SearchConfig, SearchConfig]] = []

    def scripted_game(*, black_model: Any, black_search: Any, white_search: Any, **_ignored: Any) -> int:
        searches.append((black_search, white_search))
        model_is_black = black_model is loaded[0]
        return BLACK if model_wins(len(searches) - 1) == model_is_black else WHITE

    monkeypatch.setattr(eval_match, "load_model_group", spy)
    monkeypatch.setattr(eval_match, "play_game", scripted_game)
    out = tmp_path / "match.json"
    models = [model] if isinstance(model, Path) else model
    argv = ["eval_match.py", "--model", *map(str, models), "--opponent", str(opponent)]
    monkeypatch.setattr(sys, "argv", [*argv, "--device", "cpu", "--json-out", str(out), *extra_args])
    eval_match.main()
    return json.loads(out.read_text(encoding="utf-8")), searches


def _run_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model: Path,
    opponent: Path,
    extra_args: list[str],
    model_wins: Callable[[int], bool],
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    """Collect Black and White simulation budgets for each game."""
    result, searches = _run_match_searches(monkeypatch, tmp_path, model, opponent, extra_args, model_wins)
    return result, [(black.num_simulations, white.num_simulations) for black, white in searches]


# ----------------------------------------------------------------------
# Per-side simulation budgets
# ----------------------------------------------------------------------
def test_per_side_simulations_reach_the_right_side(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    args = ["--games", "2", "--model-simulations", "3", "--opponent-simulations", "5"]
    result, sims = _run_match(monkeypatch, tmp_path, a, b, args, lambda i: True)
    assert sims == [(3, 5), (5, 3)]  # The model plays Black in game zero and White in game one.
    assert (result["model_sims"], result["opponent_sims"]) == (3, 5)
    assert (result["model_sims_forced"], result["opponent_sims_forced"]) == (3, 5)
    assert result["sims_forced"] is None


@pytest.mark.parametrize(
    ("flags", "joint", "model_sims", "opponent_sims"),
    [
        pytest.param(["--simulations", "4", "--opponent-simulations", "9"], 4, 4, 9, id="flag-of-opponent"),
        pytest.param(["--simulations", "4", "--model-simulations", "9"], 4, 9, 4, id="flag-of-model"),
        pytest.param(
            ["--simulations", "7", "--model-simulations", "3", "--opponent-simulations", "5"], 7, 3, 5, id="both-flags"
        ),
    ],
)
def test_per_side_flag_wins_over_the_joint_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoints: tuple[Path, Path],
    flags: list[str],
    joint: int,
    model_sims: int,
    opponent_sims: int,
) -> None:
    a, b = checkpoints
    result, sims = _run_match(monkeypatch, tmp_path, a, b, ["--games", "2", *flags], lambda i: True)
    assert sims == [(model_sims, opponent_sims), (opponent_sims, model_sims)]
    assert (result["model_sims"], result["opponent_sims"]) == (model_sims, opponent_sims)
    assert result["sims_forced"] == joint
    assert (result["model_sims_forced"], result["opponent_sims_forced"]) == (model_sims, opponent_sims)


def test_joint_flag_still_aligns_both_sides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    result, sims = _run_match(monkeypatch, tmp_path, a, b, ["--games", "2", "--simulations", "4"], lambda i: True)
    assert sims == [(4, 4), (4, 4)]
    assert (result["sims_forced"], result["model_sims_forced"], result["opponent_sims_forced"]) == (4, 4, 4)


def test_without_overrides_each_side_keeps_its_saved_simulations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints  # a.pt saves six simulations and b.pt saves ten.
    result, sims = _run_match(monkeypatch, tmp_path, b, a, ["--games", "2"], lambda i: True)
    assert sims == [(10, 6), (6, 10)]
    assert (result["model_sims"], result["opponent_sims"]) == (10, 6)
    assert result["sims_forced"] is None
    assert result["model_sims_forced"] is None and result["opponent_sims_forced"] is None


@pytest.mark.parametrize("flag", ["--simulations", "--model-simulations", "--opponent-simulations"])
@pytest.mark.parametrize("value", ["0", "-3", "abc"])
def test_invalid_simulation_counts_are_rejected(
    monkeypatch: pytest.MonkeyPatch, checkpoints: tuple[Path, Path], flag: str, value: str
) -> None:
    a, b = checkpoints
    monkeypatch.setattr(sys, "argv", ["eval_match.py", "--model", str(a), "--opponent", str(b), flag, value])
    with pytest.raises(SystemExit) as exit_info:
        eval_match.main()
    assert exit_info.value.code == 2


# ----------------------------------------------------------------------
# Sequential stopping at Wald boundaries
# ----------------------------------------------------------------------
def test_sprt_accepts_h1_and_stops_after_a_complete_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    result, sims = _run_match(monkeypatch, tmp_path, a, b, ["--games", "120", "--sprt"], lambda i: True)
    # The 31st consecutive win crosses the boundary. The pair completes on game 32.
    assert result["games"] == 32 and len(sims) == 32
    assert (result["wins"], result["losses"]) == (32, 0)
    sprt = result["sprt"]
    assert sprt["decision"] == "H1"
    assert (sprt["games_played"], sprt["max_games"]) == (32, 120)
    assert (sprt["wr0"], sprt["wr1"], sprt["alpha"], sprt["beta"]) == (0.5, 0.55, 0.05, 0.05)
    assert sprt["llr"] == pytest.approx(32 * 0.09531018, abs=1e-3)
    assert sprt["upper_bound"] == pytest.approx(2.944439, abs=1e-4)
    assert sprt["lower_bound"] == pytest.approx(-2.944439, abs=1e-4)


def test_sprt_accepts_h0_after_28_straight_losses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    result, _ = _run_match(monkeypatch, tmp_path, a, b, ["--games", "120", "--sprt"], lambda i: False)
    assert result["games"] == 28 and (result["wins"], result["losses"]) == (0, 28)
    assert result["sprt"]["decision"] == "H0"


def test_sprt_reports_inconclusive_when_the_maximum_is_reached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    result, sims = _run_match(
        monkeypatch, tmp_path, a, b, ["--games", "40", "--sprt"], lambda i: i % 2 == 0
    )
    assert result["games"] == 40 and len(sims) == 40 and result["wins"] == 20
    assert result["sprt"]["decision"] == "inconclusive"
    assert result["sprt"]["games_played"] == 40


def test_sprt_checks_the_last_game_even_when_the_maximum_is_odd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    result, _ = _run_match(monkeypatch, tmp_path, a, b, ["--games", "31", "--sprt"], lambda i: True)
    assert result["games"] == 31
    assert result["sprt"]["decision"] == "H1"  # After 31 wins, LLR 2.955 exceeds 2.944.


@pytest.mark.parametrize(
    ("sprt_args", "model_wins", "expected_games", "expected_decision"),
    [
        (["--sprt-alpha", "0.01", "--sprt-beta", "0.30"], True, 46, "H1"),  # Upper boundary ln(70) requires 45 wins.
        (["--sprt-alpha", "0.01", "--sprt-beta", "0.30"], False, 12, "H0"),  # Lower boundary ln(0.3/0.99) requires 12 losses.
        (["--sprt-wr1", "0.6"], True, 18, "H1"),  # A log increment of ln(1.2) crosses at 17 wins. Check after the 18th game.
        (["--sprt-wr0", "0.45"], True, 16, "H1"),  # An increment of ln(0.55/0.45) crosses at 15 wins. Check after game 16.
    ],
)
def test_sprt_parameters_are_passed_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoints: tuple[Path, Path],
    sprt_args: list[str],
    model_wins: bool,
    expected_games: int,
    expected_decision: str,
) -> None:
    a, b = checkpoints
    args = ["--games", "120", "--sprt", *sprt_args]
    result, _ = _run_match(monkeypatch, tmp_path, a, b, args, lambda i: model_wins)
    assert result["games"] == expected_games
    assert result["sprt"]["decision"] == expected_decision


def test_sprt_json_records_the_chosen_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    args = ["--games", "120", "--sprt", "--sprt-alpha", "0.01", "--sprt-beta", "0.30", "--sprt-wr1", "0.6"]
    result, _ = _run_match(monkeypatch, tmp_path, a, b, args, lambda i: True)
    sprt = result["sprt"]
    assert (sprt["wr0"], sprt["wr1"], sprt["alpha"], sprt["beta"]) == (0.5, 0.6, 0.01, 0.30)


def test_without_sprt_every_game_is_played_and_there_is_no_sprt_section(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    result, sims = _run_match(monkeypatch, tmp_path, a, b, ["--games", "40"], lambda i: True)
    assert result["games"] == 40 and len(sims) == 40
    assert "sprt" not in result


def test_statistics_describe_the_games_actually_played(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    result, _ = _run_match(monkeypatch, tmp_path, a, b, ["--games", "120", "--sprt"], lambda i: True)
    assert result["win_rate"] == 1.0
    assert result["as_black_games"] == 16
    assert result["as_black_win_rate"] == 1.0 and result["as_white_win_rate"] == 1.0
    # The Wilson lower bound uses the 32 completed games, giving 0.893.
    assert 0.85 < result["win_rate_ci95"][0] < 0.93


# ----------------------------------------------------------------------
# SPRT arguments and messages
# ----------------------------------------------------------------------
def _argv(checkpoints: tuple[Path, Path], *extra: str) -> list[str]:
    a, b = checkpoints
    return ["eval_match.py", "--model", str(a), "--opponent", str(b), "--device", "cpu", *extra]


def _error_message(capsys: pytest.CaptureFixture[str]) -> str:
    """Use argparse's final error line so the options in usage text cannot satisfy assertions."""
    return capsys.readouterr().err.strip().splitlines()[-1]


@pytest.mark.parametrize("option", ["--sprt-wr0", "--sprt-wr1", "--sprt-alpha", "--sprt-beta"])
def test_sprt_options_require_the_sprt_flag(
    monkeypatch: pytest.MonkeyPatch,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    option: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(checkpoints, option, "0.4"))
    with pytest.raises(SystemExit) as exit_info:
        eval_match.main()
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert option in message
    assert "require --sprt" in message


def test_every_sprt_option_given_without_sprt_is_named_in_the_error(
    monkeypatch: pytest.MonkeyPatch, checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(checkpoints, "--sprt-wr0", "0.4", "--sprt-alpha", "0.1"))
    with pytest.raises(SystemExit) as exit_info:
        eval_match.main()
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert "--sprt-wr0" in message and "--sprt-alpha" in message
    assert "--sprt-wr1" not in message and "--sprt-beta" not in message


def test_invalid_sprt_hypotheses_are_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(checkpoints, "--sprt", "--sprt-wr0", "0.6", "--sprt-wr1", "0.5"))
    with pytest.raises(SystemExit) as exit_info:
        eval_match.main()
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert "0 < wr0 < wr1 < 1" in message
    assert "wr0=0.6" in message and "wr1=0.5" in message


@pytest.mark.parametrize("option", ["--sprt-wr0", "--sprt-wr1", "--sprt-alpha", "--sprt-beta"])
@pytest.mark.parametrize("value", ["0", "-0.2", "nan", "inf", "-inf", "abc"])
def test_invalid_sprt_numbers_are_rejected_at_the_cli_boundary(
    monkeypatch: pytest.MonkeyPatch,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(checkpoints, "--sprt", f"{option}={value}"))
    with pytest.raises(SystemExit) as exit_info:
        eval_match.main()
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert option in message and "expected" in message


@pytest.mark.parametrize("option", ["--sprt-wr0", "--sprt-wr1", "--sprt-alpha", "--sprt-beta"])
@pytest.mark.parametrize("value", ["1.0", "1.25"])
def test_sprt_probabilities_at_or_above_one_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    option: str,
    value: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(checkpoints, "--sprt", f"{option}={value}"))
    with pytest.raises(SystemExit) as exit_info:
        eval_match.main()
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert value in message


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "abc"])
def test_invalid_opponent_rating_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(checkpoints, f"--opponent-rating={value}"))
    with pytest.raises(SystemExit) as exit_info:
        eval_match.main()
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert "--opponent-rating" in message and "expected" in message


@pytest.mark.parametrize("rating", [-250.0, 0.0, 1727.5])
def test_finite_opponent_rating_of_any_sign_is_accepted_and_shifts_elo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoints: tuple[Path, Path],
    rating: float,
) -> None:
    a, b = checkpoints
    result, _ = _run_match(
        monkeypatch, tmp_path, a, b, ["--games", "2", f"--opponent-rating={rating}"], lambda i: i == 0
    )
    assert result["opponent_rating"] == rating
    assert result["elo_estimate"] == pytest.approx(rating, abs=0.1)  # An even score gives an Elo difference of zero.


def test_console_reports_the_sprt_verdict_and_the_expected_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    a, b = checkpoints
    _run_match(monkeypatch, tmp_path, a, b, ["--games", "120", "--sprt"], lambda i: True)
    output = capsys.readouterr().out
    assert "SPRT" in output and "accepted H1" in output
    # Warn when the 120-game cap is below the expected test length of about 527.
    assert "expected test length" in output


def test_no_length_warning_when_the_maximum_is_large_enough(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    a, b = checkpoints
    _run_match(monkeypatch, tmp_path, a, b, ["--games", "2000", "--sprt"], lambda i: True)
    assert "expected test length" not in capsys.readouterr().out


# ----------------------------------------------------------------------
# Per-side c_puct and FPU on fixed weights
# ----------------------------------------------------------------------
def test_without_search_overrides_each_side_keeps_its_saved_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints  # a.pt: c_puct 1.5, FPU None; b.pt: c_puct 1.25, FPU 0.2
    result, searches = _run_match_searches(monkeypatch, tmp_path, a, b, ["--games", "2"], lambda i: True)
    model_side, opponent_side = searches[0]  # The model plays Black first.
    assert (model_side.c_puct, model_side.fpu_reduction) == (1.5, None)
    assert (opponent_side.c_puct, opponent_side.fpu_reduction) == (1.25, 0.2)
    assert (result["model_c_puct"], result["model_fpu"]) == (1.5, None)
    assert (result["opponent_c_puct"], result["opponent_fpu"]) == (1.25, 0.2)


def test_model_with_a_saved_fpu_keeps_it_when_the_flag_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    """An omitted override must preserve the model's saved FPU reduction of 0.2."""
    a, b = checkpoints
    result, searches = _run_match_searches(monkeypatch, tmp_path, b, a, ["--games", "2"], lambda i: True)
    model_side, opponent_side = searches[0]
    assert (model_side.c_puct, model_side.fpu_reduction) == (1.25, 0.2)
    assert (opponent_side.c_puct, opponent_side.fpu_reduction) == (1.5, None)
    assert (result["model_c_puct"], result["model_fpu"]) == (1.25, 0.2)


def test_search_overrides_reach_the_right_side_in_both_colours(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    flags = ["--model-c-puct", "2.5", "--model-fpu", "0.3", "--opponent-c-puct", "1.1", "--opponent-fpu", "0.1"]
    result, searches = _run_match_searches(monkeypatch, tmp_path, a, b, ["--games", "2", *flags], lambda i: True)
    model_black, opponent_white = searches[0]
    opponent_black, model_white = searches[1]  # Swap colors without swapping search configurations.
    assert (model_black.c_puct, model_black.fpu_reduction) == (2.5, 0.3)
    assert (model_white.c_puct, model_white.fpu_reduction) == (2.5, 0.3)
    assert (opponent_white.c_puct, opponent_white.fpu_reduction) == (1.1, 0.1)
    assert (opponent_black.c_puct, opponent_black.fpu_reduction) == (1.1, 0.1)
    assert (result["model_c_puct"], result["model_fpu"]) == (2.5, 0.3)
    assert (result["opponent_c_puct"], result["opponent_fpu"]) == (1.1, 0.1)


def test_overriding_one_parameter_keeps_the_other_saved_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    flags = ["--opponent-c-puct", "2.0", "--model-fpu", "0.4"]
    result, _ = _run_match_searches(monkeypatch, tmp_path, a, b, ["--games", "2", *flags], lambda i: True)
    assert (result["model_c_puct"], result["model_fpu"]) == (1.5, 0.4)  # Use the model checkpoint's exploration constant.
    assert (result["opponent_c_puct"], result["opponent_fpu"]) == (2.0, 0.2)  # Use the opponent checkpoint's FPU.


@pytest.mark.parametrize("spelling", ["none", "NONE", "None"])
def test_fpu_none_overrides_a_saved_fpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path], spelling: str
) -> None:
    a, b = checkpoints  # b.pt has saved FPU reduction 0.2.
    result, searches = _run_match_searches(
        monkeypatch, tmp_path, a, b, ["--games", "2", "--opponent-fpu", spelling], lambda i: True
    )
    assert (result["opponent_c_puct"], result["opponent_fpu"]) == (1.25, None)
    assert (result["model_c_puct"], result["model_fpu"]) == (1.5, None)
    assert searches[0][1].fpu_reduction is None


def test_search_overrides_do_not_touch_the_simulation_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints  # a.pt saves six simulations and b.pt saves ten.
    flags = ["--model-c-puct", "2.0", "--opponent-fpu", "0.4"]
    result, sims = _run_match(monkeypatch, tmp_path, a, b, ["--games", "2", *flags], lambda i: True)
    assert sims == [(6, 10), (10, 6)]
    assert (result["model_sims"], result["opponent_sims"]) == (6, 10)


def test_search_override_applies_to_an_ensemble_through_its_first_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    result, searches = _run_match_searches(
        monkeypatch, tmp_path, [a, b], a, ["--games", "2", "--model-c-puct", "2.0"], lambda i: True
    )
    assert result["model_members"] == [str(a), str(b)]
    assert (result["model_c_puct"], result["model_fpu"]) == (2.0, None)  # Use the first member's config plus the explicit override.
    assert searches[0][0].c_puct == 2.0
    assert (result["opponent_c_puct"], result["opponent_fpu"]) == (1.5, None)


@pytest.mark.parametrize("flag", ["--model-c-puct", "--opponent-c-puct"])
@pytest.mark.parametrize("value", ["0", "-1.5", "abc", "nan", "inf"])
def test_invalid_c_puct_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(checkpoints, flag, value))
    with pytest.raises(SystemExit) as exit_info:
        eval_match.main()
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    # Distinguish a rejected value from an unknown option, which also exits with code two.
    assert flag in message and "expected" in message


@pytest.mark.parametrize("flag", ["--model-fpu", "--opponent-fpu"])
@pytest.mark.parametrize("value", ["-0.1", "abc", "nan", "inf"])
def test_invalid_fpu_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _argv(checkpoints, flag, value))
    with pytest.raises(SystemExit) as exit_info:
        eval_match.main()
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert flag in message and "expected" in message


def test_zero_fpu_is_a_valid_reduction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints  # FPU zero assigns parent value to an unvisited edge. None uses zero Q.
    result, _ = _run_match_searches(
        monkeypatch, tmp_path, a, b, ["--games", "2", "--model-fpu", "0"], lambda i: True
    )
    assert result["model_fpu"] == 0.0


def test_main_accepts_an_explicit_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    monkeypatch.setattr(sys, "argv", ["eval_match.py", "--no-such-flag"])  # Explicit arguments bypass sys.argv.
    monkeypatch.setattr(eval_match, "play_game", lambda **_kwargs: BLACK)
    out = tmp_path / "explicit.json"
    explicit = ["--model", str(a), "--opponent", str(b), "--games", "2", "--device", "cpu", "--json-out", str(out)]
    eval_match.main(explicit)
    assert json.loads(out.read_text(encoding="utf-8"))["games"] == 2


# ----------------------------------------------------------------------
# Result output
# ----------------------------------------------------------------------
def test_json_result_is_written_in_utf8_whatever_the_locale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    """Write non-ASCII filenames to JSON as UTF-8 regardless of the host's default encoding."""
    a, b = checkpoints
    model = tmp_path / "model_🪨.pt"  # The filename includes characters outside common Windows code pages.
    shutil.copy(a, model)
    result, _ = _run_match_searches(monkeypatch, tmp_path, model, b, ["--games", "2"], lambda i: True)
    assert result["model"] == str(model)
    assert result["model_members"] == [str(model)]


def _two_games_argv(checkpoints: tuple[Path, Path], json_out: Path) -> list[str]:
    a, b = checkpoints
    return ["--model", str(a), "--opponent", str(b), "--games", "2", "--device", "cpu", "--json-out", str(json_out)]


def test_json_out_creates_the_missing_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    """Create missing output directory parents before saving a match result."""
    monkeypatch.setattr(eval_match, "play_game", lambda **_kwargs: BLACK)
    out = tmp_path / "new" / "nested" / "result.json"
    eval_match.main(_two_games_argv(checkpoints, out))
    assert json.loads(out.read_text(encoding="utf-8"))["games"] == 2


def test_output_directory_exists_before_the_first_game(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    """Create the output directory before spending time on match play."""
    out = tmp_path / "late" / "result.json"
    seen: list[bool] = []

    def scripted_game(**_kwargs: Any) -> int:
        seen.append(out.parent.is_dir())
        return BLACK

    monkeypatch.setattr(eval_match, "play_game", scripted_game)
    eval_match.main(_two_games_argv(checkpoints, out))
    assert seen == [True, True]


def test_unusable_output_location_is_a_usage_error_before_any_game(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report a file in place of the output directory as an immediate usage error."""
    blocker = tmp_path / "blocker"
    blocker.write_text("file in place of directory", encoding="utf-8")

    def forbidden(**_kwargs: Any) -> int:
        raise AssertionError("games must not start")

    monkeypatch.setattr(eval_match, "play_game", forbidden)
    with pytest.raises(SystemExit) as excinfo:
        eval_match.main(_two_games_argv(checkpoints, blocker / "result.json"))
    assert excinfo.value.code == 2
    message = _error_message(capsys)
    assert "--json-out" in message and "blocker" in message


def test_match_without_json_out_prints_the_result_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitting --json-out writes the report to the console only."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(eval_match, "play_game", lambda **_kwargs: BLACK)
    a, b = checkpoints
    eval_match.main(["--model", str(a), "--opponent", str(b), "--games", "2", "--device", "cpu"])
    assert '"games": 2' in capsys.readouterr().out
    assert list(work.iterdir()) == []


def test_json_out_without_a_directory_part_still_works(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    """Accept a simple filename in the current directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(eval_match, "play_game", lambda **_kwargs: BLACK)
    eval_match.main(_two_games_argv(checkpoints, Path("bare.json")))
    assert json.loads((tmp_path / "bare.json").read_text(encoding="utf-8"))["games"] == 2


# ----------------------------------------------------------------------
# Search-tree reuse
# ----------------------------------------------------------------------
def test_tree_reuse_flags_follow_the_models_color(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    seen: list[dict[str, Any]] = []

    def scripted_game(**kwargs: Any) -> int:
        seen.append(kwargs)
        return BLACK

    monkeypatch.setattr(eval_match, "play_game", scripted_game)
    out = tmp_path / "match.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_match.py",
            "--model", str(a),
            "--opponent", str(b),
            "--games", "2",
            "--model-tree-reuse",
            "--device", "cpu",
            "--json-out", str(out),
        ],
    )
    eval_match.main()
    result = json.loads(out.read_text(encoding="utf-8"))
    # The model keeps reuse enabled after alternating colors.
    assert (seen[0]["black_tree_reuse"], seen[0]["white_tree_reuse"]) == (True, False)
    assert (seen[1]["black_tree_reuse"], seen[1]["white_tree_reuse"]) == (False, True)
    assert result["model_tree_reuse"] is True
    assert result["opponent_tree_reuse"] is False


def test_play_game_with_tree_reuse_completes(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    """Complete a real game with tree reuse on both sides."""
    winner = eval_match.play_game(
        black_model=tiny_model,
        black_search=tiny_config.search,
        white_model=tiny_model,
        white_search=tiny_config.search,
        game=tiny_config.game,
        rng=np.random.default_rng(5),
        opening_moves=2,
        black_tree_reuse=True,
        white_tree_reuse=True,
    )
    assert winner in (BLACK, WHITE, None)


# ----------------------------------------------------------------------
# Time budgeting
# ----------------------------------------------------------------------
def test_time_budget_flags_follow_the_models_color(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    seen: list[dict[str, Any]] = []

    def scripted_game(**kwargs: Any) -> int:
        seen.append(kwargs)
        return WHITE

    monkeypatch.setattr(eval_match, "play_game", scripted_game)
    out = tmp_path / "match.json"
    eval_match.main(
        [
            "--model", str(a),
            "--opponent", str(b),
            "--games", "2",
            "--model-time-budget", "0.25",
            "--device", "cpu",
            "--json-out", str(out),
        ]
    )
    result = json.loads(out.read_text(encoding="utf-8"))
    # The time budget follows the model when colors swap.
    assert (seen[0]["black_time_budget"], seen[0]["white_time_budget"]) == (0.25, None)
    assert (seen[1]["black_time_budget"], seen[1]["white_time_budget"]) == (None, 0.25)
    assert result["model_time_budget"] == 0.25
    assert result["opponent_time_budget"] is None
    # A scripted game reports zero mean simulations when it supplies no search statistics.
    assert result["model_avg_sims"] == 0.0


def test_play_game_time_budget_limits_simulations(
    tiny_model: AlphaGoZeroNet, tiny_config: AlphaGoZeroConfig
) -> None:
    """A real game with zero time budget completes one evaluation batch per move."""
    search = replace(tiny_config.search, num_simulations=64, eval_batch_size=4)
    stats: dict[str, float] = {}
    winner = eval_match.play_game(
        black_model=tiny_model,
        black_search=search,
        white_model=tiny_model,
        white_search=search,
        game=tiny_config.game,
        rng=np.random.default_rng(5),
        opening_moves=0,
        black_time_budget=0.0,
        white_time_budget=0.0,
        stats=stats,
    )
    assert winner in (BLACK, WHITE, None)
    for side in ("black", "white"):
        moves = stats[f"{side}_moves"]
        sims = stats[f"{side}_sims"]
        assert moves > 0
        assert sims == moves * 4  # One leaf batch per move instead of 64 simulations.
