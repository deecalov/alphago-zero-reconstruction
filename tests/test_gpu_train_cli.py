from __future__ import annotations

"""Verify gpu_train CLI arguments reach validated configuration without running CUDA training."""

import sys
import warnings

import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig, make_demo_config
from examples import gpu_train

# Build a tiny CPU model for the parameter count.
_TINY_NET_ARGS = ["--board", "5", "--channels", "8", "--res-blocks", "1"]
_LEAGUE_ARGS = ["--league-prob", "0.5", "--league-checkpoint", "champion.pt", "--parallel-games", "4"]


class _PipelineReached(Exception):
    """Stop after the configuration has passed validation and before training begins."""


def _build_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], *cli_args: str
) -> tuple[AlphaGoZeroConfig, str]:
    captured: dict[str, AlphaGoZeroConfig] = {}

    def fake_pipeline(config: AlphaGoZeroConfig) -> dict[str, object]:
        captured["config"] = config
        raise _PipelineReached

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu_train, "run_training_pipeline", fake_pipeline)
    monkeypatch.setattr(sys, "argv", ["gpu_train.py", *_TINY_NET_ARGS, *cli_args])
    with pytest.raises(_PipelineReached):
        gpu_train.main()
    return captured["config"], capsys.readouterr().out


def test_research_flags_are_off_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _build_config(monkeypatch, capsys)
    assert config.training.weight_decay_exclude_norm_bias is False
    assert config.training.exclude_league_from_resign_calibration is False
    assert "WD_EXCLUDE_NORM_BIAS False" in output
    assert "LEAGUE_EXCLUDED_FROM_CALIBRATION False" in output


def test_research_flags_reach_training_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _build_config(
        monkeypatch, capsys, "--wd-exclude-norm-bias", "--exclude-league-from-calibration", *_LEAGUE_ARGS
    )
    assert config.training.weight_decay_exclude_norm_bias is True
    assert config.training.exclude_league_from_resign_calibration is True
    assert "WD_EXCLUDE_NORM_BIAS True" in output
    assert "LEAGUE_EXCLUDED_FROM_CALIBRATION True" in output


def test_exclude_league_flag_without_league_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="exclude_league_from_resign_calibration"):
        _build_config(monkeypatch, capsys, "--exclude-league-from-calibration")


def test_gumbel_search_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _build_config(monkeypatch, capsys)
    assert config.search.search_algo == "puct"
    assert config.search.gumbel_max_considered == 16
    assert "SEARCH_ALGO puct" in output


def test_gumbel_flags_reach_search_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _build_config(
        monkeypatch, capsys,
        "--search-algo", "gumbel", "--gumbel-max-considered", "8", "--gumbel-scale", "0.5",
        "--gumbel-c-visit", "40", "--gumbel-c-scale", "0.2", "--sims", "32", "--eval-batch", "4",
    )
    search = config.search
    assert (search.search_algo, search.num_simulations) == ("gumbel", 32)
    assert (search.gumbel_max_considered, search.gumbel_scale) == (8, 0.5)
    assert (search.gumbel_c_visit, search.gumbel_c_scale) == (40.0, 0.2)
    assert "SEARCH_ALGO gumbel GUMBEL_M 8 GUMBEL_SCALE 0.5 GUMBEL_C_VISIT 40.0 GUMBEL_C_SCALE 0.2" in output


def test_gumbel_with_the_default_eval_batch_warns_about_schedule_distortion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Warn that the default leaf batch of 16 can distort Sequential Halving."""
    with pytest.warns(UserWarning, match=r"eval_batch_size=16\b"):
        config, _ = _build_config(monkeypatch, capsys, "--search-algo", "gumbel", "--sims", "32")
    assert config.search.eval_batch_size == 16  # Warn while preserving the explicitly configured value.


def test_b3_flags_are_off_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _build_config(monkeypatch, capsys)
    assert config.search.forced_playout_fraction == 0.0
    assert config.search.policy_target_prune_fraction == 0.0
    assert "FORCED_PLAYOUT 0.0" in output
    assert "TARGET_PRUNE 0.0" in output


def test_b3_flags_reach_search_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _build_config(
        monkeypatch, capsys,
        "--forced-playout-fraction", "0.5", "--policy-target-prune-fraction", "0.1",
    )
    assert config.search.forced_playout_fraction == 0.5
    assert config.search.policy_target_prune_fraction == 0.1
    assert "FORCED_PLAYOUT 0.5" in output
    assert "TARGET_PRUNE 0.1" in output


def test_gumbel_with_forced_playouts_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="forced playouts"):
        _build_config(
            monkeypatch, capsys,
            "--search-algo", "gumbel", "--forced-playout-fraction", "0.5",
            "--sims", "32", "--eval-batch", "4",
        )


def test_b3_fraction_of_one_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="policy_target_prune_fraction"):
        _build_config(monkeypatch, capsys, "--policy-target-prune-fraction", "1.0")


def test_gumbel_with_playout_cap_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="playout cap"):
        _build_config(
            monkeypatch, capsys,
            "--search-algo", "gumbel", "--playout-cap-prob", "0.25", "--playout-cap-cheap", "8",
        )


def test_gumbel_with_a_single_candidate_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(ValueError, match="gumbel_max_considered"):
        _build_config(monkeypatch, capsys, "--search-algo", "gumbel", "--gumbel-max-considered", "1")


def test_unknown_search_algo_is_rejected_by_the_parser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["gpu_train.py", *_TINY_NET_ARGS, "--search-algo", "alphabeta"])
    with pytest.raises(SystemExit) as raised:
        gpu_train.main()
    assert raised.value.code == 2
    assert "--search-algo" in capsys.readouterr().err


# ----------------------------------------------------------------------
# Evaluation gates, including unconditional promotion
# ----------------------------------------------------------------------
def test_eval_gate_keeps_the_binomial_default_and_prints_the_header(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _build_config(monkeypatch, capsys)
    assert config.training.evaluation_gate == "binomial"
    assert "EVAL_GATE binomial EVAL_GAMES 128 EVAL_OPENING_MOVES 6" in output


def test_eval_gate_none_reaches_the_config_and_ignores_the_game_count(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A four-game sample is too small for the binomial gate but irrelevant with the gate disabled."""
    config, output = _build_config(monkeypatch, capsys, "--eval-gate", "none", "--eval-games", "4")
    assert config.training.evaluation_gate == "none"
    assert "EVAL_GATE none EVAL_GAMES 4" in output
    with pytest.raises(ValueError, match="evaluation_games"):
        _build_config(monkeypatch, capsys, "--eval-gate", "binomial", "--eval-games", "4")


def test_unknown_eval_gate_is_rejected_by_the_parser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["gpu_train.py", *_TINY_NET_ARGS, "--eval-gate", "off"])
    with pytest.raises(SystemExit) as raised:
        gpu_train.main()
    assert raised.value.code == 2
    assert "--eval-gate" in capsys.readouterr().err


# ----------------------------------------------------------------------
# Virtual loss and leaf batching
# ----------------------------------------------------------------------
def _virtual_loss_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    """Collect only warnings about the --virtual-loss setting."""
    return [str(item.message) for item in caught if "--virtual-loss" in str(item.message)]


def test_virtual_loss_keeps_the_config_value_without_the_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _build_config(monkeypatch, capsys)
    default = make_demo_config().search.virtual_loss
    assert config.search.virtual_loss == default  # Leave the configuration value intact when the flag is omitted.
    assert f"EVAL_BATCH 16 VIRTUAL_LOSS {default}" in output


def test_virtual_loss_flag_reaches_search_config_and_the_header(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, output = _build_config(monkeypatch, capsys, "--eval-batch", "4", "--virtual-loss", "0.5")
    assert config.search.virtual_loss == 0.5
    assert config.search.eval_batch_size == 4
    assert "EVAL_BATCH 4 VIRTUAL_LOSS 0.5" in output


def test_virtual_loss_zero_is_applied_and_not_mistaken_for_an_absent_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Zero is a valid override. Check for None to distinguish an omitted flag."""
    config, output = _build_config(monkeypatch, capsys, "--eval-batch", "4", "--virtual-loss", "0")
    assert config.search.virtual_loss == 0.0
    assert "VIRTUAL_LOSS 0.0" in output


def test_virtual_loss_combines_with_the_gumbel_stage_zero_command_without_warnings(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Gumbel with leaf batch four, many games and virtual loss 0.5 should emit no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config, _ = _build_config(
            monkeypatch, capsys,
            "--search-algo", "gumbel", "--sims", "32", "--eval-batch", "4", "--parallel-games", "32",
            "--virtual-loss", "0.5",
        )
    search = config.search
    assert (search.search_algo, search.eval_batch_size, search.virtual_loss) == ("gumbel", 4, 0.5)
    about_search = [
        str(item.message) for item in caught
        if "--virtual-loss" in str(item.message) or "eval_batch_size" in str(item.message)
    ]
    assert about_search == []


def test_virtual_loss_with_batch_one_warns_that_it_has_no_effect(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.warns(UserWarning, match="--virtual-loss 0\\.5 has no effect with --eval-batch 1"):
        config, _ = _build_config(monkeypatch, capsys, "--eval-batch", "1", "--virtual-loss", "0.5")
    assert config.search.virtual_loss == 0.5  # Warn while preserving the explicitly configured value.


@pytest.mark.parametrize(
    "cli_args",
    [
        pytest.param(("--eval-batch", "1"), id="batch-one-without-the-flag"),
        pytest.param(("--eval-batch", "2", "--virtual-loss", "0.5"), id="batch-two-with-the-flag"),
        pytest.param(("--eval-batch", "16", "--virtual-loss", "0"), id="batch-sixteen-with-zero"),
    ],
)
def test_virtual_loss_warning_is_not_raised_when_it_would_be_noise(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli_args: tuple[str, ...]
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build_config(monkeypatch, capsys, *cli_args)
    assert _virtual_loss_warnings(caught) == []


def test_help_renders_and_states_the_virtual_loss_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Generate the complete help text to catch stray percent escapes and verify the configuration default."""
    monkeypatch.setattr(sys, "argv", ["gpu_train.py", "--help"])
    with pytest.raises(SystemExit) as raised:
        gpu_train.main()
    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())  # Normalize argparse's line wrapping.
    assert "--virtual-loss" in help_text
    assert f"default {make_demo_config().search.virtual_loss})" in help_text


@pytest.mark.parametrize("value", ["-0.5", "nan", "inf", "-inf", "abc", ""])
def test_virtual_loss_rejects_invalid_values_in_the_parser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    """Reject bad arguments before CUDA checks. Pass negative values with '='."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["gpu_train.py", *_TINY_NET_ARGS, f"--virtual-loss={value}"])
    with pytest.raises(SystemExit) as raised:
        gpu_train.main()
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "--virtual-loss" in error and "expected" in error


# ----------------------------------------------------------------------
# Exercise numeric training, search and anchor arguments.
# ----------------------------------------------------------------------
# Use '=' so -inf and -1e-3 stay attached to the option.
_INVALID_NUMERIC_FLAG_VALUES = [
    pytest.param(flag, value, id=f"{flag}={value}")
    for flag, value in [
        ("--komi", "nan"), ("--komi", "inf"), ("--komi", "-1"),
        ("--lr", "nan"), ("--lr", "inf"), ("--lr", "-1e-3"),
        ("--fpu", "nan"), ("--fpu", "inf"), ("--fpu", "-0.1"),
        ("--ownership-weight", "nan"), ("--ownership-weight", "inf"), ("--ownership-weight", "-0.1"),
        ("--score-weight", "nan"), ("--score-weight", "inf"), ("--score-weight", "-0.1"),
        ("--anchor-rating", "nan"), ("--anchor-rating", "inf"), ("--anchor-rating", "-inf"),
        ("--max-hours", "nan"), ("--max-hours", "inf"), ("--max-hours", "0"), ("--max-hours", "-1"),
    ]
]


@pytest.mark.parametrize(("flag", "value"), _INVALID_NUMERIC_FLAG_VALUES)
def test_numeric_flags_reject_invalid_values_in_the_parser(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flag: str, value: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", ["gpu_train.py", *_TINY_NET_ARGS, f"{flag}={value}"])
    with pytest.raises(SystemExit) as raised:
        gpu_train.main()
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert flag in error and "expected" in error


def test_numeric_flags_keep_the_defaults_without_the_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _ = _build_config(monkeypatch, capsys)
    demo = make_demo_config()
    assert config.game.komi == demo.game.komi  # Leave the configuration value intact when the flag is omitted.
    assert config.training.learning_rate == demo.training.learning_rate
    assert config.search.fpu_reduction is None
    assert (config.training.ownership_loss_weight, config.training.score_loss_weight) == (0.15, 0.05)
    assert config.training.benchmark_anchor_rating == 0.0
    assert config.runtime.time_budget_sec is None


def test_numeric_flags_reach_the_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config, _ = _build_config(
        monkeypatch, capsys,
        "--komi", "7.5", "--lr", "1e-4", "--fpu", "0.25", "--ownership-weight", "0.2",
        "--score-weight", "0.1", "--anchor-rating", "-50", "--max-hours", "8",
    )
    assert config.game.komi == 7.5
    assert config.training.learning_rate == 1e-4
    assert config.search.fpu_reduction == 0.25
    assert (config.training.ownership_loss_weight, config.training.score_loss_weight) == (0.2, 0.1)
    assert config.training.benchmark_anchor_rating == -50.0  # A relative Elo rating may have either sign.
    assert config.runtime.time_budget_sec == 8 * 3600.0  # The CLI uses hours and the configuration stores seconds.


def test_numeric_flags_accept_zero_as_a_real_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preserve explicit zero for komi, learning rate, weights, FPU and rating."""
    config, _ = _build_config(
        monkeypatch, capsys,
        "--komi", "0", "--lr", "0", "--fpu", "0", "--ownership-weight", "0", "--score-weight", "0",
        "--anchor-rating", "0",
    )
    assert config.game.komi == 0.0
    assert config.training.learning_rate == 0.0
    assert config.search.fpu_reduction == 0.0  # FPU zero uses the parent value at an unvisited edge.
    assert (config.training.ownership_loss_weight, config.training.score_loss_weight) == (0.0, 0.0)
    assert config.training.benchmark_anchor_rating == 0.0


@pytest.mark.parametrize("text", ["none", "NONE"])
def test_fpu_flag_accepts_none_as_the_classic_q_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], text: str
) -> None:
    config, _ = _build_config(monkeypatch, capsys, "--fpu", text)
    assert config.search.fpu_reduction is None


def test_max_hours_overflowing_to_infinity_is_rejected_by_validate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A finite 1e306-hour input overflows when converted to seconds."""
    with pytest.raises(ValueError, match="time_budget_sec"):
        _build_config(monkeypatch, capsys, "--max-hours", "1e306")
