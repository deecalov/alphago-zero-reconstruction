from __future__ import annotations

"""Experiment configuration validation tests."""

from dataclasses import fields

import pytest

from alphago_zero.config import make_demo_config, make_paper_like_config


def test_demo_config_is_valid() -> None:
    make_demo_config().validate()  # A valid configuration must pass.


def test_paper_like_config_is_valid() -> None:
    make_paper_like_config().validate()


def test_board_size_mismatch_raises() -> None:
    cfg = make_demo_config()
    cfg.network.board_size = 7  # Deliberately mismatch game.board_size=5.
    with pytest.raises(ValueError):
        cfg.validate()


def test_num_simulations_must_be_positive() -> None:
    cfg = make_demo_config()
    cfg.search.num_simulations = 0
    with pytest.raises(ValueError):
        cfg.validate()


def test_resignation_window_must_be_positive() -> None:
    cfg = make_demo_config()
    cfg.search.resignation_window = 0
    with pytest.raises(ValueError):
        cfg.validate()


def test_resign_min_move_must_be_non_negative() -> None:
    cfg = make_demo_config()
    cfg.search.resign_min_move = -1
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_resign_score_margin_must_be_none_or_non_negative_finite(bad: float) -> None:
    cfg = make_demo_config()
    cfg.search.resign_score_margin = bad
    with pytest.raises(ValueError):
        cfg.validate()


def test_resign_live_game_guards_accept_valid_values() -> None:
    cfg = make_demo_config()
    cfg.search.resign_min_move = 20
    cfg.search.resign_score_margin = 12.5
    cfg.validate()  # A valid configuration must pass.


def test_num_input_planes_must_be_17() -> None:
    cfg = make_demo_config()
    cfg.network.num_input_planes = 16
    with pytest.raises(ValueError):
        cfg.validate()


def test_batch_size_must_be_positive() -> None:
    cfg = make_demo_config()
    cfg.training.batch_size = 0
    with pytest.raises(ValueError):
        cfg.validate()


def test_replace_threshold_out_of_range_raises() -> None:
    cfg = make_demo_config()
    cfg.training.replace_threshold = 1.5
    with pytest.raises(ValueError):
        cfg.validate()


def test_dirichlet_epsilon_out_of_range_raises() -> None:
    cfg = make_demo_config()
    cfg.search.root_dirichlet_epsilon = 1.5
    with pytest.raises(ValueError):
        cfg.validate()


def test_resign_threshold_out_of_range_raises() -> None:
    cfg = make_demo_config()
    cfg.search.resign_threshold = 0.5  # The valid threshold range is [-1, 0].
    with pytest.raises(ValueError):
        cfg.validate()


def test_negative_komi_raises() -> None:
    cfg = make_demo_config()
    cfg.game.komi = -1.0
    with pytest.raises(ValueError):
        cfg.validate()


def test_eval_batch_size_must_be_at_least_one() -> None:
    cfg = make_demo_config()
    cfg.search.eval_batch_size = 0
    with pytest.raises(ValueError, match="eval_batch_size"):
        cfg.validate()


def test_virtual_loss_must_be_non_negative() -> None:
    cfg = make_demo_config()
    cfg.search.virtual_loss = -0.5
    with pytest.raises(ValueError, match="virtual_loss"):
        cfg.validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_virtual_loss_must_be_finite(value: float) -> None:
    # Check finiteness explicitly because NaN can evade a lower-bound test.
    cfg = make_demo_config()
    cfg.search.virtual_loss = value
    with pytest.raises(ValueError, match="virtual_loss"):
        cfg.validate()


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 3.0])
def test_virtual_loss_accepts_zero_and_positive_values(value: float) -> None:
    cfg = make_demo_config()
    cfg.search.virtual_loss = value
    cfg.validate()  # Zero disables the penalty and is valid.


def test_checkpoint_every_must_be_non_negative() -> None:
    cfg = make_demo_config()
    cfg.training.checkpoint_every = -1
    with pytest.raises(ValueError, match="checkpoint_every"):
        cfg.validate()


def test_time_budget_must_be_positive_or_none() -> None:
    cfg = make_demo_config()
    cfg.runtime.time_budget_sec = 0.0
    with pytest.raises(ValueError, match="time_budget_sec"):
        cfg.validate()
    cfg.runtime.time_budget_sec = None
    cfg.validate()


# ----------------------------------------------------------------------
# Exercise numeric fields exposed through gpu_train as well as
# (c_puct, root_dirichlet_alpha, deterministic_temperature, target_false_positive_rate,
# optimizer and resignation settings. NaN evades simple lower-bound comparisons.
# ----------------------------------------------------------------------
_FINITE_NON_NEGATIVE_FIELDS = [
    pytest.param("game", "komi", id="komi"),
    pytest.param("search", "fpu_reduction", id="fpu_reduction"),
    pytest.param("search", "deterministic_temperature", id="deterministic_temperature"),
    pytest.param("search", "resignation_adjustment", id="resignation_adjustment"),
    pytest.param("training", "learning_rate", id="learning_rate"),
    pytest.param("training", "momentum", id="momentum"),
    pytest.param("training", "weight_decay", id="weight_decay"),
    pytest.param("training", "ownership_loss_weight", id="ownership_loss_weight"),
    pytest.param("training", "score_loss_weight", id="score_loss_weight"),
]


@pytest.mark.parametrize(("section", "field"), _FINITE_NON_NEGATIVE_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.5])
def test_finite_non_negative_fields_reject_nan_inf_and_negative_values(
    section: str, field: str, value: float
) -> None:
    cfg = make_demo_config()
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(ValueError, match=field):
        cfg.validate()


@pytest.mark.parametrize(("section", "field"), _FINITE_NON_NEGATIVE_FIELDS)
@pytest.mark.parametrize("value", [0.0, 0.25, 7.5])
def test_finite_non_negative_fields_accept_zero_and_positive_values(
    section: str, field: str, value: float
) -> None:
    cfg = make_demo_config()
    setattr(getattr(cfg, section), field, value)
    cfg.validate()  # Zero is meaningful for komi, learning rate, loss weights and FPU reduction.


def test_fpu_reduction_none_is_accepted() -> None:
    cfg = make_demo_config()
    cfg.search.fpu_reduction = None  # Classic unvisited Q of zero.
    cfg.validate()


def test_learning_rate_is_validated_even_when_the_paper_schedule_ignores_it() -> None:
    """Validate stored values even if the current learning-rate schedule overrides them."""
    cfg = make_demo_config()
    cfg.training.lr_schedule = "paper"
    cfg.training.learning_rate = float("nan")
    with pytest.raises(ValueError, match="learning_rate"):
        cfg.validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_time_budget_rejects_nan_inf_zero_and_negative_values(value: float) -> None:
    """A nonfinite time budget can silently disable the intended deadline."""
    cfg = make_demo_config()
    cfg.runtime.time_budget_sec = value
    with pytest.raises(ValueError, match="time_budget_sec"):
        cfg.validate()


@pytest.mark.parametrize("value", [1e-6, 1.0, 3600.0, 48 * 3600.0])
def test_time_budget_accepts_positive_finite_values(value: float) -> None:
    cfg = make_demo_config()
    cfg.runtime.time_budget_sec = value
    cfg.validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_benchmark_anchor_rating_must_be_finite(value: float) -> None:
    """A nonfinite anchor makes the estimated rating unusable."""
    cfg = make_demo_config()
    cfg.training.benchmark_anchor_rating = value
    with pytest.raises(ValueError, match="benchmark_anchor_rating"):
        cfg.validate()


@pytest.mark.parametrize("value", [-50.0, 0.0, 1727.0])
def test_benchmark_anchor_rating_accepts_any_finite_number(value: float) -> None:
    cfg = make_demo_config()
    cfg.training.benchmark_anchor_rating = value
    cfg.validate()  # Negative anchor ratings are valid on a relative scale.


_FINITE_POSITIVE_FIELDS = [
    pytest.param("search", "c_puct", id="c_puct"),
    pytest.param("search", "root_dirichlet_alpha", id="root_dirichlet_alpha"),
]


@pytest.mark.parametrize(("section", "field"), _FINITE_POSITIVE_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -0.5])
def test_finite_positive_fields_reject_nan_inf_zero_and_negative_values(
    section: str, field: str, value: float
) -> None:
    """Reject NaN and infinity before they reach PUCT scoring or Dirichlet sampling."""
    cfg = make_demo_config()
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(ValueError, match=field):
        cfg.validate()


@pytest.mark.parametrize(("section", "field"), _FINITE_POSITIVE_FIELDS)
@pytest.mark.parametrize("value", [1e-6, 0.03, 0.30, 1.5, 4.0])
def test_finite_positive_fields_accept_positive_finite_values(
    section: str, field: str, value: float
) -> None:
    cfg = make_demo_config()
    setattr(getattr(cfg, section), field, value)
    cfg.validate()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01, 2.5])
def test_target_false_positive_rate_rejects_values_outside_zero_to_one(value: float) -> None:
    """A false-resignation target outside [0, 1] can push the threshold in the wrong direction."""
    cfg = make_demo_config()
    cfg.search.target_false_positive_rate = value
    with pytest.raises(ValueError, match="target_false_positive_rate"):
        cfg.validate()


@pytest.mark.parametrize("value", [0.0, 0.05, 0.5, 1.0])
def test_target_false_positive_rate_accepts_zero_to_one_inclusive(value: float) -> None:
    cfg = make_demo_config()
    cfg.search.target_false_positive_rate = value
    cfg.validate()


# ----------------------------------------------------------------------
# Root quotas and policy pruning fractions must lie in [0, 1).
# ----------------------------------------------------------------------
_B3_FRACTION_FIELDS = ["forced_playout_fraction", "policy_target_prune_fraction"]


@pytest.mark.parametrize("field", _B3_FRACTION_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.01, 1.0, 1.5])
def test_b3_fractions_reject_values_outside_zero_to_one(field: str, value: float) -> None:
    cfg = make_demo_config()
    setattr(cfg.search, field, value)
    with pytest.raises(ValueError, match=field):
        cfg.validate()


@pytest.mark.parametrize("field", _B3_FRACTION_FIELDS)
@pytest.mark.parametrize("value", [0.0, 0.1, 0.5, 0.999])
def test_b3_fractions_accept_zero_to_one_exclusive(field: str, value: float) -> None:
    cfg = make_demo_config()
    setattr(cfg.search, field, value)
    cfg.validate()


def test_forced_playouts_are_incompatible_with_gumbel() -> None:
    """Gumbel root allocation belongs to Sequential Halving alone."""
    cfg = make_demo_config()
    cfg.search.search_algo = "gumbel"
    cfg.search.forced_playout_fraction = 0.5
    with pytest.raises(ValueError, match="forced_playout_fraction"):
        cfg.validate()


def test_target_pruning_is_compatible_with_gumbel() -> None:
    """Target pruning applies to both Gumbel improved policies and PUCT visit targets."""
    cfg = make_demo_config()
    cfg.search.search_algo = "gumbel"
    cfg.search.policy_target_prune_fraction = 0.1
    cfg.validate()


def test_every_float_field_in_the_config_rejects_nan_and_infinities() -> None:
    """Enumerate dataclass fields to detect any new float field missing validation tests."""
    demo = make_demo_config()
    checked = 0
    for section_field in fields(demo):
        section_name = section_field.name
        for item in fields(getattr(demo, section_name)):
            if "float" not in str(item.type):
                continue
            checked += 1
            for bad in (float("nan"), float("inf"), float("-inf")):
                cfg = make_demo_config()
                setattr(getattr(cfg, section_name), item.name, bad)
                with pytest.raises(ValueError, match=item.name):
                    cfg.validate()
    assert checked == 29  # Cover every floating-point configuration field.


# ----------------------------------------------------------------------
# Reserve less than the full game set for holdout and require a positive buffer capacity.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("value", [-0.01, 1.0, 1.5, float("nan"), float("inf")])
def test_holdout_fraction_rejects_values_outside_zero_to_one(value: float) -> None:
    cfg = make_demo_config()
    cfg.training.holdout_game_fraction = value
    with pytest.raises(ValueError, match="holdout_game_fraction"):
        cfg.validate()


@pytest.mark.parametrize("value", [0.0, 0.05, 0.5, 0.999])
def test_holdout_fraction_accepts_zero_to_one_exclusive(value: float) -> None:
    cfg = make_demo_config()
    cfg.training.holdout_game_fraction = value
    cfg.validate()


@pytest.mark.parametrize("value", [0, -1])
def test_holdout_buffer_size_must_be_positive(value: int) -> None:
    cfg = make_demo_config()
    cfg.training.holdout_buffer_size = value
    with pytest.raises(ValueError, match="holdout_buffer_size"):
        cfg.validate()
