"""Wald sequential probability ratio test for matches between two models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SprtDecision(Enum):
    """Sequential decision with a string value for logs and JSON."""

    ACCEPT_H1 = "H1"  # Evidence for the higher win-rate hypothesis.
    ACCEPT_H0 = "H0"  # Evidence for the lower win-rate hypothesis.
    CONTINUE = "continue"  # Continue collecting outcomes.


@dataclass(frozen=True, slots=True)
class SprtConfig:
    """Win-rate hypotheses and error probabilities."""

    wr0: float = 0.50
    wr1: float = 0.55
    alpha: float = 0.05
    beta: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.wr0 < self.wr1 < 1.0:
            raise ValueError(
                f"SPRT hypotheses require 0 < wr0 < wr1 < 1 (H1 strictly above H0), "
                f"got wr0={self.wr0}, wr1={self.wr1}"
            )
        for name, value in (("alpha", self.alpha), ("beta", self.beta)):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1, got {value}")
        if self.alpha + self.beta >= 1.0:
            raise ValueError(
                f"alpha + beta must be less than 1 for ordered boundaries, "
                f"got alpha={self.alpha}, beta={self.beta}"
            )

    @property
    def lower_bound(self) -> float:
        """Accept H0 when LLR reaches ln(beta / (1 - alpha)) from above."""
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def upper_bound(self) -> float:
        """Accept H1 when LLR reaches ln((1 - beta) / alpha) from below."""
        return math.log((1.0 - self.beta) / self.alpha)

    @property
    def win_step(self) -> float:
        """Log-likelihood increment for a win, ln(wr1 / wr0)."""
        return math.log(self.wr1 / self.wr0)

    @property
    def loss_step(self) -> float:
        """Log-likelihood increment for a loss, ln((1 - wr1) / (1 - wr0))."""
        return math.log((1.0 - self.wr1) / (1.0 - self.wr0))


def _check_counts(wins: int, losses: int) -> None:
    if wins < 0 or losses < 0:
        raise ValueError(f"wins and losses must be nonnegative: {wins}, {losses}")


def sprt_llr(wins: int, losses: int, config: SprtConfig) -> float:
    """Compute the log likelihood ratio of H1 to H0 for the observed wins and losses."""
    _check_counts(wins, losses)
    return wins * config.win_step + losses * config.loss_step


def sprt_decision(wins: int, losses: int, config: SprtConfig) -> SprtDecision:
    """Choose H1, H0 or continuation given the current match score."""
    llr = sprt_llr(wins, losses, config)
    if llr >= config.upper_bound:
        return SprtDecision.ACCEPT_H1
    if llr <= config.lower_bound:
        return SprtDecision.ACCEPT_H0
    return SprtDecision.CONTINUE


def expected_games(config: SprtConfig) -> tuple[float, float]:
    """Approximate expected sample sizes under the two hypotheses using Wald's formula."""
    lower, upper = config.lower_bound, config.upper_bound

    def mean_step(win_rate: float) -> float:
        return win_rate * config.win_step + (1.0 - win_rate) * config.loss_step

    under_h0 = (config.alpha * upper + (1.0 - config.alpha) * lower) / mean_step(config.wr0)
    under_h1 = ((1.0 - config.beta) * upper + config.beta * lower) / mean_step(config.wr1)
    return under_h0, under_h1
