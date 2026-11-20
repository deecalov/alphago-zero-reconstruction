"""Shared argparse types for numeric search and training options. Reject infinities and NaNs before starting
an experiment. For FPU, zero and None select different search behavior.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def _integer(text: str) -> int:
    """Parse a command-line integer."""
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None


def positive_int(text: str) -> int:
    """Parse a strictly positive integer."""
    value = _integer(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a value greater than zero, got {value}")
    return value


def non_negative_int(text: str) -> int:
    """Parse a nonnegative integer, such as the number of sampled opening moves."""
    value = _integer(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a nonnegative value, got {value}")
    return value


def finite_float(text: str) -> float:
    """Parse a finite floating-point value of either sign, for example a komi or rating."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"expected a finite number, got {text!r}")
    return value


def positive_float(text: str) -> float:
    """Parse a strictly positive finite value, such as c_puct."""
    value = finite_float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a value greater than zero, got {text!r}")
    return value


def non_negative_float(text: str) -> float:
    """Parse a finite nonnegative value. A virtual loss of zero disables the penalty."""
    value = finite_float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"expected a nonnegative value, got {text!r}")
    return value


def fpu_value(text: str) -> float | None:
    """Parse 'none' for an unvisited Q of zero, or a finite nonnegative FPU reduction."""
    if text.strip().lower() == "none":
        return None
    value = finite_float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError(f"expected \'none\' or a nonnegative number, got {text!r}")
    return value


def ensure_directory(parser: argparse.ArgumentParser, folder: Path, flag: str) -> None:
    """Create the output directory and its parents before starting a match. Report unusable paths as
    command-line errors while allowing an existing directory.
    """
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"{flag}: cannot create directory {folder} ({error})")
