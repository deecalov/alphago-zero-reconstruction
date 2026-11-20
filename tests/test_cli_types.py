"""Command-line argument type and validation tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from alphago_zero.cli_types import (
    ensure_directory,
    finite_float,
    fpu_value,
    non_negative_float,
    non_negative_int,
    positive_float,
    positive_int,
)


# ----------------------------------------------------------------------
# non_negative_int
# ----------------------------------------------------------------------
@pytest.mark.parametrize(("text", "expected"), [("0", 0), ("6", 6), (" 12 ", 12)])
def test_non_negative_int_accepts_zero_and_above(text: str, expected: int) -> None:
    assert non_negative_int(text) == expected


@pytest.mark.parametrize("text", ["-1", "abc", "1.5", "", "inf"])
def test_non_negative_int_rejects_everything_else(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expected"):
        non_negative_int(text)


# ----------------------------------------------------------------------
# positive_int
# ----------------------------------------------------------------------
@pytest.mark.parametrize(("text", "expected"), [("1", 1), ("160", 160), (" 7 ", 7)])
def test_positive_int_accepts_integers_above_zero(text: str, expected: int) -> None:
    assert positive_int(text) == expected


@pytest.mark.parametrize("text", ["0", "-3", "abc", "1.5", "", "nan"])
def test_positive_int_rejects_everything_else(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expected"):
        positive_int(text)


# ----------------------------------------------------------------------
# positive_float
# ----------------------------------------------------------------------
@pytest.mark.parametrize(("text", "expected"), [("1.5", 1.5), ("2", 2.0), ("0.001", 0.001), ("1e-3", 0.001)])
def test_positive_float_accepts_finite_numbers_above_zero(text: str, expected: float) -> None:
    assert positive_float(text) == expected


@pytest.mark.parametrize("text", ["0", "0.0", "-1.5", "abc", "", "nan", "inf", "-inf"])
def test_positive_float_rejects_zero_negative_and_non_finite(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expected"):
        positive_float(text)


# ----------------------------------------------------------------------
# finite_float
# ----------------------------------------------------------------------
@pytest.mark.parametrize(("text", "expected"), [("7.5", 7.5), ("0", 0.0), ("-6.5", -6.5), ("1e3", 1000.0), (" 2 ", 2.0)])
def test_finite_float_accepts_any_finite_number_including_zero_and_negative(text: str, expected: float) -> None:
    assert finite_float(text) == expected


@pytest.mark.parametrize("text", ["abc", "", "nan", "inf", "-inf", "1,5"])
def test_finite_float_rejects_text_and_non_finite(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expected"):
        finite_float(text)


# ----------------------------------------------------------------------
# non_negative_float
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"), [("0", 0.0), ("0.0", 0.0), ("0.5", 0.5), ("1", 1.0), ("1e-3", 0.001), (" 2 ", 2.0)]
)
def test_non_negative_float_accepts_zero_and_above(text: str, expected: float) -> None:
    value = non_negative_float(text)
    assert value == expected
    assert isinstance(value, float)  # An integer-looking zero is parsed as a float for virtual loss.


@pytest.mark.parametrize("text", ["-0.1", "-1", "abc", "", "nan", "inf", "-inf", "1,5"])
def test_non_negative_float_rejects_negative_text_and_non_finite(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expected"):
        non_negative_float(text)


# ----------------------------------------------------------------------
# ensure_directory
# ----------------------------------------------------------------------
def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(prog="tool")


def test_ensure_directory_creates_missing_parents(tmp_path: Path) -> None:
    folder = tmp_path / "a" / "b" / "c"
    ensure_directory(_parser(), folder, "--out")
    assert folder.is_dir()


def test_ensure_directory_accepts_an_existing_directory(tmp_path: Path) -> None:
    ensure_directory(_parser(), tmp_path, "--out")
    assert tmp_path.is_dir()


def test_ensure_directory_reports_an_unusable_path_as_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        ensure_directory(_parser(), blocker / "sub", "--sgf-dir")
    assert excinfo.value.code == 2
    error_line = capsys.readouterr().err.strip().splitlines()[-1]
    assert "--sgf-dir: cannot create directory" in error_line and "sub" in error_line


# ----------------------------------------------------------------------
# fpu_value
# ----------------------------------------------------------------------
@pytest.mark.parametrize("text", ["none", "NONE", "None", " none "])
def test_fpu_value_none_means_no_reduction(text: str) -> None:
    assert fpu_value(text) is None


@pytest.mark.parametrize(("text", "expected"), [("0.2", 0.2), ("0", 0.0), ("0.0", 0.0), ("1", 1.0)])
def test_fpu_value_accepts_non_negative_numbers(text: str, expected: float) -> None:
    value = fpu_value(text)
    assert value == expected
    assert value is not None  # FPU zero uses the parent's value for unvisited edges.


@pytest.mark.parametrize("text", ["-0.1", "abc", "", "nan", "inf", "-inf"])
def test_fpu_value_rejects_negative_and_non_finite(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="expected"):
        fpu_value(text)
