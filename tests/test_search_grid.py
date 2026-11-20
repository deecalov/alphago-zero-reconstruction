"""Parameter grid, caching, reporting and confirmation tests for fixed-weight search comparisons."""

from __future__ import annotations

import dataclasses
import json
import shlex
from pathlib import Path
from typing import Any

import pytest

from alphago_zero.game import BLACK
from examples import eval_match, search_grid
from examples.search_grid import ControlPoint, GridOptions, GridPoint, Point, PointResult

# Shared fixtures provide two small checkpoints.

# The stub's saved search uses c_puct 1.5 and FPU 0.25.
SAVED: tuple[float, float | None] = (1.5, 0.25)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _options(tmp_path: Path, **changes: Any) -> GridOptions:
    """Create a 2x2 grid over c_puct {1.5, 2.0} and FPU {none, 0.2} without a control."""
    base = GridOptions(
        model=("m.pt",),
        out_dir=tmp_path / "grid",
        points=tuple(search_grid.build_points([1.5, 2.0], [None, 0.2])),
        games=120,
        simulations=None,
        opening_moves=6,
        device="cpu",
        seed=7,
    )
    return dataclasses.replace(base, **changes)


def _flags(argv: list[str]) -> dict[str, list[str]]:
    """Map option names to their argument lists independently of flag order."""
    parsed: dict[str, list[str]] = {}
    current = ""
    for token in argv:
        if token.startswith("--"):
            current = token
            parsed[current] = []
        else:
            parsed[current].append(token)
    return parsed


def _point_of(argv: list[str]) -> Point:
    """Infer the point from match arguments. No model overrides means a control."""
    flags = _flags(argv)
    if "--model-c-puct" not in flags:
        return ControlPoint()
    fpu = flags["--model-fpu"][0]
    return GridPoint(float(flags["--model-c-puct"][0]), None if fpu == "none" else float(fpu))


class FakeMatch:
    """Stub eval_match.main and write a scripted JSON result to --json-out."""

    def __init__(
        self, wins_by_key: dict[str, int] | None = None, saved: tuple[float, float | None] = SAVED
    ) -> None:
        self.calls: list[list[str]] = []
        self._wins_by_key = wins_by_key or {}
        self._saved = saved

    def __call__(self, argv: list[str]) -> None:
        self.calls.append(list(argv))
        point = _point_of(argv)
        model_search = (point.c_puct, point.fpu) if isinstance(point, GridPoint) else self._saved
        flags = _flags(argv)
        games = int(flags["--games"][0])
        wins = self._wins_by_key.get(point.key, games // 2)
        rate = wins / games
        low, high = eval_match.wilson_interval(wins, games)[1:]
        payload = {
            "games": games,
            "wins": wins,
            "losses": games - wins,
            "win_rate": round(rate, 4),
            "win_rate_ci95": [round(low, 4), round(high, 4)],
            "elo_estimate": round(eval_match.elo_from_win_rate(rate, 0.0), 1),
            "model_c_puct": model_search[0],
            "model_fpu": model_search[1],
            "opponent_c_puct": self._saved[0],
            "opponent_fpu": self._saved[1],
        }
        Path(flags["--json-out"][0]).write_text(json.dumps(payload), encoding="utf-8")


def _result(
    point: Point,
    *,
    effective: tuple[float, float | None] | None = None,
    wins: int = 60,
    games: int = 120,
    ci: tuple[float, float] = (0.41, 0.59),
    elo: float = 0.0,
    is_control: bool = False,
    source: str = "played",
) -> PointResult:
    """Give each point its actual search settings, reading the control settings from the checkpoint."""
    if effective is None:
        effective = (point.c_puct, point.fpu) if isinstance(point, GridPoint) else SAVED
    return PointResult(
        point=point,
        c_puct=effective[0],
        fpu=effective[1],
        games=games,
        wins=wins,
        losses=games - wins,
        win_rate=wins / games,
        ci_low=ci[0],
        ci_high=ci[1],
        elo=elo,
        is_control=is_control,
        source=source,
    )


def _error_message(capsys: pytest.CaptureFixture[str]) -> str:
    """Use the final stderr line so argparse usage cannot satisfy the check."""
    return capsys.readouterr().err.strip().splitlines()[-1]


def _commands(out: str) -> list[dict[str, list[str]]]:
    """Extract flags from each match command printed by dry run."""
    lines = [line for line in out.splitlines() if "examples.eval_match" in line]
    return [_flags(shlex.split(line)[3:]) for line in lines]


# ----------------------------------------------------------------------
# Grid points
# ----------------------------------------------------------------------
def test_build_points_is_the_product_with_c_puct_in_the_outer_loop() -> None:
    points = search_grid.build_points([1.1, 1.5], [None, 0.2])
    assert points == [GridPoint(1.1, None), GridPoint(1.1, 0.2), GridPoint(1.5, None), GridPoint(1.5, 0.2)]


@pytest.mark.parametrize(
    ("c_values", "fpu_values"),
    [([1.5, 1.5], [None]), ([1.5], [0.2, 0.2]), ([1.5], [None, None]), ([1.5], [0.2, 0.20])],
)
def test_build_points_rejects_repeated_values(c_values: list[float], fpu_values: list[float | None]) -> None:
    with pytest.raises(ValueError, match="duplicated"):
        search_grid.build_points(c_values, fpu_values)


@pytest.mark.parametrize(("c_values", "fpu_values"), [([], [None]), ([1.5], [])])
def test_build_points_rejects_an_empty_axis(c_values: list[float], fpu_values: list[float | None]) -> None:
    with pytest.raises(ValueError, match="empty"):
        search_grid.build_points(c_values, fpu_values)


@pytest.mark.parametrize(
    ("point", "key"),
    [
        (GridPoint(1.1, None), "c1.1_fpu-none"),
        (GridPoint(2.0, 0.2), "c2_fpu0.2"),
        (GridPoint(1.25, 0.4), "c1.25_fpu0.4"),
        (GridPoint(1.5, 0.0), "c1.5_fpu0"),  # FPU zero differs from an unspecified reduction.
        (ControlPoint(), "control"),
    ],
)
def test_point_keys_name_the_result_files(point: Point, key: str) -> None:
    assert point.key == key


def test_plan_points_puts_the_control_before_the_grid() -> None:
    points = search_grid.plan_points([1.5], [None, 0.2], control=True)
    assert points == [ControlPoint(), GridPoint(1.5, None), GridPoint(1.5, 0.2)]


def test_plan_points_without_control_is_the_grid_alone() -> None:
    assert search_grid.plan_points([1.5], [None, 0.2], control=False) == search_grid.build_points([1.5], [None, 0.2])


def test_plan_points_still_validates_the_axes() -> None:
    with pytest.raises(ValueError, match="duplicated"):
        search_grid.plan_points([1.5, 1.5], [None], control=True)


def test_default_grid_has_the_documented_twelve_points() -> None:
    points = search_grid.build_points(search_grid.DEFAULT_C_PUCT, search_grid.DEFAULT_FPU)
    assert search_grid.DEFAULT_C_PUCT == (1.1, 1.5, 2.0, 2.5)
    assert search_grid.DEFAULT_FPU == (None, 0.2, 0.4)
    assert len(points) == 12
    assert len({point.key for point in points}) == 12


def test_point_seed_is_stable_distinct_per_point_and_follows_the_base_seed() -> None:
    points = search_grid.plan_points(search_grid.DEFAULT_C_PUCT, search_grid.DEFAULT_FPU, control=True)
    seeds = [search_grid.point_seed(7, point) for point in points]
    assert seeds == [search_grid.point_seed(7, point) for point in points]  # Determinism.
    assert len(points) == 13 and len(set(seeds)) == 13  # Every point, including the control, has its own random stream.
    shifted = [search_grid.point_seed(8, point) for point in points]
    assert [b - a for a, b in zip(seeds, shifted, strict=True)] == [1] * len(points)


# ----------------------------------------------------------------------
# Match argument construction
# ----------------------------------------------------------------------
def test_match_argv_carries_the_experiment_to_eval_match(tmp_path: Path) -> None:
    options = _options(
        tmp_path,
        model=("a.pt", "b.pt"),
        games=96,
        simulations=160,
        opening_moves=4,
        device="cuda",
    )
    point = GridPoint(2.0, 0.2)
    assert _flags(search_grid.match_argv(options, point)) == {
        "--model": ["a.pt", "b.pt"],
        "--opponent": ["a.pt", "b.pt"],  # The opponent uses the same weights with the saved search configuration.
        "--games": ["96"],
        "--opening-moves": ["4"],
        "--device": ["cuda"],
        "--seed": [str(search_grid.point_seed(options.seed, point))],
        "--simulations": ["160"],
        "--model-c-puct": ["2.0"],
        "--model-fpu": ["0.2"],
        "--json-out": [str(options.out_dir / "c2_fpu0.2.json")],
    }


def test_control_argv_has_no_search_overrides_at_all(tmp_path: Path) -> None:
    """The control must supply identical weights and omit search overrides for both sides."""
    options = _options(tmp_path, model=("a.pt",), games=96, simulations=160, opening_moves=4, device="cuda")
    point = ControlPoint()
    assert _flags(search_grid.match_argv(options, point)) == {
        "--model": ["a.pt"],
        "--opponent": ["a.pt"],
        "--games": ["96"],
        "--opening-moves": ["4"],
        "--device": ["cuda"],
        "--seed": [str(search_grid.point_seed(options.seed, point))],
        "--simulations": ["160"],
        "--json-out": [str(options.out_dir / "control.json")],
    }


def test_match_argv_without_simulations_leaves_each_side_on_its_saved_search(tmp_path: Path) -> None:
    flags = _flags(search_grid.match_argv(_options(tmp_path, simulations=None), GridPoint(1.5, None)))
    assert "--simulations" not in flags


@pytest.mark.parametrize(("fpu", "text"), [(None, "none"), (0.0, "0.0"), (0.4, "0.4")])
def test_match_argv_writes_fpu_so_that_none_and_zero_stay_different(
    tmp_path: Path, fpu: float | None, text: str
) -> None:
    flags = _flags(search_grid.match_argv(_options(tmp_path), GridPoint(1.5, fpu)))
    assert flags["--model-fpu"] == [text]


@pytest.mark.parametrize("point", [GridPoint(2.0, 0.2), ControlPoint()], ids=["grid-point", "control"])
def test_match_argv_never_touches_the_search_of_the_opponent(tmp_path: Path, point: Point) -> None:
    flags = _flags(search_grid.match_argv(_options(tmp_path), point))
    assert "--opponent-c-puct" not in flags
    assert "--opponent-fpu" not in flags


# ----------------------------------------------------------------------
# Loading a point result
# ----------------------------------------------------------------------
def _write_result(tmp_path: Path, point: Point, **changes: Any) -> Path:
    """Generate a result JSON through the stub, then apply selected field changes."""
    options = _options(tmp_path)
    options.out_dir.mkdir(parents=True, exist_ok=True)
    FakeMatch({point.key: 71})(search_grid.match_argv(options, point))
    path = search_grid.result_path(options, point)
    if changes:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(changes)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_point_result_reads_the_match_json(tmp_path: Path) -> None:
    point = GridPoint(2.0, 0.2)
    result = search_grid.load_point_result(point, _write_result(tmp_path, point), "played")
    assert (result.games, result.wins, result.losses) == (120, 71, 49)
    assert result.win_rate == pytest.approx(71 / 120, abs=1e-4)
    low, high = eval_match.wilson_interval(71, 120)[1:]
    assert (result.ci_low, result.ci_high) == (pytest.approx(low, abs=1e-4), pytest.approx(high, abs=1e-4))
    assert result.elo == pytest.approx(eval_match.elo_from_win_rate(71 / 120, 0.0), abs=0.1)
    assert (result.c_puct, result.fpu) == (2.0, 0.2)
    assert result.point == point
    assert result.source == "played"


def test_control_result_carries_the_search_saved_in_the_checkpoint(tmp_path: Path) -> None:
    point = ControlPoint()
    result = search_grid.load_point_result(point, _write_result(tmp_path, point), "played")
    assert result.is_control is True
    assert (result.c_puct, result.fpu) == SAVED  # Show the checkpoint's actual saved settings in the control row.


@pytest.mark.parametrize(
    ("saved", "is_control"),
    [
        ((2.0, 0.2), True),  # With identical settings any measured difference is match noise.
        ((1.5, 0.2), False),  # Different c_puct.
        ((2.0, None), False),  # Different FPU.
        ((2.0, 0.0), False),  # Distinguish zero FPU reduction from None.
    ],
)
def test_grid_point_equal_to_the_saved_search_is_also_a_control(
    tmp_path: Path, saved: tuple[float, float | None], is_control: bool
) -> None:
    point = GridPoint(2.0, 0.2)
    path = _write_result(tmp_path, point, opponent_c_puct=saved[0], opponent_fpu=saved[1])
    assert search_grid.load_point_result(point, path, "cached").is_control is is_control


@pytest.mark.parametrize("changes", [{"model_c_puct": 1.5}, {"model_fpu": None}, {"model_fpu": 0.4}])
def test_result_of_another_point_is_rejected(tmp_path: Path, changes: dict[str, Any]) -> None:
    point = GridPoint(2.0, 0.2)
    path = _write_result(tmp_path, point, **changes)
    with pytest.raises(ValueError, match="another grid point") as excinfo:
        search_grid.load_point_result(point, path, "cached")
    assert path.name in str(excinfo.value)


@pytest.mark.parametrize("changes", [{"model_c_puct": 2.0}, {"model_fpu": None}, {"opponent_fpu": 0.4}])
def test_control_result_with_unequal_searches_is_rejected(tmp_path: Path, changes: dict[str, Any]) -> None:
    """A control with unequal per-side search settings belongs to a different experiment."""
    point = ControlPoint()
    path = _write_result(tmp_path, point, **changes)
    with pytest.raises(ValueError, match="another grid point") as excinfo:
        search_grid.load_point_result(point, path, "cached")
    assert path.name in str(excinfo.value)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param('{"games": 120, "wins"', id="truncated"),  # Interrupted write.
        pytest.param("", id="empty"),  # Failure immediately after creating the result file.
        pytest.param("[]", id="not-an-object"),
        pytest.param("120", id="a-number"),
    ],
)
def test_corrupt_json_is_reported_with_the_file_name(tmp_path: Path, content: str) -> None:
    point = GridPoint(2.0, 0.2)
    path = _write_result(tmp_path, point)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt") as excinfo:
        search_grid.load_point_result(point, path, "cached")
    assert path.name in str(excinfo.value)


def test_missing_result_file_is_reported_with_the_file_name(tmp_path: Path) -> None:
    """Name the missing result file if a runner exits before writing JSON."""
    point = GridPoint(2.0, 0.2)
    path = _options(tmp_path).out_dir / "c2_fpu0.2.json"
    with pytest.raises(ValueError, match="unreadable") as excinfo:
        search_grid.load_point_result(point, path, "played")
    assert path.name in str(excinfo.value)


@pytest.mark.parametrize("interval", [[0.5], 0.5, [0.1, 0.2, 0.3]])
def test_confidence_interval_must_be_a_pair(tmp_path: Path, interval: Any) -> None:
    point = GridPoint(2.0, 0.2)
    path = _write_result(tmp_path, point, win_rate_ci95=interval)
    with pytest.raises(ValueError, match="win_rate_ci95") as excinfo:
        search_grid.load_point_result(point, path, "cached")
    assert path.name in str(excinfo.value)


def test_json_without_a_required_field_is_reported(tmp_path: Path) -> None:
    point = GridPoint(2.0, 0.2)
    path = _write_result(tmp_path, point)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["win_rate_ci95"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="win_rate_ci95") as excinfo:
        search_grid.load_point_result(point, path, "cached")
    assert path.name in str(excinfo.value)


# ----------------------------------------------------------------------
# Verdicts
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("ci", "verdict"),
    [
        ((0.5001, 0.70), "better"),
        ((0.50, 0.70), "parity"),  # A lower bound exactly equal to one half gives no evidence of a lead.
        ((0.45, 0.55), "parity"),
        ((0.30, 0.50), "parity"),
        ((0.30, 0.4999), "worse"),
    ],
)
def test_verdict_is_decided_by_the_interval_not_by_the_point(ci: tuple[float, float], verdict: str) -> None:
    assert _result(GridPoint(2.0, 0.2), ci=ci).verdict == verdict


# ----------------------------------------------------------------------
# Grid execution
# ----------------------------------------------------------------------
def test_run_grid_plays_every_point_in_grid_order(tmp_path: Path) -> None:
    options = _options(tmp_path)
    fake = FakeMatch()
    results = search_grid.run_grid(options, runner=fake, echo=lambda _line: None)
    assert fake.calls == [search_grid.match_argv(options, point) for point in options.points]
    assert [result.point for result in results] == list(options.points)
    assert {result.source for result in results} == {"played"}
    assert all(search_grid.result_path(options, point).exists() for point in options.points)


def test_run_grid_plays_the_control_first_and_without_overrides(tmp_path: Path) -> None:
    points = tuple(search_grid.plan_points([1.5], [None, 0.2], control=True))
    fake = FakeMatch()
    results = search_grid.run_grid(_options(tmp_path, points=points), runner=fake, echo=lambda _line: None)
    assert [_point_of(call) for call in fake.calls] == [ControlPoint(), GridPoint(1.5, None), GridPoint(1.5, 0.2)]
    assert "--model-c-puct" not in _flags(fake.calls[0])
    assert [result.is_control for result in results] == [True, False, False]  # The checkpoint search uses 1.5 / 0.25.


def test_run_grid_skips_points_that_already_have_a_result(tmp_path: Path) -> None:
    options = _options(tmp_path)
    first = options.points[0]
    options.out_dir.mkdir(parents=True)
    FakeMatch({first.key: 80})(search_grid.match_argv(options, first))  # Cache the first point ahead of time.
    fake = FakeMatch()
    lines: list[str] = []
    results = search_grid.run_grid(options, runner=fake, echo=lines.append)
    assert [_point_of(call) for call in fake.calls] == list(options.points[1:])
    assert [result.source for result in results] == ["cached", "played", "played", "played"]
    assert results[0].wins == 80  # Read the previous result without replaying it.
    assert any(first.key in line and "skipping" in line for line in lines)


def test_second_run_replays_nothing(tmp_path: Path) -> None:
    options = _options(tmp_path)
    search_grid.run_grid(options, runner=FakeMatch(), echo=lambda _line: None)
    fake = FakeMatch()
    results = search_grid.run_grid(options, runner=fake, echo=lambda _line: None)
    assert fake.calls == []
    assert {result.source for result in results} == {"cached"}


def test_grid_can_be_extended_in_the_same_directory(tmp_path: Path) -> None:
    small = _options(tmp_path, points=tuple(search_grid.build_points([1.5], [None, 0.2])))
    search_grid.run_grid(small, runner=FakeMatch(), echo=lambda _line: None)
    fake = FakeMatch()
    extended = _options(tmp_path, points=(ControlPoint(), *_options(tmp_path).points))
    search_grid.run_grid(extended, runner=fake, echo=lambda _line: None)
    assert [_point_of(call) for call in fake.calls] == [ControlPoint(), GridPoint(2.0, None), GridPoint(2.0, 0.2)]


def test_run_grid_writes_the_manifest_of_the_experiment(tmp_path: Path) -> None:
    options = _options(tmp_path, model=("a.pt", "b.pt"), simulations=160)
    search_grid.run_grid(options, runner=FakeMatch(), echo=lambda _line: None)
    manifest = json.loads((options.out_dir / "grid_manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "model": ["a.pt", "b.pt"],
        "games": 120,
        "simulations": 160,
        "opening_moves": 6,
        "seed": 7,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"games": 240},
        {"simulations": 160},
        {"opening_moves": 0},
        {"seed": 8},
        {"model": ("other.pt",)},
    ],
)
def test_directory_of_another_experiment_is_refused(tmp_path: Path, changes: dict[str, Any]) -> None:
    search_grid.run_grid(_options(tmp_path), runner=FakeMatch(), echo=lambda _line: None)
    fake = FakeMatch()
    with pytest.raises(ValueError, match="another experiment") as excinfo:
        search_grid.run_grid(_options(tmp_path, **changes), runner=fake, echo=lambda _line: None)
    message = str(excinfo.value)
    changed = next(iter(changes))
    assert changed in message  # Report the mismatched field
    others = {"model", "games", "simulations", "opening_moves", "seed"} - {changed}
    assert not any(name in message for name in others)  # and no matching fields.
    assert fake.calls == []  # Do not launch any match in a directory belonging to a different experiment.


def test_device_is_not_part_of_the_experiment(tmp_path: Path) -> None:
    search_grid.run_grid(_options(tmp_path, device="cpu"), runner=FakeMatch(), echo=lambda _line: None)
    fake = FakeMatch()
    search_grid.run_grid(_options(tmp_path, device="cuda"), runner=fake, echo=lambda _line: None)
    assert fake.calls == []  # Device choice does not affect the experiment identity.


def test_run_grid_defaults_to_the_real_eval_match(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeMatch()
    monkeypatch.setattr(eval_match, "main", fake)
    search_grid.run_grid(_options(tmp_path), echo=lambda _line: None)
    assert len(fake.calls) == 4


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
def _summary_results() -> list[PointResult]:
    """Construct a control at 1.5 / 0.25 and three points classified as better, even and worse."""
    return [
        _result(ControlPoint(), wins=64, ci=(0.44, 0.62), elo=23.2, is_control=True),
        _result(GridPoint(2.0, 0.2), wins=71, ci=(0.503, 0.675), elo=64.4),
        _result(GridPoint(2.5, 0.4), wins=58, ci=(0.39, 0.57), elo=-11.6),
        _result(GridPoint(1.1, None), wins=40, ci=(0.25, 0.42), elo=-120.6),
    ]


BEST = ("2", "0.2")
CONTROL = ("1.5", "0.25")
MIDDLE = ("2.5", "0.4")
WORST = ("1.1", "none")


def _row(text: str, cells: tuple[str, str]) -> str:
    """Find a table row by its first two cells, c_puct and FPU."""
    for line in text.splitlines():
        if tuple(line.split()[:2]) == cells:
            return line
    raise AssertionError(f"row {cells} missing from summary:\n{text}")


def test_summary_lists_points_by_win_rate_best_first(tmp_path: Path) -> None:
    text = search_grid.format_summary(_summary_results(), _options(tmp_path))
    order = [text.index(_row(text, cells)) for cells in (BEST, CONTROL, MIDDLE, WORST)]
    assert order == sorted(order)


def test_summary_rows_show_counts_interval_and_verdict(tmp_path: Path) -> None:
    text = search_grid.format_summary(_summary_results(), _options(tmp_path))
    best = _row(text, BEST)
    assert "71" in best.split() and "120" in best.split()
    assert "0.592" in best and "[0.503, 0.675]" in best
    assert "better" in best
    assert "+64" in best.split()  # Signed Elo.
    assert "parity" in _row(text, MIDDLE)
    assert "-12" in _row(text, MIDDLE).split()
    assert "worse" in _row(text, WORST)
    assert "-121" in _row(text, WORST).split()


def test_summary_marks_only_the_control_row_and_shows_the_saved_search_in_it(tmp_path: Path) -> None:
    text = search_grid.format_summary(_summary_results(), _options(tmp_path))
    assert "control" in _row(text, CONTROL)  # Read the control row's search parameters from the checkpoint.
    for cells in (BEST, MIDDLE, WORST):
        assert "control" not in _row(text, cells)


def test_summary_marks_a_grid_point_that_equals_the_saved_search_too(tmp_path: Path) -> None:
    results = [_result(GridPoint(1.5, None), is_control=True), _result(GridPoint(2.0, 0.2), wins=70, ci=(0.5, 0.67))]
    text = search_grid.format_summary(results, _options(tmp_path))
    assert "control" in _row(text, ("1.5", "none"))
    assert "control" not in _row(text, BEST)


def test_summary_names_the_best_non_control_point_as_the_candidate(tmp_path: Path) -> None:
    results = _summary_results()
    # A control cannot become the candidate even if it has the best observed win rate.
    results[0] = _result(ControlPoint(), wins=75, ci=(0.52, 0.70), is_control=True)
    text = search_grid.format_summary(results, _options(tmp_path))
    candidate = next(line for line in text.splitlines() if line.startswith("Candidate"))
    assert "c_puct=2" in candidate and "fpu=0.2" in candidate
    assert "better" in candidate


def test_summary_gives_a_confirmation_command_with_more_games_and_a_new_seed(tmp_path: Path) -> None:
    options = _options(tmp_path, model=("a.pt", "b.pt"), simulations=160, device="cuda")
    text = search_grid.format_summary(_summary_results(), options)
    command = next(line for line in text.splitlines() if "examples.eval_match" in line)
    tokens = shlex.split(command)
    assert tokens[:3] == ["python", "-m", "examples.eval_match"]
    flags = _flags(tokens[3:])
    point = GridPoint(2.0, 0.2)
    assert flags["--games"] == ["240"]  # Twice the original match size.
    assert flags["--seed"] != [str(search_grid.point_seed(options.seed, point))]
    assert (flags["--model-c-puct"], flags["--model-fpu"]) == (["2.0"], ["0.2"])
    assert (flags["--model"], flags["--opponent"], flags["--simulations"]) == (["a.pt", "b.pt"], ["a.pt", "b.pt"], ["160"])
    assert Path(flags["--json-out"][0]).parent == options.out_dir / "confirm"


def test_summary_warns_about_the_winners_curse_when_points_were_compared(tmp_path: Path) -> None:
    text = search_grid.format_summary(_summary_results(), _options(tmp_path))
    assert "Selection bias" in text
    assert "confirm" in text.lower()


def test_summary_of_two_non_control_points_already_warns_about_selection(tmp_path: Path) -> None:
    results = [
        _result(ControlPoint(), is_control=True),
        _result(GridPoint(2.0, 0.2), wins=70, ci=(0.5, 0.67)),
        _result(GridPoint(2.5, 0.4), wins=60),
    ]
    text = search_grid.format_summary(results, _options(tmp_path))
    assert "Selection bias" in text
    assert "best of 2 settings" in text  # Exclude the control from the number of alternatives.


def test_summary_of_one_non_control_point_has_no_selection_warning(tmp_path: Path) -> None:
    results = [_result(ControlPoint(), is_control=True), _result(GridPoint(2.0, 0.2), wins=70, ci=(0.5, 0.67))]
    text = search_grid.format_summary(results, _options(tmp_path))
    assert "Candidate" in text
    assert "Selection bias" not in text  # With one alternative there is no maximum-selection effect.


def test_summary_without_non_control_points_has_no_candidate(tmp_path: Path) -> None:
    text = search_grid.format_summary([_result(ControlPoint(), is_control=True)], _options(tmp_path))
    assert "Candidate" not in text
    assert "examples.eval_match" not in text


def test_summary_explains_the_control_row_only_when_there_is_one(tmp_path: Path) -> None:
    with_control = search_grid.format_summary(_summary_results(), _options(tmp_path))
    assert "noise" in with_control
    without = search_grid.format_summary(_summary_results()[1:], _options(tmp_path))
    assert "noise" not in without
    assert "No grid point measures the control" in without  # Warn separately when the grid contains no control.


def test_summary_payload_is_json_and_mirrors_the_table(tmp_path: Path) -> None:
    results = _summary_results()
    payload = search_grid.summary_payload(results, _options(tmp_path))
    json.dumps(payload)  # Serialize directly to JSON.
    assert [entry["key"] for entry in payload["points"]] == [
        "c2_fpu0.2",
        "control",
        "c2.5_fpu0.4",
        "c1.1_fpu-none",
    ]
    best, control = payload["points"][0], payload["points"][1]
    assert best["verdict"] == "better"
    assert best["is_control"] is False
    assert best["ci95"] == [0.503, 0.675]
    assert (best["c_puct"], best["fpu"], best["wins"], best["games"]) == (2.0, 0.2, 71, 120)
    assert (control["c_puct"], control["fpu"], control["is_control"]) == (1.5, 0.25, True)  # Effective search settings.
    assert payload["candidate"] == "c2_fpu0.2"
    assert payload["controls"] == ["control"]
    assert payload["options"]["games"] == 120


def test_summary_payload_lists_every_control(tmp_path: Path) -> None:
    results = [
        _result(ControlPoint(), is_control=True),
        _result(GridPoint(1.5, 0.25), wins=62, is_control=True),  # A grid point matching the checkpoint is also a control.
        _result(GridPoint(2.0, 0.2), wins=70, ci=(0.5, 0.67)),
    ]
    payload = search_grid.summary_payload(results, _options(tmp_path))
    # Sort by win rate: 62 wins rank above 60 regardless of execution order.
    assert payload["controls"] == ["c1.5_fpu0.25", "control"]


def test_summary_payload_candidate_is_never_the_control(tmp_path: Path) -> None:
    results = _summary_results()
    results[0] = _result(ControlPoint(), wins=75, ci=(0.52, 0.70), is_control=True)
    payload = search_grid.summary_payload(results, _options(tmp_path))
    assert payload["points"][0]["key"] == "control"  # The control leads on raw win rate
    assert payload["candidate"] == "c2_fpu0.2"  # but the best non-control setting is the candidate.


def test_summary_payload_without_candidate_or_control_uses_empty_values(tmp_path: Path) -> None:
    payload = search_grid.summary_payload([_result(GridPoint(2.0, 0.2))], _options(tmp_path))
    assert payload["controls"] == []
    only_control = search_grid.summary_payload([_result(ControlPoint(), is_control=True)], _options(tmp_path))
    assert only_control["candidate"] is None


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _forbid_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(argv: list[str]) -> None:
        raise AssertionError("match must not start")

    monkeypatch.setattr(eval_match, "main", refuse)


def test_dry_run_prints_the_control_and_one_command_per_default_point_and_touches_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_matches(monkeypatch)
    out_dir = tmp_path / "grid"
    search_grid.main(["--model", "m.pt", "--out-dir", str(out_dir), "--dry-run"])
    output = capsys.readouterr().out
    assert "matches 13, games per match: 120" in output.splitlines()[0]  # Use a numeric count independent of pluralization.
    commands = _commands(output)
    assert len(commands) == 13  # One control plus a four-by-three grid.
    control, first = commands[0], commands[1]
    assert "--model-c-puct" not in control and "--model-fpu" not in control  # No overrides in the control.
    assert (first["--model-c-puct"], first["--model-fpu"]) == (["1.1"], ["none"])
    for command in (control, first):
        assert command["--model"] == command["--opponent"] == ["m.pt"]  # Use the same weights for the opponent.
        assert command["--games"] == ["120"]
        assert command["--opening-moves"] == ["6"]  # Without opening sampling, deterministic fixed-weight matches can repeat just two games.
        assert command["--device"] == ["cuda"]
        assert "--simulations" not in command  # Omitting an explicit budget preserves each side's saved settings.
    assert not out_dir.exists()


def test_no_control_flag_drops_only_the_control_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_matches(monkeypatch)
    search_grid.main(["--model", "m.pt", "--out-dir", str(tmp_path / "grid"), "--dry-run", "--no-control"])
    commands = _commands(capsys.readouterr().out)
    assert len(commands) == 12
    assert all("--model-c-puct" in command for command in commands)


def test_cli_axes_and_options_reach_the_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _forbid_matches(monkeypatch)
    argv = [
        "--model", "a.pt", "b.pt",
        "--out-dir", str(tmp_path / "grid"),
        "--c-puct", "1.25", "2",
        "--fpu", "none", "0",
        "--games", "40",
        "--simulations", "100",
        "--opening-moves", "3",
        "--device", "cpu",
        "--seed", "11",
        "--dry-run",
    ]  # fmt: skip
    search_grid.main(argv)
    commands = _commands(capsys.readouterr().out)
    control, grid = commands[0], commands[1:]
    assert "--model-c-puct" not in control
    assert [(p["--model-c-puct"][0], p["--model-fpu"][0]) for p in grid] == [
        ("1.25", "none"),
        ("1.25", "0.0"),
        ("2.0", "none"),
        ("2.0", "0.0"),
    ]
    for command in commands:
        assert command["--model"] == command["--opponent"] == ["a.pt", "b.pt"]
        assert command["--games"] == ["40"] and command["--simulations"] == ["100"]
        assert command["--opening-moves"] == ["3"] and command["--device"] == ["cpu"]


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        (["--c-puct", "0"], "expected"),
        (["--c-puct", "nan"], "expected"),
        (["--fpu", "-0.1"], "expected"),
        (["--games", "0"], "expected"),
        (["--opening-moves", "-1"], "expected"),
        (["--c-puct", "1.5", "1.5"], "duplicated"),
        (["--fpu", "none", "None"], "duplicated"),
    ],
)
def test_cli_rejects_a_bad_grid(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, extra: list[str], fragment: str
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        search_grid.main(["--model", "m.pt", "--out-dir", str(tmp_path / "grid"), "--dry-run", *extra])
    assert excinfo.value.code == 2
    assert fragment in _error_message(capsys)


def _scripted_games(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Stub game outcomes with Black always winning and count played games."""
    played: list[int] = []

    def scripted_game(**_ignored: Any) -> int:
        played.append(1)
        return BLACK

    monkeypatch.setattr(eval_match, "play_game", scripted_game)
    return played


def test_cli_end_to_end_with_the_real_eval_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    a, _b = checkpoints  # a.pt stores c_puct 1.5 and FPU None.
    played = _scripted_games(monkeypatch)
    out_dir = tmp_path / "grid"
    argv = [
        "--model", str(a),
        "--out-dir", str(out_dir),
        "--c-puct", "1.5", "2",
        "--fpu", "none", "0.2",
        "--games", "2",
        "--opening-moves", "0",
        "--device", "cpu",
        "--seed", "5",
    ]  # fmt: skip
    search_grid.main(argv)
    assert len(played) == 5 * 2  # One control and four points with two games per match.
    control = json.loads((out_dir / "control.json").read_text(encoding="utf-8"))
    assert (control["model_c_puct"], control["model_fpu"]) == (1.5, None)  # The control used the checkpoint's search settings.
    assert (control["opponent_c_puct"], control["opponent_fpu"]) == (1.5, None)
    assert control["seed"] == search_grid.point_seed(5, ControlPoint())
    for point in search_grid.build_points([1.5, 2.0], [None, 0.2]):
        match = json.loads((out_dir / f"{point.key}.json").read_text(encoding="utf-8"))
        assert (match["model_c_puct"], match["model_fpu"]) == (point.c_puct, point.fpu)
        assert (match["opponent_c_puct"], match["opponent_fpu"]) == (1.5, None)  # The opponent retained its saved search.
        assert match["seed"] == search_grid.point_seed(5, point)
    summary = json.loads((out_dir / "grid_summary.json").read_text(encoding="utf-8"))
    # Both the explicit control and the matching (1.5, None) point measure the control.
    assert summary["controls"] == ["control", "c1.5_fpu-none"]
    assert len(summary["points"]) == 5
    assert "D2 search grid" in capsys.readouterr().out

    search_grid.main(argv)  # Reuse results on a second run.
    assert len(played) == 5 * 2
    assert "skipping" in capsys.readouterr().out


def test_cli_control_shows_a_checkpoint_search_that_is_not_in_the_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """b.pt stores 1.25 / 0.2, outside this grid. The control must still measure it."""
    _a, b = checkpoints
    _scripted_games(monkeypatch)
    out_dir = tmp_path / "grid"
    search_grid.main(
        ["--model", str(b), "--out-dir", str(out_dir), "--c-puct", "2", "--fpu", "none",
         "--games", "2", "--opening-moves", "0", "--device", "cpu"]
    )  # fmt: skip
    summary = json.loads((out_dir / "grid_summary.json").read_text(encoding="utf-8"))
    assert summary["controls"] == ["control"]
    control = next(entry for entry in summary["points"] if entry["key"] == "control")
    assert (control["c_puct"], control["fpu"]) == (1.25, 0.2)
    assert "control" in _row(capsys.readouterr().out, ("1.25", "0.2"))


def test_cli_stops_with_the_file_name_when_a_result_is_corrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, _b = checkpoints
    _scripted_games(monkeypatch)
    out_dir = tmp_path / "grid"
    argv = ["--model", str(a), "--out-dir", str(out_dir), "--c-puct", "1.5", "--fpu", "none",
            "--games", "2", "--opening-moves", "0", "--device", "cpu"]  # fmt: skip
    search_grid.main(argv)
    (out_dir / "c1.5_fpu-none.json").write_text("{", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        search_grid.main(argv)
    assert "c1.5_fpu-none.json" in str(excinfo.value)
    assert excinfo.value.code not in (0, None)


def test_cli_refuses_a_directory_of_another_experiment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, _b = checkpoints
    played = _scripted_games(monkeypatch)
    out_dir = tmp_path / "grid"
    base = ["--model", str(a), "--out-dir", str(out_dir), "--c-puct", "1.5", "--fpu", "none",
            "--opening-moves", "0", "--device", "cpu"]  # fmt: skip
    search_grid.main([*base, "--games", "2"])
    before = len(played)
    with pytest.raises(SystemExit) as excinfo:
        search_grid.main([*base, "--games", "4"])
    assert "another experiment" in str(excinfo.value)
    assert len(played) == before


def test_cli_confirmation_command_from_the_summary_runs_in_the_real_eval_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """Execute the printed confirmation arguments with the real eval_match entry point."""
    a, _b = checkpoints
    played = _scripted_games(monkeypatch)
    out_dir = tmp_path / "grid"
    search_grid.main(
        ["--model", str(a), "--out-dir", str(out_dir), "--c-puct", "2", "--fpu", "0.2",
         "--games", "2", "--opening-moves", "0", "--device", "cpu", "--seed", "5"]
    )  # fmt: skip
    command = next(line for line in capsys.readouterr().out.splitlines() if "examples.eval_match" in line)
    assert not (out_dir / "confirm").exists()  # The match must create the missing confirmation directory.
    before = len(played)
    eval_match.main(shlex.split(command)[3:])  # Remove the Python module invocation before passing argv.
    assert len(played) - before == 4  # Use twice the games from the grid.
    confirm = json.loads((out_dir / "confirm" / "c2_fpu0.2.json").read_text(encoding="utf-8"))
    assert confirm["games"] == 4
    assert (confirm["model_c_puct"], confirm["model_fpu"]) == (2.0, 0.2)
    assert confirm["seed"] == search_grid.point_seed(5 + search_grid.CONFIRM_SEED_SHIFT, GridPoint(2.0, 0.2))
