"""Compare c_puct and FPU settings on fixed weights, then confirm the best candidate with a fresh seed."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Final

from alphago_zero.cli_types import fpu_value, non_negative_int, positive_float, positive_int
from examples import eval_match

DEFAULT_C_PUCT: Final[tuple[float, ...]] = (1.1, 1.5, 2.0, 2.5)
DEFAULT_FPU: Final[tuple[float | None, ...]] = (None, 0.2, 0.4)

CONTROL_KEY: Final = "control"
MANIFEST_NAME: Final = "grid_manifest.json"
SUMMARY_NAME: Final = "grid_summary.json"
CONFIRM_DIR: Final = "confirm"
CONFIRM_SEED_SHIFT: Final = 1000  # Use a different seed for confirmation.
SEED_SPREAD: Final = 100_000

_REQUIRED_FIELDS: Final = (
    "games",
    "wins",
    "losses",
    "win_rate",
    "win_rate_ci95",
    "elo_estimate",
    "model_c_puct",
    "model_fpu",
    "opponent_c_puct",
    "opponent_fpu",
)


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One search grid point. None assigns zero Q to unvisited edges."""

    c_puct: float
    fpu: float | None

    @property
    def key(self) -> str:
        """Build a distinct filename for each setting, including FPU zero versus None."""
        fpu = "-none" if self.fpu is None else f"{self.fpu:.12g}"
        return f"c{self.c_puct:.12g}_fpu{fpu}"

    def override_argv(self) -> list[str]:
        """Return eval_match arguments for this model-side override."""
        fpu = "none" if self.fpu is None else str(self.fpu)
        return ["--model-c-puct", str(self.c_puct), "--model-fpu", fpu]

    def mismatch(self, raw: dict[str, Any]) -> str | None:
        """Explain mismatched cached results, or return None for a match."""
        if (raw["model_c_puct"], raw["model_fpu"]) == (self.c_puct, self.fpu):
            return None
        return (
            f"c_puct={raw['model_c_puct']}, fpu={raw['model_fpu']}, "
            f"expected c_puct={self.c_puct}, fpu={self.fpu}"
        )


@dataclass(frozen=True, slots=True)
class ControlPoint:
    """Control match with identical weights and saved search settings on both sides."""

    @property
    def key(self) -> str:
        return CONTROL_KEY

    def override_argv(self) -> list[str]:
        """Use checkpoint search settings without overrides for the control."""
        return []

    def mismatch(self, raw: dict[str, Any]) -> str | None:
        """Reject a cached control whose sides used different search settings."""
        model = (raw["model_c_puct"], raw["model_fpu"])
        opponent = (raw["opponent_c_puct"], raw["opponent_fpu"])
        if model == opponent:
            return None
        return (
            f"model c_puct={model[0]}, fpu={model[1]}, opponent c_puct={opponent[0]}, "
            f"fpu={opponent[1]}. The control requires identical search settings"
        )


Point = GridPoint | ControlPoint


@dataclass(frozen=True, slots=True)
class GridOptions:
    """Experiment inputs, game count and grid points. The opponent uses the same weights."""

    model: tuple[str, ...]
    out_dir: Path
    points: tuple[Point, ...]
    games: int = 120
    simulations: int | None = None
    opening_moves: int = 6
    device: str = "cuda"
    seed: int = 1

    def manifest(self) -> dict[str, Any]:
        """Experiment identity excludes device and grid points so further points can be appended."""
        return {
            "model": list(self.model),
            "games": self.games,
            "simulations": self.simulations,
            "opening_moves": self.opening_moves,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class PointResult:
    """A grid point result with the effective settings read from the match JSON."""

    point: Point
    c_puct: float
    fpu: float | None
    games: int
    wins: int
    losses: int
    win_rate: float
    ci_low: float
    ci_high: float
    elo: float
    is_control: bool
    source: str  # Record whether this run played the match or found a compatible cached result.

    @property
    def verdict(self) -> str:
        """Use the Wilson interval to classify a result relative to an even score."""
        if self.ci_low > 0.5:
            return "better"
        if self.ci_high < 0.5:
            return "worse"
        return "parity"


# ----------------------------------------------------------------------
# Grid construction and match commands
# ----------------------------------------------------------------------
def build_points(c_puct_values: Sequence[float], fpu_values: Sequence[float | None]) -> list[GridPoint]:
    """Build the Cartesian product in c_puct-major order. Reject empty axes and duplicate values."""
    for name, values in (("c_puct", c_puct_values), ("fpu", fpu_values)):
        if not values:
            raise ValueError(f"axis {name} is empty. Supply at least one value")
        if len(set(values)) != len(values):
            raise ValueError(f"axis values {name} are duplicated: {list(values)}")
    return [GridPoint(c, fpu) for c, fpu in product(c_puct_values, fpu_values)]


def plan_points(
    c_puct_values: Sequence[float], fpu_values: Sequence[float | None], *, control: bool
) -> list[Point]:
    """Place the optional control before the parameter grid."""
    grid: list[Point] = [*build_points(c_puct_values, fpu_values)]
    return [ControlPoint(), *grid] if control else grid


def point_seed(base_seed: int, point: Point) -> int:
    """Derive the seed from a stable point key independently of grid order."""
    return base_seed + zlib.crc32(point.key.encode("utf-8")) % SEED_SPREAD


def result_path(options: GridOptions, point: Point) -> Path:
    """Return the result file for one grid point."""
    return options.out_dir / f"{point.key}.json"


def match_argv(options: GridOptions, point: Point) -> list[str]:
    """Build an eval_match command that overrides the model search and keeps the opponent at checkpoint settings."""
    argv = [
        "--model", *options.model,
        "--opponent", *options.model,
        "--games", str(options.games),
        "--opening-moves", str(options.opening_moves),
        "--device", options.device,
        "--seed", str(point_seed(options.seed, point)),
    ]  # fmt: skip
    if options.simulations is not None:
        argv += ["--simulations", str(options.simulations)]
    return [*argv, *point.override_argv(), "--json-out", str(result_path(options, point))]


def _command(argv: Sequence[str]) -> str:
    return shlex.join(["python", "-m", "examples.eval_match", *argv])


def confirm_command(options: GridOptions, point: Point) -> str:
    """Double the games and use a different seed and output directory for confirmation."""
    confirm = dataclasses.replace(
        options,
        games=options.games * 2,
        seed=options.seed + CONFIRM_SEED_SHIFT,
        out_dir=options.out_dir / CONFIRM_DIR,
    )
    return _command(match_argv(confirm, point))


# ----------------------------------------------------------------------
# Result loading and match execution
# ----------------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and identify corrupt or unreadable results by filename."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name}: corrupt or unreadable file ({exc}). Remove this result to replay the match") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: corrupt file, expected a JSON object")
    return raw


def load_point_result(point: Point, path: Path, source: str) -> PointResult:
    """Read a complete result and verify that it belongs to the requested point."""
    raw = _read_json(path)
    missing = [name for name in _REQUIRED_FIELDS if name not in raw]
    if missing:
        raise ValueError(f"{path.name}: result is missing fields: {', '.join(missing)}")
    problem = point.mismatch(raw)
    if problem is not None:
        raise ValueError(f"{path.name}: result belongs to another grid point ({problem})")
    try:
        ci_low, ci_high = raw["win_rate_ci95"]
    except (TypeError, ValueError):
        raise ValueError(f"{path.name}: win_rate_ci95 must be a [lower, upper] pair") from None
    model = (raw["model_c_puct"], raw["model_fpu"])
    return PointResult(
        point=point,
        c_puct=float(model[0]),
        fpu=None if model[1] is None else float(model[1]),
        games=int(raw["games"]),
        wins=int(raw["wins"]),
        losses=int(raw["losses"]),
        win_rate=float(raw["win_rate"]),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        elo=float(raw["elo_estimate"]),
        is_control=model == (raw["opponent_c_puct"], raw["opponent_fpu"]),
        source=source,
    )


def _claim_out_dir(options: GridOptions) -> None:
    """Write a manifest on first use and verify experiment identity on later runs."""
    options.out_dir.mkdir(parents=True, exist_ok=True)
    path = options.out_dir / MANIFEST_NAME
    current = options.manifest()
    if not path.exists():
        path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    saved = _read_json(path)
    differing = [name for name, value in current.items() if saved.get(name) != value]
    if differing:
        raise ValueError(
            f"{path.name}: directory contains another experiment (different fields: {', '.join(differing)}). "
            "Use a new --out-dir for a different experiment"
        )


def _echo(line: str) -> None:
    print(line, flush=True)


def run_grid(
    options: GridOptions,
    runner: Callable[[list[str]], None] | None = None,
    echo: Callable[[str], None] = _echo,
) -> list[PointResult]:
    """Run missing points and read valid results for the full grid."""
    run = runner if runner is not None else eval_match.main
    _claim_out_dir(options)
    total = len(options.points)
    results: list[PointResult] = []
    for index, point in enumerate(options.points, start=1):
        path = result_path(options, point)
        if path.exists():
            echo(f"[{index}/{total}] {point.key}: result already exists ({path.name}), skipping")
            source = "cached"
        else:
            echo(f"[{index}/{total}] {point.key}: match, {options.games} games")
            run(match_argv(options, point))
            source = "played"
        results.append(load_point_result(point, path, source))
    return results


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
_ROW = "{c:<8}{fpu:<7}{games:>6}{wins:>7}{rate:>10}  {ci:<16}{elo:>6}  {verdict}"


def _ranked(results: Sequence[PointResult]) -> list[PointResult]:
    """Sort by decreasing win rate and preserve run order on ties."""
    return sorted(results, key=lambda result: result.win_rate, reverse=True)


def _candidate(ranked: Sequence[PointResult]) -> PointResult | None:
    """Select the best non-control point. A control measures match noise."""
    return next((result for result in ranked if not result.is_control), None)


def _fpu_text(fpu: float | None) -> str:
    return "none" if fpu is None else f"{fpu:g}"


def _interval(result: PointResult) -> str:
    return f"[{result.ci_low:.3f}, {result.ci_high:.3f}]"


def _table_row(result: PointResult) -> str:
    verdict = result.verdict + (" (control)" if result.is_control else "")
    return _ROW.format(
        c=f"{result.c_puct:g}",
        fpu=_fpu_text(result.fpu),
        games=result.games,
        wins=result.wins,
        rate=f"{result.win_rate:.3f}",
        ci=_interval(result),
        elo=f"{result.elo:+.0f}",
        verdict=verdict,
    )


def _control_notes(ranked: Sequence[PointResult]) -> list[str]:
    if any(result.is_control for result in ranked):
        return [
            "The control uses identical weights and checkpoint settings on both sides. "
            "Its deviation from 0.5 measures match noise."
        ]
    return [
        "No grid point measures the control: --no-control was specified and no setting "
        "matched the checkpoint. An identical-player comparison is unavailable."
    ]


def _candidate_notes(ranked: Sequence[PointResult], options: GridOptions) -> list[str]:
    candidate = _candidate(ranked)
    if candidate is None:
        return []
    notes = [
        "",
        f"Candidate: c_puct={candidate.c_puct:g} fpu={_fpu_text(candidate.fpu)} — "
        f"win-rate {candidate.win_rate:.3f} {_interval(candidate)}, verdict: {candidate.verdict}",
    ]
    compared = sum(1 for result in ranked if not result.is_control)
    selection = (
        f"Selection bias: the best of {compared} settings tends to overestimate its strength "
        "(for 12 equally strong settings with 120 games each, the best averages about 0.57). "
        "Treat it as a candidate and confirm it "
        if compared >= 2
        else "Confirm "
    )
    notes.append(selection + "in an independent match with twice as many games and a fresh seed:")
    notes.append("  " + confirm_command(options, candidate.point))
    return notes


def format_summary(results: Sequence[PointResult], options: GridOptions) -> str:
    """Rank points and print the control, candidate and confirmation command."""
    ranked = _ranked(results)
    sims = "from checkpoints" if options.simulations is None else str(options.simulations)
    header = _ROW.format(
        c="c_puct", fpu="fpu", games="games", wins="wins", rate="win-rate", ci="95% Wilson CI", elo="Elo",
        verdict="verdict",
    )  # fmt: skip
    lines = [
        f"D2 search grid: weights {' + '.join(options.model)}; opponent uses the same weights and checkpoint search",
        f"games per point: {options.games}, simulations: {sims}, opening-moves: {options.opening_moves}, "
        f"seed: {options.seed}",
        "Elo is relative to this opponent. Verdicts use Wilson interval bounds.",
        "",
        header,
        *(_table_row(result) for result in ranked),
        "",
        *_control_notes(ranked),
        *_candidate_notes(ranked, options),
    ]
    return "\n".join(lines)


def _entry(result: PointResult) -> dict[str, Any]:
    return {
        "key": result.point.key,
        "c_puct": result.c_puct,
        "fpu": result.fpu,
        "games": result.games,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": round(result.win_rate, 4),
        "ci95": [round(result.ci_low, 4), round(result.ci_high, 4)],
        "elo": round(result.elo, 1),
        "verdict": result.verdict,
        "is_control": result.is_control,
        "source": result.source,
    }


def summary_payload(results: Sequence[PointResult], options: GridOptions) -> dict[str, Any]:
    """Build grid_summary.json in the same order as the printed table."""
    ranked = _ranked(results)
    candidate = _candidate(ranked)
    return {
        "options": options.manifest(),
        "points": [_entry(result) for result in ranked],
        "candidate": None if candidate is None else candidate.point.key,
        "controls": [result.point.key for result in ranked if result.is_control],
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare c_puct and FPU settings with fixed weights using eval_match",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        metavar="CKPT",
        help="weights (.pt) or ensemble members. The opponent uses the same weights and checkpoint search",
    )
    parser.add_argument("--out-dir", required=True, help="experiment directory for the manifest, results and summary")
    parser.add_argument(
        "--c-puct",
        nargs="+",
        type=positive_float,
        default=None,
        metavar="C",
        help=f"c_puct values (default {' '.join(f'{c:g}' for c in DEFAULT_C_PUCT)})",
    )
    parser.add_argument(
        "--fpu",
        nargs="+",
        type=fpu_value,
        default=None,
        metavar="F",
        help="FPU reductions (nonnegative numbers or none) "
        f"(default {' '.join('none' if f is None else f'{f:g}' for f in DEFAULT_FPU)})",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="omit the identical-search control used to measure match noise",
    )
    parser.add_argument("--games", type=positive_int, default=120, help="games per grid point with alternating colors")
    parser.add_argument(
        "--simulations",
        type=positive_int,
        default=None,
        help="set equal simulation budgets for both sides (defaults to checkpoint budgets)",
    )
    parser.add_argument(
        "--opening-moves",
        type=non_negative_int,
        default=6,
        help="opening moves sampled from visits. Zero can reduce a deterministic self-match to two games",
    )
    parser.add_argument("--device", default="cuda", help="cuda|cpu")
    parser.add_argument("--seed", type=int, default=12345, help="base seed with a stable offset for each point")
    parser.add_argument("--dry-run", action="store_true", help="show match commands without running them")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        points = plan_points(
            list(DEFAULT_C_PUCT if args.c_puct is None else args.c_puct),
            list(DEFAULT_FPU if args.fpu is None else args.fpu),
            control=not args.no_control,
        )
    except ValueError as exc:
        parser.error(str(exc))
    options = GridOptions(
        model=tuple(args.model),
        out_dir=Path(args.out_dir),
        points=tuple(points),
        games=args.games,
        simulations=args.simulations,
        opening_moves=args.opening_moves,
        device=args.device,
        seed=args.seed,
    )
    if args.dry_run:
        print(f"# D2: matches {len(points)}, games per match: {options.games}; --dry-run: no matches played")
        for point in options.points:
            print(_command(match_argv(options, point)))
        return
    try:
        results = run_grid(options)
    except ValueError as exc:
        raise SystemExit(f"search_grid: {exc}") from exc
    print(format_summary(results, options))
    summary_path = options.out_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary_payload(results, options), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written {summary_path}", flush=True)


if __name__ == "__main__":
    main()
