"""Play a model against an external GTP engine, optionally with a separate scoring referee."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Final

from alphago_zero.cli_types import ensure_directory, finite_float, non_negative_int, positive_float, positive_int
from alphago_zero.gtp import action_to_vertex
from alphago_zero.gtp_client import GTPClient, GTPCommandError, GTPError, split_command
from alphago_zero.gtp_match import ExternalGame, MatchError, MatchResult, MatchSettings, run_match
from alphago_zero.sgf import game_to_sgf, save_sgf
from examples import eval_match

PLAY_GTP: Final = Path(__file__).with_name("play_gtp.py")
VOID_WARNING_SHARE: Final = 0.10
INTERRUPTED: Final = "interrupted by user (Ctrl+C). The unfinished game was excluded"
_OUTCOME_TEXT: Final = {1.0: "win", 0.5: "draw", 0.0: "loss"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play a model against an external GTP engine such as Pachi or GNU Go",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--checkpoint", type=Path, nargs="+", help="model checkpoints (multiple paths form an ensemble)")
    model.add_argument("--model-command", help="GTP engine launch command in place of --checkpoint")
    parser.add_argument("--simulations", type=positive_int, default=None, help="model MCTS simulations (with --checkpoint)")
    parser.add_argument("--device", default=None, help="cpu or cuda (with --checkpoint, default cpu)")
    parser.add_argument(
        "--model-args",
        default=None,
        help="extra play_gtp.py arguments as a single string (with --checkpoint). Use '=' for leading dashes: "
        "--model-args=--no-resign",
    )
    parser.add_argument("--model-name", default="model", help="model name for console, JSON and SGF")
    parser.add_argument("--opponent-command", required=True, help="opponent command, for example \"pachi -t =2000 -d0\"")
    parser.add_argument("--opponent-name", default=None, help="opponent name (defaults to program name)")
    parser.add_argument("--opponent-rating", type=finite_float, default=0.0, help="Elo rating of this opponent configuration")
    parser.add_argument("--referee-command", default=None, help="independent final_score referee (defaults to opponent)")
    parser.add_argument("--referee-name", default="referee", help="referee name for diagnostics")
    parser.add_argument("--games", type=positive_int, default=60, help="number of games with alternating colors")
    parser.add_argument("--board-size", type=positive_int, default=9)
    parser.add_argument("--komi", type=finite_float, default=7.5)
    parser.add_argument("--opening-moves", type=non_negative_int, default=4, help="random opening moves shared by each pair of games")
    parser.add_argument("--max-moves", type=positive_int, default=None, help="move limit (defaults to twice the board area)")
    parser.add_argument("--move-timeout", type=positive_float, default=120.0, help="timeout in seconds for genmove and final_score")
    parser.add_argument("--engine-timeout", type=positive_float, default=60.0, help="timeout in seconds for other commands")
    parser.add_argument("--startup-timeout", type=positive_float, default=180.0, help="startup timeout in seconds")
    parser.add_argument("--max-consecutive-voids", type=positive_int, default=3, help="consecutive void games allowed before aborting")
    parser.add_argument("--seed", type=int, default=0, help="seed for openings and the model")
    parser.add_argument("--json-out", default=None, help="write match results to this JSON file")
    parser.add_argument("--sgf-dir", default=None, help="directory for game records in SGF format")
    parser.add_argument("--dry-run", action="store_true", help="print commands and settings without starting engines")
    return parser


def build_model_command(args: argparse.Namespace) -> list[str]:
    """Build the model command from --model-command or the checkpoint-based GTP frontend."""
    if args.model_command is not None:
        return split_command(args.model_command)
    command = [sys.executable, str(PLAY_GTP), "--checkpoint", *map(str, args.checkpoint)]
    command += ["--device", args.device or "cpu", "--seed", str(args.seed)]
    if args.simulations is not None:
        command += ["--simulations", str(args.simulations)]
    if args.model_args and args.model_args.strip():
        command += split_command(args.model_args)
    return command


def _check_usage(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject checkpoint-only options that would otherwise be ignored by a custom model command."""
    if args.model_command is not None:
        ignored = [
            flag
            for flag, value in (("--simulations", args.simulations), ("--device", args.device), ("--model-args", args.model_args))
            if value is not None
        ]
        if ignored:
            parser.error(f"{', '.join(ignored)} apply to --checkpoint and cannot be used with --model-command")


def _settings(parser: argparse.ArgumentParser, args: argparse.Namespace) -> MatchSettings:
    try:
        return MatchSettings(
            games=args.games,
            board_size=args.board_size,
            komi=args.komi,
            opening_moves=args.opening_moves,
            move_timeout=args.move_timeout,
            max_moves=args.max_moves,
            seed=args.seed,
            max_consecutive_voids=args.max_consecutive_voids,
        )
    except ValueError as error:
        parser.error(str(error))


def _opponent_name(args: argparse.Namespace, command: Sequence[str]) -> str:
    return args.opponent_name or Path(command[0]).stem


def _make_client(command: Sequence[str], name: str, args: argparse.Namespace) -> GTPClient:
    return GTPClient(command, name=name, default_timeout=args.engine_timeout, startup_timeout=args.startup_timeout)


def _identify(client: GTPClient) -> dict[str, str]:
    """Read the engine name and optional version, reporting startup connection errors."""
    name = client.command("name")
    try:
        version = client.command("version")
    except GTPCommandError:
        version = ""
    return {"name": name, "version": version}


def game_record(game: ExternalGame, board_size: int) -> dict[str, Any]:
    outcome = game.model_outcome
    return {
        "number": game.index + 1,
        "model_is_black": game.model_is_black,
        "outcome": "void" if outcome is None else _OUTCOME_TEXT[outcome],
        "result": game.result,
        "resigned": game.resigned,
        "move_count": len(game.moves),
        "opening": [action_to_vertex(action, board_size) for action in game.opening],
        "void_reason": game.void_reason,
        "seconds": round(game.seconds, 2),
    }


def format_game_line(game: ExternalGame, total: int) -> str:
    side = "black" if game.model_is_black else "white"
    head = f"[{game.index + 1:>{len(str(total))}}/{total}] model={side:<6}"
    tail = f"moves={len(game.moves)} {game.seconds:.1f}s"
    if game.model_outcome is None:
        return f"{head} VOID: {game.void_reason} ({tail})"
    return f"{head} {game.result:<8} {_OUTCOME_TEXT[game.model_outcome]:<9} {tail}"


def summary_payload(
    args: argparse.Namespace,
    settings: MatchSettings,
    result: MatchResult,
    *,
    commands: dict[str, list[str] | None],
    engines: dict[str, dict[str, str] | None],
    opponent_name: str,
    aborted: str | None,
) -> dict[str, Any]:
    valid = result.valid_games
    point, lower, upper = eval_match.wilson_interval(result.points, valid)
    black_games, black_points = result.side_record(as_black=True)
    white_games, white_points = result.side_record(as_black=False)

    def elo(value: float) -> float | None:
        return round(eval_match.elo_from_win_rate(value, args.opponent_rating), 1) if valid else None

    return {
        "model": args.model_name,
        "opponent": opponent_name,
        "games_requested": settings.games,
        "games": valid,  # Use valid games as the denominator for scores and intervals.
        "games_played": len(result.games),
        "wins": result.wins,
        "losses": result.losses,
        "draws": result.draws,
        "voids": result.voids,
        "win_rate": round(point, 4) if valid else None,
        "win_rate_ci95": [round(lower, 4), round(upper, 4)] if valid else None,
        "as_black_games": black_games,
        "as_black_win_rate": round(black_points / black_games, 4) if black_games else None,
        "as_white_games": white_games,
        "as_white_win_rate": round(white_points / white_games, 4) if white_games else None,
        "opponent_rating": args.opponent_rating,
        "elo_estimate": elo(point),
        "elo_ci95": [elo(lower), elo(upper)],
        "board_size": settings.board_size,
        "komi": settings.komi,
        "opening_moves": settings.opening_moves,
        "max_moves": settings.max_moves,
        "move_timeout": settings.move_timeout,
        "seed": settings.seed,
        "scoring": "referee" if commands["referee"] is not None else "opponent",
        "commands": commands,
        "engines": engines,
        "aborted": aborted,
        "games_detail": [game_record(game, settings.board_size) for game in result.games],
    }


def format_summary(payload: dict[str, Any]) -> str:
    """Report match results, confidence interval, Elo and any warnings."""
    lines = [
        f"Result: {payload['model']} against {payload['opponent']} wins: {payload['wins']}, "
        f"losses: {payload['losses']}, draws: {payload['draws']}, void: {payload['voids']} "
        f"(played {payload['games_played']} of {payload['games_requested']})"
    ]
    if payload["win_rate"] is not None:
        lower, upper = payload["win_rate_ci95"]
        elo_low, elo_high = payload["elo_ci95"]
        lines.append(f"win-rate {payload['win_rate']:.3f}, 95% Wilson [{lower:.3f}, {upper:.3f}] across {payload['games']} games")
        lines.append(
            f"Elo relative to the opponent configuration (rating {payload['opponent_rating']:g}): "
            f"{payload['elo_estimate']:+.0f} [{elo_low:+.0f}, {elo_high:+.0f}]"
        )
    if payload["games_played"] and payload["voids"] / payload["games_played"] > VOID_WARNING_SHARE:
        lines.append(
            f"Warning: void games: {payload['voids']} of {payload['games_played']} (> 10 %): "
            "check rules, engine commands and timeouts. See games_detail in the JSON report"
        )
    if payload["aborted"]:
        lines.append(f"MATCH ABORTED: {payload['aborted']}")
    return "\n".join(lines)


def _write_sgf(game: ExternalGame, args: argparse.Namespace, settings: MatchSettings, opponent_name: str) -> None:
    model, opponent = args.model_name, opponent_name
    sgf = game_to_sgf(
        board_size=settings.board_size,
        komi=settings.komi,
        moves=list(game.moves),
        result=game.result,
        black_name=model if game.model_is_black else opponent,
        white_name=opponent if game.model_is_black else model,
        comment=game.void_reason,
    )
    save_sgf(Path(args.sgf_dir) / f"game_{game.index + 1:03d}.sgf", sgf)


def _print_header(
    args: argparse.Namespace, settings: MatchSettings, commands: dict[str, list[str] | None], opponent_name: str
) -> None:
    print(f"model={args.model_name} command={commands['model']}", flush=True)
    print(f"opponent={opponent_name} command={commands['opponent']} rating={args.opponent_rating:g}", flush=True)
    print(f"referee={'none (opponent scores)' if commands['referee'] is None else commands['referee']}", flush=True)
    print(
        f"board={settings.board_size} komi={settings.komi:g} games={settings.games} "
        f"opening_moves={settings.opening_moves} seed={settings.seed} move_timeout={settings.move_timeout:g}",
        flush=True,
    )


def _prepare_outputs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Create result directories before committing time to the series."""
    if args.json_out:
        ensure_directory(parser, Path(args.json_out).parent, "--json-out")
    if args.sgf_dir:
        ensure_directory(parser, Path(args.sgf_dir), "--sgf-dir")


def _play(
    args: argparse.Namespace, settings: MatchSettings, commands: dict[str, list[str] | None], opponent_name: str
) -> tuple[MatchResult, dict[str, dict[str, str] | None], str | None]:
    """Start engines, play the series and close all subprocesses. Return results, engine metadata and any
    abort reason.
    """
    engines: dict[str, dict[str, str] | None] = {"model": None, "opponent": None, "referee": None}
    games: list[ExternalGame] = []

    def on_game(game: ExternalGame) -> None:
        games.append(game)
        print(format_game_line(game, settings.games), flush=True)
        if args.sgf_dir:
            _write_sgf(game, args, settings, opponent_name)

    with ExitStack() as stack:
        model_command, opponent_command, referee_command = commands["model"], commands["opponent"], commands["referee"]
        assert model_command is not None and opponent_command is not None
        model = stack.enter_context(_make_client(model_command, args.model_name, args))
        opponent = stack.enter_context(_make_client(opponent_command, opponent_name, args))
        referee = (
            stack.enter_context(_make_client(referee_command, args.referee_name, args))
            if referee_command is not None
            else None
        )
        engines["model"], engines["opponent"] = _identify(model), _identify(opponent)
        engines["referee"] = _identify(referee) if referee is not None else None
        print(f"engines: {json.dumps(engines, ensure_ascii=False)}", flush=True)
        try:
            return run_match(model, opponent, settings, referee=referee, on_game=on_game), engines, None
        except MatchError as error:
            return MatchResult(error.games), engines, str(error)
        except KeyboardInterrupt:
            return MatchResult(tuple(games)), engines, INTERRUPTED


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _check_usage(parser, args)
    settings = _settings(parser, args)
    try:
        commands: dict[str, list[str] | None] = {
            "model": build_model_command(args),
            "opponent": split_command(args.opponent_command),
            "referee": split_command(args.referee_command) if args.referee_command is not None else None,
        }
    except ValueError as error:
        parser.error(str(error))
    opponent_name = _opponent_name(args, commands["opponent"] or [])
    _print_header(args, settings, commands, opponent_name)
    if args.dry_run:
        print("--dry-run: no engines started", flush=True)
        return 0
    _prepare_outputs(parser, args)
    try:
        result, engines, aborted = _play(args, settings, commands, opponent_name)
    except GTPError as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    payload = summary_payload(
        args, settings, result, commands=commands, engines=engines, opponent_name=opponent_name, aborted=aborted
    )
    print(format_summary(payload), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written {args.json_out}", flush=True)
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
