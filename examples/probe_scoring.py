"""Compare three scorers on recorded final positions: raw area, network ownership and an external referee."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from alphago_zero.cli_types import finite_float, positive_int
from alphago_zero.config import (
    AlphaGoZeroConfig,
    GameConfig,
    NetworkConfig,
    RuntimeConfig,
    SearchConfig,
    TrainingConfig,
)
from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.gtp import action_to_vertex
from alphago_zero.gtp_client import GTPClient, GTPError, split_command
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.sgf import ParsedGame, parse_sgf, replay_game


@dataclass(slots=True)
class GameScoring:
    """Three verdicts for one final position. Positive margins favor Black."""

    source: str
    moves: int
    end_kind: str  # double_pass / resign / max_moves / unfinished
    our_margin: float | None = None
    ownership_margin: float | None = None
    judge_margin: float | None = None
    error: str | None = None


def winner_of(margin: float) -> str:
    """Map a signed margin to B, W or 0 for a draw."""
    if margin > 0:
        return "B"
    if margin < 0:
        return "W"
    return "0"


def parse_final_score(text: str) -> float:
    """Convert a numeric GTP final_score to a signed Black margin."""
    token = text.strip()
    if token == "0":
        return 0.0
    if len(token) > 2 and token[0] in "BW" and token[1] == "+":
        try:
            margin = float(token[2:])
        except ValueError as exc:
            raise ValueError(f"unrecognized final_score: {text!r}") from exc
        return margin if token[0] == "B" else -margin
    raise ValueError(f"unrecognized final_score: {text!r}")


def score_from_ownership(ownership_black: np.ndarray, *, komi: float, threshold: float) -> float:
    """Estimate a Black margin by assigning only sufficiently confident ownership predictions."""
    black_points = int((ownership_black > threshold).sum())
    white_points = int((ownership_black < -threshold).sum())
    return float(black_points - white_points) - float(komi)


@torch.no_grad()
def ownership_margin(
    model: AlphaGoZeroNet, state: GoState, *, device: torch.device, threshold: float
) -> float | None:
    """Return the network ownership margin, or None when the model lacks auxiliary heads."""
    encoded = torch.from_numpy(state.encode()).unsqueeze(0).to(device)
    outputs = model.forward_full(encoded)
    ownership = outputs.get("ownership")
    if ownership is None:
        return None
    board_ownership = ownership[0].detach().cpu().numpy().reshape(state.board_size, state.board_size)
    if state.to_play == WHITE:
        # Convert from player-to-move to Black's perspective.
        board_ownership = -board_ownership
    return score_from_ownership(board_ownership, komi=state.komi, threshold=threshold)


def end_kind_of(state: GoState, result: str | None) -> str:
    """Identify resignation from the result property and other endings from the board."""
    if result is not None and result.strip().endswith("+R"):
        return "resign"
    if state.consecutive_passes >= 2:
        return "double_pass"
    if state.move_number >= state.max_moves:
        return "max_moves"
    return "unfinished"


def compare(records: list[GameScoring], left: str, right: str) -> dict[str, Any]:
    """Compare two scorers on games with both verdicts."""
    pairs = [
        (getattr(record, left), getattr(record, right))
        for record in records
        if getattr(record, left) is not None and getattr(record, right) is not None
    ]
    if not pairs:
        return {"compared": 0, "winner_divergences": 0, "winner_divergence_share": None, "mean_abs_margin_delta": None}
    divergences = sum(1 for a, b in pairs if winner_of(a) != winner_of(b))
    return {
        "compared": len(pairs),
        "winner_divergences": divergences,
        "winner_divergence_share": divergences / len(pairs),
        "mean_abs_margin_delta": float(np.mean([abs(a - b) for a, b in pairs])),
    }


def summarize(records: list[GameScoring]) -> dict[str, Any]:
    """Group game endings and report pairwise scorer differences on double-pass games."""
    scored = [record for record in records if record.end_kind == "double_pass" and record.error is None]
    return {
        "games_total": len(records),
        "double_pass": len(scored),
        "resign": sum(1 for r in records if r.end_kind == "resign"),
        "max_moves": sum(1 for r in records if r.end_kind == "max_moves"),
        "unfinished": sum(1 for r in records if r.end_kind == "unfinished"),
        "errors": sum(1 for r in records if r.error is not None),
        "our_vs_judge": compare(scored, "our_margin", "judge_margin"),
        "our_vs_ownership": compare(scored, "our_margin", "ownership_margin"),
        "ownership_vs_judge": compare(scored, "ownership_margin", "judge_margin"),
    }


def collect_sgf_paths(inputs: list[Path], *, limit: int | None) -> list[Path]:
    """Collect SGF files recursively and sort the paths for reproducibility."""
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("*.sgf")))
        elif item.suffix.lower() == ".sgf":
            paths.append(item)
        else:
            raise ValueError(f"--sgf: expected an .sgf file or directory: {item}")
    if limit is not None:
        paths = paths[:limit]
    return paths


def _section(cls: type[Any], raw: dict[str, Any]) -> Any:
    """Reconstruct a checkpoint config section using its recognized fields."""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in names})


def load_model(checkpoint: Path, *, device: torch.device) -> AlphaGoZeroNet:
    """Load the scoring network from its configuration and state dictionary."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    raw = payload["config"]
    config = AlphaGoZeroConfig(
        game=_section(GameConfig, raw["game"]),
        network=_section(NetworkConfig, raw["network"]),
        search=_section(SearchConfig, raw["search"]),
        training=_section(TrainingConfig, raw["training"]),
        runtime=_section(RuntimeConfig, raw["runtime"]),
    )
    model = AlphaGoZeroNet(config.network).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def judge_margin(
    judge: GTPClient, *, path: Path, parsed: ParsedGame, mode: str
) -> float:
    """Ask an external referee for the final margin by loading the SGF or replaying its moves."""
    if mode == "loadsgf":
        judge.command(f"loadsgf {path}")
    else:
        size = parsed.board_size
        judge.command(f"boardsize {size}")
        judge.command("clear_board")
        judge.command(f"komi {parsed.komi:g}")
        for color, action in parsed.moves:
            who = "b" if color == BLACK else "w"
            judge.command(f"play {who} {action_to_vertex(action, size)}")
    return parse_final_score(judge.command("final_score"))


def score_games(
    paths: list[Path],
    *,
    model: AlphaGoZeroNet | None,
    device: torch.device,
    ownership_threshold: float,
    judge_command: list[str] | None,
    judge_mode: str = "loadsgf",
) -> list[GameScoring]:
    """Evaluate all games with available scorers and retain parse failures as error records."""
    if judge_command is None:
        return _score_games(paths, model=model, device=device,
                            ownership_threshold=ownership_threshold, judge=None, judge_mode=judge_mode)
    with GTPClient(judge_command, name="judge", default_timeout=120, startup_timeout=180) as judge:
        return _score_games(paths, model=model, device=device,
                            ownership_threshold=ownership_threshold, judge=judge, judge_mode=judge_mode)


def _score_games(
    paths: list[Path],
    *,
    model: AlphaGoZeroNet | None,
    device: torch.device,
    ownership_threshold: float,
    judge: GTPClient | None,
    judge_mode: str = "loadsgf",
) -> list[GameScoring]:
    records: list[GameScoring] = []
    for path in paths:
        record = GameScoring(source=str(path), moves=0, end_kind="unfinished")
        records.append(record)
        try:
            parsed = parse_sgf(path.read_text(encoding="utf-8"))
            state = replay_game(parsed)
        except (OSError, ValueError) as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            continue
        record.moves = len(parsed.moves)
        record.end_kind = end_kind_of(state, parsed.result)
        if record.end_kind != "double_pass":
            continue
        black_score, white_score = state.area_score()
        record.our_margin = black_score - white_score
        if model is not None:
            record.ownership_margin = ownership_margin(
                model, state, device=device, threshold=ownership_threshold
            )
        if judge is not None:
            try:
                record.judge_margin = judge_margin(judge, path=path, parsed=parsed, mode=judge_mode)
            except (GTPError, ValueError) as exc:  # Keep this referee error in the record and continue with the next game.
                record.error = f"judge: {type(exc).__name__}: {exc}"
    return records


def _format_share(block: dict[str, Any]) -> str:
    if not block["compared"]:
        return "no games in common"
    share = block["winner_divergence_share"]
    return (
        f"games {block['compared']}, winner disagreements {block['winner_divergences']} "
        f"({share:.1%}), mean absolute margin difference {block['mean_abs_margin_delta']:.2f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare area_score, the ownership head and an external referee on recorded games"
    )
    parser.add_argument("--sgf", type=Path, action="append", required=True,
                        help=".sgf file or game directory (repeatable)")
    parser.add_argument("--limit", type=positive_int, default=None,
                        help="maximum games after sorting (default all)")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional checkpoint with an ownership head")
    parser.add_argument("--ownership-threshold", type=finite_float, default=0.5,
                        help="ownership confidence threshold in [0, 1) (default 0.5)")
    parser.add_argument("--judge-command", type=str, default=None,
                        help="external GTP scoring command, e.g. \"gnugo --mode gtp --chinese-rules --positional-superko\"")
    parser.add_argument("--judge-mode", choices=("loadsgf", "replay"), default="loadsgf",
                        help="set the referee position using loadsgf (GNU Go) or replay (Pachi)")
    parser.add_argument("--device", default="cuda", help="device for ownership inference (default cuda)")
    parser.add_argument("--json-out", type=Path, default=None, help="write all game records to JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.ownership_threshold < 1.0:
        raise SystemExit("--ownership-threshold must be in [0, 1)")

    paths = collect_sgf_paths(args.sgf, limit=args.limit)
    if not paths:
        raise SystemExit("--sgf: no .sgf files found")

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device=device) if args.checkpoint else None
    judge_command = split_command(args.judge_command) if args.judge_command else None

    records = score_games(
        paths,
        model=model,
        device=device,
        ownership_threshold=args.ownership_threshold,
        judge_command=judge_command,
        judge_mode=args.judge_mode,
    )
    summary = summarize(records)

    print(f"Total games: {summary['games_total']} "
          f"(double pass: {summary['double_pass']}, resign: {summary['resign']}, "
          f"max_moves: {summary['max_moves']}, incomplete: {summary['unfinished']}, "
          f"errors: {summary['errors']})")
    print(f"area vs referee:     {_format_share(summary['our_vs_judge'])}")
    print(f"area vs ownership: {_format_share(summary['our_vs_ownership'])}")
    print(f"ownership vs referee: {_format_share(summary['ownership_vs_judge'])}")

    if args.json_out is not None:
        payload = {
            "summary": summary,
            "games": [
                {
                    "source": r.source,
                    "moves": r.moves,
                    "end_kind": r.end_kind,
                    "our_margin": r.our_margin,
                    "ownership_margin": r.ownership_margin,
                    "judge_margin": r.judge_margin,
                    "error": r.error,
                }
                for r in records
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
