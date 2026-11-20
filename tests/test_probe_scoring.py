"""Check terminal scoring diagnostics against independent model and referee stubs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.sgf import game_to_sgf, parse_sgf
from examples import probe_scoring
from examples.probe_scoring import (
    GameScoring,
    end_kind_of,
    parse_final_score,
    score_from_ownership,
    summarize,
    winner_of,
)


def test_parse_final_score_variants() -> None:
    assert parse_final_score("B+3.5") == 3.5
    assert parse_final_score("W+12") == -12.0
    assert parse_final_score("  W+0.5 ") == -0.5
    assert parse_final_score("0") == 0.0
    for bad in ("B+R", "?", "", "B+", "X+1.5"):
        with pytest.raises(ValueError):
            parse_final_score(bad)


def test_winner_of_sign() -> None:
    assert winner_of(2.5) == "B"
    assert winner_of(-0.5) == "W"
    assert winner_of(0.0) == "0"


def test_score_from_ownership_counts_only_confident_points() -> None:
    ownership = np.array(
        [[0.9, 0.8, -0.9], [0.2, -0.7, 0.0], [-0.2, 0.6, -0.6]], dtype=np.float32
    )
    # Three confident black points and three confident white points leave the rest neutral.
    assert score_from_ownership(ownership, komi=1.5, threshold=0.5) == -1.5
    # At confidence 0.95 no points qualify, leaving only komi.
    assert score_from_ownership(ownership, komi=1.5, threshold=0.95) == -1.5


def test_end_kind_classification() -> None:
    state = GoState.new_game(board_size=5, komi=2.5)
    passed = state.play(state.pass_action).play(state.pass_action)
    assert end_kind_of(passed, "B+1") == "double_pass"
    assert end_kind_of(passed, "B+R") == "resign"  # The SGF result property takes precedence over the board state.

    truncated = GoState.new_game(board_size=5, komi=2.5, max_moves=1).play(12)
    assert end_kind_of(truncated, None) == "max_moves"
    assert end_kind_of(state, None) == "unfinished"


def test_summarize_counts_buckets_and_divergences() -> None:
    records = [
        GameScoring(source="a", moves=10, end_kind="double_pass",
                    our_margin=5.0, judge_margin=3.0, ownership_margin=5.0),
        GameScoring(source="b", moves=10, end_kind="double_pass",
                    our_margin=-2.0, judge_margin=4.0, ownership_margin=-2.0),
        GameScoring(source="c", moves=10, end_kind="resign"),
        GameScoring(source="d", moves=10, end_kind="double_pass", error="corrupt SGF"),
    ]
    summary = summarize(records)
    assert summary["games_total"] == 4
    assert summary["double_pass"] == 2  # Exclude errored records from paired scoring comparisons.
    assert summary["resign"] == 1
    assert summary["errors"] == 1

    versus_judge = summary["our_vs_judge"]
    assert versus_judge["compared"] == 2
    assert versus_judge["winner_divergences"] == 1
    assert versus_judge["winner_divergence_share"] == 0.5
    assert versus_judge["mean_abs_margin_delta"] == 4.0  # (|5-3| + |-2-4|) / 2

    versus_ownership = summary["our_vs_ownership"]
    assert versus_ownership["winner_divergences"] == 0
    assert versus_ownership["mean_abs_margin_delta"] == 0.0


class _StubModel:
    """Stub forward_full with a specified ownership map or no ownership head."""

    def __init__(self, ownership: np.ndarray | None) -> None:
        self._ownership = ownership

    def forward_full(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._ownership is None:
            return {}
        flat = torch.from_numpy(self._ownership.astype(np.float32).reshape(1, -1))
        return {"ownership": flat}


def test_ownership_margin_respects_player_perspective() -> None:
    board = np.zeros((3, 3), dtype=np.int8)
    device = torch.device("cpu")
    # Positive ownership everywhere favors Black when Black is to play.
    all_mine = np.full((3, 3), 0.9, dtype=np.float32)
    black_state = GoState.from_board(board, to_play=BLACK, komi=0.5)
    margin = probe_scoring.ownership_margin(_StubModel(all_mine), black_state, device=device, threshold=0.5)  # type: ignore[arg-type]
    assert margin == 9.0 - 0.5
    # The same player-relative map favors White when White is to play.
    white_state = GoState.from_board(board, to_play=WHITE, komi=0.5)
    margin = probe_scoring.ownership_margin(_StubModel(all_mine), white_state, device=device, threshold=0.5)  # type: ignore[arg-type]
    assert margin == -9.0 - 0.5
    # A network without auxiliary heads returns None for its ownership verdict.
    plain = probe_scoring.ownership_margin(_StubModel(None), black_state, device=device, threshold=0.5)  # type: ignore[arg-type]
    assert plain is None


class _FakeJudge:
    """Use a GTP referee with a constant final_score."""

    def __init__(self, command: list[str], **_kwargs: Any) -> None:
        self.argv = command

    def __enter__(self) -> "_FakeJudge":
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def command(self, line: str) -> str:
        if line.startswith("loadsgf"):
            return ""
        if line == "final_score":
            return "W+2.5"
        return ""


class _RecordingJudge:
    """Record replay commands with a referee that lacks loadsgf and scores B+1.5."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def command(self, line: str) -> str:
        self.lines.append(line)
        if line.startswith("loadsgf"):
            raise AssertionError("loadsgf must not be used in replay mode")
        return "B+1.5" if line == "final_score" else ""


def test_judge_replay_mode_plays_the_moves() -> None:
    parsed = parse_sgf(
        game_to_sgf(board_size=5, komi=2.5, moves=[12, 6, 25, 25], winner=BLACK, score_margin=1.0)
    )
    judge = _RecordingJudge()
    margin = probe_scoring.judge_margin(judge, path=Path("x.sgf"), parsed=parsed, mode="replay")  # type: ignore[arg-type]
    assert margin == 1.5
    assert judge.lines == [
        "boardsize 5",
        "clear_board",
        "komi 2.5",
        "play b C3",
        "play w B4",
        "play b pass",
        "play w pass",
        "final_score",
    ]


def test_main_end_to_end_with_fake_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    games = tmp_path / "games"
    games.mkdir()
    (games / "scored.sgf").write_text(
        game_to_sgf(board_size=5, komi=2.5, moves=[12, 6, 25, 25], winner=BLACK, score_margin=1.0),
        encoding="utf-8",
    )
    (games / "resigned.sgf").write_text(
        game_to_sgf(board_size=5, komi=2.5, moves=[12, 6], winner=BLACK, resigned=True),
        encoding="utf-8",
    )
    (games / "broken.sgf").write_text("this is not SGF", encoding="utf-8")
    monkeypatch.setattr(probe_scoring, "GTPClient", _FakeJudge)

    out = tmp_path / "report.json"
    code = probe_scoring.main(
        ["--sgf", str(games), "--judge-command", "fake-judge", "--json-out", str(out), "--device", "cpu"]
    )
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["games_total"] == 3
    assert summary["double_pass"] == 1
    assert summary["resign"] == 1
    assert summary["errors"] == 1

    by_source = {game["source"].split("/")[-1].split("\\")[-1]: game for game in payload["games"]}
    scored = by_source["scored.sgf"]
    # One stone each and neutral empty space give scores 1 and 1 + 2.5 komi.
    assert scored["our_margin"] == -2.5
    assert scored["judge_margin"] == -2.5
    assert scored["ownership_margin"] is None  # No checkpoint supplied.
    assert summary["our_vs_judge"]["compared"] == 1
    assert summary["our_vs_judge"]["winner_divergences"] == 0
