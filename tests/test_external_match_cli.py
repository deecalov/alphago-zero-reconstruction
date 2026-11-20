"""Exercise the external-match CLI with real subprocesses controlled by a fake GTP engine."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from alphago_zero.game import BLACK, WHITE
from alphago_zero.gtp_match import ExternalGame, MatchResult, MatchSettings
from examples import eval_match, external_match

FAKE = Path(__file__).with_name("fake_gtp_engine.py")
FAST = ["--startup-timeout", "60", "--move-timeout", "30", "--engine-timeout", "30"]


def engine(*flags: str) -> str:
    """Build the fake engine command for either side of a match."""
    return " ".join(f'"{part}"' for part in [sys.executable, str(FAKE), *flags])


def base_argv(*extra: str, scores: str = "B+1.5,B+2.5,W+3.5,0", games: str = "4") -> list[str]:
    """Script a 9x9 series with distinct board zones and a repeating score sequence."""
    return [
        "--model-command", engine("--zone", "bottom", "--name", "ModelFake"),
        "--opponent-command", engine("--zone", "top", "--name", "OppFake", "--final-scores", scores),
        "--opponent-name", "pachi_fake",
        "--games", games, "--board-size", "9", "--opening-moves", "2", "--seed", "1",
        *FAST, *extra,
    ]


def run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = external_match.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def make_game(index: int, model_is_black: bool, winner: int | None, **fields: object) -> ExternalGame:
    values: dict[str, object] = {
        "opening": (), "moves": (1, 2, 3), "result": "B+3.5", "margin": 3.5, "resigned": False,
        "void_reason": None if winner is not None else "reason", "seconds": 12.34,
    }
    values.update(fields)
    return ExternalGame(index=index, model_is_black=model_is_black, winner=winner, **values)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# End-to-end series with fake engines
# ----------------------------------------------------------------------
def test_a_short_match_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "res" / "match.json"
    code, stdout, _ = run_cli(base_argv("--json-out", str(out), "--opponent-rating", "1500"), capsys)
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    # Alternate the model's color and score each scripted referee result from its perspective.
    assert (data["wins"], data["losses"], data["draws"], data["voids"]) == (1, 2, 1, 0)
    assert data["games"] == 4 and data["games_played"] == 4 and data["games_requested"] == 4
    assert data["win_rate"] == 0.375 and data["aborted"] is None
    point, lower, upper = eval_match.wilson_interval(1.5, 4)
    assert data["win_rate_ci95"] == [round(lower, 4), round(upper, 4)]
    assert data["elo_estimate"] == round(eval_match.elo_from_win_rate(point, 1500.0), 1)
    assert data["elo_ci95"] == [round(eval_match.elo_from_win_rate(value, 1500.0), 1) for value in (lower, upper)]
    assert data["opponent_rating"] == 1500.0
    assert (data["as_black_games"], data["as_black_win_rate"]) == (2, 0.5)
    assert (data["as_white_games"], data["as_white_win_rate"]) == (2, 0.25)
    assert data["model"] == "model" and data["opponent"] == "pachi_fake" and data["scoring"] == "opponent"
    assert data["engines"]["model"]["name"] == "ModelFake" and data["engines"]["opponent"]["name"] == "OppFake"
    assert data["engines"]["referee"] is None and data["commands"]["referee"] is None
    assert [g["outcome"] for g in data["games_detail"]] == ["win", "loss", "loss", "draw"]
    assert [g["number"] for g in data["games_detail"]] == [1, 2, 3, 4]
    assert [g["model_is_black"] for g in data["games_detail"]] == [True, False, True, False]
    assert data["games_detail"][0]["opening"] == data["games_detail"][1]["opening"]
    assert all(len(g["opening"]) == 2 for g in data["games_detail"])
    assert sum(1 for line in stdout.splitlines() if line.startswith("[")) == 4
    assert f"written {out}" in stdout and "Result:" in stdout and "engines: " in stdout
    assert "win" in out.read_text(encoding="utf-8")


def test_sgf_files_are_written_for_every_game(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sgf_dir, out = tmp_path / "records" / "sgf", tmp_path / "match.json"
    argv = base_argv("--sgf-dir", str(sgf_dir), "--model-name", "champion-●", "--json-out", str(out))
    code, _, _ = run_cli(argv, capsys)
    assert code == 0
    files = sorted(sgf_dir.glob("*.sgf"))
    assert [f.name for f in files] == ["game_001.sgf", "game_002.sgf", "game_003.sgf", "game_004.sgf"]
    texts = [f.read_text(encoding="utf-8") for f in files]
    first, second, _third, fourth = texts
    assert "SZ[9]" in first and "KM[7.5]" in first
    assert "PB[champion-●]" in first and "PW[pachi_fake]" in first and "RE[B+1.5]" in first
    assert "PB[pachi_fake]" in second and "PW[champion-●]" in second  # Colors swap for the second game.
    assert "RE[0]" in fourth
    assert "champion-●" in out.read_text(encoding="utf-8")  # Preserve non-ASCII names in the JSON text.
    records = json.loads(out.read_text(encoding="utf-8"))["games_detail"]
    for text, record in zip(texts, records, strict=True):  # The SGF contains the complete game, including the shared opening.
        assert text.count(";B[") + text.count(";W[") == record["move_count"] > 2


def test_a_dedicated_referee_decides_the_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, referee_log = tmp_path / "match.json", tmp_path / "referee.log"
    referee = engine("--name", "RefFake", "--final-scores", "W+5.5", "--log", str(referee_log))
    code, _, _ = run_cli(
        base_argv("--referee-command", referee, "--referee-name", "ref", "--json-out", str(out), scores="B+99"), capsys
    )
    assert code == 0
    assert referee_log.read_text(encoding="utf-8").splitlines()[-1] == "quit"  # Close the referee together with the players.
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scoring"] == "referee" and data["engines"]["referee"]["name"] == "RefFake"
    assert (data["wins"], data["losses"]) == (2, 2)  # White always wins by 5.5, so alternating colors alternate model wins.
    assert {g["result"] for g in data["games_detail"]} == {"W+5.5"}


def test_a_series_of_void_games_aborts_with_a_partial_result(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "aborted.json"
    argv = base_argv("--json-out", str(out), games="6")
    argv[argv.index("--opponent-command") + 1] = engine("--zone", "top", "--reject", "play")
    code, stdout, _ = run_cli(argv, capsys)
    assert code == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["aborted"] and "3 consecutive void games" in data["aborted"]
    assert (data["voids"], data["games"], data["games_played"], data["games_requested"]) == (3, 0, 3, 6)
    assert data["win_rate"] is None and data["win_rate_ci95"] is None and data["elo_estimate"] is None
    assert all(g["outcome"] == "void" and g["void_reason"] for g in data["games_detail"])
    assert "MATCH ABORTED" in stdout and "Warning" in stdout


def test_the_void_limit_is_configurable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "aborted.json"
    argv = base_argv("--json-out", str(out), "--max-consecutive-voids", "2", games="6")
    argv[argv.index("--opponent-command") + 1] = engine("--zone", "top", "--reject", "play")
    code, _, _ = run_cli(argv, capsys)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and data["voids"] == 2 and "2 consecutive void games" in data["aborted"]


def test_non_default_settings_reach_the_json_the_sgf_and_the_header(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, sgf_dir = tmp_path / "match.json", tmp_path / "sgf"
    argv = [
        "--model-command", engine("--zone", "bottom"),
        "--opponent-command", engine("--zone", "top", "--final-scores", "B+1.5"),
        "--games", "2", "--board-size", "7", "--komi", "0.5", "--opening-moves", "2", "--max-moves", "8",
        "--seed", "5", "--json-out", str(out), "--sgf-dir", str(sgf_dir), *FAST,
    ]
    code, stdout, _ = run_cli(argv, capsys)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0
    assert (data["board_size"], data["komi"], data["opening_moves"], data["max_moves"]) == (7, 0.5, 2, 8)
    assert (data["move_timeout"], data["seed"], data["games_requested"]) == (30.0, 5, 2)
    assert [g["move_count"] for g in data["games_detail"]] == [8, 8]  # The move limit truncated this game.
    assert "board=7 komi=0.5 games=2 opening_moves=2 seed=5 move_timeout=30" in stdout
    sgf = (sgf_dir / "game_001.sgf").read_text(encoding="utf-8")
    assert "SZ[7]" in sgf and "KM[0.5]" in sgf


def test_an_interrupted_match_keeps_the_games_finished_so_far(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    finished = [make_game(0, True, BLACK), make_game(1, False, BLACK)]  # One model win and one loss.

    def interrupted(*_players: object, on_game: Callable[[ExternalGame], None], **_options: object) -> MatchResult:
        for game in finished:
            on_game(game)
        raise KeyboardInterrupt

    monkeypatch.setattr(external_match, "run_match", interrupted)
    out, model_log = tmp_path / "partial.json", tmp_path / "model.log"
    argv = base_argv("--json-out", str(out), games="6")
    argv[argv.index("--model-command") + 1] = engine("--log", str(model_log))
    try:
        code, stdout, _ = run_cli(argv, capsys)
    except KeyboardInterrupt:  # Catch interrupts inside main so pytest can report the result.
        pytest.fail("main must handle Ctrl+C and retain completed games")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1 and data["aborted"] == external_match.INTERRUPTED
    assert (data["games"], data["games_played"], data["games_requested"]) == (2, 2, 6)
    assert (data["wins"], data["losses"], data["win_rate"]) == (1, 1, 0.5)
    assert "MATCH ABORTED" in stdout and "Ctrl+C" in stdout
    assert model_log.read_text(encoding="utf-8").splitlines()[-1] == "quit"  # Close all engines after interruption.


def test_a_missing_opponent_program_is_a_clear_error_and_the_model_engine_is_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "model.log"
    out = tmp_path / "never.json"
    argv = base_argv("--json-out", str(out))
    argv[argv.index("--model-command") + 1] = engine("--log", str(log))
    argv[argv.index("--opponent-command") + 1] = "definitely-not-an-engine-xyz --gtp"
    code, _, err = run_cli(argv, capsys)
    assert code == 1 and "definitely-not-an-engine-xyz" in err and not out.exists()
    assert log.read_text(encoding="utf-8").splitlines() == ["quit"]  # Close the already running model process.


def test_both_engines_are_closed_after_a_normal_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model_log, opponent_log = tmp_path / "m.log", tmp_path / "o.log"
    argv = base_argv(games="2")
    argv[argv.index("--model-command") + 1] = engine("--zone", "bottom", "--log", str(model_log))
    argv[argv.index("--opponent-command") + 1] = engine("--zone", "top", "--log", str(opponent_log))
    code, _, _ = run_cli(argv, capsys)
    assert code == 0
    for log in (model_log, opponent_log):
        assert log.read_text(encoding="utf-8").splitlines()[-1] == "quit"


def test_the_opening_is_reported_as_gtp_vertices(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "match.json"
    assert run_cli(base_argv("--json-out", str(out), games="2"), capsys)[0] == 0
    openings = [g["opening"] for g in json.loads(out.read_text(encoding="utf-8"))["games_detail"]]
    assert all(re.fullmatch(r"[A-HJ][2-8]", vertex) for opening in openings for vertex in opening)  # An interior point on a 9x9 board.


def test_a_rejected_version_command_is_not_fatal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "match.json"
    argv = base_argv("--json-out", str(out), games="2")
    argv[argv.index("--opponent-command") + 1] = engine("--zone", "top", "--name", "NoVersion", "--reject", "version")
    code, _, _ = run_cli(argv, capsys)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and data["engines"]["opponent"] == {"name": "NoVersion", "version": ""}


def test_void_games_leave_their_reason_in_the_sgf_comment(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sgf_dir = tmp_path / "sgf"
    argv = base_argv("--sgf-dir", str(sgf_dir), games="3")
    argv[argv.index("--opponent-command") + 1] = engine("--zone", "top", "--reject", "play")
    assert run_cli(argv, capsys)[0] == 1
    text = (sgf_dir / "game_001.sgf").read_text(encoding="utf-8")
    assert "RE[Void]" in text and "C[" in text and "rejected command" in text


def test_timeouts_are_handed_to_the_engine_clients(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    created: list[dict[str, object]] = []
    original = external_match.GTPClient.__init__

    def recording_init(self: external_match.GTPClient, command: object, **kwargs: object) -> None:
        created.append({"name": kwargs["name"], "default": kwargs["default_timeout"], "startup": kwargs["startup_timeout"]})
        original(self, command, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(external_match.GTPClient, "__init__", recording_init)
    argv = base_argv(games="2")
    argv[argv.index("--startup-timeout") + 1] = "22"
    argv[argv.index("--engine-timeout") + 1] = "11"
    assert run_cli(argv, capsys)[0] == 0
    assert created == [
        {"name": "model", "default": 11.0, "startup": 22.0},
        {"name": "pachi_fake", "default": 11.0, "startup": 22.0},
    ]


def test_documented_defaults() -> None:
    args = external_match.build_parser().parse_args(["--model-command", "x", "--opponent-command", "y"])
    assert (args.games, args.board_size, args.komi, args.opening_moves) == (60, 9, 7.5, 4)
    assert (args.move_timeout, args.engine_timeout, args.startup_timeout) == (120.0, 60.0, 180.0)
    assert (args.max_consecutive_voids, args.seed, args.opponent_rating) == (3, 0, 0.0)
    assert (args.model_name, args.referee_name, args.max_moves) == ("model", "referee", None)
    assert (args.json_out, args.sgf_dir, args.dry_run) == (None, None, False)


# ----------------------------------------------------------------------
# Dry run and usage errors
# ----------------------------------------------------------------------
def test_the_default_opponent_name_is_the_program_stem(capsys: pytest.CaptureFixture[str]) -> None:
    code, stdout, _ = run_cli(
        ["--model-command", "x", "--opponent-command", "/opt/engines/pachi.exe -t =2000", "--dry-run"], capsys
    )
    assert code == 0 and "opponent=pachi " in stdout and "rating=0\n" in stdout
    assert "referee=none (opponent scores)" in stdout
    assert "board=9 komi=7.5 games=60 opening_moves=4 seed=0 move_timeout=120" in stdout


def test_an_unusable_sgf_directory_is_a_usage_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        external_match.main(
            ["--model-command", "x", "--opponent-command", "y", "--sgf-dir", str(blocker / "sgf")]
        )
    assert excinfo.value.code == 2
    assert "--sgf-dir: cannot create directory" in capsys.readouterr().err


def test_dry_run_prints_the_commands_and_starts_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "deep" / "dir" / "out.json"
    code, stdout, _ = run_cli(
        [
            "--checkpoint", "a.pt", "--simulations", "160", "--device", "cuda",
            "--opponent-command", "definitely-not-an-engine-xyz -t =2000 -d0", "--referee-command", "gnugo --mode gtp",
            "--opponent-rating", "1500", "--json-out", str(out), "--sgf-dir", str(tmp_path / "sgf"), "--dry-run",
        ],
        capsys,
    )
    assert code == 0 and "--dry-run" in stdout and "rating=1500\n" in stdout
    assert "opponent=definitely-not-an-engine-xyz" in stdout  # Default to the executable name.
    assert "play_gtp.py" in stdout and "'--simulations', '160'" in stdout and "'gnugo'" in stdout
    assert not out.parent.exists() and not (tmp_path / "sgf").exists()  # Dry run has no side effects.


def test_an_unusable_json_directory_is_a_usage_error_before_any_engine_starts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file.txt"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        external_match.main(
            ["--model-command", "definitely-not-an-engine-xyz", "--opponent-command", "also-missing",
             "--json-out", str(blocker / "sub" / "out.json")]
        )
    assert excinfo.value.code == 2
    assert "--json-out: cannot create directory" in capsys.readouterr().err  # Distinguish a usage error from a startup failure.


@pytest.mark.parametrize(
    ("extra", "fragment"),
    [
        (["--simulations", "10"], "--simulations"),
        (["--device", "cuda"], "--device"),
        (["--model-args=--no-resign"], "--model-args"),
        (["--simulations", "10", "--device", "cuda"], "--simulations, --device"),  # Name every incompatible flag.
    ],
)
def test_checkpoint_only_flags_are_rejected_with_model_command(
    extra: list[str], fragment: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        external_match.main(["--model-command", "x", "--opponent-command", "y", *extra])
    assert excinfo.value.code == 2
    assert fragment in capsys.readouterr().err.strip().splitlines()[-1]


@pytest.mark.parametrize(
    "argv",
    [
        ["--opponent-command", "y"],  # Missing model.
        ["--checkpoint", "a.pt", "--model-command", "x", "--opponent-command", "y"],  # Both model sources specified.
        ["--model-command", "x"],  # Missing opponent.
        ["--model-command", "x", "--opponent-command", "y", "--games", "0"],
        ["--checkpoint", "a.pt", "--opponent-command", "y", "--simulations", "0"],
        ["--model-command", "x", "--opponent-command", "y", "--opponent-rating", "nan"],
        ["--model-command", "x", "--opponent-command", "y", "--engine-timeout", "0"],
        ["--model-command", "x", "--opponent-command", "y", "--startup-timeout", "-1"],
        ["--model-command", "x", "--opponent-command", "y", "--move-timeout", "nan"],
        ["--model-command", "x", "--opponent-command", "y", "--komi", "inf"],
        ["--model-command", "x", "--opponent-command", "y", "--opening-moves", "-1"],
        ["--model-command", "x", "--opponent-command", "y", "--opening-moves", "30", "--board-size", "9"],
        ["--model-command", "x", "--opponent-command", "y", "--max-moves", "2", "--opening-moves", "2"],
        ["--model-command", "'unbalanced", "--opponent-command", "y"],
        ["--model-command", "x", "--opponent-command", "   "],
    ],
)
def test_invalid_usage_exits_with_code_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        external_match.main(argv)
    assert excinfo.value.code == 2


# ----------------------------------------------------------------------
# Model command and reporting
# ----------------------------------------------------------------------
def test_model_command_runs_play_gtp_with_the_checkpoints_and_flags() -> None:
    args = external_match.build_parser().parse_args(
        ["--checkpoint", "a.pt", "b.pt", "--opponent-command", "y", "--simulations", "160",
         "--device", "cuda", "--seed", "5", "--model-args=--no-resign"]  # Pass a dash-prefixed option value with '='.
    )
    assert external_match.build_model_command(args) == [
        sys.executable, str(external_match.PLAY_GTP), "--checkpoint", "a.pt", "b.pt",
        "--device", "cuda", "--seed", "5", "--simulations", "160", "--no-resign",
    ]


def test_model_command_defaults_to_cpu_and_the_checkpoint_search() -> None:
    args = external_match.build_parser().parse_args(["--checkpoint", "a.pt", "--opponent-command", "y"])
    assert external_match.build_model_command(args) == [
        sys.executable, str(external_match.PLAY_GTP), "--checkpoint", "a.pt", "--device", "cpu", "--seed", "0",
    ]


def test_blank_model_args_are_ignored() -> None:
    args = external_match.build_parser().parse_args(
        ["--checkpoint", "a.pt", "--opponent-command", "y", "--model-args", "   "]
    )
    assert external_match.build_model_command(args)[-2:] == ["--seed", "0"]  # No extra options should produce no error.


def test_model_command_passes_a_custom_command_through() -> None:
    args = external_match.build_parser().parse_args(["--model-command", "engine --gtp", "--opponent-command", "y"])
    assert external_match.build_model_command(args) == ["engine", "--gtp"]


def test_a_game_line_shows_side_result_outcome_and_size() -> None:
    line = external_match.format_game_line(make_game(6, True, BLACK), 60)
    assert line.startswith("[ 7/60]") and "model=black" in line
    assert "B+3.5" in line and "win" in line and "moves=3" in line and "12.3s" in line
    loss = external_match.format_game_line(make_game(0, False, BLACK), 4)
    assert loss.startswith("[1/4]") and "model=white" in loss and "loss" in loss
    assert "draw" in external_match.format_game_line(make_game(0, True, 0, result="0"), 4)


def test_a_game_line_layout_is_aligned() -> None:
    win = external_match.format_game_line(make_game(6, True, BLACK), 60)
    assert win == "[ 7/60] model=black  B+3.5    win       moves=3 12.3s"
    loss = external_match.format_game_line(make_game(1, False, BLACK, result="B+9.5"), 10)
    assert loss == "[ 2/10] model=white  B+9.5    loss      moves=3 12.3s"
    draw = external_match.format_game_line(make_game(2, True, 0, result="0"), 10)
    assert draw == "[ 3/10] model=black  0        draw      moves=3 12.3s"


def test_a_void_game_line_carries_the_reason() -> None:
    line = external_match.format_game_line(make_game(2, False, None, void_reason="disputed move: pachi"), 10)
    assert "VOID: disputed move: pachi" in line and "win" not in line
    assert line == "[ 3/10] model=white  VOID: disputed move: pachi (moves=3 12.3s)"


def test_a_game_record_describes_the_game() -> None:
    game = make_game(6, False, BLACK, opening=(0, 6), moves=(0, 6, 12, 25), result="B+0.5", resigned=True, seconds=1.2345)
    assert external_match.game_record(game, 5) == {
        "number": 7, "model_is_black": False, "outcome": "loss", "result": "B+0.5", "resigned": True,
        "move_count": 4, "opening": ["A5", "B4"], "void_reason": None, "seconds": 1.23,
    }
    void = external_match.game_record(make_game(0, True, None, void_reason="failed"), 9)
    assert void["outcome"] == "void" and void["void_reason"] == "failed" and void["number"] == 1


def _payload(games: tuple[ExternalGame, ...], rating: float = 0.0) -> dict[str, object]:
    args = external_match.build_parser().parse_args(
        ["--model-command", "x", "--opponent-command", "y", "--opponent-rating", str(rating)]
    )
    commands: dict[str, list[str] | None] = {"model": ["x"], "opponent": ["y"], "referee": None}
    return external_match.summary_payload(
        args, MatchSettings(games=len(games) or 1), MatchResult(games), commands=commands,
        engines={"model": None, "opponent": None, "referee": None}, opponent_name="y", aborted=None,
    )


def test_the_summary_has_no_rates_without_valid_games() -> None:
    payload = _payload((make_game(0, True, None), make_game(1, False, None)))
    assert payload["voids"] == 2 and payload["games"] == 0
    assert payload["win_rate"] is None and payload["win_rate_ci95"] is None
    assert payload["elo_estimate"] is None and payload["elo_ci95"] == [None, None]
    assert payload["as_black_win_rate"] is None and payload["as_white_win_rate"] is None


def test_the_summary_counts_a_draw_as_half_a_point() -> None:
    payload = _payload((make_game(0, True, BLACK), make_game(1, False, 0, result="0")))
    assert payload["win_rate"] == 0.75 and payload["draws"] == 1


def test_the_summary_rounds_the_rates_and_splits_them_by_color() -> None:
    black_only = _payload((make_game(0, True, BLACK), make_game(2, True, BLACK), make_game(4, True, WHITE)))
    assert black_only["win_rate"] == 0.6667
    assert (black_only["as_black_games"], black_only["as_black_win_rate"]) == (3, 0.6667)
    assert (black_only["as_white_games"], black_only["as_white_win_rate"]) == (0, None)
    mixed = _payload(
        (
            make_game(0, True, BLACK), make_game(2, True, BLACK), make_game(4, True, WHITE),  # Black scored two points in three games.
            make_game(1, False, WHITE), make_game(3, False, BLACK), make_game(5, False, BLACK),
            make_game(7, False, BLACK), make_game(9, False, WHITE), make_game(11, False, BLACK),  # White scored two points in six games.
        )
    )
    assert (mixed["as_black_games"], mixed["as_black_win_rate"]) == (3, 0.6667)
    assert (mixed["as_white_games"], mixed["as_white_win_rate"]) == (6, 0.3333)
    assert mixed["win_rate"] == 0.4444


def test_the_text_summary_shows_the_score_the_interval_and_the_relative_elo() -> None:
    games = (
        make_game(0, True, BLACK), make_game(1, False, WHITE), make_game(2, True, BLACK), make_game(3, False, BLACK),
        make_game(4, True, None),  # Exclude the void game from the interval.
    )
    text = external_match.format_summary(_payload(games, rating=1500.0))
    point, lower, upper = eval_match.wilson_interval(3.0, 4)
    elo, elo_low, elo_high = (eval_match.elo_from_win_rate(value, 1500.0) for value in (point, lower, upper))
    assert "Result: model against y wins: 3, losses: 1, draws: 0, void: 1 (played 5 of 5)" in text
    assert f"win-rate 0.750, 95% Wilson [{lower:.3f}, {upper:.3f}] across 4 games" in text
    assert f"(rating 1500): {elo:+.0f} [{elo_low:+.0f}, {elo_high:+.0f}]" in text


def test_the_text_summary_of_an_empty_result_does_not_divide_by_zero() -> None:
    payload = _payload(())
    payload["aborted"] = "setup rejected"
    text = external_match.format_summary(payload)
    assert "played 0 of 1" in text and "MATCH ABORTED: setup rejected" in text
    assert "Warning" not in text and "win-rate" not in text


def test_the_text_summary_warns_when_voids_exceed_a_tenth() -> None:
    games = tuple(make_game(i, i % 2 == 0, BLACK if i % 2 == 0 else WHITE) for i in range(8))
    quiet = external_match.format_summary(_payload(games))
    assert "Warning" not in quiet and "win-rate 1.000" in quiet and "Elo relative to" in quiet
    noisy = external_match.format_summary(_payload((*games, make_game(8, True, None), make_game(9, False, None))))
    assert "Warning" in noisy and "2 of 10" in noisy


def test_the_warning_threshold_is_strictly_above_ten_percent() -> None:
    games = tuple(make_game(i, i % 2 == 0, BLACK if i % 2 == 0 else WHITE) for i in range(9))
    exactly_ten_percent = external_match.format_summary(_payload((*games, make_game(9, True, None))))
    assert "Warning" not in exactly_ten_percent


def test_the_text_summary_reports_an_abort() -> None:
    payload = _payload((make_game(0, True, None),))
    payload["aborted"] = "stop reason"
    assert "MATCH ABORTED: stop reason" in external_match.format_summary(payload)


# ----------------------------------------------------------------------
# Actual play_gtp frontend
# ----------------------------------------------------------------------
def test_the_real_play_gtp_engine_plays_a_match(
    checkpoints: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Play a tiny 5x5 checkpoint through the real GTP frontend against the fake opponent."""
    out = tmp_path / "real.json"
    code, _, _ = run_cli(
        [
            "--checkpoint", str(checkpoints[0]), "--simulations", "4", "--model-name", "tiny",
            "--opponent-command", engine("--zone", "top", "--name", "OppFake", "--own-moves", "2"),
            "--opponent-name", "fake", "--games", "2", "--board-size", "5", "--komi", "2.5",
            "--opening-moves", "2", "--seed", "3", "--json-out", str(out),
            "--startup-timeout", "180", "--move-timeout", "60", "--engine-timeout", "60",
        ],
        capsys,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and data["aborted"] is None and data["games_played"] == 2
    assert data["engines"]["model"]["name"] == "alphago-zero-reconstruction"
    assert data["wins"] + data["losses"] + data["draws"] + data["voids"] == 2
    assert data["commands"]["model"][1].endswith("play_gtp.py")
