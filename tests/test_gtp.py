from __future__ import annotations

"""Go Text Protocol vertex conversion, command handling and play integration tests."""

from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.game import GoState
from alphago_zero.gtp import GTPEngine, action_to_vertex, vertex_to_action
from alphago_zero.mcts import AlphaGoZeroMCTS, MCTSNode, SearchOutput
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.sgf import game_to_sgf
from examples import external_match, play_gtp


@pytest.fixture
def engine(tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet) -> GTPEngine:
    return GTPEngine(
        model=tiny_model,
        config=tiny_config,
        rng=np.random.default_rng(0),
        allow_resign=False,  # Disable resignation in tests that inspect generated moves.
    )


# ----------------------------------------------------------------------
# Vertex conversion
# ----------------------------------------------------------------------
def test_vertex_roundtrip_all_actions() -> None:
    board_size = 5
    for action in range(board_size * board_size + 1):
        vertex = action_to_vertex(action, board_size)
        assert vertex_to_action(vertex, board_size) == action


def test_vertex_conventions() -> None:
    # A1 is the lower left corner, row size-1 and column zero.
    assert vertex_to_action("A1", 5) == 4 * 5 + 0
    # E5 is the top right on a 5x5 board.
    assert vertex_to_action("E5", 5) == 0 * 5 + 4
    # GTP skips I, so J is column index eight on a 19x19 board.
    assert vertex_to_action("J10", 19) == (19 - 10) * 19 + 8
    # Accept pass in either case.
    assert vertex_to_action("PASS", 5) == 25
    assert action_to_vertex(25, 5) == "pass"


def test_vertex_rejects_garbage() -> None:
    for bad in ("", "Z9", "A0", "A6", "I3", "5A", "qq"):
        with pytest.raises(ValueError):
            vertex_to_action(bad, 5)


# ----------------------------------------------------------------------
# Protocol framing
# ----------------------------------------------------------------------
def test_protocol_admin_commands(engine: GTPEngine) -> None:
    response, quit_flag = engine.handle_line("protocol_version")
    assert response == "= 2\n\n" and not quit_flag

    response, _ = engine.handle_line("name")
    assert response is not None and response.startswith("= ")

    response, _ = engine.handle_line("known_command genmove")
    assert response == "= true\n\n"
    response, _ = engine.handle_line("known_command teleport")
    assert response == "= false\n\n"

    response, _ = engine.handle_line("list_commands")
    assert response is not None
    for command in ("genmove", "play", "boardsize", "quit"):
        assert command in response


def test_protocol_id_echo_and_errors(engine: GTPEngine) -> None:
    response, _ = engine.handle_line("7 name")
    assert response is not None and response.startswith("=7 ")

    response, _ = engine.handle_line("9 frobnicate")
    assert response == "?9 unknown command\n\n"

    # Blank lines and comments produce no response.
    assert engine.handle_line("") == (None, False)
    assert engine.handle_line("# just a comment") == (None, False)


def test_quit_terminates(engine: GTPEngine) -> None:
    response, quit_flag = engine.handle_line("quit")
    assert response == "=\n\n"
    assert quit_flag


def test_boardsize_rules(engine: GTPEngine) -> None:
    # Reject board sizes incompatible with the 5x5 network.
    response, _ = engine.handle_line("boardsize 9")
    assert response == "? unacceptable size\n\n"
    response, _ = engine.handle_line("boardsize 5")
    assert response == "=\n\n"


# ----------------------------------------------------------------------
# Game command sequence
# ----------------------------------------------------------------------
def test_play_and_illegal_moves(engine: GTPEngine) -> None:
    response, _ = engine.handle_line("play b C3")
    assert response == "=\n\n"
    assert engine.state.move_number == 1

    # The point is already occupied.
    response, _ = engine.handle_line("play w C3")
    assert response is not None and response.startswith("? illegal move")

    # White is to move, so another Black move is out of turn.
    response, _ = engine.handle_line("play b D3")
    assert response is not None and response.startswith("? illegal move")

    response, _ = engine.handle_line("play w pass")
    assert response == "=\n\n"


def test_genmove_advances_state(engine: GTPEngine) -> None:
    response, _ = engine.handle_line("genmove b")
    assert response is not None and response.startswith("= ")
    vertex = response[2:].strip()
    assert engine.state.move_number == 1
    # Convert the generated vertex back to an action.
    vertex_to_action(vertex, engine.board_size)

    # Reject a second genmove for the same color.
    response, _ = engine.handle_line("genmove b")
    assert response is not None and response.startswith("?")


def test_clear_board_resets(engine: GTPEngine) -> None:
    engine.handle_line("play b C3")
    engine.handle_line("clear_board")
    assert engine.state.move_number == 0


def test_komi_command(engine: GTPEngine) -> None:
    # Accept the saved komi of 2.5 and reset the game.
    engine.handle_line("play b C3")
    response, _ = engine.handle_line("komi 2.5")
    assert response == "=\n\n"
    assert engine.komi == 2.5
    assert engine.state.move_number == 0

    # Reject a different komi while preserving the current game.
    engine.handle_line("play b C3")
    response, _ = engine.handle_line("komi 5.5")
    assert response == "? unacceptable komi\n\n"
    assert engine.komi == 2.5
    assert engine.state.move_number == 1

    response, _ = engine.handle_line("komi seven")
    assert response is not None and response.startswith("?")
    response, _ = engine.handle_line("komi nan")
    assert response == "? unacceptable komi\n\n"


def test_opening_moves_sampling_diversifies_games(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    # Different seeds should vary openings when sampling from visits.
    def first_moves(seed: int) -> list[str]:
        eng = GTPEngine(
            model=tiny_model,
            config=tiny_config,
            rng=np.random.default_rng(seed),
            allow_resign=False,
            opening_moves=4,
        )
        moves: list[str] = []
        for _ in range(4):
            response, _ = eng.handle_line(f"genmove {'b' if len(moves) % 2 == 0 else 'w'}")
            assert response is not None
            moves.append(response.strip())
        return moves

    variants = {tuple(first_moves(seed)) for seed in range(6)}
    assert len(variants) > 1


def test_opening_moves_zero_is_deterministic(
    tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> None:
    def game(seed: int) -> list[str]:
        eng = GTPEngine(
            model=tiny_model,
            config=tiny_config,
            rng=np.random.default_rng(seed),
            allow_resign=False,
            opening_moves=0,
        )
        moves: list[str] = []
        for i in range(6):
            response, _ = eng.handle_line(f"genmove {'b' if i % 2 == 0 else 'w'}")
            assert response is not None
            moves.append(response.strip())
        return moves

    # The same seed reproduces a complete game.
    assert game(7) == game(7)


def test_time_commands_accepted_as_noop(engine: GTPEngine) -> None:
    # Accept time_settings and time_left sent by match controllers.
    response, _ = engine.handle_line("time_settings 0 0 0")
    assert response == "=\n\n"
    response, _ = engine.handle_line("time_left b 0 0")
    assert response == "=\n\n"


def test_final_score_empty_board(engine: GTPEngine) -> None:
    # The empty board scores only White's komi.
    response, _ = engine.handle_line("final_score")
    assert response == f"= W+{engine.komi:g}\n\n"


def test_showboard_contains_grid(engine: GTPEngine) -> None:
    engine.handle_line("play b C3")
    response, _ = engine.handle_line("showboard")
    assert response is not None
    assert "X" in response or "B" in response  # The board rendering contains a black stone.


# ----------------------------------------------------------------------
# GTP frontend search overrides
# ----------------------------------------------------------------------
@pytest.fixture
def gtp_checkpoints(
    tmp_path: Path, tiny_config: AlphaGoZeroConfig, tiny_model: AlphaGoZeroNet
) -> tuple[Path, Path]:
    """Create two 5x5 checkpoints with distinct saved search settings."""
    base = replace(
        tiny_config,
        search=replace(
            tiny_config.search,
            num_simulations=6,
            c_puct=1.75,
            fpu_reduction=0.25,
        ),
    )
    second = replace(
        base,
        search=replace(base.search, num_simulations=10, c_puct=1.25, fpu_reduction=None),
    )
    paths: list[Path] = []
    for name, cfg in [("a.pt", base), ("b.pt", second)]:
        path = tmp_path / name
        torch.save(
            {"kind": "test", "config": asdict(cfg), "model_state_dict": tiny_model.state_dict()},
            path,
        )
        paths.append(path)
    return paths[0], paths[1]


def _run_play_gtp(monkeypatch: pytest.MonkeyPatch, *cli_args: str) -> GTPEngine:
    """Run the frontend with a captured engine and no stdin loop."""
    built: list[GTPEngine] = []
    monkeypatch.setattr(GTPEngine, "run", lambda self: built.append(self))
    play_gtp.main([*cli_args, "--device", "cpu"])
    assert len(built) == 1
    return built[0]


def _cli_error_message(capsys: pytest.CaptureFixture[str]) -> str:
    """Read the last stderr line as the argparse error."""
    return capsys.readouterr().err.strip().splitlines()[-1]


def test_play_gtp_cli_keeps_checkpoint_search_when_no_overrides_are_given(
    monkeypatch: pytest.MonkeyPatch, gtp_checkpoints: tuple[Path, Path]
) -> None:
    a, b = gtp_checkpoints
    eng_a = _run_play_gtp(monkeypatch, "--checkpoint", str(a))
    assert (
        eng_a.config.search.num_simulations,
        eng_a.config.search.c_puct,
        eng_a.config.search.fpu_reduction,
    ) == (6, 1.75, 0.25)
    assert (eng_a.search.config.c_puct, eng_a.search.config.fpu_reduction) == (1.75, 0.25)
    assert eng_a.allow_resign is True

    eng_b = _run_play_gtp(monkeypatch, "--checkpoint", str(b), "--no-resign")
    assert (
        eng_b.config.search.num_simulations,
        eng_b.config.search.c_puct,
        eng_b.config.search.fpu_reduction,
    ) == (10, 1.25, None)
    assert eng_b.allow_resign is False


def test_play_gtp_cli_overrides_simulations_c_puct_fpu_and_opening_moves(
    monkeypatch: pytest.MonkeyPatch, gtp_checkpoints: tuple[Path, Path]
) -> None:
    a, _ = gtp_checkpoints
    eng = _run_play_gtp(
        monkeypatch,
        "--checkpoint",
        str(a),
        "--simulations",
        "12",
        "--c-puct",
        "2.25",
        "--fpu",
        "0.1",
        "--opening-moves",
        "4",
    )
    assert (
        eng.config.search.num_simulations,
        eng.config.search.c_puct,
        eng.config.search.fpu_reduction,
    ) == (12, 2.25, 0.1)
    assert (eng.search.config.c_puct, eng.search.config.fpu_reduction) == (2.25, 0.1)
    assert eng.opening_moves == 4


def test_play_gtp_cli_overriding_one_search_parameter_keeps_the_others(
    monkeypatch: pytest.MonkeyPatch, gtp_checkpoints: tuple[Path, Path]
) -> None:
    a, _ = gtp_checkpoints  # Saved settings are six simulations, c_puct 1.75 and FPU 0.25.
    only_cpuct = _run_play_gtp(monkeypatch, "--checkpoint", str(a), "--c-puct", "2.5")
    assert (
        only_cpuct.config.search.num_simulations,
        only_cpuct.config.search.c_puct,
        only_cpuct.config.search.fpu_reduction,
    ) == (6, 2.5, 0.25)

    only_fpu = _run_play_gtp(monkeypatch, "--checkpoint", str(a), "--fpu", "0.4")
    assert (
        only_fpu.config.search.num_simulations,
        only_fpu.config.search.c_puct,
        only_fpu.config.search.fpu_reduction,
    ) == (6, 1.75, 0.4)


@pytest.mark.parametrize("spelling", ["none", "NONE", "None"])
def test_play_gtp_cli_fpu_none_overrides_a_saved_fpu(
    monkeypatch: pytest.MonkeyPatch, gtp_checkpoints: tuple[Path, Path], spelling: str
) -> None:
    a, _ = gtp_checkpoints  # a.pt stores FPU 0.25.
    eng = _run_play_gtp(monkeypatch, "--checkpoint", str(a), "--fpu", spelling)
    assert (eng.config.search.c_puct, eng.config.search.fpu_reduction) == (1.75, None)
    assert eng.search.config.fpu_reduction is None


def test_play_gtp_cli_zero_fpu_is_a_valid_reduction(
    monkeypatch: pytest.MonkeyPatch, gtp_checkpoints: tuple[Path, Path]
) -> None:
    _, b = gtp_checkpoints  # b.pt stores None. An explicit FPU of zero uses parent value.
    eng = _run_play_gtp(monkeypatch, "--checkpoint", str(b), "--fpu", "0")
    assert eng.config.search.fpu_reduction == 0.0
    assert eng.search.config.fpu_reduction == 0.0


def test_play_gtp_cli_applies_c_puct_and_fpu_to_an_ensemble(
    monkeypatch: pytest.MonkeyPatch, gtp_checkpoints: tuple[Path, Path]
) -> None:
    a, b = gtp_checkpoints
    eng = _run_play_gtp(
        monkeypatch, "--checkpoint", str(a), str(b), "--c-puct", "2.0", "--fpu", "0.15"
    )
    assert (eng.config.search.c_puct, eng.config.search.fpu_reduction) == (2.0, 0.15)


@pytest.mark.parametrize("value", ["0", "-1.5", "abc", "nan", "inf"])
def test_play_gtp_cli_rejects_invalid_c_puct(
    gtp_checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str], value: str
) -> None:
    a, _ = gtp_checkpoints
    with pytest.raises(SystemExit) as exit_info:
        play_gtp.main(["--checkpoint", str(a), "--c-puct", value])
    assert exit_info.value.code == 2
    message = _cli_error_message(capsys)
    assert "--c-puct" in message and "expected" in message


@pytest.mark.parametrize("value", ["-0.1", "abc", "nan", "inf"])
def test_play_gtp_cli_rejects_invalid_fpu(
    gtp_checkpoints: tuple[Path, Path], capsys: pytest.CaptureFixture[str], value: str
) -> None:
    a, _ = gtp_checkpoints
    with pytest.raises(SystemExit) as exit_info:
        play_gtp.main(["--checkpoint", str(a), "--fpu", value])
    assert exit_info.value.code == 2
    message = _cli_error_message(capsys)
    assert "--fpu" in message and "expected" in message


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--simulations", "0"), ("--simulations", "-3"), ("--opening-moves", "-1")],
)
def test_play_gtp_cli_rejects_invalid_simulations_and_opening_moves(
    gtp_checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    a, _ = gtp_checkpoints
    with pytest.raises(SystemExit) as exit_info:
        play_gtp.main(["--checkpoint", str(a), flag, value])
    assert exit_info.value.code == 2
    message = _cli_error_message(capsys)
    assert flag in message and "expected" in message


# ----------------------------------------------------------------------
# Undo, SGF loading and analysis
# ----------------------------------------------------------------------
def test_undo_walks_back_through_played_moves(engine: GTPEngine) -> None:
    initial_hash = engine.state.position_hash
    engine.handle_line("play b C3")
    after_one = engine.state.position_hash
    engine.handle_line("play w A1")

    response, _ = engine.handle_line("undo")
    assert response == "=\n\n"
    assert engine.state.position_hash == after_one
    assert engine.state.move_number == 1

    engine.handle_line("undo")
    assert engine.state.position_hash == initial_hash
    assert engine.state.move_number == 0

    response, _ = engine.handle_line("undo")
    assert response is not None and response.startswith("?")
    assert "cannot undo" in response


def test_undo_after_genmove_and_clear_board(engine: GTPEngine) -> None:
    engine.handle_line("genmove b")
    assert engine.state.move_number == 1
    engine.handle_line("undo")
    assert engine.state.move_number == 0

    engine.handle_line("genmove b")
    engine.handle_line("clear_board")
    response, _ = engine.handle_line("undo")
    assert response is not None and "cannot undo" in response


def test_reg_genmove_is_a_genmove_alias(engine: GTPEngine) -> None:
    response, _ = engine.handle_line("known_command reg_genmove")
    assert response == "= true\n\n"
    response, _ = engine.handle_line("reg_genmove b")
    assert response is not None and response.startswith("= ")
    assert engine.state.move_number == 1


def _sample_sgf() -> str:
    return game_to_sgf(
        board_size=5, komi=2.5, moves=[12, 6, 18, 25, 0], winner=1, score_margin=3.0
    )


def test_loadsgf_replays_game_and_undo_crosses_it(engine: GTPEngine, tmp_path: Path) -> None:
    path = tmp_path / "game.sgf"
    path.write_text(_sample_sgf(), encoding="utf-8")

    response, _ = engine.handle_line(f"loadsgf {path}")
    assert response == "=\n\n"
    expected = GoState.new_game(board_size=5, komi=2.5)
    for action in (12, 6, 18, 25, 0):
        expected = expected.play(action)
    assert engine.state.position_hash == expected.position_hash
    assert engine.state.move_number == 5

    # Load through the second move and allow undo through the replayed history.
    response, _ = engine.handle_line(f"loadsgf {path} 2")
    assert response == "=\n\n"
    assert engine.state.move_number == 2
    engine.handle_line("undo")
    assert engine.state.move_number == 1


def test_loadsgf_rejects_mismatched_rules_and_bad_args(engine: GTPEngine, tmp_path: Path) -> None:
    foreign_komi = tmp_path / "komi.sgf"
    foreign_komi.write_text(
        game_to_sgf(board_size=5, komi=7.5, moves=[12], winner=1), encoding="utf-8"
    )
    response, _ = engine.handle_line(f"loadsgf {foreign_komi}")
    assert response == "? unacceptable komi\n\n"

    foreign_size = tmp_path / "size.sgf"
    foreign_size.write_text(
        game_to_sgf(board_size=9, komi=2.5, moves=[12], winner=1), encoding="utf-8"
    )
    response, _ = engine.handle_line(f"loadsgf {foreign_size}")
    assert response == "? unacceptable size\n\n"

    response, _ = engine.handle_line(f"loadsgf {tmp_path / 'missing.sgf'}")
    assert response is not None and response.startswith("?")

    path = tmp_path / "game.sgf"
    path.write_text(_sample_sgf(), encoding="utf-8")
    response, _ = engine.handle_line(f"loadsgf {path} 99")
    assert response is not None and response.startswith("?")
    response, _ = engine.handle_line(f"loadsgf {path} abc")
    assert response is not None and response.startswith("?")


def test_analyze_commands_offer_best_moves(engine: GTPEngine) -> None:
    response, _ = engine.handle_line("gogui-analyze_commands")
    assert response == "= gfx/Best moves/agz_best_moves\n\n"
    response, _ = engine.handle_line("known_command agz_best_moves")
    assert response == "= true\n\n"

    response, _ = engine.handle_line("list_commands")
    assert response is not None
    for command in ("undo", "loadsgf", "reg_genmove", "gogui-analyze_commands"):
        assert command in response

    response, _ = engine.handle_line("agz_best_moves")
    assert response is not None and response.startswith("= TEXT winrate ")
    body = response.removeprefix("= ").removesuffix("\n\n")
    lines = body.splitlines()
    assert "for black" in lines[0] and "sims)" in lines[0]
    labels = [line for line in lines[1:] if line.startswith("LABEL ")]
    assert labels, "a visited search must report at least one candidate move"
    legal = set(engine.state.legal_actions())
    for label in labels:
        _, vertex, stats = label.split(" ", 2)
        assert vertex_to_action(vertex, 5) in legal
        visits, _, winrate = stats.partition("v/")
        assert int(visits) > 0 and 0 <= int(winrate.rstrip("%")) <= 100


def test_best_moves_on_terminal_position(engine: GTPEngine) -> None:
    engine.handle_line("play b pass")
    engine.handle_line("play w pass")
    assert engine.state.is_terminal()
    response, _ = engine.handle_line("agz_best_moves")
    assert response is not None and "TEXT terminal position" in response


# ----------------------------------------------------------------------
# Search-tree reuse
# ----------------------------------------------------------------------
@pytest.fixture
def captured_engine_roots(monkeypatch: pytest.MonkeyPatch) -> list[MCTSNode | None]:
    """Collect the root supplied to every MCTS.run in the test."""
    roots: list[MCTSNode | None] = []
    real_run = AlphaGoZeroMCTS.run

    def spy(self: AlphaGoZeroMCTS, *args: Any, **kwargs: Any) -> SearchOutput:
        roots.append(kwargs.get("root"))
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(AlphaGoZeroMCTS, "run", spy)
    return roots


def _root_aligned(engine: GTPEngine) -> bool:
    """A retained root must match the current game position."""
    root = engine._root
    return root is None or (
        root.state.to_play == engine.state.to_play
        and bool(np.array_equal(root.state.board, engine.state.board))
    )


def test_genmove_reuses_tree_between_moves(
    engine: GTPEngine, captured_engine_roots: list[MCTSNode | None]
) -> None:
    engine.handle_line("genmove b")
    assert captured_engine_roots == [None]  # Start the first search from scratch.
    assert engine._root is not None and _root_aligned(engine)

    engine.handle_line("genmove w")
    assert captured_engine_roots[-1] is not None  # Start the next search from the saved subtree.
    assert _root_aligned(engine)

    engine.handle_line("genmove b")
    assert captured_engine_roots[-1] is not None
    assert _root_aligned(engine)

    # After an opponent move, advance the root or clear it if its child is absent.
    engine.handle_line("play w pass")
    assert _root_aligned(engine)


def test_tree_reuse_disabled_engine_starts_fresh(
    tiny_config: AlphaGoZeroConfig,
    tiny_model: AlphaGoZeroNet,
    captured_engine_roots: list[MCTSNode | None],
) -> None:
    fresh = GTPEngine(
        model=tiny_model,
        config=tiny_config,
        rng=np.random.default_rng(0),
        allow_resign=False,
        tree_reuse=False,
    )
    fresh.handle_line("genmove b")
    fresh.handle_line("genmove w")
    assert captured_engine_roots == [None, None]
    assert fresh._root is None


def test_undo_loadsgf_and_clear_reset_the_tree(engine: GTPEngine, tmp_path: Path) -> None:
    engine.handle_line("genmove b")
    assert engine._root is not None
    engine.handle_line("undo")
    assert engine._root is None

    engine.handle_line("genmove b")
    assert engine._root is not None
    engine.handle_line("clear_board")
    assert engine._root is None

    engine.handle_line("genmove b")
    assert engine._root is not None
    path = tmp_path / "game.sgf"
    path.write_text(_sample_sgf(), encoding="utf-8")
    engine.handle_line(f"loadsgf {path} 2")
    assert engine._root is None


def test_best_moves_does_not_touch_the_game_tree(engine: GTPEngine) -> None:
    engine.handle_line("genmove b")
    root_before = engine._root
    assert root_before is not None
    engine.handle_line("agz_best_moves 3")
    assert engine._root is root_before


# ----------------------------------------------------------------------
# Time management
# ----------------------------------------------------------------------
@pytest.fixture
def captured_outputs(monkeypatch: pytest.MonkeyPatch) -> list[SearchOutput]:
    """Capture search outputs to measure actual simulation counts."""
    outputs: list[SearchOutput] = []
    real_run = AlphaGoZeroMCTS.run

    def spy(self: AlphaGoZeroMCTS, *args: Any, **kwargs: Any) -> SearchOutput:
        output = real_run(self, *args, **kwargs)
        outputs.append(output)
        return output

    monkeypatch.setattr(AlphaGoZeroMCTS, "run", spy)
    return outputs


class _StepClock:
    """Advance a fake clock by a fixed amount on each read."""

    def __init__(self, step: float) -> None:
        self._step = step
        self._now = 0.0

    def __call__(self) -> float:
        self._now += self._step
        return self._now


class _UniformModel:
    """Use uniform priors and zero values for predictable clock-test visits."""

    def eval(self) -> None:
        return None

    def predict_single(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        return np.zeros(26, dtype=np.float32), 0.0

    def predict_batch(
        self, planes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        batch = planes.shape[0]
        return (
            np.zeros((batch, 26), dtype=np.float32),
            np.zeros(batch, dtype=np.float32),
        )


def _timed_engine(tiny_config: AlphaGoZeroConfig, clock: _StepClock) -> GTPEngine:
    config = replace(
        tiny_config,
        search=replace(tiny_config.search, num_simulations=64, eval_batch_size=4),
    )
    return GTPEngine(
        model=_UniformModel(),  # type: ignore[arg-type]
        config=config,
        rng=np.random.default_rng(0),
        allow_resign=False,
        clock=clock,
    )


def test_without_time_settings_full_simulations(
    tiny_config: AlphaGoZeroConfig, captured_outputs: list[SearchOutput]
) -> None:
    timed = _timed_engine(tiny_config, _StepClock(1.0))
    timed.handle_line("genmove b")
    assert int(captured_outputs[-1].visit_counts.sum()) == 64  # Without a clock the simulation budget sets the cap.


def test_time_settings_budget_stops_the_search(
    tiny_config: AlphaGoZeroConfig, captured_outputs: list[SearchOutput]
) -> None:
    timed = _timed_engine(tiny_config, _StepClock(1.0))
    response, _ = timed.handle_line("time_settings 26 0 0")
    assert response == "=\n\n"
    # Divide 26 seconds by 13 estimated moves. A one-second clock step makes the deadline predictable.
    timed.handle_line("genmove b")
    assert int(captured_outputs[-1].visit_counts.sum()) == 8
    # Subtract elapsed fake time from Black's remaining time.
    assert timed._time_left[1] == pytest.approx(22.0)


def test_time_left_overrides_the_remaining_time(
    tiny_config: AlphaGoZeroConfig, captured_outputs: list[SearchOutput]
) -> None:
    timed = _timed_engine(tiny_config, _StepClock(1.0))
    timed.handle_line("time_settings 1000 0 0")
    timed.handle_line("time_left b 13 0")  # Thirteen seconds remaining gives one second per move.
    timed.handle_line("genmove b")
    assert int(captured_outputs[-1].visit_counts.sum()) == 4  # Stop after a single batch.


def test_byoyomi_adds_per_move_budget(tiny_config: AlphaGoZeroConfig) -> None:
    timed = _timed_engine(tiny_config, _StepClock(1.0))
    timed.handle_line("time_settings 0 30 10")  # Canadian overtime gives 30 seconds per ten stones.
    assert timed._move_time_budget(1) == pytest.approx(3.0)
    timed.handle_line("time_settings 0 7 0")  # Zero stones gives seven seconds for each move.
    assert timed._move_time_budget(1) == pytest.approx(7.0)


# ----------------------------------------------------------------------
# Live-game resignation safeguards
# ----------------------------------------------------------------------
class _LosingModel:
    """Stub predictor with policy and value but no score head."""

    def eval(self) -> None:
        return None

    def predict_single(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        return np.zeros(26, dtype=np.float32), 1.0

    def predict_batch(
        self, planes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        batch = planes.shape[0]
        return (
            np.zeros((batch, 26), dtype=np.float32),
            np.ones(batch, dtype=np.float32),
        )


class _ScoredLosingModel(_LosingModel):
    """Add a score head returning a configured normalized margin."""

    def __init__(self, score: float) -> None:
        self._score = score

    def forward_full(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"score": torch.full((1, 1), self._score)}


def _resign_engine(
    tiny_config: AlphaGoZeroConfig, model: object, **search_changes: Any
) -> GTPEngine:
    config = replace(
        tiny_config,
        search=replace(tiny_config.search, **search_changes),
    )
    return GTPEngine(
        model=model,  # type: ignore[arg-type]
        config=config,
        rng=np.random.default_rng(0),
        allow_resign=True,
    )


def _lost_output(root_q: float = -1.0, best_child_q: float = -1.0) -> SearchOutput:
    """Fix root search estimates to test resignation independently of search."""
    state = GoState.new_game(board_size=5, komi=2.5)
    return SearchOutput(
        policy=np.zeros(state.action_size, dtype=np.float32),
        visit_counts=np.zeros(state.action_size, dtype=np.float32),
        root=MCTSNode(state=state),
        root_value=0.0,
        best_child_q=best_child_q,
        root_q=root_q,
    )


def test_should_resign_value_criterion(tiny_config: AlphaGoZeroConfig) -> None:
    engine = _resign_engine(tiny_config, _LosingModel())
    assert engine._should_resign(_lost_output()) is True
    assert engine._should_resign(_lost_output(root_q=-0.9)) is False
    assert engine._should_resign(_lost_output(best_child_q=-0.9)) is False


def test_resign_min_move_blocks_the_resign(tiny_config: AlphaGoZeroConfig) -> None:
    engine = _resign_engine(tiny_config, _LosingModel(), resign_min_move=100)
    assert engine._should_resign(_lost_output()) is False  # The game has just begun.


def test_resign_score_margin_blocks_close_score(tiny_config: AlphaGoZeroConfig) -> None:
    # A -2.5 point prediction is inside the ten-point margin even when value would allow resignation.
    engine = _resign_engine(
        tiny_config, _ScoredLosingModel(-0.1), resign_score_margin=10.0
    )
    assert engine._should_resign(_lost_output()) is False


def test_resign_score_margin_allows_hopeless_score(tiny_config: AlphaGoZeroConfig) -> None:
    # A -22.5 point prediction exceeds the required deficit.
    engine = _resign_engine(
        tiny_config, _ScoredLosingModel(-0.9), resign_score_margin=10.0
    )
    assert engine._should_resign(_lost_output()) is True


def test_resign_score_margin_without_score_head_keeps_value_rule(
    tiny_config: AlphaGoZeroConfig,
) -> None:
    # Skip the margin guard when the predictor has no score head.
    engine = _resign_engine(tiny_config, _LosingModel(), resign_score_margin=10.0)
    assert engine._should_resign(_lost_output()) is True


def test_genmove_resigns_end_to_end(
    tiny_config: AlphaGoZeroConfig,
    tiny_model: AlphaGoZeroNet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return resign from genmove when the supplied estimates satisfy all conditions."""
    engine = GTPEngine(
        model=tiny_model,
        config=tiny_config,
        rng=np.random.default_rng(0),
        allow_resign=True,
    )
    monkeypatch.setattr(engine.search, "run", lambda **_kwargs: _lost_output())
    response, _ = engine.handle_line("genmove b")
    assert response == "= resign\n\n"
    assert engine.state.move_number == 0  # The resignation ends play without a board move.


def test_play_gtp_applies_calibrated_resign_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_config: AlphaGoZeroConfig,
    tiny_model: AlphaGoZeroNet,
) -> None:
    payload = {
        "kind": "best",
        "config": asdict(tiny_config),
        "model_state_dict": tiny_model.state_dict(),
        "resign_threshold": -0.87,
    }
    path = tmp_path / "calibrated.pt"
    torch.save(payload, path)
    eng = _run_play_gtp(monkeypatch, "--checkpoint", str(path))
    assert eng.config.search.resign_threshold == pytest.approx(-0.87)


def test_play_gtp_ignores_missing_or_broken_calibrated_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tiny_config: AlphaGoZeroConfig,
    tiny_model: AlphaGoZeroNet,
) -> None:
    for broken in (None, "high", 0.5):
        payload: dict[str, object] = {
            "kind": "best",
            "config": asdict(tiny_config),
            "model_state_dict": tiny_model.state_dict(),
        }
        if broken is not None:
            payload["resign_threshold"] = broken
        path = tmp_path / "plain.pt"
        torch.save(payload, path)
        eng = _run_play_gtp(monkeypatch, "--checkpoint", str(path))
        # Fall back to the checkpoint config's initial threshold of -0.95.
        assert eng.config.search.resign_threshold == pytest.approx(-0.95)


# ----------------------------------------------------------------------
# Difficulty levels
# ----------------------------------------------------------------------
def test_policy_only_genmove_plays_raw_policy(
    tiny_config: AlphaGoZeroConfig, captured_engine_roots: list[MCTSNode | None]
) -> None:
    casual = GTPEngine(
        model=_UniformModel(),  # type: ignore[arg-type]
        config=tiny_config,
        rng=np.random.default_rng(0),
        allow_resign=True,
        policy_only=True,
    )
    response, _ = casual.handle_line("genmove b")
    first_legal = GoState.new_game(board_size=5, komi=2.5).legal_actions()[0]
    assert response == f"= {action_to_vertex(first_legal, 5)}\n\n"
    assert captured_engine_roots == []  # Raw-policy mode made no search call.
    assert casual._root is None and casual.state.move_number == 1


def test_policy_only_best_moves_show_policy_probabilities(
    tiny_config: AlphaGoZeroConfig,
) -> None:
    casual = GTPEngine(
        model=_UniformModel(),  # type: ignore[arg-type]
        config=tiny_config,
        rng=np.random.default_rng(0),
        policy_only=True,
    )
    response, _ = casual.handle_line("agz_best_moves 2")
    assert response is not None and "policy only" in response
    labels = [line for line in response.splitlines() if line.startswith("LABEL ")]
    assert len(labels) == 2
    assert all(label.rsplit(" ", 1)[1].endswith("%") for label in labels)


def test_play_gtp_level_presets(
    monkeypatch: pytest.MonkeyPatch, gtp_checkpoints: tuple[Path, Path]
) -> None:
    a, _b = gtp_checkpoints  # Checkpoint A stores six simulations.

    casual = _run_play_gtp(monkeypatch, "--checkpoint", str(a), "--level", "casual")
    assert casual.policy_only is True

    club = _run_play_gtp(monkeypatch, "--checkpoint", str(a), "--level", "club")
    assert club.policy_only is False
    assert club.config.search.num_simulations == 64  # Use the difficulty preset.

    explicit = _run_play_gtp(
        monkeypatch, "--checkpoint", str(a), "--level", "club", "--simulations", "9"
    )
    assert explicit.config.search.num_simulations == 9  # An explicit flag overrides the preset.

    champion = _run_play_gtp(monkeypatch, "--checkpoint", str(a), "--level", "champion")
    assert champion.policy_only is False
    assert champion.config.search.num_simulations == 6  # Use the checkpoint settings directly.

    default = _run_play_gtp(monkeypatch, "--checkpoint", str(a))
    assert default.policy_only is False
    assert default.config.search.num_simulations == 6


def test_play_gtp_rejects_unknown_level(
    monkeypatch: pytest.MonkeyPatch, gtp_checkpoints: tuple[Path, Path]
) -> None:
    a, _b = gtp_checkpoints
    with pytest.raises(SystemExit) as excinfo:
        _run_play_gtp(monkeypatch, "--checkpoint", str(a), "--level", "grandmaster")
    assert excinfo.value.code == 2


def test_time_commands_reject_garbage(engine: GTPEngine) -> None:
    for line in (
        "time_settings 10 x 0",
        "time_settings 10 5",
        "time_settings -1 0 0",
        "time_left x 10 0",
        "time_left b z 0",
    ):
        response, _ = engine.handle_line(line)
        assert response is not None and response.startswith("?"), line
    # Rejected time-control commands leave the time budget unset.
    assert engine._move_time_budget(1) is None


def test_external_match_forwards_c_puct_and_fpu_to_play_gtp_via_model_args(
    monkeypatch: pytest.MonkeyPatch, gtp_checkpoints: tuple[Path, Path]
) -> None:
    """Forward external_match model arguments through play_gtp to the search configuration."""
    a, _ = gtp_checkpoints
    args = external_match.build_parser().parse_args(
        [
            "--checkpoint",
            str(a),
            "--opponent-command",
            "gnugo --mode gtp",
            "--simulations",
            "8",
            "--model-args=--c-puct 2.1 --fpu 0.15",
        ]
    )
    cmd = external_match.build_model_command(args)
    assert cmd[:2] == [sys.executable, str(external_match.PLAY_GTP)]
    built: list[GTPEngine] = []
    monkeypatch.setattr(GTPEngine, "run", lambda self: built.append(self))
    play_gtp.main(cmd[2:])
    assert len(built) == 1
    assert (
        built[0].config.search.num_simulations,
        built[0].config.search.c_puct,
        built[0].config.search.fpu_reduction,
    ) == (8, 2.1, 0.15)

