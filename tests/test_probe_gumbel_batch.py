"""Test the probe measuring changes in Sequential Halving as leaf batches grow."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gumbel_support import FakeNet, make_state

from alphago_zero import mcts as mcts_module
from alphago_zero.config import AlphaGoZeroConfig, GameConfig, SearchConfig
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.pipeline import _save_checkpoint
from examples import probe_gumbel_batch as probe

_GAME = GameConfig(board_size=5, komi=2.5, max_moves_multiplier=2)


def _write_checkpoint(path: Path, cfg: AlphaGoZeroConfig) -> None:
    torch.manual_seed(0)
    model = AlphaGoZeroNet(cfg.network)
    _save_checkpoint(path, model, None, extra={"config": asdict(cfg), "kind": "final_best", "global_step": 0})


def _output(move: int, improved: list[float], root_q: float) -> SimpleNamespace:
    """Specify the chosen action, improved policy and root estimate directly."""
    policy = np.zeros(len(improved), dtype=np.float32)
    policy[move] = 1.0
    return SimpleNamespace(policy=policy, improved_policy=np.asarray(improved, dtype=np.float32), root_q=root_q)


# ---------------------------------------------------------------------------
# Position sampling
# ---------------------------------------------------------------------------
def test_sample_positions_are_deterministic_and_cover_the_whole_opening_range() -> None:
    positions = probe.sample_positions(_GAME, 200, max_opening_moves=8, seed=2)
    assert len(positions) == 200
    # Sample opening lengths uniformly from zero through eight inclusive.
    assert {p.move_number for p in positions} == set(range(9))
    assert all(not p.is_terminal() for p in positions)
    # Exclude pass moves from random openings.
    assert all(p.consecutive_passes == 0 for p in positions)

    again = probe.sample_positions(_GAME, 200, max_opening_moves=8, seed=2)
    assert all(np.array_equal(a.encode(), b.encode()) for a, b in zip(positions, again, strict=True))
    other = probe.sample_positions(_GAME, 200, max_opening_moves=8, seed=3)
    assert any(not np.array_equal(a.encode(), b.encode()) for a, b in zip(positions, other, strict=True))


def test_sample_positions_follow_the_rules_of_the_game_config() -> None:
    """Use the checkpoint's board size, komi, move limit and superko rule."""
    game = GameConfig(board_size=5, komi=3.5, max_moves_multiplier=3, superko_rule="situational")
    positions = probe.sample_positions(game, 6, max_opening_moves=5, seed=1)
    assert len(positions) == 6
    for position in positions:
        assert (position.board_size, position.komi) == (5, 3.5)
        assert (position.max_moves, position.superko_rule) == (game.max_moves, "situational")


def test_sample_positions_without_opening_moves_are_empty_boards() -> None:
    positions = probe.sample_positions(_GAME, 5, max_opening_moves=0, seed=1)
    assert len(positions) == 5
    assert all(p.move_number == 0 and not p.encode()[:2].any() for p in positions)


def test_sample_positions_skip_terminal_positions() -> None:
    """Discard opening positions that already reached the move limit."""
    tight = GameConfig(board_size=5, komi=2.5, max_moves_multiplier=1)  # The limit is 25 moves.
    positions = probe.sample_positions(tight, 100, max_opening_moves=40, seed=1)
    assert 0 < len(positions) < 100
    assert all(not p.is_terminal() and p.move_number < tight.max_moves for p in positions)


# ---------------------------------------------------------------------------
# Board symmetries
# ---------------------------------------------------------------------------
def test_identity_symmetry_disables_rotation_without_spending_the_rng_and_restores_it() -> None:
    original = mcts_module.random_dihedral_index
    rng = np.random.default_rng(0)
    before = rng.bit_generator.state
    with probe.identity_symmetry():
        assert [mcts_module.random_dihedral_index(rng) for _ in range(20)] == [0] * 20
        assert rng.bit_generator.state == before  # Keep the root noise RNG sequence unchanged.
    assert mcts_module.random_dihedral_index is original
    with pytest.raises(RuntimeError, match="boom"), probe.identity_symmetry():
        raise RuntimeError("boom")
    assert mcts_module.random_dihedral_index is original  # Restore the original function after an exception too.
    assert {original(rng) for _ in range(64)} - {0}  # The original function returns nonidentity symmetries as well.


# ---------------------------------------------------------------------------
# Metrics and summary calculations
# ---------------------------------------------------------------------------
def test_improved_policy_is_required_from_the_search() -> None:
    with pytest.raises(RuntimeError, match="π'"):
        probe._improved_policy(SimpleNamespace(improved_policy=None))
    pi = probe._improved_policy(SimpleNamespace(improved_policy=np.array([0.25, 0.75], dtype=np.float32)))
    assert pi.dtype == np.float64 and pi.tolist() == [0.25, 0.75]


def _scripted_search(calls: list[tuple[Any, ...]], symmetries: list[list[int]]) -> Any:
    """Stub search varies its answer by batch size and records every argument."""

    def fake(model: Any, state: Any, *, simulations: int, batch: int, max_considered: int,
             virtual_loss: float, rng: np.random.Generator) -> SimpleNamespace:  # fmt: skip
        calls.append((batch, simulations, max_considered, virtual_loss, float(rng.random())))
        symmetries.append([mcts_module.random_dihedral_index(np.random.default_rng(i)) for i in range(16)])
        if batch == 4:  # Choose a different move and change all four probabilities and the root value.
            return _output(3, [0.25, 0.25, 0.25, 0.25], -0.10)
        return _output(2, [0.5, 0.5, 0.0, 0.0], 0.20)  # Reference search and matching batch eight.

    return fake


def _measure_scripted(
    monkeypatch: pytest.MonkeyPatch, seed: int, symmetries: list[list[int]] | None = None
) -> tuple[list[probe.PairOutcome], list[tuple[Any, ...]]]:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(probe, "_run_search", _scripted_search(calls, [] if symmetries is None else symmetries))
    outcomes = probe.measure_pairs(
        object(),
        [make_state(), make_state((12,))],
        simulations=24, batches=(4, 8), noise_seeds=2, max_considered=6, virtual_loss=0.5, seed=seed,
    )  # fmt: skip
    return outcomes, calls


def test_measure_pairs_computes_the_metrics_from_their_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes, _ = _measure_scripted(monkeypatch, seed=9)
    # Iterate positions, noise samples and batch sizes in that order.
    assert [(o.position, o.batch) for o in outcomes] == [(0, 4), (0, 8), (0, 4), (0, 8), (1, 4), (1, 8), (1, 4), (1, 8)]
    assert {o.simulations for o in outcomes} == {24}
    for outcome in outcomes:
        if outcome.batch == 4:
            assert outcome.same_move is False  # Move three instead of reference move two.
            assert outcome.tv_distance == pytest.approx(0.5)  # 0.5 * (0.25 + 0.25 + 0.25 + 0.25)
            assert outcome.root_q_gap == pytest.approx(0.30)  # Use the absolute value of -0.10 - 0.20.
        else:
            assert outcome.same_move is True
            assert outcome.tv_distance == 0.0 and outcome.root_q_gap == 0.0


def test_measure_pairs_passes_the_options_and_gives_reference_and_candidate_one_noise_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, calls = _measure_scripted(monkeypatch, seed=9)
    # For each of four position/noise pairs run batch one, then the candidate batches.
    assert [call[0] for call in calls] == [1, 4, 8] * 4
    assert {call[1:4] for call in calls} == {(24, 6, 0.5)}  # Simulations, candidate count and virtual loss.
    fingerprints = [calls[i][4] for i in range(0, 12, 3)]
    for group in range(4):  # All searches in a pair use the same root noise.
        assert len({calls[3 * group + k][4] for k in range(3)}) == 1
    assert len(set(fingerprints)) == 4  # Different position/noise pairs use distinct noise.

    _, other_calls = _measure_scripted(monkeypatch, seed=10)
    assert {c[4] for c in other_calls}.isdisjoint({c[4] for c in calls})  # Changing the seed changes the noise sequence.


def test_every_search_of_the_measurement_runs_without_board_symmetries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable random rotations so batch size alone can explain a difference."""
    symmetries: list[list[int]] = []
    _measure_scripted(monkeypatch, seed=9, symmetries=symmetries)
    assert len(symmetries) == 12
    assert all(seen == [0] * 16 for seen in symmetries)
    # Verify that the original sampler really can choose another rotation.
    assert any(mcts_module.random_dihedral_index(np.random.default_rng(i)) for i in range(16))


def test_run_search_builds_a_gumbel_search_from_its_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class SpySearch:
        def __init__(self, model: Any, config: SearchConfig, rng: np.random.Generator) -> None:
            captured.update(model=model, config=config, rng=rng)

        def run(self, state: Any, **kwargs: Any) -> str:
            captured.update(state=state, run_kwargs=kwargs)
            return "output"

    monkeypatch.setattr(probe, "GumbelMCTS", SpySearch)
    model, state, rng = object(), make_state(), np.random.default_rng(0)
    result = probe._run_search(
        model, state, simulations=24, batch=8, max_considered=6, virtual_loss=0.5, rng=rng
    )  # fmt: skip
    assert result == "output"
    config = captured["config"]
    assert (config.search_algo, config.num_simulations, config.eval_batch_size) == ("gumbel", 24, 8)
    assert (config.gumbel_max_considered, config.virtual_loss) == (6, 0.5)
    assert captured["model"] is model and captured["rng"] is rng and captured["state"] is state
    # Enable root Gumbel noise. Move temperature does not enter winner selection.
    assert captured["run_kwargs"] == {"temperature": 1.0, "add_root_noise": True}


def _outcome(simulations: int, batch: int, position: int, same: bool, tv: float, gap: float) -> probe.PairOutcome:
    return probe.PairOutcome(simulations, batch, position, same, tv, gap)


def test_summarize_outcomes_by_hand() -> None:
    outcomes = [
        # For (32, 2), agreement is two of two noise samples on the first position and one of two on the
        # second.
        _outcome(32, 2, 0, True, 0.0, 0.01), _outcome(32, 2, 0, True, 0.1, 0.02),
        _outcome(32, 2, 1, True, 0.2, 0.03), _outcome(32, 2, 1, False, 0.9, 0.10),
        # A single position cannot provide a standard error for (32, 4).
        _outcome(32, 4, 0, False, 0.5, 0.10), _outcome(32, 4, 0, True, 0.3, 0.20),
        # Keep first-seen ordering for the (64, 2) group.
        _outcome(64, 2, 0, True, 0.1, 0.02), _outcome(64, 2, 1, True, 0.1, 0.02),
    ]  # fmt: skip
    rows = probe.summarize_outcomes(outcomes)
    assert [(r["simulations"], r["batch"]) for r in rows] == [(32, 2), (32, 4), (64, 2)]

    first = rows[0]
    assert (first["pairs"], first["positions"]) == (4, 2)
    assert first["same_move"] == pytest.approx(0.75)
    # Position agreement fractions of one and one half determine the sample standard error.
    assert first["same_move_se"] == pytest.approx(0.25)
    assert first["tv_distance"] == pytest.approx(0.3)
    assert first["root_q_gap"] == pytest.approx(0.04)

    second = rows[1]
    assert (second["pairs"], second["positions"]) == (2, 1)
    assert second["same_move"] == pytest.approx(0.5)
    assert second["same_move_se"] is None
    assert second["tv_distance"] == pytest.approx(0.4)
    assert second["root_q_gap"] == pytest.approx(0.15)

    third = rows[2]
    assert (third["same_move"], third["same_move_se"]) == (1.0, 0.0)  # Perfect agreement at every position has zero spread.


# ---------------------------------------------------------------------------
# Real search with a deterministic predictor
# ---------------------------------------------------------------------------
def _measure_fake(seed: int = 5) -> list[probe.PairOutcome]:
    positions = probe.sample_positions(_GAME, 10, max_opening_moves=8, seed=2)
    return probe.measure_pairs(
        FakeNet(0, good_point=12), positions,
        simulations=32, batches=(1, 2, 16), noise_seeds=2, max_considered=16, virtual_loss=1.0, seed=seed,
    )  # fmt: skip


def test_batch_one_reproduces_the_reference_exactly_and_larger_batches_distort_more() -> None:
    outcomes = _measure_fake()
    assert len(outcomes) == 10 * 2 * 3
    rows = {row["batch"]: row for row in probe.summarize_outcomes(outcomes)}
    assert (rows[1]["same_move"], rows[1]["tv_distance"], rows[1]["root_q_gap"]) == (1.0, 0.0, 0.0)  # Control.
    assert rows[16]["tv_distance"] > rows[2]["tv_distance"] > 0.0
    assert rows[16]["root_q_gap"] > rows[2]["root_q_gap"] > 0.0
    assert rows[16]["same_move"] < 1.0  # Batch sixteen changes the selected move in some positions.


def test_measurement_is_deterministic_and_depends_on_the_seed() -> None:
    first, second, other = _measure_fake(5), _measure_fake(5), _measure_fake(6)
    assert first == second
    assert first != other


def test_measure_pairs_restores_symmetries_after_the_run() -> None:
    original = mcts_module.random_dihedral_index
    _measure_fake()
    assert mcts_module.random_dihedral_index is original


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _report_data() -> dict[str, Any]:
    return {
        "checkpoint": "runs/x/final_best.pt", "checkpoint_sha256_12": "ABCDEF012345",
        "board_size": 9, "komi": 7.5, "device": "cpu", "positions": 60, "noise_seeds": 2,
        "max_opening_moves": 20, "max_considered": 16, "virtual_loss": 1.0, "seed": 1,
        "elapsed_sec": 125.4,
        "rows": [
            {"simulations": 32, "batch": 2, "pairs": 120, "positions": 60, "same_move": 0.95,
             "same_move_se": 0.0229, "tv_distance": 0.0394, "root_q_gap": 0.00572},
            {"simulations": 64, "batch": 16, "pairs": 5, "positions": 1, "same_move": 0.6,
             "same_move_se": None, "tv_distance": 0.296, "root_q_gap": 0.0596},
        ],
    }  # fmt: skip


def test_format_report_shows_every_setting_and_row() -> None:
    lines = probe.format_report(_report_data()).splitlines()
    assert lines[0].startswith("== Sequential Halving distortion")
    header = "\n".join(lines[1:5])
    for fragment in ("runs/x/final_best.pt", "ABCDEF012345", "board 9x9", "komi 7.5", "device cpu",
                     "positions: 60", "up to 20 moves", "noise samples per position: 2", "m = 16", "virtual loss 1.0",
                     "batch 1", "elapsed 125 s"):  # fmt: skip
        assert fragment in header, fragment
    first = next(line for line in lines if line.split()[:2] == ["32", "2"])
    assert first.split() == ["32", "2", "120", "95.0", "%", "2.3", "%", "0.039", "0.0057"]
    second = next(line for line in lines if line.split()[:2] == ["64", "16"])
    assert second.split() == ["64", "16", "5", "60.0", "%", "—", "0.296", "0.0596"]  # Use a dash when no standard error is available.
    assert "Use eval_match to measure playing strength" in lines[-1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_main_runs_on_a_checkpoint_and_writes_consistent_json_and_report(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "net.pt"
    _write_checkpoint(checkpoint, tiny_config)
    out = tmp_path / "nested" / "batch.json"

    code = probe.main([
        "--checkpoint", str(checkpoint), "--positions", "3", "--noise-seeds", "1",
        "--simulations", "8", "8", "--batches", "2", "4", "2", "--max-opening-moves", "4",
        "--seed", "3", "--json-out", str(out),
    ])  # fmt: skip

    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (data["board_size"], data["komi"], data["device"], data["positions"]) == (5, 2.5, "cpu", 3)
    assert (data["noise_seeds"], data["max_opening_moves"], data["seed"]) == (1, 4, 3)
    assert (data["max_considered"], data["virtual_loss"]) == (16, 1.0)
    assert len(data["checkpoint_sha256_12"]) == 12 and data["checkpoint_sha256_12"].isupper()
    assert data["elapsed_sec"] > 0.0
    # Ignore repeated arguments and emit one row per budget/batch pair.
    assert [(r["simulations"], r["batch"], r["pairs"]) for r in data["rows"]] == [(8, 2, 3), (8, 4, 3)]
    captured = capsys.readouterr()
    assert captured.out.rstrip("\n") == probe.format_report(data)
    assert "completed: 8 simulations" in captured.err

    # The library call with the same arguments must give identical values.
    model, _search, game = probe.load_checkpoint(checkpoint, torch.device("cpu"))
    positions = probe.sample_positions(game, 3, max_opening_moves=4, seed=3)
    direct = probe.measure_pairs(
        model, positions, simulations=8, batches=[2, 4], noise_seeds=1,
        max_considered=16, virtual_loss=1.0, seed=3,
    )  # fmt: skip
    assert probe.summarize_outcomes(direct) == data["rows"]


def test_main_passes_every_option_to_the_measurement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "net.pt"
    checkpoint.write_bytes(b"not a real checkpoint")
    model = FakeNet(0)
    seen: dict[str, Any] = {"measure": []}
    monkeypatch.setattr(probe, "load_checkpoint", lambda path, device: (model, SearchConfig(), _GAME))
    # Leave the process-global torch thread count unchanged in this test.
    monkeypatch.setattr(torch, "set_num_threads", lambda count: seen.__setitem__("threads", count))

    def fake_sample(game: GameConfig, count: int, **kwargs: Any) -> list[str]:
        seen["sample"] = (game, count, kwargs)
        return ["state-a", "state-b"]

    def fake_measure(model_arg: Any, positions: Any, **kwargs: Any) -> list[probe.PairOutcome]:
        seen["measure"].append((model_arg, list(positions), kwargs))
        return [_outcome(kwargs["simulations"], 2, 0, True, 0.0, 0.0)]

    monkeypatch.setattr(probe, "sample_positions", fake_sample)
    monkeypatch.setattr(probe, "measure_pairs", fake_measure)

    code = probe.main([
        "--checkpoint", str(checkpoint), "--positions", "7", "--noise-seeds", "3",
        "--simulations", "16", "48", "--batches", "2", "6", "--max-considered", "8",
        "--virtual-loss", "0.25", "--max-opening-moves", "11", "--seed", "42", "--threads", "1",
    ])  # fmt: skip

    assert code == 0
    assert seen["threads"] == 1
    assert seen["sample"] == (_GAME, 7, {"max_opening_moves": 11, "seed": 42})
    assert [call[0] for call in seen["measure"]] == [model, model]
    assert [call[1] for call in seen["measure"]] == [["state-a", "state-b"]] * 2
    for call, simulations in zip(seen["measure"], (16, 48), strict=True):
        assert call[2] == {
            "simulations": simulations, "batches": [2, 6], "noise_seeds": 3,
            "max_considered": 8, "virtual_loss": 0.25, "seed": 42,
        }  # fmt: skip
    printed = capsys.readouterr().out
    assert printed.startswith("== Sequential Halving distortion")
    # Report the actual number of positions returned by sampling, along with the effective settings.
    assert "positions: 2 (random openings up to 11 moves), noise samples per position: 3, m = 8, virtual loss 0" in printed


@pytest.mark.parametrize(
    "argv",
    [
        [],  # Missing checkpoint.
        ["--checkpoint", "net.pt", "--batches", "1", "4"],  # Batch one is the reference.
        ["--checkpoint", "net.pt", "--batches", "0"],
        ["--checkpoint", "net.pt", "--max-considered", "1"],  # Sequential Halving needs at least two candidates.
        ["--checkpoint", "net.pt", "--virtual-loss", "-0.5"],
        ["--checkpoint", "net.pt", "--virtual-loss", "nan"],
        ["--checkpoint", "net.pt", "--positions", "0"],
        ["--checkpoint", "net.pt", "--noise-seeds", "0"],
        ["--checkpoint", "net.pt", "--simulations", "0"],
        ["--checkpoint", "net.pt", "--simulations"],  # Option supplied without a value.
        ["--checkpoint", "net.pt", "--max-opening-moves", "-1"],
        ["--checkpoint", "net.pt", "--threads", "0"],
    ],
)
def test_cli_rejects_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        probe.main(argv)
    assert raised.value.code == 2


def test_cli_zero_virtual_loss_is_allowed() -> None:
    args = probe._parse_args(["--checkpoint", "net.pt", "--virtual-loss", "0"])
    assert args.virtual_loss == 0.0


def test_cli_rejects_an_unusable_json_directory_before_any_work(tmp_path: Path) -> None:
    blocker = tmp_path / "file.txt"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        probe.main(["--checkpoint", str(tmp_path / "missing.pt"), "--json-out", str(blocker / "out.json")])
    assert raised.value.code == 2  # Reject the output path before loading the missing checkpoint.
