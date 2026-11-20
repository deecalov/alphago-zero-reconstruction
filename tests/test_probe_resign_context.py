from __future__ import annotations

"""Test actual and hypothetical resignation measurements."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig, NetworkConfig
from alphago_zero.game import GoState
from alphago_zero.mcts import EdgeStats, MCTSNode
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.parallel_selfplay import play_parallel_self_play_games
from alphago_zero.pipeline import _save_checkpoint
from alphago_zero.selfplay import ResignationController
from examples import probe_resign_context as probe

_EVENT_KEYS = {
    "kind", "game_id", "move_number", "to_play", "n_legal", "root_q", "best_child_q", "threshold",
    "move_is_full", "most_visited_is_pass", "most_visited_terminal", "best_q_terminal",
    "all_visited_terminal", "consecutive_passes", "most_visited_q", "resignation_disabled",
}


class _ConstantValueNet(AlphaGoZeroNet):
    """Return a uniform policy and a constant value for every position."""

    def __init__(self, config: NetworkConfig, value: float) -> None:
        super().__init__(config)
        self._constant_value = value

    def predict_single(self, state_planes: np.ndarray) -> tuple[np.ndarray, float]:
        action_size = state_planes.shape[-1] * state_planes.shape[-2] + 1
        return np.zeros(action_size, dtype=np.float32), self._constant_value

    def predict_batch(self, states_planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = states_planes.shape[0]
        action_size = states_planes.shape[-1] * states_planes.shape[-2] + 1
        return (
            np.zeros((batch, action_size), dtype=np.float32),
            np.full(batch, self._constant_value, dtype=np.float32),
        )


def _probe_config(tiny_config: AlphaGoZeroConfig, *, mode: str) -> AlphaGoZeroConfig:
    cfg = tiny_config
    cfg.search.num_simulations = 4
    cfg.search.eval_batch_size = 1
    cfg.search.resign_min_visits = 1
    cfg.search.disable_resign_prob = 1.0 if mode == "disabled" else 0.0
    cfg.training.parallel_games = 2
    cfg.validate()
    return cfg


def _always_losing_probe(tiny_config: AlphaGoZeroConfig, *, mode: str, games: int, seed: int) -> dict[str, Any]:
    cfg = _probe_config(tiny_config, mode=mode)
    model = _ConstantValueNet(cfg.network, 1.0)
    data = probe.run_probe(model, cfg, games=games, seed=seed)
    data["args"] = {"mode": mode, "sims": "4", "seed": str(seed)}
    return data


def _write_checkpoint(path: Path, cfg: AlphaGoZeroConfig, *, unknown_training_key: bool = False) -> AlphaGoZeroNet:
    torch.manual_seed(0)
    model = AlphaGoZeroNet(cfg.network)
    raw = asdict(cfg)
    if unknown_training_key:
        raw["training"]["future_field"] = 1  # Simulated field from a newer checkpoint schema.
    _save_checkpoint(path, model, None, extra={"config": raw, "kind": "final_best", "global_step": 0})
    return model


# ---------------------------------------------------------------------------
# Classification and event recording from constructed data
# ---------------------------------------------------------------------------
def _root_after_one_pass() -> tuple[MCTSNode, SimpleNamespace]:
    start = GoState.new_game(board_size=5, komi=2.5)
    state = start.play(start.pass_action)  # One pass has already occurred, so another ends the game.
    pass_action = state.pass_action
    edges = {
        pass_action: EdgeStats(
            prior=0.5, visit_count=30, value_sum=-27.0, child=MCTSNode(state=state.play(pass_action))
        ),
        0: EdgeStats(prior=0.3, visit_count=5, value_sum=-2.5, child=MCTSNode(state=state.play(0))),
        1: EdgeStats(prior=0.2),
    }
    return MCTSNode(state=state, edges=edges, is_expanded=True), SimpleNamespace(state=state, move_is_full=True)


def test_classify_marks_terminal_edges() -> None:
    root, slot = _root_after_one_pass()
    record = probe.classify(root, slot, threshold=-0.95, min_visits=8)

    assert record["move_number"] == slot.state.move_number
    assert record["to_play"] == slot.state.to_play
    assert record["consecutive_passes"] == 1
    assert record["n_legal"] == 3
    assert record["root_q"] == pytest.approx(-29.5 / 35)
    assert record["best_child_q"] == pytest.approx(-0.5)  # The best-Q edge is nonterminal.
    assert record["most_visited_q"] == pytest.approx(-0.9)
    assert record["threshold"] == -0.95 and record["move_is_full"] is True
    # The most visited move is the second pass, but the best-Q edge continues play.
    assert record["most_visited_is_pass"] is True
    assert record["most_visited_terminal"] is True
    assert record["best_q_terminal"] is False
    assert record["all_visited_terminal"] is False
    assert probe._is_exact(record)


def test_classify_requires_a_searched_root() -> None:
    root, slot = _root_after_one_pass()
    for edge in root.edges.values():
        edge.visit_count = 0
    with pytest.raises(ValueError, match="no visited edges"):
        probe.classify(root, slot, threshold=-0.95, min_visits=1)


def test_recording_set_logs_every_add_but_keeps_set_semantics() -> None:
    class FakeDriver:
        def __init__(self) -> None:
            self.calls: list[tuple[object, str]] = []

        def log(self, slot: object, kind: str) -> None:
            self.calls.append((slot, kind))

    driver, slot = FakeDriver(), object()
    players = probe._RecordingSet(driver, slot)  # type: ignore[arg-type]
    players.add(1)
    players.add(1)
    players.add(-1)

    # Record repeated hypothetical resignation decisions even for the same color.
    assert driver.calls == [(slot, "hypothetical")] * 3
    # Keep ordinary set membership and truth behavior for the driver.
    assert players == {1, -1} and bool(players) and 1 in players


# ---------------------------------------------------------------------------
# Instrumented driver with a stub network
# ---------------------------------------------------------------------------
def test_enabled_mode_logs_one_resign_per_game(tiny_config: AlphaGoZeroConfig) -> None:
    data = _always_losing_probe(tiny_config, mode="enabled", games=4, seed=0)

    assert len(data["games"]) == 4 and all(game["resigned"] for game in data["games"])
    events = data["events"]
    assert {event["kind"] for event in events} == {"resign"}
    assert sorted(event["game_id"] for event in events) == [0, 1, 2, 3]
    for event in events:
        assert _EVENT_KEYS <= event.keys()
        assert not event["resignation_disabled"] and event["move_is_full"]
        assert event["root_q"] < event["threshold"] and event["best_child_q"] < event["threshold"]
    json.dumps(data)  # Raw measurements must serialize directly to JSON.


def test_disabled_mode_logs_hypothetical_decisions_and_natural_end(tiny_config: AlphaGoZeroConfig) -> None:
    data = _always_losing_probe(tiny_config, mode="disabled", games=3, seed=1)

    assert not any(game["resigned"] for game in data["games"])
    events = data["events"]
    natural = [event for event in events if event["kind"] == "natural_end"]
    hypothetical = [event for event in events if event["kind"] == "hypothetical"]
    assert sorted(event["game_id"] for event in natural) == [0, 1, 2]
    assert {event["ended_by"] for event in natural} <= {"double_pass", "max_moves"}
    # The criterion would trigger on the first full move in every tested game.
    assert {event["game_id"] for event in hypothetical} == {0, 1, 2}
    assert all(event["resignation_disabled"] for event in hypothetical)
    assert data["controller_history_len"] == 3
    for event in events:
        assert _EVENT_KEYS <= event.keys()


@pytest.mark.parametrize("mode", ["enabled", "disabled"])
def test_spy_driver_does_not_change_the_games(tiny_config: AlphaGoZeroConfig, mode: str) -> None:
    # Instrumentation must preserve RNG consumption and decisions.
    cfg = _probe_config(tiny_config, mode=mode)
    model = _ConstantValueNet(cfg.network, 1.0)
    spied = probe.run_probe(model, cfg, games=4, seed=5)

    controller = ResignationController(
        threshold=cfg.search.resign_threshold,
        target_false_positive_rate=cfg.search.target_false_positive_rate,
        adjustment=cfg.search.resignation_adjustment,
        window=cfg.search.resignation_window,
    )
    plain = play_parallel_self_play_games(
        model=model, config=cfg, rng=np.random.default_rng(5), resignation_controller=controller, num_games=4
    )

    assert [(g["num_moves"], g["winner"], g["resigned"]) for g in spied["games"]] == [
        (r.num_moves, r.winner, r.resigned) for r in plain
    ]
    assert spied["final_threshold"] == controller.threshold
    assert spied["controller_history_len"] == len(controller.history)


# ---------------------------------------------------------------------------
# Checkpoint loading and CLI
# ---------------------------------------------------------------------------
def test_load_probe_setup_applies_overrides_and_keeps_the_recipe(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path
) -> None:
    tiny_config.training.parallel_games = 4
    tiny_config.training.league_prob = 0.5
    tiny_config.training.league_checkpoints = ["champion.pt"]
    tiny_config.validate()
    path = tmp_path / "net.pt"
    saved = _write_checkpoint(path, tiny_config, unknown_training_key=True)

    model, config = probe.load_probe_setup(path, device="cpu", sims=7, mode="disabled", parallel=3)

    assert config.search.num_simulations == 7 and config.search.disable_resign_prob == 1.0
    assert config.training.parallel_games == 3
    assert config.training.league_prob == 0.0  # Disable the league for pure self-play measurements.
    # Preserve the network's remaining training search settings.
    assert config.search.c_puct == tiny_config.search.c_puct
    assert config.search.resign_threshold == tiny_config.search.resign_threshold
    assert not model.training
    for key, tensor in saved.state_dict().items():
        assert torch.equal(model.state_dict()[key], tensor)

    _, enabled = probe.load_probe_setup(path, device="cpu", sims=7, mode="enabled", parallel=3)
    assert enabled.search.disable_resign_prob == 0.0


@pytest.mark.parametrize("mode", ["enabled", "disabled"])
def test_main_writes_json_and_report(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    checkpoint = tmp_path / "net.pt"
    _write_checkpoint(checkpoint, tiny_config)
    out, report = tmp_path / "nested" / "probe.json", tmp_path / "report.txt"

    code = probe.main([
        "--checkpoint", str(checkpoint), "--mode", mode, "--games", "3", "--sims", "4",
        "--parallel", "2", "--seed", "0", "--out", str(out), "--report-out", str(report),
    ])

    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["args"]["mode"] == mode and data["args"]["games"] == "3"
    assert len(data["checkpoint_sha256_12"]) == 12 and data["checkpoint_sha256_12"].isupper()
    assert len(data["games"]) == 3
    terminal = [event for event in data["events"] if event["kind"] in ("resign", "natural_end")]
    assert sorted(event["game_id"] for event in terminal) == [0, 1, 2]  # Exactly one per game.
    printed = capsys.readouterr().out
    assert printed.startswith(f"== Mode {mode}")
    assert report.read_text(encoding="utf-8").strip() == printed.strip()


def test_analyze_reads_saved_json(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "probe_a.json"
    data = _always_losing_probe(tiny_config, mode="enabled", games=4, seed=0)
    path.write_text(json.dumps(data), encoding="utf-8")

    assert probe.main(["--analyze", str(path)]) == 0
    assert "resigned: 4/4 = 100.0%" in capsys.readouterr().out


def test_summaries_cover_both_modes_and_verbosity(tiny_config: AlphaGoZeroConfig) -> None:
    enabled = _always_losing_probe(tiny_config, mode="enabled", games=4, seed=0)
    assert "-- resignation decisions:" in probe.summarize(enabled)

    disabled = _always_losing_probe(tiny_config, mode="disabled", games=3, seed=1)
    concise, detailed = probe.summarize(disabled), probe.summarize(disabled, verbose=True)
    assert "games with hypothetical resignations: 3/3" in concise
    assert "game_id, n_events" not in concise and "game_id, n_events" in detailed

    disabled["args"]["mode"] = "bogus"
    with pytest.raises(ValueError, match="unknown probe mode"):
        probe.summarize(disabled)


@pytest.mark.parametrize(
    "argv",
    [
        ["--mode", "enabled"],  # Missing checkpoint.
        ["--checkpoint", "net.pt"],  # Missing mode.
        ["--analyze", "a.json", "--mode", "enabled"],  # Analysis-only and new-run options are incompatible.
        ["--checkpoint", "net.pt", "--mode", "enabled", "--games", "0"],  # Zero games would give an empty measurement.
    ],
)
def test_cli_rejects_inconsistent_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        probe.main(argv)
    assert raised.value.code == 2
