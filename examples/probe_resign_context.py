from __future__ import annotations

"""Measure the game and search context of actual or hypothetical self-play resignations."""

import argparse
import dataclasses
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphago_zero.config import (
    AlphaGoZeroConfig,
    GameConfig,
    NetworkConfig,
    RuntimeConfig,
    SearchConfig,
    TrainingConfig,
)
from alphago_zero.mcts import EdgeStats, MCTSNode, resignation_child_q
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.parallel_selfplay import _GameSlot, _ParallelSelfPlayDriver
from alphago_zero.selfplay import ResignationController, game_end_reason
from examples.eval_match import wilson_interval

# Store these launch arguments as strings in the JSON report.
_RECORDED_ARGS = ("checkpoint", "mode", "games", "sims", "seed", "parallel", "threads", "device")


# ---------------------------------------------------------------------------
# Driver instrumentation
# ---------------------------------------------------------------------------
def _is_terminal_edge(edge: EdgeStats) -> bool:
    return edge.child is not None and edge.child.state.is_terminal()


def classify(root: MCTSNode, slot: _GameSlot, threshold: float, min_visits: int) -> dict[str, Any]:
    """Describe the position and what the selected move would do if play continued."""
    state = slot.state
    visited = [(action, edge) for action, edge in root.edges.items() if edge.visit_count > 0]
    if not visited:
        raise ValueError("the root has no visited edges. Run search first")
    total_visits = sum(edge.visit_count for _, edge in visited)
    root_q = sum(edge.value_sum for edge in root.edges.values()) / total_visits
    _, best_edge = max(visited, key=lambda item: item[1].q_value)
    most_action, most_edge = max(visited, key=lambda item: item[1].visit_count)
    return {
        "move_number": state.move_number,
        "to_play": state.to_play,
        "n_legal": len(root.edges),
        "root_q": float(root_q),
        "best_child_q": float(resignation_child_q(root, min_visits)),
        "threshold": float(threshold),
        "move_is_full": bool(slot.move_is_full),
        "most_visited_is_pass": most_action == state.pass_action,
        "most_visited_terminal": _is_terminal_edge(most_edge),
        "best_q_terminal": _is_terminal_edge(best_edge),
        "all_visited_terminal": all(_is_terminal_edge(edge) for _, edge in visited),
        "consecutive_passes": state.consecutive_passes,
        "most_visited_q": float(most_edge.q_value),
    }


class _RecordingSet(set[int]):
    """Record each hypothetical resignation event when the driver updates the player set."""

    def __init__(self, driver: SpyDriver, slot: _GameSlot) -> None:
        super().__init__()
        self._driver = driver
        self._slot = slot

    def add(self, item: int) -> None:
        self._driver.log(self._slot, "hypothetical")
        super().add(item)


class SpyDriver(_ParallelSelfPlayDriver):
    """Observe the standard driver and log events without changing move selection."""

    def __init__(self, *args: Any, events: list[dict[str, Any]], **kwargs: Any) -> None:
        # Initialize observer attributes before the base constructor starts its first game.
        self.events = events
        # Retain slots to prevent identity reuse across completed and active games.
        self._registry: list[_GameSlot] = []
        self._game_ids: dict[int, int] = {}
        super().__init__(*args, **kwargs)

    def log(self, slot: _GameSlot, kind: str, **extra: Any) -> None:
        """Record the current slot decision after its search has finished."""
        record = classify(
            slot.root,
            slot,
            self.resignation_controller.threshold,
            self.config.search.resign_min_visits,
        )
        record["kind"] = kind
        record["game_id"] = self._game_ids[id(slot)]
        record["resignation_disabled"] = slot.resignation_disabled
        record.update(extra)
        self.events.append(record)

    def _start_new_game(self) -> _GameSlot | None:
        slot = super()._start_new_game()
        if slot is not None:
            self._game_ids[id(slot)] = len(self._registry)
            self._registry.append(slot)
            slot.hypothetical_resign_players = _RecordingSet(self, slot)
        return slot

    def _finalize_game(self, slot: _GameSlot, winner: int | None, final_state: Any = None) -> None:
        if slot.resigned:
            self.log(slot, "resign", winner=winner)
        elif final_state is not None:
            # Classify endings with the self-play helper so reports use the same categories as training.
            self.log(slot, "natural_end", winner=winner, ended_by=game_end_reason(final_state, False))
        super()._finalize_game(slot, winner, final_state)


# ---------------------------------------------------------------------------
# Loading and execution
# ---------------------------------------------------------------------------
def _section(cls: type[Any], raw: dict[str, Any]) -> Any:
    """Load recognized keys into the configuration dataclass."""
    names = {field.name for field in dataclasses.fields(cls)}
    return cls(**{key: value for key, value in raw.items() if key in names})


def load_probe_setup(
    checkpoint: Path, *, device: str, sims: int, mode: str, parallel: int
) -> tuple[AlphaGoZeroNet, AlphaGoZeroConfig]:
    """Rebuild the model and probe settings from the checkpoint and explicit overrides."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    raw = payload["config"]
    config = AlphaGoZeroConfig(
        game=_section(GameConfig, raw["game"]),
        network=_section(NetworkConfig, raw["network"]),
        search=_section(SearchConfig, raw["search"]),
        training=_section(TrainingConfig, raw["training"]),
        runtime=_section(RuntimeConfig, raw["runtime"]),
    )
    config.search = dataclasses.replace(
        config.search,
        num_simulations=sims,
        disable_resign_prob=1.0 if mode == "disabled" else 0.0,
    )
    config.training = dataclasses.replace(
        config.training,
        parallel_games=parallel,
        league_prob=0.0,
        exclude_league_from_resign_calibration=False,
    )
    model = AlphaGoZeroNet(config.network).to(torch.device(device))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config


def run_probe(model: AlphaGoZeroNet, config: AlphaGoZeroConfig, *, games: int, seed: int) -> dict[str, Any]:
    """Play games with the instrumented driver and return raw measurements."""
    controller = ResignationController(
        threshold=config.search.resign_threshold,
        target_false_positive_rate=config.search.target_false_positive_rate,
        adjustment=config.search.resignation_adjustment,
        window=config.search.resignation_window,
    )
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    driver = SpyDriver(
        model=model,
        config=config,
        rng=np.random.default_rng(seed),
        resignation_controller=controller,
        num_games=games,
        events=events,
    )
    results = driver.run()
    return {
        "elapsed_sec": time.perf_counter() - started,
        "final_threshold": controller.threshold,
        "controller_history_len": len(controller.history),
        "controller_fp_rate": controller.false_positive_rate,
        "games": [
            {
                "num_moves": r.num_moves,
                "resigned": r.resigned,
                "ended_by": r.ended_by,
                "winner": r.winner,
                "resignation_disabled": r.resignation_disabled,
                "false_positive_resignation": r.false_positive_resignation,
                "score_margin": r.score_margin,
            }
            for r in results
        ],
        "events": events,
    }


def _sha256_12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12].upper()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _share(count: int, total: int) -> str:
    """Format a fraction and its 95% Wilson interval."""
    if total <= 0:
        return f"{count}/{total}"
    point, lower, upper = wilson_interval(count, total)
    return f"{count}/{total} = {point:.1%} [{lower:.1%}; {upper:.1%}]"


def _quantiles(values: list[float]) -> str:
    if not values:
        return "-"
    ordered = sorted(values)
    size = len(ordered)

    def pick(q: float) -> float:
        return ordered[min(size - 1, int(q * size))]

    return (
        f"min={ordered[0]:.0f} q25={pick(0.25):.0f} med={pick(0.5):.0f} "
        f"q75={pick(0.75):.0f} max={ordered[-1]:.0f}"
    )


def _is_exact(event: dict[str, Any]) -> bool:
    """Check whether the selected or best-Q move reaches a terminal state with known outcome."""
    return bool(event["most_visited_terminal"] or event["best_q_terminal"])


def _header(data: dict[str, Any], title: str) -> str:
    args = data.get("args", {})
    return (
        f"== Mode {title}: {len(data['games'])} games in {data['elapsed_sec']:.0f} s, "
        f"checkpoint {data.get('checkpoint_sha256_12', '?')}, sims={args.get('sims', '?')}, "
        f"seed={args.get('seed', '?')}, final threshold {data['final_threshold']:.3f}"
    )


_DETAIL_KEYS = (
    "game_id", "move_number", "n_legal", "root_q", "best_child_q", "most_visited_is_pass",
    "most_visited_terminal", "best_q_terminal", "consecutive_passes",
)


def _resign_lines(resign: list[dict[str, Any]], verbose: bool) -> list[str]:
    total = len(resign)

    def count(key: str) -> int:
        return sum(bool(event[key]) for event in resign)

    lines = [
        "-- resignation decisions:",
        f"   selected move would end the game (most_visited_terminal): {_share(count('most_visited_terminal'), total)}",
        f"   best-Q edge leads to a terminal state (best_q_terminal): {_share(count('best_q_terminal'), total)}",
        f"   all visited edges lead to terminal states: {_share(count('all_visited_terminal'), total)}",
        f"   exact terminal (most_visited OR best_q): {_share(sum(_is_exact(e) for e in resign), total)}",
        f"   only one legal move (n_legal == 1): {_share(sum(e['n_legal'] == 1 for e in resign), total)}",
        f"   most visited move is pass: {_share(count('most_visited_is_pass'), total)}",
        f"   resignations immediately after an opponent pass: "
        f"{_share(sum(e['consecutive_passes'] >= 1 for e in resign), total)}",
        f"   consecutive_passes: {dict(Counter(e['consecutive_passes'] for e in resign))}",
        f"   move number at resignation: {_quantiles([e['move_number'] for e in resign])}",
        f"   mean root_q at resignation: {statistics.mean(e['root_q'] for e in resign):.3f}",
    ]
    if verbose:
        for event in resign:
            if _is_exact(event) or event["consecutive_passes"] >= 1:
                detail = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in event.items() if k in _DETAIL_KEYS}
                lines.append(f"   * {detail}")
    return lines


def _natural_end_lines(natural: list[dict[str, Any]]) -> list[str]:
    qualified = [
        e for e in natural
        if e["move_is_full"] and e["root_q"] < e["threshold"] and e["best_child_q"] < e["threshold"]
    ]
    return [
        "-- natural endings (final decision):",
        f"   mean root_q on final move: {statistics.mean(e['root_q'] for e in natural):.3f}, "
        f"minimum {min(e['root_q'] for e in natural):.3f}",
        f"   selected move ended the game (most_visited_terminal): "
        f"{_share(sum(bool(e['most_visited_terminal']) for e in natural), len(natural))}",
        f"   final decisions meeting the resignation criterion: {len(qualified)} (expected zero in enabled mode)",
    ]


def _summarize_enabled(data: dict[str, Any], verbose: bool) -> list[str]:
    games, events = data["games"], data["events"]
    resign = [e for e in events if e["kind"] == "resign"]
    natural = [e for e in events if e["kind"] == "natural_end"]
    ended = Counter(e["ended_by"] for e in natural)
    lines = [
        _header(data, "enabled (resignation allowed)"),
        f"resigned: {_share(len(resign), len(games))}; played to completion: {len(natural)} "
        f"(double_pass={ended['double_pass']}, max_moves={ended['max_moves']})",
        f"length of resigned games: {_quantiles([g['num_moves'] for g in games if g['resigned']])}",
        f"length of completed games: {_quantiles([g['num_moves'] for g in games if not g['resigned']])}",
    ]
    if resign:
        lines += _resign_lines(resign, verbose)
    if natural:
        lines += _natural_end_lines(natural)
    return lines


def _hypothetical_lines(
    hypothetical: list[dict[str, Any]], natural: dict[int, dict[str, Any]], verbose: bool
) -> list[str]:
    by_game: dict[int, list[dict[str, Any]]] = {}
    for event in hypothetical:
        by_game.setdefault(event["game_id"], []).append(event)
    rows = []
    for game_id, events in sorted(by_game.items()):
        players = {e["to_play"] for e in events}
        winner = natural[game_id]["winner"] if game_id in natural else None
        rows.append((game_id, events, sorted(players), winner, winner in players, all(_is_exact(e) for e in events)))
    informative = [row for row in rows if not row[5]]
    lines = [
        f"   exact terminal events (most_visited OR best_q): "
        f"{_share(sum(_is_exact(e) for e in hypothetical), len(hypothetical))}",
        f"   most_visited_terminal events: "
        f"{_share(sum(bool(e['most_visited_terminal']) for e in hypothetical), len(hypothetical))}",
        f"   games with false positives (the player who would resign went on to win): "
        f"{_share(sum(row[4] for row in rows), len(rows))}",
        f"   games where all hypothetical resignations were exact terminal decisions: {_share(sum(row[5] for row in rows), len(rows))}",
    ]
    if informative:
        lines.append(
            f"   false positives among informative games only: "
            f"{_share(sum(row[4] for row in informative), len(informative))}"
        )
    if verbose:
        lines.append("   game_id, n_events, players, winner, FP, all_exact, first moves")
        lines += [
            f"   {(gid, len(evs), players, winner, fp, all_exact, [e['move_number'] for e in evs][:6])}"
            for gid, evs, players, winner, fp, all_exact in rows
        ]
    return lines


def _summarize_disabled(data: dict[str, Any], verbose: bool) -> list[str]:
    games, events = data["games"], data["events"]
    hypothetical = [e for e in events if e["kind"] == "hypothetical"]
    natural = {e["game_id"]: e for e in events if e["kind"] == "natural_end"}
    ended = Counter(e["ended_by"] for e in natural.values())
    games_with_decision = len({e["game_id"] for e in hypothetical})
    lines = [
        _header(data, "disabled (hypothetical resignations)")
        + f", false-positive window: n={data['controller_history_len']}, controller false-positive rate={data['controller_fp_rate']:.3f}",
        f"played to completion: {len(natural)} (double_pass={ended['double_pass']}, max_moves={ended['max_moves']})",
        f"game lengths: {_quantiles([g['num_moves'] for g in games])}",
        f"games with hypothetical resignations: {games_with_decision}/{len(games)}; events: {len(hypothetical)}",
    ]
    if hypothetical:
        lines += _hypothetical_lines(hypothetical, natural, verbose)
    return lines


def summarize(data: dict[str, Any], *, verbose: bool = False) -> str:
    """Format raw probe data using the mode recorded with the launch arguments."""
    mode = data["args"]["mode"]
    if mode == "enabled":
        return "\n".join(_summarize_enabled(data, verbose))
    if mode == "disabled":
        return "\n".join(_summarize_disabled(data, verbose))
    raise ValueError(f"unknown probe mode: {mode!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1")
    return value


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect resignation decisions in self-play with a trained network",
    )
    parser.add_argument("--checkpoint", type=Path, help="network checkpoint (.pt with config and model_state_dict)")
    parser.add_argument(
        "--mode", choices=("enabled", "disabled"),
        help="enabled allows resignation. Disabled records hypothetical decisions",
    )
    parser.add_argument("--games", type=_positive_int, default=48, help="number of games (default 48)")
    parser.add_argument("--sims", type=_positive_int, default=96, help="simulations per move (default 96)")
    parser.add_argument("--seed", type=int, default=1, help="game generation seed")
    parser.add_argument("--parallel", type=_positive_int, default=16, help="concurrent games (default 16)")
    parser.add_argument("--threads", type=_positive_int, default=None, help="torch threads (defaults to the current setting)")
    parser.add_argument("--device", default="cpu", help="inference device: cpu or cuda")
    parser.add_argument("--out", type=Path, default=None, help="write raw measurements to JSON")
    parser.add_argument("--report-out", type=Path, default=None, help="write the UTF-8 text report")
    parser.add_argument(
        "--analyze", type=Path, nargs="+", metavar="JSON",
        help="analyze existing JSON reports without playing new games",
    )
    parser.add_argument("--verbose", action="store_true", help="include individual game and decision details")
    args = parser.parse_args(argv)
    if args.analyze:
        if args.checkpoint is not None or args.mode is not None:
            parser.error("--analyze reads existing JSON and cannot be combined with --checkpoint/--mode")
    elif args.checkpoint is None or args.mode is None:
        parser.error("supply --checkpoint and --mode, or --analyze for existing JSON")
    return args


def _configure_stdout() -> None:
    """Use UTF-8 for reports across console encodings."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def _measure(args: argparse.Namespace) -> str:
    """Run the probe, write optional JSON and return a text report."""
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    model, config = load_probe_setup(
        args.checkpoint, device=args.device, sims=args.sims, mode=args.mode, parallel=args.parallel
    )
    data = run_probe(model, config, games=args.games, seed=args.seed)
    data["args"] = {key: str(getattr(args, key)) for key in _RECORDED_ARGS}
    data["checkpoint_sha256_12"] = _sha256_12(args.checkpoint)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return summarize(data, verbose=args.verbose)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_stdout()
    if args.analyze:
        reports = [
            summarize(json.loads(path.read_text(encoding="utf-8")), verbose=args.verbose)
            for path in args.analyze
        ]
    else:
        reports = [_measure(args)]
    text = "\n\n".join(reports)
    print(text)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
