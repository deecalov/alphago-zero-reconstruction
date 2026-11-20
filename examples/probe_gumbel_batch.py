"""Measure how leaf batch size changes Sequential Halving decisions on fixed weights, positions and noise."""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import alphago_zero.mcts as mcts_module
from alphago_zero.cli_types import ensure_directory, finite_float, non_negative_int, positive_int
from alphago_zero.config import GameConfig, SearchConfig
from alphago_zero.game import GoState
from alphago_zero.gumbel import GumbelMCTS
from alphago_zero.mcts import SearchOutput
from alphago_zero.network import PolicyValueModel
from examples.eval_match import load_checkpoint

REFERENCE_BATCH = 1  # Exact sequential reference.


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PairOutcome:
    """Result for one position and noise seed under batched and sequential search."""

    simulations: int
    batch: int
    position: int
    same_move: bool
    tv_distance: float
    root_q_gap: float


def _identity_index(rng: np.random.Generator) -> int:
    """Return the identity symmetry without advancing the RNG."""
    return 0


@contextlib.contextmanager
def identity_symmetry() -> Iterator[None]:
    """Temporarily disable random board symmetries for a paired comparison."""
    with mock.patch.object(mcts_module, "random_dihedral_index", _identity_index):
        yield


def sample_positions(
    game: GameConfig, count: int, *, max_opening_moves: int, seed: int
) -> list[GoState]:
    """Generate up to count nonterminal positions from random openings without passes."""
    rng = np.random.default_rng(seed)
    positions: list[GoState] = []
    for _ in range(count):
        state = GoState.new_game(
            board_size=game.board_size,
            komi=game.komi,
            max_moves=game.max_moves,
            superko_rule=game.superko_rule,
        )
        for _ in range(int(rng.integers(0, max_opening_moves + 1))):
            moves = [action for action in state.legal_actions() if action != state.pass_action]
            if not moves:  # Terminal positions have no moves to search.
                break
            state = state.play(int(rng.choice(moves)))
        if not state.is_terminal():
            positions.append(state)
    return positions


def _run_search(
    model: PolicyValueModel,
    state: GoState,
    *,
    simulations: int,
    batch: int,
    max_considered: int,
    virtual_loss: float,
    rng: np.random.Generator,
) -> SearchOutput:
    config = SearchConfig(
        num_simulations=simulations,
        eval_batch_size=batch,
        virtual_loss=virtual_loss,
        search_algo="gumbel",
        gumbel_max_considered=max_considered,
    )
    return GumbelMCTS(model, config, rng).run(state, temperature=1.0, add_root_noise=True)


def _improved_policy(output: SearchOutput) -> np.ndarray:
    """Return the improved root policy as float64. Gumbel search must provide it."""
    if output.improved_policy is None:
        raise RuntimeError("Gumbel search did not return the improved policy π'")
    return output.improved_policy.astype(np.float64)


def measure_pairs(
    model: PolicyValueModel,
    positions: Sequence[GoState],
    *,
    simulations: int,
    batches: Sequence[int],
    noise_seeds: int,
    max_considered: int,
    virtual_loss: float,
    seed: int,
) -> list[PairOutcome]:
    """Compare each leaf batch with batch one over the same position and noise-seed pairs."""
    outcomes: list[PairOutcome] = []
    with identity_symmetry():
        for index, state in enumerate(positions):
            search = functools.partial(
                _run_search, model, state,
                simulations=simulations, max_considered=max_considered, virtual_loss=virtual_loss,
            )  # fmt: skip
            for noise_seed in range(noise_seeds):
                # Reset the generator for each search so the candidate receives the reference noise.
                reference = search(batch=REFERENCE_BATCH, rng=np.random.default_rng([seed, index, noise_seed]))
                reference_move = int(np.argmax(reference.policy))
                reference_pi = _improved_policy(reference)
                for batch in batches:
                    candidate = search(batch=batch, rng=np.random.default_rng([seed, index, noise_seed]))
                    candidate_pi = _improved_policy(candidate)
                    outcomes.append(
                        PairOutcome(
                            simulations=simulations,
                            batch=batch,
                            position=index,
                            same_move=int(np.argmax(candidate.policy)) == reference_move,
                            tv_distance=0.5 * float(np.abs(candidate_pi - reference_pi).sum()),
                            root_q_gap=abs(float(candidate.root_q) - float(reference.root_q)),
                        )
                    )
    return outcomes


def summarize_outcomes(outcomes: Sequence[PairOutcome]) -> list[dict[str, Any]]:
    """Aggregate by simulation budget and batch size in input order. Compute uncertainty across positions."""
    groups: dict[tuple[int, int], list[PairOutcome]] = {}
    for outcome in outcomes:
        groups.setdefault((outcome.simulations, outcome.batch), []).append(outcome)
    rows: list[dict[str, Any]] = []
    for (simulations, batch), items in groups.items():
        per_position: dict[int, list[float]] = {}
        for item in items:
            per_position.setdefault(item.position, []).append(float(item.same_move))
        position_rates = [statistics.fmean(values) for values in per_position.values()]
        rows.append(
            {
                "simulations": simulations,
                "batch": batch,
                "pairs": len(items),
                "positions": len(per_position),
                "same_move": statistics.fmean(float(item.same_move) for item in items),
                "same_move_se": (
                    statistics.stdev(position_rates) / math.sqrt(len(position_rates))
                    if len(position_rates) >= 2
                    else None
                ),
                "tv_distance": statistics.fmean(item.tv_distance for item in items),
                "root_q_gap": statistics.fmean(item.root_q_gap for item in items),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_report(data: dict[str, Any]) -> str:
    """Format the same measurements written to JSON as a text report."""
    board = data["board_size"]
    lines = [
        "== Sequential Halving distortion by leaf batch size (Gumbel) ==",
        f"checkpoint: {data['checkpoint']} (sha256 {data['checkpoint_sha256_12']}), "
        f"board {board}x{board}, komi {data['komi']}, device {data['device']}",
        f"positions: {data['positions']} (random openings up to {data['max_opening_moves']} moves), "
        f"noise samples per position: {data['noise_seeds']}, m = {data['max_considered']}, "
        f"virtual loss {data['virtual_loss']}",
        f"reference: batch {REFERENCE_BATCH} (exact sequential search). Board symmetries disabled. "
        f"elapsed {data['elapsed_sec']:.0f} s",
        "",
        f"{'simulations':>9} {'batch':>5} {'pairs':>5} {'same move':>11} {'± SE':>8} {'TV(π′)':>8} {'|Δroot_q|':>10}",
    ]
    for row in data["rows"]:
        error = row["same_move_se"]
        error_text = "—" if error is None else f"{100.0 * error:.1f} %"
        lines.append(
            f"{row['simulations']:>9} {row['batch']:>5} {row['pairs']:>5} "
            f"{100.0 * row['same_move']:>9.1f} % {error_text:>8} "
            f"{row['tv_distance']:>8.3f} {row['root_q_gap']:>10.4f}"
        )
    lines += [
        "",
        "same move is the fraction of position/noise pairs agreeing with batch one. SE is computed across positions.",
        "This measures scheduling distortion. Use eval_match to measure playing strength.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _unique(values: Sequence[int]) -> list[int]:
    """Deduplicate values while preserving their first-seen order."""
    return list(dict.fromkeys(values))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the effect of leaf batch size on Gumbel Sequential Halving "
        "by comparison with sequential search on identical weights, positions and noise",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="network checkpoint (.pt with config and model_state_dict)"
    )
    parser.add_argument("--positions", type=positive_int, default=60, help="number of positions (default 60)")
    parser.add_argument("--noise-seeds", type=positive_int, default=2, help="noise samples per position (default 2)")
    parser.add_argument(
        "--simulations", type=positive_int, nargs="+", default=[32, 64],
        help="search simulation budgets (default 32 64)",
    )  # fmt: skip
    parser.add_argument(
        "--batches", type=positive_int, nargs="+", default=[2, 4, 8, 16],
        help="leaf batch sizes >= 2 (default 2 4 8 16), compared with batch one",
    )  # fmt: skip
    parser.add_argument(
        "--max-considered", type=positive_int, default=16, help="Sequential Halving candidate count (default 16)"
    )
    parser.add_argument(
        "--virtual-loss", type=finite_float, default=1.0, help="virtual loss within a batch (default 1.0)"
    )
    parser.add_argument(
        "--max-opening-moves", type=non_negative_int, default=20,
        help="sample positions after 0..N random non-pass moves (default 20)",
    )  # fmt: skip
    parser.add_argument("--seed", type=int, default=1, help="position and noise seed (default 1)")
    parser.add_argument("--device", default="cpu", help="inference device: cpu or cuda")
    parser.add_argument(
        "--threads", type=positive_int, default=None, help="torch threads (defaults to the current setting)"
    )
    parser.add_argument("--json-out", type=Path, default=None, help="write measurements to this JSON file")
    args = parser.parse_args(argv)
    if any(batch < 2 for batch in args.batches):
        parser.error(f"--batches: batch {REFERENCE_BATCH} is the reference. Supply values >= 2")
    if args.max_considered < 2:
        parser.error("--max-considered must be >= 2 for Sequential Halving")
    if args.virtual_loss < 0.0:
        parser.error("--virtual-loss must be nonnegative")
    args.simulations = _unique(args.simulations)
    args.batches = _unique(args.batches)
    if args.json_out is not None:
        ensure_directory(parser, args.json_out.parent, "--json-out")
    return args


def _configure_stdout() -> None:
    """Use UTF-8 for reports across console encodings."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def _sha256_12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12].upper()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_stdout()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    model, _search, game = load_checkpoint(args.checkpoint, torch.device(args.device))
    started = time.perf_counter()
    positions = sample_positions(
        game, args.positions, max_opening_moves=args.max_opening_moves, seed=args.seed
    )
    outcomes: list[PairOutcome] = []
    for simulations in args.simulations:
        outcomes += measure_pairs(
            model, positions, simulations=simulations, batches=args.batches,
            noise_seeds=args.noise_seeds, max_considered=args.max_considered,
            virtual_loss=args.virtual_loss, seed=args.seed,
        )  # fmt: skip
        print(f"completed: {simulations} simulations ({time.perf_counter() - started:.0f} s)", file=sys.stderr)
    data: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_12": _sha256_12(args.checkpoint),
        "board_size": game.board_size,
        "komi": game.komi,
        "device": str(args.device),
        "positions": len(positions),
        "noise_seeds": args.noise_seeds,
        "max_opening_moves": args.max_opening_moves,
        "max_considered": args.max_considered,
        "virtual_loss": args.virtual_loss,
        "seed": args.seed,
        "elapsed_sec": time.perf_counter() - started,
        "rows": summarize_outcomes(outcomes),
    }
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(format_report(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
