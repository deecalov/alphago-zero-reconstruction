from __future__ import annotations

"""Compare saved networks head to head with alternating colors, confidence intervals and optional sequential
stopping.
"""

import argparse
import dataclasses
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from alphago_zero.cli_types import (
    ensure_directory,
    finite_float,
    fpu_value,
    positive_float,
    positive_int,
)
from alphago_zero.config import GameConfig, NetworkConfig, SearchConfig
from alphago_zero.ensemble import EnsembleNet, require_same_rules
from alphago_zero.game import BLACK, WHITE, GoState
from alphago_zero.mcts import AlphaGoZeroMCTS, MCTSNode
from alphago_zero.network import AlphaGoZeroNet, PolicyValueModel
from alphago_zero.sprt import SprtConfig, SprtDecision, expected_games, sprt_decision, sprt_llr


def _filter_fields(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    """Keep fields supported by the current dataclass when loading checkpoint settings."""
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in raw.items() if k in names}


def load_checkpoint(
    path: str | Path, device: torch.device, sims_override: int | None = None
) -> tuple[AlphaGoZeroNet, SearchConfig, GameConfig]:
    """Restore the model, search configuration and game rules from a pipeline checkpoint."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    raw = payload["config"]
    network = AlphaGoZeroNet(NetworkConfig(**_filter_fields(NetworkConfig, raw["network"])))
    network = network.to(device)
    network.load_state_dict(payload["model_state_dict"])
    network.eval()
    search = SearchConfig(**_filter_fields(SearchConfig, raw["search"]))
    if sims_override is not None:
        search = dataclasses.replace(search, num_simulations=sims_override)
    game = GameConfig(**_filter_fields(GameConfig, raw["game"]))
    return network, search, game


def load_model_group(
    paths: Sequence[str | Path], device: torch.device, sims_override: int | None = None
) -> tuple[PolicyValueModel, SearchConfig, GameConfig]:
    """Load one network or combine several compatible checkpoints as an ensemble."""
    if not paths:
        raise ValueError("at least one checkpoint is required")
    loaded = [load_checkpoint(path, device, sims_override) for path in paths]
    _network, search, game = loaded[0]
    require_same_rules([member_game for _n, _s, member_game in loaded], [str(path) for path in paths])
    networks = [network for network, _search, _game in loaded]
    model: PolicyValueModel = networks[0] if len(networks) == 1 else EnsembleNet(networks)
    return model, search, game


def play_game(
    *,
    black_model: PolicyValueModel,
    black_search: SearchConfig,
    white_model: PolicyValueModel,
    white_search: SearchConfig,
    game: GameConfig,
    rng: np.random.Generator,
    opening_moves: int,
    black_tree_reuse: bool = False,
    white_tree_reuse: bool = False,
    black_time_budget: float | None = None,
    white_time_budget: float | None = None,
    stats: dict[str, float] | None = None,
) -> int | None:
    """Play a game with independent search settings for each side. Return the winner or None for a draw."""
    state = GoState.new_game(
        board_size=game.board_size,
        komi=game.komi,
        max_moves=game.max_moves,
        superko_rule=game.superko_rule,
    )
    black_mcts = AlphaGoZeroMCTS(black_model, black_search, rng)
    white_mcts = AlphaGoZeroMCTS(white_model, white_search, rng)
    black_root: MCTSNode | None = None
    white_root: MCTSNode | None = None

    while not state.is_terminal():
        mover_is_black = state.to_play == BLACK
        if mover_is_black:
            search, temperature = black_mcts, black_search.deterministic_temperature
            root = black_root if black_tree_reuse else None
            budget = black_time_budget
        else:
            search, temperature = white_mcts, white_search.deterministic_temperature
            root = white_root if white_tree_reuse else None
            budget = white_time_budget

        output = search.run(
            root_state=state,
            temperature=temperature,
            add_root_noise=False,
            root=root,
            time_budget_seconds=budget,
            early_exit_unreachable=budget is not None,
        )
        if stats is not None:
            side = "black" if mover_is_black else "white"
            stats[f"{side}_sims"] = stats.get(f"{side}_sims", 0.0) + float(
                output.visit_counts.sum()
            )
            stats[f"{side}_moves"] = stats.get(f"{side}_moves", 0.0) + 1.0

        if state.move_number < opening_moves:
            distribution = output.visit_distribution
            total = float(distribution.sum())
            if total > 0.0:
                action = int(rng.choice(len(distribution), p=distribution / total))
            else:
                action = int(np.argmax(output.policy))
        else:
            action = int(np.argmax(output.policy))

        # Advance both retained trees along the move that was actually played.
        if black_tree_reuse:
            black_root = black_mcts.advance_root(
                output.root if mover_is_black else black_root, action
            )
        if white_tree_reuse:
            white_root = white_mcts.advance_root(
                output.root if not mover_is_black else white_root, action
            )
        state = state.play(action)

    return state.winner_color()


def _describe(paths: Sequence[str]) -> str:
    """Format a checkpoint or ensemble label for match output."""
    if len(paths) == 1:
        return str(paths[0])
    return f"{' + '.join(paths)} (ensemble of {len(paths)})"


# Use a sentinel for an omitted FPU override because None is itself a valid setting.
_UNSET: Final[object] = object()


def _apply_search_overrides(search: SearchConfig, c_puct: float | None, fpu: object) -> SearchConfig:
    """Apply explicit per-side PUCT and FPU overrides to the saved search settings."""
    changes: dict[str, Any] = {}
    if c_puct is not None:
        changes["c_puct"] = c_puct
    if fpu is not _UNSET:
        changes["fpu_reduction"] = fpu
    return dataclasses.replace(search, **changes)


def _sprt_from_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> SprtConfig | None:
    """Build the sequential test configuration and reject unused test arguments."""
    given = {
        "--sprt-wr0": args.sprt_wr0,
        "--sprt-wr1": args.sprt_wr1,
        "--sprt-alpha": args.sprt_alpha,
        "--sprt-beta": args.sprt_beta,
    }
    if not args.sprt:
        extra = [flag for flag, value in given.items() if value is not None]
        if extra:
            parser.error(f"SPRT options ({', '.join(extra)}) require --sprt")
        return None
    defaults = SprtConfig()
    try:
        return SprtConfig(
            wr0=defaults.wr0 if args.sprt_wr0 is None else args.sprt_wr0,
            wr1=defaults.wr1 if args.sprt_wr1 is None else args.sprt_wr1,
            alpha=defaults.alpha if args.sprt_alpha is None else args.sprt_alpha,
            beta=defaults.beta if args.sprt_beta is None else args.sprt_beta,
        )
    except ValueError as error:
        parser.error(str(error))


def _prepare_output_dir(parser: argparse.ArgumentParser, json_out: str | None) -> None:
    """Prepare the JSON output directory before starting the match."""
    if not json_out:
        return
    ensure_directory(parser, Path(json_out).parent, "--json-out")


def _print_sprt_header(sprt: SprtConfig, max_games: int) -> None:
    """Print hypotheses and decision boundaries. Report when a small game cap makes an inconclusive result
    likely.
    """
    gap0 = elo_from_win_rate(sprt.wr0, 0.0)
    gap1 = elo_from_win_rate(sprt.wr1, 0.0)
    print(
        f"SPRT: H0 wr={sprt.wr0:.2f} ({gap0:+.0f} Elo) versus H1 wr={sprt.wr1:.2f} ({gap1:+.0f} Elo), "
        f"alpha={sprt.alpha:g}, beta={sprt.beta:g}, LLR bounds "
        f"[{sprt.lower_bound:+.3f}, {sprt.upper_bound:+.3f}]. Check after each pair of games",
        flush=True,
    )
    under_h0, under_h1 = expected_games(sprt)
    if max_games < max(under_h0, under_h1):
        print(
            f"SPRT: expected test length approximately {under_h0:.0f} games at wr={sprt.wr0:.2f} and "
            f"≈ {under_h1:.0f} at wr={sprt.wr1:.2f} (Wald approximation). Maximum --games="
            f"{max_games} is lower, so a small advantage may yield an inconclusive result",
            flush=True,
        )


def _sprt_verdict_text(decision: SprtDecision, sprt: SprtConfig, played: int) -> str:
    """Format the sequential test decision."""
    if decision is SprtDecision.ACCEPT_H1:
        return f"accepted H1: win rate at least {sprt.wr1:.2f} (after {played} games)"
    if decision is SprtDecision.ACCEPT_H0:
        return f"accepted H0: win rate at most {sprt.wr0:.2f} (after {played} games)"
    return f"no decision after {played} games (reached --games cap): inconclusive"


def _sprt_summary(
    sprt: SprtConfig, wins: int, losses: int, decision: SprtDecision, max_games: int
) -> dict[str, Any]:
    """Return hypotheses, thresholds, final likelihood ratio and decision for the JSON report."""
    return {
        "wr0": sprt.wr0,
        "wr1": sprt.wr1,
        "alpha": sprt.alpha,
        "beta": sprt.beta,
        "lower_bound": round(sprt.lower_bound, 4),
        "upper_bound": round(sprt.upper_bound, 4),
        "llr": round(sprt_llr(wins, losses, sprt), 4),
        "decision": "inconclusive" if decision is SprtDecision.CONTINUE else decision.value,
        "games_played": wins + losses,
        "max_games": max_games,
    }


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return the observed fraction and its 95% Wilson interval."""
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def elo_from_win_rate(p: float, opponent_rating: float) -> float:
    """Estimate Elo from the score against an opponent with a known rating."""
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return opponent_rating + 400.0 * math.log10(p / (1.0 - p))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-head match between AlphaGo Zero checkpoints",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        metavar="CKPT",
        help="model checkpoint (.pt) or several checkpoints for an ensemble averaging policy "
        "and value in one search. The first checkpoint supplies settings and rules",
    )
    parser.add_argument(
        "--opponent",
        nargs="+",
        required=True,
        metavar="CKPT",
        help="opponent checkpoint (.pt) or an ensemble as for --model",
    )
    parser.add_argument("--games", type=int, default=120, help="number of games with alternating colors")
    parser.add_argument(
        "--opponent-rating",
        type=finite_float,
        default=0.0,
        help="known finite opponent Elo for estimating the model rating",
    )
    parser.add_argument(
        "--opening-moves",
        type=int,
        default=6,
        help="number of opening moves sampled from search visits",
    )
    parser.add_argument(
        "--simulations",
        type=positive_int,
        default=None,
        help="set equal MCTS simulation budgets for both sides "
        "to compare weights at equal search depth. By default each uses its saved "
        "num_simulations. Set an individual budget with "
        "--model-simulations or --opponent-simulations",
    )
    parser.add_argument(
        "--model-simulations",
        type=positive_int,
        default=None,
        help="model simulation budget, overriding --simulations. "
        "Together with --opponent-simulations, compare compute budgets such as "
        "an ensemble at 160 simulations against one network with a larger budget",
    )
    parser.add_argument(
        "--opponent-simulations",
        type=positive_int,
        default=None,
        help="opponent simulation budget, overriding --simulations",
    )
    parser.add_argument(
        "--model-c-puct",
        type=positive_float,
        default=None,
        help="model c_puct override for comparing search settings on identical weights. "
        "Defaults to the checkpoint value",
    )
    parser.add_argument(
        "--opponent-c-puct",
        type=positive_float,
        default=None,
        help="opponent c_puct override (defaults to checkpoint)",
    )
    parser.add_argument(
        "--model-fpu",
        type=fpu_value,
        default=_UNSET,
        help="model first-play urgency: use a nonnegative reduction "
        "from parent value, or none for an unvisited Q of zero. "
        "Defaults to the checkpoint value",
    )
    parser.add_argument(
        "--opponent-fpu",
        type=fpu_value,
        default=_UNSET,
        help="opponent FPU reduction or none (defaults to checkpoint)",
    )
    parser.add_argument(
        "--model-tree-reuse",
        action="store_true",
        help="reuse the model's search tree between moves "
        "(disabled by default for weight comparisons)",
    )
    parser.add_argument(
        "--opponent-tree-reuse",
        action="store_true",
        help="reuse the opponent's search tree between moves (default off)",
    )
    parser.add_argument(
        "--model-time-budget",
        type=positive_float,
        default=None,
        help="model search time budget in seconds per move. "
        "Search stops at its deadline or earlier when the winner is certain. "
        "The simulation limit still applies. Disabled by default",
    )
    parser.add_argument(
        "--opponent-time-budget",
        type=positive_float,
        default=None,
        help="opponent time budget in seconds per move (default off)",
    )
    parser.add_argument(
        "--sprt",
        action="store_true",
        help="run a sequential probability ratio test of the model's win rate "
        "under H0 (--sprt-wr0) and H1 (--sprt-wr1). Use --games as the "
        "maximum and check after every pair of games",
    )
    parser.add_argument(
        "--sprt-wr0",
        type=positive_float,
        default=None,
        help="H0 model win rate (default 0.50, strictly between zero and one)",
    )
    parser.add_argument(
        "--sprt-wr1",
        type=positive_float,
        default=None,
        help="H1 model win rate (default 0.55, strictly between --sprt-wr0 and one)",
    )
    parser.add_argument(
        "--sprt-alpha",
        type=positive_float,
        default=None,
        help="probability of accepting H1 when H0 holds (default 0.05, strictly between zero and one)",
    )
    parser.add_argument(
        "--sprt-beta",
        type=positive_float,
        default=None,
        help="probability of accepting H0 when H1 holds (default 0.05, strictly between zero and one)",
    )
    parser.add_argument("--device", default="cuda", help="cuda|cpu")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--json-out", default=None, help="write match results to this JSON file")
    args = parser.parse_args(argv)
    sprt = _sprt_from_args(parser, args)
    _prepare_output_dir(parser, args.json_out)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    rng = np.random.default_rng(args.seed)

    # Per-side budgets override the shared budget. None preserves the checkpoint setting.
    model_sims = args.simulations if args.model_simulations is None else args.model_simulations
    opponent_sims = args.simulations if args.opponent_simulations is None else args.opponent_simulations
    model, model_search, model_game = load_model_group(args.model, device, model_sims)
    opponent, opponent_search, _opponent_game = load_model_group(args.opponent, device, opponent_sims)
    game = model_game  # Use the primary model's game rules.
    model_search = _apply_search_overrides(model_search, args.model_c_puct, args.model_fpu)
    opponent_search = _apply_search_overrides(opponent_search, args.opponent_c_puct, args.opponent_fpu)

    print(
        f"model={_describe(args.model)} sims={model_search.num_simulations} "
        f"c_puct={model_search.c_puct} fpu={model_search.fpu_reduction} "
        f"tree_reuse={args.model_tree_reuse} time_budget={args.model_time_budget}",
        flush=True,
    )
    print(
        f"opponent={_describe(args.opponent)} sims={opponent_search.num_simulations} "
        f"c_puct={opponent_search.c_puct} fpu={opponent_search.fpu_reduction} "
        f"tree_reuse={args.opponent_tree_reuse} time_budget={args.opponent_time_budget} "
        f"rating={args.opponent_rating}",
        flush=True,
    )
    print(
        f"board={game.board_size} komi={game.komi} superko={game.superko_rule} "
        f"games={args.games} opening_moves={args.opening_moves} device={device}",
        flush=True,
    )

    if sprt is not None:
        _print_sprt_header(sprt, args.games)

    wins = 0
    losses = 0
    draws = 0
    as_black = 0
    black_points = 0.0
    played = 0
    # Record actual simulations to support comparisons at equal work per move.
    model_sims_total = 0.0
    model_moves_total = 0.0
    opponent_sims_total = 0.0
    opponent_moves_total = 0.0
    decision = SprtDecision.CONTINUE
    started = time.time()

    for i in range(args.games):
        model_is_black = i % 2 == 0
        game_stats: dict[str, float] = {}
        if model_is_black:
            winner = play_game(
                black_model=model,
                black_search=model_search,
                white_model=opponent,
                white_search=opponent_search,
                game=game,
                rng=rng,
                opening_moves=args.opening_moves,
                black_tree_reuse=args.model_tree_reuse,
                white_tree_reuse=args.opponent_tree_reuse,
                black_time_budget=args.model_time_budget,
                white_time_budget=args.opponent_time_budget,
                stats=game_stats,
            )
            model_color = BLACK
            as_black += 1
        else:
            winner = play_game(
                black_model=opponent,
                black_search=opponent_search,
                white_model=model,
                white_search=model_search,
                game=game,
                rng=rng,
                opening_moves=args.opening_moves,
                black_tree_reuse=args.opponent_tree_reuse,
                white_tree_reuse=args.model_tree_reuse,
                black_time_budget=args.opponent_time_budget,
                white_time_budget=args.model_time_budget,
                stats=game_stats,
            )
            model_color = WHITE
        model_side = "black" if model_is_black else "white"
        opponent_side = "white" if model_is_black else "black"
        model_sims_total += game_stats.get(f"{model_side}_sims", 0.0)
        model_moves_total += game_stats.get(f"{model_side}_moves", 0.0)
        opponent_sims_total += game_stats.get(f"{opponent_side}_sims", 0.0)
        opponent_moves_total += game_stats.get(f"{opponent_side}_moves", 0.0)

        # A draw earns the model half a point.
        if winner is None:
            draws += 1
            points = 0.5
        elif winner == model_color:
            wins += 1
            points = 1.0
        else:
            losses += 1
            points = 0.0
        if model_is_black:
            black_points += points
        played = i + 1

        if played % 10 == 0:
            rate = (time.time() - started) / played
            llr_note = f" llr={sprt_llr(wins, losses, sprt):+.3f}" if sprt is not None else ""
            print(
                f"  [{played}/{args.games}] wins={wins} losses={losses} draws={draws} "
                f"wr={(wins + 0.5 * draws) / played:.3f}{llr_note} ({rate:.1f}s/game)",
                flush=True,
            )

        # Check SPRT boundaries after each alternating-color pair and after the last game.
        if sprt is not None and (played % 2 == 0 or played == args.games):
            decision = sprt_decision(wins, losses, sprt)
            if decision is not SprtDecision.CONTINUE:
                break

    if sprt is not None:
        print(f"SPRT: {_sprt_verdict_text(decision, sprt, played)}", flush=True)

    point, lower, upper = wilson_interval(wins + 0.5 * draws, played)
    result: dict[str, Any] = {
        # Keep the existing path field for a single model and list all ensemble members separately.
        "model": " + ".join(args.model),
        "opponent": " + ".join(args.opponent),
        "model_members": list(args.model),
        "opponent_members": list(args.opponent),
        "games": played,  # Actual games played may be below the sequential test cap.
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(point, 4),
        "win_rate_ci95": [round(lower, 4), round(upper, 4)],
        "as_black_games": as_black,
        "as_black_win_rate": round(black_points / max(1, as_black), 4),
        "as_white_win_rate": round(
            (wins + 0.5 * draws - black_points) / max(1, played - as_black), 4
        ),
        "opponent_rating": args.opponent_rating,
        "elo_estimate": round(elo_from_win_rate(point, args.opponent_rating), 1),
        "elo_ci95": [
            round(elo_from_win_rate(lower, args.opponent_rating), 1),
            round(elo_from_win_rate(upper, args.opponent_rating), 1),
        ],
        "model_sims": model_search.num_simulations,
        "opponent_sims": opponent_search.num_simulations,
        # Effective per-side search settings after applying overrides.
        "model_c_puct": model_search.c_puct,
        "model_fpu": model_search.fpu_reduction,
        # Per-side tree reuse settings.
        "model_tree_reuse": args.model_tree_reuse,
        "opponent_tree_reuse": args.opponent_tree_reuse,
        # Record time budgets and measured work per move.
        "model_time_budget": args.model_time_budget,
        "opponent_time_budget": args.opponent_time_budget,
        "model_avg_sims": round(model_sims_total / max(1.0, model_moves_total), 1),
        "opponent_avg_sims": round(
            opponent_sims_total / max(1.0, opponent_moves_total), 1
        ),
        "opponent_c_puct": opponent_search.c_puct,
        "opponent_fpu": opponent_search.fpu_reduction,
        "sims_forced": args.simulations,
        # Record the simulation override actually applied to each side.
        "model_sims_forced": model_sims,
        "opponent_sims_forced": opponent_sims,
        "seed": args.seed,
    }
    if sprt is not None:
        result["sprt"] = _sprt_summary(sprt, wins, losses, decision, args.games)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"written {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
