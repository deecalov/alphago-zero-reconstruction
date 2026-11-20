"""Run a trained network as a GTP engine for a Go GUI or match controller."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Final

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alphago_zero.cli_types import (  # noqa: E402
    fpu_value,
    non_negative_float,
    non_negative_int,
    positive_float,
    positive_int,
)
from alphago_zero.config import (  # noqa: E402
    AlphaGoZeroConfig,
    GameConfig,
    NetworkConfig,
    RuntimeConfig,
    SearchConfig,
    TrainingConfig,
)
from alphago_zero.ensemble import EnsembleNet, require_same_rules  # noqa: E402
from alphago_zero.gtp import GTPEngine  # noqa: E402
from alphago_zero.network import AlphaGoZeroNet, PolicyValueModel  # noqa: E402

# Distinguish an omitted FPU override from an explicit --fpu none.
_UNSET: Final[object] = object()

# Difficulty presets override checkpoint settings. Explicit simulation counts take precedence.
_LEVEL_SIMULATIONS: Final[dict[str, int]] = {"club": 64}


def _config_from_checkpoint(payload: dict) -> AlphaGoZeroConfig:
    """Restore experiment settings from checkpoint metadata."""
    raw = payload["config"]
    return AlphaGoZeroConfig(
        game=GameConfig(**raw["game"]),
        network=NetworkConfig(**raw["network"]),
        search=SearchConfig(**raw["search"]),
        training=TrainingConfig(**raw["training"]),
        runtime=RuntimeConfig(**raw["runtime"]),
    )


def _apply_search_overrides(
    search: SearchConfig,
    simulations: int | None,
    c_puct: float | None,
    fpu: object,
    resign_min_move: int | None = None,
    resign_score_margin: float | None = None,
) -> SearchConfig:
    """Override only explicitly supplied search parameters."""
    changes: dict[str, Any] = {}
    if simulations is not None:
        changes["num_simulations"] = simulations
    if c_puct is not None:
        changes["c_puct"] = c_puct
    if fpu is not _UNSET:
        changes["fpu_reduction"] = fpu
    if resign_min_move is not None:
        changes["resign_min_move"] = resign_min_move
    if resign_score_margin is not None:
        changes["resign_score_margin"] = resign_score_margin
    return replace(search, **changes)


def _checkpoint_resign_threshold(payload: dict) -> float | None:
    """Read the calibrated resignation threshold if the checkpoint contains one."""
    value = payload.get("resign_threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    threshold = float(value)
    return threshold if -1.0 <= threshold <= 0.0 else None


def _load_network(payload: dict, device: torch.device) -> AlphaGoZeroNet:
    """Rebuild each ensemble member with its saved architecture."""
    network = AlphaGoZeroNet(NetworkConfig(**payload["config"]["network"])).to(device)
    network.load_state_dict(payload["model_state_dict"])
    network.eval()
    return network


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AlphaGo Zero reconstruction GTP engine")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="checkpoint path (.pt) or several checkpoints for an ensemble averaging policy and value "
        "in one search. The first checkpoint supplies configuration and rules",
    )
    parser.add_argument(
        "--simulations",
        type=positive_int,
        default=None,
        help="override MCTS simulations per move (defaults to checkpoint)",
    )
    parser.add_argument(
        "--c-puct",
        type=positive_float,
        default=None,
        help="override the positive PUCT exploration constant (defaults to checkpoint)",
    )
    parser.add_argument(
        "--fpu",
        type=fpu_value,
        default=_UNSET,
        help="override first-play urgency with a nonnegative reduction "
        "from parent value, or none for an unvisited Q of zero. "
        "defaults to checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="inference device: cpu or cuda"
    )
    parser.add_argument(
        "--level",
        choices=("casual", "club", "champion"),
        default=None,
        help="difficulty: casual plays the raw "
        "policy without search or resignation. Club uses 64 PUCT simulations unless "
        "--simulations overrides it. Champion uses checkpoint settings. "
        "Omitting the preset also uses checkpoint settings",
    )
    parser.add_argument(
        "--no-resign", action="store_true", help="disable engine resignation"
    )
    parser.add_argument(
        "--resign-min-move",
        type=non_negative_int,
        default=None,
        help="earliest move at which resignation is allowed (defaults to checkpoint)",
    )
    parser.add_argument(
        "--resign-score-margin",
        type=non_negative_float,
        default=None,
        help="resign only with a predicted deficit greater than N points "
        "(requires a score head, defaults to the checkpoint margin if any)",
    )
    parser.add_argument(
        "--opening-moves",
        type=non_negative_int,
        default=0,
        help="sample the first N moves from visits to vary openings",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="engine RNG seed"
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    payloads = [torch.load(path, map_location=args.device, weights_only=False) for path in args.checkpoint]
    configs = [_config_from_checkpoint(payload) for payload in payloads]
    require_same_rules([cfg.game for cfg in configs], [str(path) for path in args.checkpoint])
    config = configs[0]
    # Apply the preset simulation count only when the flag was omitted.
    simulations = args.simulations
    if simulations is None and args.level in _LEVEL_SIMULATIONS:
        simulations = _LEVEL_SIMULATIONS[args.level]
    config.search = _apply_search_overrides(
        config.search,
        simulations,
        args.c_puct,
        args.fpu,
        resign_min_move=args.resign_min_move,
        resign_score_margin=args.resign_score_margin,
    )
    # Prefer the trained controller's calibrated threshold over the original configuration.
    calibrated_threshold = _checkpoint_resign_threshold(payloads[0])
    if calibrated_threshold is not None:
        config.search = replace(config.search, resign_threshold=calibrated_threshold)

    networks = [_load_network(payload, device) for payload in payloads]
    model: PolicyValueModel = networks[0] if len(networks) == 1 else EnsembleNet(networks)

    engine = GTPEngine(
        model=model,
        config=config,
        rng=np.random.default_rng(args.seed),
        allow_resign=not args.no_resign,
        opening_moves=args.opening_moves,
        policy_only=args.level == "casual",
    )
    # Reserve stdout for GTP. Send diagnostics to stderr.
    loaded = " + ".join(str(path) for path in args.checkpoint)
    print(
        f"loaded {loaded} (kind={payloads[0].get('kind')}, members={len(networks)}, "
        f"level={args.level or 'default'}, "
        f"resign_threshold={config.search.resign_threshold:g})",
        file=sys.stderr,
    )
    engine.run()


if __name__ == "__main__":
    main()
