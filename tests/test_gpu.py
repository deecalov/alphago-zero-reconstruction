"""Actual CUDA integration: inference, league, capped self-play, SGD and resume."""

import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from alphago_zero.config import AlphaGoZeroConfig
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.pipeline import _save_checkpoint, run_training_pipeline


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
def test_cuda_training_league_and_resume(
    tiny_config: AlphaGoZeroConfig, tmp_path: Path, amp: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    cfg = copy.deepcopy(tiny_config)
    cfg.runtime.device = "cuda"
    cfg.runtime.use_amp = amp
    cfg.network.channels = 16
    cfg.network.aux_heads = True
    cfg.search.num_simulations = 8
    cfg.search.eval_batch_size = 2
    cfg.search.playout_cap_prob = 0.5
    cfg.search.playout_cap_cheap_sims = 2
    cfg.search.disable_resign_prob = 1.0
    cfg.training.parallel_games = 2
    cfg.training.self_play_games_per_iteration = 2
    cfg.training.gradient_steps_per_iteration = 3
    cfg.training.replay_buffer_size = 32  # wrap before checkpointing
    cfg.training.aux_warmup_steps = 3
    cfg.training.evaluation_gate = "binomial"
    cfg.training.evaluation_games = 8
    cfg.training.evaluation_opening_moves = 3
    cfg.training.num_iterations = 2

    anchor_cfg = copy.deepcopy(cfg)
    anchor_cfg.network.aux_heads = False
    anchor_cfg.search = replace(cfg.search, num_simulations=11, eval_batch_size=3)
    anchor = AlphaGoZeroNet(anchor_cfg.network).cuda().eval()
    anchor_path = tmp_path / "frozen.pt"
    _save_checkpoint(anchor_path, anchor, None, extra={"config": asdict(anchor_cfg)})
    cfg.training.league_checkpoints = [anchor_path]
    cfg.training.league_prob = 1.0

    cfg.runtime.work_dir = tmp_path / "continuous"
    summary = run_training_pipeline(cfg)
    assert summary["device"] == "cuda"
    assert summary["global_step"] == 6
    metrics = json.loads(Path(summary["metrics_log_path"]).read_text(encoding="utf-8"))["iterations"]
    assert all(row["league_games"] == 2 for row in metrics)
    assert all(row["evaluation"]["required_wins"] == 7 for row in metrics)
    assert all(row["gpu"]["torch_max_allocated_mb"] > 0 for row in metrics)
    assert all(np.isfinite(row["train"]["total_loss"]) for row in metrics)
    assert metrics[-1]["train"]["ownership_loss"] > 0

    resumed = copy.deepcopy(cfg)
    resumed.runtime.work_dir = tmp_path / "resumed"
    resumed.training.num_iterations = 1
    run_training_pipeline(resumed)
    resume_path = Path(resumed.runtime.work_dir) / "resume_state.pt"
    resumed.runtime.resume_from = resume_path
    resumed.training.num_iterations = 2
    continued = run_training_pipeline(resumed)
    assert continued["start_iteration"] == 2 and continued["global_step"] == 6
    assert (Path(resumed.runtime.work_dir) / "resume_state.previous.pt").exists()

    expected = torch.load(Path(summary["final_candidate_path"]), map_location="cpu", weights_only=False)
    actual = torch.load(Path(continued["final_candidate_path"]), map_location="cpu", weights_only=False)
    for key, tensor in expected["model_state_dict"].items():
        torch.testing.assert_close(actual["model_state_dict"][key], tensor, rtol=0, atol=0)

    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 2
    assert payload["replay_buffer"]["states"].shape[0] == 32
    assert np.all(payload["replay_buffer"]["aux_mask"] == 1)
