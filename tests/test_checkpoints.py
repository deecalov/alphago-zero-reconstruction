from __future__ import annotations

"""Checkpoint round-trip and architecture compatibility tests."""

from pathlib import Path

import pytest
import torch

from alphago_zero.config import NetworkConfig
from alphago_zero.checkpoint_io import atomic_output
from alphago_zero.network import AlphaGoZeroNet
from alphago_zero.pipeline import _save_checkpoint, load_checkpoint_into_model


def test_save_and_load_roundtrip(tiny_network_config: NetworkConfig, tmp_path: Path) -> None:
    model = AlphaGoZeroNet(tiny_network_config)
    path = tmp_path / "ckpt.pt"
    _save_checkpoint(path, model, optimizer=None, extra={"kind": "test", "global_step": 5})

    restored = AlphaGoZeroNet(tiny_network_config)
    payload = load_checkpoint_into_model(path, restored)

    assert payload["kind"] == "test"
    assert payload["global_step"] == 5
    assert "model_summary" in payload

    # Compare weights bit for bit.
    for original, loaded in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(original, loaded)


def test_load_into_mismatched_config_raises(
    tiny_network_config: NetworkConfig, tmp_path: Path
) -> None:
    model = AlphaGoZeroNet(tiny_network_config)
    path = tmp_path / "ckpt.pt"
    _save_checkpoint(path, model, optimizer=None, extra={})

    # A different channel count gives an incompatible tensor shape for strict loading.
    other_config = NetworkConfig(board_size=5, channels=16, num_res_blocks=1, value_head_hidden=8)
    other_model = AlphaGoZeroNet(other_config)
    with pytest.raises(RuntimeError):
        load_checkpoint_into_model(path, other_model)

def test_atomic_snapshot_preserves_current_and_previous_on_write_failure(tmp_path: Path) -> None:
    path = tmp_path / "resume_state.pt"
    for generation in (1, 2):
        with atomic_output(path, keep_previous=True) as stream:
            torch.save({"generation": generation}, stream)
    previous = tmp_path / "resume_state.previous.pt"
    with pytest.raises(OSError, match="disk full"):
        with atomic_output(path, keep_previous=True) as stream:
            stream.write(b"partial checkpoint")
            raise OSError("disk full")
    assert torch.load(path, weights_only=True)["generation"] == 2
    assert torch.load(previous, weights_only=True)["generation"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_failed_serialization_preserves_existing_weights(
    monkeypatch: pytest.MonkeyPatch, tiny_model: AlphaGoZeroNet, tmp_path: Path,
) -> None:
    path = tmp_path / "best.pt"
    _save_checkpoint(path, tiny_model, None)
    original = path.read_bytes()

    def fail_save(payload: object, stream: object) -> None:
        stream.write(b"incomplete")  # type: ignore[attr-defined]
        raise OSError("write failed")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="write failed"):
        _save_checkpoint(path, tiny_model, None)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
