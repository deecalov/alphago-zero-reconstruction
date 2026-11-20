from __future__ import annotations

"""Teacher labeling, student optimization and distillation CLI tests."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from alphago_zero.network import AlphaGoZeroNet
from examples import distill_merge


def _write_buffer(path: Path, states: np.ndarray) -> Path:
    torch.save({"replay_buffer": {"states": states}}, path)
    return path


def _error_message(capsys: pytest.CaptureFixture[str]) -> str:
    """Return the last stderr line, where argparse reports the actual error."""
    return capsys.readouterr().err.strip().splitlines()[-1]


def _base_argv(
    checkpoints: tuple[Path, Path], buffer_path: Path, out_path: Path, *extra: str
) -> list[str]:
    a, b = checkpoints
    return [
        "--teacher-a",
        str(a),
        "--teacher-b",
        str(b),
        "--buffer",
        str(buffer_path),
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--batch-size",
        "4",
        "--label-batch",
        "4",
        "--out",
        str(out_path),
        *extra,
    ]


# ----------------------------------------------------------------------
# Corpus and ensemble target generation
# ----------------------------------------------------------------------
def test_load_positions_concatenates_shuffles_and_truncates(tmp_path: Path) -> None:
    s1 = np.zeros((5, 17, 5, 5), dtype=np.uint8)
    s2 = np.ones((7, 17, 5, 5), dtype=np.uint8)
    b1 = _write_buffer(tmp_path / "b1.pt", s1)
    b2 = _write_buffer(tmp_path / "b2.pt", s2)

    full = distill_merge.load_positions([str(b1), str(b2)], max_positions=0, seed=42)
    assert full.shape == (12, 17, 5, 5) and full.dtype == np.uint8
    # Both source buffers contribute positions.
    assert int((full[:, 0, 0, 0] == 1).sum()) == 7

    same_seed = distill_merge.load_positions([str(b1), str(b2)], max_positions=0, seed=42)
    assert np.array_equal(full, same_seed)

    truncated = distill_merge.load_positions([str(b1), str(b2)], max_positions=4, seed=42)
    assert truncated.shape == (4, 17, 5, 5)
    assert np.array_equal(truncated, full[:4])

    not_truncated = distill_merge.load_positions([str(b1), str(b2)], max_positions=20, seed=42)
    assert not_truncated.shape == (12, 17, 5, 5)


def test_load_positions_rejects_wrong_tensor_rank(tmp_path: Path) -> None:
    bad = _write_buffer(tmp_path / "bad.pt", np.zeros((4, 5, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="expected states"):
        distill_merge.load_positions([str(bad)], max_positions=0, seed=1)


def test_label_corpus_averages_softmax_policies_and_values(checkpoints: tuple[Path, Path]) -> None:
    a, b = checkpoints
    device = torch.device("cpu")
    teacher_a, _ = distill_merge.load_teacher(a, device)
    teacher_b, _ = distill_merge.load_teacher(b, device)
    rng = np.random.default_rng(7)
    corpus = rng.integers(0, 2, size=(6, 17, 5, 5), dtype=np.uint8)

    policies, values = distill_merge.label_corpus([teacher_a, teacher_b], corpus, batch_size=2, device=device)
    policies_single_batch, values_single_batch = distill_merge.label_corpus(
        [teacher_a, teacher_b], corpus, batch_size=64, device=device
    )
    assert np.allclose(policies, policies_single_batch, atol=1e-6)
    assert np.allclose(values, values_single_batch, atol=1e-6)
    assert np.allclose(policies.sum(axis=1), 1.0, atol=1e-5)

    with torch.no_grad():
        x = torch.from_numpy(corpus).float()
        logits_a, val_a = teacher_a(x)
        logits_b, val_b = teacher_b(x)
        expected_policy = 0.5 * (F.softmax(logits_a, dim=1) + F.softmax(logits_b, dim=1)).numpy()
        expected_value = 0.5 * (val_a.squeeze(1) + val_b.squeeze(1)).numpy()
    assert np.allclose(policies, expected_policy, atol=1e-6)
    assert np.allclose(values, expected_value, atol=1e-6)


# ----------------------------------------------------------------------
# Student training and value-loss weight
# ----------------------------------------------------------------------
def test_train_student_updates_weights_and_ignores_values_when_value_weight_is_zero(
    tiny_model: AlphaGoZeroNet,
) -> None:
    rng = np.random.default_rng(11)
    # Of 300 positions, 256 form the minimum holdout and 44 remain for training.
    corpus = rng.integers(0, 2, size=(300, 17, 5, 5), dtype=np.uint8)
    raw = rng.random((300, tiny_model.action_size), dtype=np.float32)
    target_policy = raw / raw.sum(axis=1, keepdims=True)
    target_value_a = np.ones((300,), dtype=np.float32)
    target_value_b = -np.ones((300,), dtype=np.float32)

    student_a = AlphaGoZeroNet(tiny_model.config)
    student_a.load_state_dict(tiny_model.state_dict())
    student_b = AlphaGoZeroNet(tiny_model.config)
    student_b.load_state_dict(tiny_model.state_dict())

    report_a = distill_merge.train_student(
        student_a,
        corpus,
        target_policy,
        target_value_a,
        epochs=1,
        batch_size=16,
        lr=1e-2,
        value_weight=0.0,
        device=torch.device("cpu"),
        seed=5,
    )
    report_b = distill_merge.train_student(
        student_b,
        corpus,
        target_policy,
        target_value_b,
        epochs=1,
        batch_size=16,
        lr=1e-2,
        value_weight=0.0,
        device=torch.device("cpu"),
        seed=5,
    )
    assert report_a["holdout_size"] == 256
    assert len(report_a["epochs"]) == 1
    # With zero value weight, opposite value targets must produce identical SGD updates.
    for key, tensor_a in student_a.state_dict().items():
        assert torch.equal(tensor_a, student_b.state_dict()[key]), key
    # Positive learning rate moves the student away from initialization.
    assert any(not torch.equal(tensor_a, tiny_model.state_dict()[k]) for k, tensor_a in student_a.state_dict().items())
    assert report_b["epochs"][0]["train_policy_ce"] == report_a["epochs"][0]["train_policy_ce"]


# ----------------------------------------------------------------------
# End-to-end CLI and parameter forwarding
# ----------------------------------------------------------------------
def test_main_writes_a_distilled_checkpoint_with_recorded_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    buf = _write_buffer(tmp_path / "buf.pt", np.zeros((10, 17, 5, 5), dtype=np.uint8))
    out = tmp_path / "nested" / "student.pt"
    captured: dict[str, Any] = {}

    def fake_train(student: AlphaGoZeroNet, corpus: np.ndarray, tp: np.ndarray, tv: np.ndarray, **kwargs: Any) -> dict:
        captured["corpus_len"] = int(corpus.shape[0])
        captured["tp_shape"] = tp.shape
        captured["tv_shape"] = tv.shape
        captured.update(kwargs)
        return {"epochs": [{"epoch": 1, "train_policy_ce": 3.1, "train_value_mse": 0.2}], "holdout_size": 4}

    monkeypatch.setattr(distill_merge, "train_student", fake_train)
    distill_merge.main(
        _base_argv(
            checkpoints,
            buf,
            out,
            "--max-positions",
            "6",
            "--epochs",
            "2",
            "--batch-size",
            "3",
            "--lr",
            "0.001",
            "--value-weight",
            "0",
            "--seed",
            "99",
        )
    )
    assert captured["corpus_len"] == 6
    assert (captured["epochs"], captured["batch_size"], captured["lr"], captured["value_weight"]) == (2, 3, 0.001, 0.0)
    assert captured["seed"] == 99

    saved = torch.load(out, map_location="cpu", weights_only=False)
    assert saved["kind"] == "distilled_checkpoint"
    assert saved["config"]["game"]["board_size"] == 5
    meta = saved["distill"]
    assert (meta["teacher_a"], meta["teacher_b"]) == (str(a), str(b))
    assert meta["buffers"] == [str(buf)]
    assert (meta["positions"], meta["epochs"], meta["lr"], meta["value_weight"], meta["seed"]) == (6, 2, 0.001, 0.0, 99)


def test_student_init_overrides_the_default_teacher_a_weights(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoints: tuple[Path, Path]
) -> None:
    a, b = checkpoints
    buf = _write_buffer(tmp_path / "buf.pt", np.zeros((4, 17, 5, 5), dtype=np.uint8))
    out = tmp_path / "student.pt"
    monkeypatch.setattr(
        distill_merge, "train_student", lambda *_a, **_kw: {"epochs": [], "holdout_size": 2}
    )
    # Explicit --max-positions 0 selects the entire corpus just like the default.
    distill_merge.main(_base_argv(checkpoints, buf, out, "--max-positions", "0", "--student-init", str(b)))
    saved = torch.load(out, map_location="cpu", weights_only=False)
    assert saved["distill"]["positions"] == 4
    teacher_b_state = torch.load(b, map_location="cpu", weights_only=False)["model_state_dict"]
    teacher_a_state = torch.load(a, map_location="cpu", weights_only=False)["model_state_dict"]
    assert all(torch.equal(saved["model_state_dict"][k], teacher_b_state[k]) for k in teacher_b_state)
    assert any(not torch.equal(saved["model_state_dict"][k], teacher_a_state[k]) for k, _ in teacher_a_state.items())


# ----------------------------------------------------------------------
# Numeric arguments and output directory validation
# ----------------------------------------------------------------------
_INVALID_CLI_VALUES: list[tuple[str, str]] = [
    ("--lr", "0"),
    ("--lr", "-1e-4"),
    ("--lr", "nan"),
    ("--lr", "inf"),
    ("--lr", "-inf"),
    ("--lr", "abc"),
    ("--value-weight", "-0.1"),
    ("--value-weight", "nan"),
    ("--value-weight", "inf"),
    ("--value-weight", "-inf"),
    ("--value-weight", "abc"),
    ("--max-positions", "-1"),
    ("--max-positions", "1.5"),
    ("--max-positions", "abc"),
    ("--epochs", "0"),
    ("--epochs", "-2"),
    ("--epochs", "abc"),
    ("--batch-size", "0"),
    ("--batch-size", "-8"),
    ("--batch-size", "abc"),
    ("--label-batch", "0"),
    ("--label-batch", "-16"),
    ("--label-batch", "abc"),
]


@pytest.mark.parametrize(("flag", "value"), _INVALID_CLI_VALUES)
def test_invalid_numeric_flags_are_rejected_before_loading_teachers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    def forbidden(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("teachers must not load with an invalid flag")

    monkeypatch.setattr(distill_merge, "load_teacher", forbidden)
    out = tmp_path / "student.pt"
    argv = _base_argv(checkpoints, tmp_path / "missing_buf.pt", out, f"{flag}={value}")
    with pytest.raises(SystemExit) as exit_info:
        distill_merge.main(argv)
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert flag in message and "expected" in message


def test_unusable_out_directory_is_rejected_before_loading_teachers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoints: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("file in place of a directory", encoding="utf-8")

    def forbidden(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("teachers must not load when the --out directory cannot be created")

    monkeypatch.setattr(distill_merge, "load_teacher", forbidden)
    argv = _base_argv(checkpoints, tmp_path / "buf.pt", blocker / "student.pt")
    with pytest.raises(SystemExit) as exit_info:
        distill_merge.main(argv)
    assert exit_info.value.code == 2
    message = _error_message(capsys)
    assert "--out" in message and "blocker" in message
