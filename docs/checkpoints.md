# Checkpoints and recovery

The pipeline writes artifacts beneath `runtime.work_dir`. Choose a separate work directory for each experiment so current metrics, snapshots and final weights stay together.

```text
config.json                   Resolved configuration
summary.json                  Final counters, environment and artifact paths
final_best.pt                 Promoted model for play and evaluation
final_candidate.pt            Most recently trained candidate
resume_state.pt               Full training snapshot
resume_state.previous.pt      Previous complete snapshot after rotation
checkpoints/best.pt           Best model at a checkpoint boundary
checkpoints/candidate_iter_*.pt
logs/metrics.json             Seed, environment and per-iteration records
sgf/                          Optional self-play game records
```

`training.save_every_iteration` and `training.checkpoint_every` control intermediate weight files. `runtime.save_resume_every` controls full snapshots independently. An interval of zero disables resume snapshots. A time budget stops the loop at an iteration boundary and still writes final artifacts.

## Inference

```python
from alphago_zero import AlphaGoZeroNet, load_checkpoint_into_model, make_demo_config

config = make_demo_config()
model = AlphaGoZeroNet(config.network)
payload = load_checkpoint_into_model("runs/tiny_demo/final_best.pt", model)
model.eval()
print(payload["kind"], payload["global_step"])
```

Construct the model with the saved architecture. For an arbitrary trusted pipeline checkpoint, the GTP and match frontends read the network configuration from its payload. The files use PyTorch object deserialization with `weights_only=False` and should come from a trusted source.

## Continue or start a new experiment

| Goal | Setting | State retained |
| --- | --- | --- |
| Continue the same run | `runtime.resume_from` / `--resume` | Learner and best weights, optimizer and momentum, replay and write position, calibration, random streams, iteration counters, metrics and available anchor/holdout state |
| Train fresh weights from accumulated games | `runtime.warmstart_buffer_from` / `--warmstart-buffer` | Replay examples only. A smaller capacity retains the newest examples in chronological order. Board representation must remain compatible. |
| Initialize weights for a new run | `runtime.init_weights_from` / `--init-weights` | Tensors with matching names and shapes. New heads retain their initialization. Can be combined with a replay warm start. |

```python
from alphago_zero import make_demo_config, run_training_pipeline

config = make_demo_config("runs/experiment")
config.runtime.resume_from = "runs/experiment/resume_state.pt"
config.training.num_iterations = 10
run_training_pipeline(config)
```

The iteration count is an absolute target. Resume validates the saved game, model and optimizer settings before writing the new configuration. Changing an allowed training or search setting also changes the expected trajectory. Legacy snapshots without embedded configuration require their original adjacent `config.json`.

Writes use a temporary file in the destination directory, flush it and replace the destination atomically. If the latest full snapshot is unusable, point `resume_from` explicitly at `resume_state.previous.pt`. Allow disk space for the old, new and backup copies during rotation. Checkpoints and snapshots may include optimizer state or large replay data. [Resume tests](../tests/test_resume.py)
