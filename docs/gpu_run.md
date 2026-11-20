# GPU training and learning verification

Use an environment in which PyTorch recognizes the intended CUDA device. From the repository root, inspect the selected build and hardware, run the GPU integration tests and start a short training run.

```sh
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pytest tests/test_gpu.py -q --require-cuda
python -m examples.gpu_train --board 9 --komi 7.5 --channels 32 --res-blocks 2 --sims 24 --eval-batch 8 --parallel-games 4 --self-play-games 4 --iterations 2 --grad-steps 8 --eval-games 8 --work-dir runs/gpu_smoke
```

The GPU training command requires CUDA. The general Python pipeline can fall back to CPU. Verify the actual device in `summary.json`. Per-iteration metrics record self-play and training time, GPU memory and available device telemetry. `--amp` enables bfloat16 autocast for optimizer steps. Run the CUDA tests on the target device before using it for a long experiment. [Configuration](configuration.md)

## Learning verification

`examples.verify_learning` separates prediction on unseen games from strength on the board. It requires real CUDA and a new output directory. It writes the configuration, environment, source hashes, seeds and pass criteria to `protocol.json` before training.

The diagnostic enables deterministic torch algorithms. Set the cuBLAS workspace configuration before launching Python. In PowerShell, run the following commands.

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
python -m examples.verify_learning --output runs/learning_check
```

In a POSIX shell, use `CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m examples.verify_learning --output runs/learning_check`.

1. The initial network generates 64 training games and 32 independent holdout games. A copy trains for 400 updates. A second copy sees the same batches at learning rate zero so BatchNorm statistics can change without parameter updates. Passing requires lower holdout policy KL and value MSE than both controls.
2. A fresh 5×5 pipeline starts at the same seed. The default protocol runs 20 iterations of 16 self-play games and 64 optimizer steps with 32 search simulations. Only the internal promotion gate selects the final best checkpoint.
3. Play 120 independent games between that checkpoint and the initial weights, alternating colors and sampling six opening moves. The strength criterion is a lower 95% Wilson bound above 50%. The command exits with 0 when both criteria pass and 2 when a completed experiment misses either criterion.

The outputs include `diagnostic.json`, `pipeline/logs/metrics.json`, `match.json` with individual game outcomes and `result.json`. Run in a fresh directory on each repetition so a later result cannot overwrite an earlier failure.

### Recorded 5×5 experiment

On September 17, 2026, seed 20260917 was tested on an RTX 3090 Ti and PyTorch 2.10.0+cu128. Training produced 320 games, 10,347 positions and 1,280 optimizer steps. Iteration 19 passed the gate. Its independent match against the initial weights finished **81–39**, with 40 wins as Black and 41 as White (67.5%, Wilson 95% interval 58.7–75.2%). The separately tested final candidate scored 67–53 (55.8%, interval 46.9–64.4%).

| Model on 32 held-out games (1,146 positions) | Policy KL | Value MSE |
| --- | ---: | ---: |
| Initial weights | 1.5244 | 1.0191 |
| BatchNorm control, learning rate zero | 1.6672 | 1.2128 |
| 400 fixed-data optimizer steps | 1.0381 | 1.0962 |
| Pipeline's promoted model, supplementary evaluation | 0.6167 | 1.0325 |
| Last candidate, supplementary evaluation | 0.6190 | 0.9861 |

The match met its strength criterion. The fixed-data learner improved policy KL, but its value MSE exceeded the initial model's. The recorded result was `strength_passed=true`, `diagnostic_passed=false`, `passed=false`. This experiment covers one seed and one 5×5 budget.
