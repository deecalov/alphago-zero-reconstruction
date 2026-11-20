# Testing

From the repository root, install the development dependencies in the active environment and run the suite.

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pytest -q -m "not slow"
python -m ruff check .
python -m mypy alphago_zero
```

The unfiltered command includes slow tests. Running just the fast selection is convenient while editing. Most tests use small networks, seeded RNGs and temporary run directories. For coverage, run `python -m pytest --cov=alphago_zero --cov-report=term-missing --cov-fail-under=90`. The timing-budget tests skip under instrumentation.

## Environment-dependent tests

| Check | Command and prerequisite |
| --- | --- |
| CUDA training and resume | With a CUDA-enabled torch build and GPU, run `python -m pytest tests/test_gpu.py -q --require-cuda`. This checks FP32 and bfloat16, including league, replay overflow and resumed training. |
| Independent rules engine | Put `gnugo` on PATH, or set `AGZ_GNUGO_COMMAND` to its full launch command. Run `python -m pytest tests/test_gnugo_integration.py tests/test_classic_positions.py -q --require-gnugo`. Use `--mode gtp --level 1 --chinese-rules --positional-superko` in an explicit launch command. |
| Learning and playing strength | `python -m examples.verify_learning --output runs/learning_check` runs a longer CUDA experiment outside pytest. A successful integration test by itself cannot establish improved playing strength. [Protocol and recorded result](gpu_run.md#learning-verification) |

For SGF scoring tests, GNU Go must be able to read paths supplied by the test process. If the engine and Python run under different operating systems, run both in the same environment or translate the path in the launcher.

## Regression coverage

| Area | Examples of what the suite checks |
| --- | --- |
| Go rules | Differential games against a slow simulator and a frozen group-based engine. Classical positions exercise ko cycles, life and death, ladders and seki. |
| PUCT and Gumbel | Sign of backup, visit counts, batch collisions, virtual loss compensation, exact move selection and Sequential Halving schedules. The fast-path tests compare arrays, layout, complete trees and RNG state against frozen implementations. |
| Training | Network heads, policy/value losses, optional target masks, SGD groups, replay ring order, historical snapshots and exact resume. |
| Self-play | Sequential and parallel behavior, worker exceptions, effective slots, league routing, learner-only resignation and held-out game assignment. |
| Playing interfaces | GTP framing, timeouts, undo, SGF replay, engine termination, external adjudication, pair openings, void games and CLI entry points. |
| Strength tools | Elo and intervals, sequential stopping, matched search budgets, search-grid cache identity, distillation and checkpoint averaging. |

When comparing static type checks across environments, record NumPy and mypy versions. Newer NumPy stubs can require newer Python grammar than the project's Python 3.11 target. The earlier documented type-check environment used NumPy 2.4.2.
