# Measuring throughput

Measure the stage that accounts for iteration time before changing the training configuration. `examples/profile_baseline.py` measures move generation, search, sequential and parallel self-play, and SGD separately. It finishes with a cumulative-time profile.

```sh
python -m examples.profile_baseline --device cpu
```

Use `--full` to include the slower 19×19 engine benchmark. Keep board size, network, simulations, batch size, thread count, seed and backend fixed when comparing revisions. Faster inference can leave Python tree traversal as the new limiting stage. A benchmark with a stub predictor isolates CPU work and cannot predict complete GPU training time.

## Where the CPU time went

The board engine uses cached geometry and move candidates, incremental Zobrist updates and a separate final superko check. The search caches permutations for common encoder shapes, constructs priors in legal move order and avoids recomputing Gumbel prior statistics within a move. Preserving floating-point operation order matters because strict comparisons and dictionary order determine ties.

A recorded CPU comparison across the corresponding board and search optimizations measured PUCT at 160 simulations going from 45.6 to 14.7 ms per move, and Gumbel at 32 from 10.74 to 4.71 ms. Stub self-play increased from 129 to 296 moves/s on that benchmark. These figures describe that workload; use the profiler on the target model and hardware. [Fast-path regression tests](../tests/test_search_fast_paths.py)

## Effective slots and batches

With 24 games and a single process, requesting 32 slots opens 24. With 24 games distributed over three workers, each can fill at most eight. When the game count is uneven, the earliest workers receive the remainder. A nominal capacity increase can therefore leave the experiment unchanged.

| Metric | Interpretation |
| --- | --- |
| `parallel_games`, `slots` | Requested slots per driver and total slots actually opened across workers |
| `rounds`, `fill` | Scheduling rounds and the mean occupied fraction of available slots |
| `network_calls`, `positions`, `mean_batch` | Number of inference calls, evaluated positions and realized positions per call |
| Stage and worker timings | Time spent generating games versus optimizing and evaluating, and imbalance between processes |

Once no fresh game remains to refill a slot, long games run in increasingly thin batches. In one 24-game/24-slot measurement, a resignation threshold of −0.999 yielded mean batches of 65–78 positions. At −0.54 the shorter games produced batches of 25–37. Record game length and resignation settings alongside utilization before attributing a throughput change to the GPU.

Worker processes can use additional CPU cores but each holds a device-side network copy and has fewer games to batch. Compare both elapsed iteration time and games generated under the same search settings. [Architecture](architecture.md)
