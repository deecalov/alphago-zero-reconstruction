# Reproducibility

Record the run directory, seed, source revision, resolved configuration, Python/PyTorch/NumPy versions and actual device. The pipeline stores environment and seed in its summary, metrics and checkpoints. Save the local configuration before comparing results created with different presets.

The run seeds Python, NumPy and torch, including CUDA when available. Self-play and minibatch selection use explicit NumPy generators. Benchmarking and holdout monitoring use separately derived streams. Worker seeds derive from the run seed, iteration number and worker index. Changing the number of workers or simultaneously active games can change the games played even at the same seed.

The CPU pipeline uses one torch thread for small workloads. Test exact resume by comparing uninterrupted training with a run interrupted at an iteration boundary. Ignore wall-clock and device telemetry in that comparison. A new hardware backend, torch build or parallel search schedule can change floating-point tie-breaking and all subsequent moves.

When checking playing strength, record the opponent weights, search settings, opening procedure, color assignment, stopping rule and match seed. Keep checkpoint selection independent of the final match. A deterministic replay of one seed and an independent replication across seeds answer different questions. [Resume](checkpoints.md) · [Learning verification](gpu_run.md#learning-verification)
