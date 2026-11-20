# Measuring playing strength

For a weight comparison, fix the board, komi, search budget, opening procedure and color balance. Use a frozen opponent and record the seed. The training gate selects among candidates on a small sample. An independent match supplies evidence about the selected checkpoint.

```sh
python -m examples.eval_match --model runs/tiny_demo/final_best.pt --opponent runs/tiny_demo/final_candidate.pt --games 20 --simulations 32 --opening-moves 4 --device cpu --json-out runs/match.json
```

Alternating colors limits first-player and komi effects. Sample opening moves from visits to avoid replaying the same deterministic game. For ensembles, pass multiple paths to one side. The first member supplies that side's search settings. Equal simulations can cost more for an ensemble, so record simulations and time per move when interpreting a comparison.

`--model-c-puct`, `--opponent-c-puct`, per-side FPU, tree reuse, simulation and time overrides support search comparisons on fixed weights. `examples.search_grid` runs a parameter grid against the saved settings, checks the experiment manifest before reusing a result and prints a confirmation command. Selecting the maximum from many noisy matches biases its observed score upward. Confirm a candidate on a fresh seed.

## Ratings, intervals and sequential decisions

The match report gives the model a point per win and half a point per draw, with a 95% Wilson interval for its score fraction. Relative Elo uses `400 * log10(p / (1 - p))`. An absolute number depends on the assigned opponent rating and its entire search configuration. The built-in ladder starts with a random player, adds the initial raw-policy model and can include a rated frozen checkpoint. Sweeping weak anchors provides little resolution near the top of the ladder.

`--sprt` tests the specified win-rate hypotheses using a log likelihood ratio. Check boundaries after pairs of games, with `--games` as a cap. A run can reach its cap with an inconclusive result. Record the hypotheses, error levels, actual game count, final decision and interval. [Sequential test](../alphago_zero/sprt.py)

## Playing through GTP and SGF

`examples.play_gtp` connects a checkpoint to a GUI through standard input and output. Its `casual` preset plays the raw policy. `club` uses 64 PUCT simulations unless explicitly overridden. `champion` uses checkpoint search settings. The engine supports time controls, undo, SGF loading and GoGui analysis. Use `list_commands` to inspect the commands offered by a running instance.

The SGF reader follows the first variation, reports skipped variations and unsupported properties, and replays setup stones and moves to reconstruct history. Use an explicit result when exporting a draw or void game.

## External opponents

With GNU Go available as a GTP program, this example plays four games on the demo board. Use checkpoints trained at the matching komi.

```sh
python -m examples.external_match --checkpoint runs/tiny_demo/final_best.pt --board-size 5 --komi 2.5 --games 4 --opponent-command "gnugo --mode gtp --level 1 --chinese-rules --positional-superko" --json-out runs/external/match.json --sgf-dir runs/external/sgf
```

`--dry-run` prints the launch commands and settings. The controller gives each color pair the same opening. A separate referee can provide `final_score`, otherwise the opponent scores. An illegal move, timeout, broken connection or inconsistent score produces a void game with a recorded reason. Void games remain in the report and are excluded from the performance estimate. Record engine versions, arguments, budget and scoring rules when using GNU Go or Pachi as an anchor.

### Recorded external controls

In a historical rules comparison, GNU Go and the local engine agreed on 297,510 legality queries across 5,662 positions on 5×5 and 9×9 boards. The sample included 8,103 prohibited suicides and 105 ko or superko prohibitions. Omitting GNU Go's positional-superko setting produced five disagreements across 2,877 positions in a separate 5×5 check.

A historical single-threaded Pachi 12.90 match series used fixed playout counts. Against Pachi at 2,000 playouts per move, one frozen checkpoint scored 30–30. An ensemble of that checkpoint and a second network scored 39–21 (65.0%, 95% Wilson interval 52.4–75.8%). These scores identify the Pachi build and budget as part of the opponent and can be used to design a repeat measurement.
