# Game and engine audit

The October 2026 audit examined how the rules engine, search and playing interfaces agree on outcomes. The table records implemented checks and the boundaries that matter when reading a game record or interpreting a result.

| Topic | Implemented check or behavior |
| --- | --- |
| Draws | With integer komi, equal area scores give no winner. Both sides receive z=0. External and checkpoint match reporting counts half a point for a draw, and SGF exports `RE[0]`. Training promotion uses the configured win-count rule. |
| Rules and setup | Public constructors validate board dimensions, stone values, turn, finite komi and superko variant. Replaying SGF reconstructs history across setup and moves. |
| Scoring | `examples.probe_scoring` can compare raw area, a network ownership estimate and an external GTP referee on recorded endings. Track double pass, resignation and move truncation separately. |
| Board semantics | Independent tests cover seki, snapback, nakade, bent four, double and triple ko, ladders and false eyes. GNU Go can check legal placements and terminal SGFs. |
| Time and resignation | Live search can reuse a subtree and respect a clock. The engine reads a calibrated checkpoint threshold and can require a minimum move number or predicted score deficit before resigning. |
| Human and GUI play | GTP includes raw-policy and search presets, undo, partial SGF loads, replay and GoGui move analysis. Protocol output uses stdout and diagnostics use stderr. |

## A scored board can still contain dead groups

In a recorded comparison of 582 scorer verdicts, no winner disagreement appeared in that sample. The classical bent-four position supplies a counterexample to treating raw final area and adjudicated territory as generally interchangeable: leaving the group on the board counts its stones, while an external referee can mark it dead. Track score margin as well as winner when comparing scorers.

Inspect a record with the actual rules, final board and termination reason. A truncated game can have a known area score on an unresolved position. A resignation has a winner but lacks a scored terminal board for the auxiliary targets. [Go rules](go_rules.md) · [Classical positions](../tests/test_classic_positions.py) · [Scoring probe](../examples/probe_scoring.py)
