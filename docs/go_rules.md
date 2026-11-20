# Go rules and position encoding

`GoState` uses +1 for Black, −1 for White and 0 for empty. A move is an integer `row * N + col`. Pass has index `N * N`. GTP vertices count rows from the bottom and skip column I. SGF coordinates count from the upper left with the column first.

## Placement and repetition

A placement must be on an empty point. Remove adjacent opposing groups whose last liberty it fills, then check the friendly group's liberties. A capture can make a surrounded placement legal. Pure suicide is illegal.

The default **positional superko** rejects a placement that recreates a previous board. **Situational superko** keys the board together with the next player. A fixed Zobrist table and incremental XOR updates track repetitions. Pass is exempt under both variants. Two passes in a row end play. The move limit also ends play and defaults to twice the board area.

`GoState.from_board` validates an arbitrary starting diagram and seeds its current repetition key. To enforce earlier ko history or fill historical input planes, replay the moves that produced the position. `play` creates a successor and shares immutable historical arrays. Do not mutate the arrays of a state already used by search.

## Scoring

Count stones on the board plus each empty region bordered exclusively by that color. Regions touching both colors are neutral. Add komi to White. Equal scores produce a draw and a training outcome of zero. A resignation decides the winner without supplying a terminal ownership or score target. The ending reason distinguishes `double_pass`, `max_moves` and `resign`.

Area scoring evaluates the stones left on the board. Players must resolve dead groups before passing if they want those captures reflected in the score. For example, a bent four left in the corner can receive a different verdict from a referee that adjudicates dead stones. The classical-position tests cover this distinction alongside seki, snapback, nakade, ko cycles, ladders and false eyes. [Scoring audit](game_improvement_audit.md) · [Position tests](../tests/test_classic_positions.py)

## Network input

The 17 float32 planes interleave current-player and opponent stones at each of eight times, newest first. The last plane is all ones for Black to play and all zeros for White. Komi is used by the rules and stays outside the input. The GTP frontend consequently requires the checkpoint's board size and komi.

Move generation caches a byte mask of nonsuicidal empty points and the captures associated with each point. The superko check uses the state's own history after that analysis. Tests compare the fast path with two independent historical implementations, including the legal action order, resulting board, captures and hash. [Game engine](../alphago_zero/game.py) · [Differential tests](../tests/test_game_fast_engine.py)
