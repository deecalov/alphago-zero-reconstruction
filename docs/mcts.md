# Monte Carlo tree search

A node represents a position. Each legal edge stores a prior P, a visit count N and a value sum W. For a visited edge Q = W/N is the expected outcome for **the player choosing that edge**. If a child predicts +0.8 for White to move, the edge Black took to reach it receives −0.8. The sign flips before every parent-edge update.

## One PUCT simulation

1. Select the edge maximizing `Q + c_puct * P * sqrt(1 + total_visits) / (1 + edge_visits)`. The extra one inside the square root lets priors distinguish moves on the first visit. Optional first-play urgency sets unvisited Q from the parent value minus a reduction, clipped at −1. With the default `None` it is zero.
2. At an unexpanded leaf, rotate or reflect the encoded board, evaluate it and undo the transformation on the logits. Normalize over legal moves, including pass. A terminal leaf uses the known outcome for the player to move.
3. Back up the leaf value with alternating sign. Root Q averages backed-up estimates by visits and can differ from the raw root prediction.

During full-budget exploratory self-play the root mixes network priors with Dirichlet noise. After search, early moves sample a temperature-adjusted distribution. The training policy uses normalized visits independently of move temperature. Deterministic play can still train on a broad target.

## Batching, subtree reuse and deadlines

`eval_batch_size` leaf selections can be pending at once. Selection increments visits and temporarily subtracts virtual loss. If several paths reach one leaf, evaluate it once and back up each path. Backup removes each temporary penalty. A batch of one restores the penalty before the next choice. To increase inference batch width while limiting within-tree distortion, batch leaves across multiple games. [Performance](performance.md)

A PUCT tree can advance to the child corresponding to the played move. A model switch, SGF load or undo can invalidate the cached root. With a time budget, check deadlines between batches. At least the first batch runs. A sufficiently large lead in visits can settle the winning action early when the other actions cannot catch up.

## Gumbel search and optional targets

`search_algo="gumbel"` applies Gumbel-Top-k and Sequential Halving to the learner's root. Interior nodes use completed Q values to form an improved policy. The root winner supplies the move and `softmax(logits + transformed_completed_Q)` supplies the target. In-flight evaluations count as visits, so large leaf batches can eliminate candidates before their values return. Each move starts from a fresh root. Frozen league opponents, promotion games and benchmarks use PUCT.

PUCT can reserve a minimum visit share for underexplored root actions. Target pruning drops entries below a fraction of the highest mass and renormalizes. Pruning changes what the learner sees while preserving the recorded visits and the move played. Both controls default to zero. [Configuration](configuration.md) · [Experiments](rnd_experiments.md)
