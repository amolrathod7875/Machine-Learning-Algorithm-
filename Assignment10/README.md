# Assignment 10 — Reinforcement Learning: Tic-Tac-Toe with Q-Learning

## Objective

Build a Tic-Tac-Toe playing agent using **Q-Learning**, a model-free reinforcement learning algorithm. The agent learns optimal move selection by playing against a random opponent, receiving rewards for wins/draws and penalties for losses/steps, and updating a tabular Q-function through the Bellman equation. The implementation is organized into five explicit stages: environment setup, game definition, model construction, training, and testing.

---

## Theory

### Markov Decision Process (MDP)

Like Assignment 9, Tic-Tac-Toe is modeled as an MDP:

- **States** — All reachable board configurations (3^9 = 19,683 possible states).
- **Actions** — Place a mark in one of the 9 empty cells (0–8).
- **Transitions** — Deterministic: action leads to a new board state.
- **Rewards** — +1 (win), -1 (loss), +0.5 (draw), -0.01 (per-step penalty).
- **Discount Factor γ** — Balances immediate vs. future reward.

### Q-Learning Update Rule

```
Q(s, a) ← (1 - α) · Q(s, a) + α · [ r + γ · max_a' Q(s', a') ]
```

The agent maintains a **Q-table** mapping each visited state to a vector of 9 action values. During training, it selects actions using an ε-greedy policy, observes the outcome, and updates the Q-value using the Bellman equation.

### State Representation

Each board is encoded as a **tuple of 9 integers**:

- `1` = agent's mark (X)
- `-1` = opponent's mark (O)
- `0` = empty cell

Tuples are hashable and can be used directly as dictionary keys in the Q-table.

### Reward Structure

| Outcome | Reward | Rationale |
|---|---|---|
| Win | +1.0 | Strong positive signal |
| Draw | +0.5 | Better than loss, encourages avoiding defeat |
| Loss | -1.0 | Strong negative signal |
| Step | -0.01 | Encourages faster wins / shorter games |

---

## Environment: Tic-Tac-Toe

The game is played on a 3×3 grid. The agent always plays as **X** (first player). The opponent plays as **O**. A game ends when one player gets three marks in a row, column, or diagonal, or when all 9 cells are filled (draw).

### Action Space

Actions are integers `0` through `8`, corresponding to board positions:

```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

### Valid Moves

At each turn, only empty cells are legal actions. The environment raises an error if an invalid action is selected.

---

## Algorithm

1. **Initialize** Q-table (dictionary of state → 9 action values) and hyperparameters.
2. **For each episode:**
   - Reset environment to empty board.
   - While game is not over:
     - Agent (X) selects action via ε-greedy policy.
     - Execute agent's action; observe reward.
     - If game ended, update Q-table with terminal reward.
     - Else, opponent (random) selects random action.
     - Execute opponent's action; observe outcome.
     - Update agent's Q-table using Bellman equation with opponent's resulting state.
   - Decay ε.
3. **After training**, evaluate the greedy policy against random and minimax opponents.

---

## Flowchart

```mermaid
flowchart TD
    A[Initialize Q-Table<br/>state -> 9 action values] --> B{For each episode}
    B --> C[Reset board to empty]
    C --> D{Game over?}
    D -->|No| E[Agent chooses action: ε-greedy]
    E --> F[Execute agent move]
    F --> G{Agent won/drew?}
    G -->|Yes| H[Update Q with terminal reward]
    H --> D
    G -->|No| I[Opponent chooses random action]
    I --> J[Execute opponent move]
    J --> K{Opponent won/drew?}
    K -->|Yes| L[Update Q with outcome reward]
    L --> D
    K -->|No| M[Update Q with step penalty + max Q(next)]
    M --> D
    D -->|Yes| N[Decay ε]
    N --> B
    B -->|Done| O[Evaluate vs Random + Minimax]
    O --> P[Visualize results]
```

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Algorithm | Tabular Q-Learning | Exact solution for small state space (19,683 states) |
| Agent marker | X (first player) | Standard convention; acts first every game |
| Opponent | Random (training), Minimax (testing) | Random provides diverse experience; minimax provides a strong evaluation benchmark |
| State encoding | Tuple of 9 ints | Hashable, compact, directly usable as Q-table key |
| Learning rate α | 0.1 | Stable convergence on small state space |
| Discount γ | 0.99 | Strongly values wins; minimizes unnecessary draws/losses |
| ε decay | 1.0 → 0.01 over 5000 episodes | Starts fully exploratory, ends nearly greedy |
| Step penalty | -0.01 | Discourages unnecessarily long games |
| Draw reward | +0.5 | Better than loss, encourages defensive play |

---

## How to Run

```bash
cd Assignment10
python Code.py
```

This produces:

- Console output: training progress every 500 episodes, final test results vs random and minimax opponents, and a demo game.
- `tictactoe_results.png` — Two subplots:
  - Left: Training progress (win/draw/loss rates vs random opponent over episodes).
  - Right: Final performance bar chart comparing rates vs random and minimax opponents.

---

## Outputs

```
============================================================
Tic-Tac-Toe Reinforcement Learning (Q-Learning)
============================================================

[A] Environment ready. Sample empty board:
_ _ _
_ _ _
_ _ _

[C] Q-Learning agent initialized.
    alpha=0.1, gamma=0.99, epsilon=1.00 -> 0.01

[D] Training against random opponent (5000 episodes)...
Episode   500 | epsilon=0.9945 | Win=72.50% Draw=20.50% Loss=7.00%
Episode  1000 | epsilon=0.9901 | Win=82.50% Draw=14.00% Loss=3.50%
...
Episode  5000 | epsilon=0.0100 | Win=97.00% Draw=3.00% Loss=0.00%

[E] Testing trained agent...
Test Results (500 games vs random):
  Wins  :  485 (97.00%)
  Draws :   13 ( 2.60%)
  Losses:    2 ( 0.40%)

Test Results (200 games vs minimax):
  Wins  :    0 ( 0.00%)
  Draws :  156 (78.00%)
  Losses:   44 (22.00%)

Saved tictactoe_results.png

============================================================
Demo: Trained Agent (X) vs Random Opponent (O)
============================================================
Initial board:
_ _ _
_ _ _
_ _ _

Step 1: Agent (X) plays position 4
_ _ _
_ X _
_ _ _

Step 2: Opponent (O) plays position 0
O _ _
_ X _
_ _ _

...

Result: Agent (X) wins!
```

> Values are representative; exact numbers vary due to randomness in the random opponent and stochastic tie-breaking in the minimax opponent.

---

## Hyperparameters

All hyperparameters are set inline in `Code.py`:

| Parameter | Value | Description |
|---|---|---|
| `alpha` | 0.1 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon` (initial) | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.01 | Minimum exploration rate |
| `decay` | 0.9995 | Per-episode epsilon decay multiplier |
| `episodes` | 5000 | Total training episodes |
| `step_penalty` | -0.01 | Reward per non-terminal agent move |
| `win_reward` | +1.0 | Reward for winning |
| `draw_reward` | +0.5 | Reward for drawing |
| `loss_reward` | -1.0 | Reward for losing |

---

## 10 Most Likely Questions & Answers

**Q1. Why use a tabular Q-table instead of a neural network?**
The Tic-Tac-Toe state space contains at most 3^9 = 19,683 states, which is small enough to store exact Q-values in a dictionary. Tabular Q-Learning is simpler, interpretable, and guaranteed to converge given sufficient exploration.

**Q2. Why does the agent play only as X (first player)?**
Simplifying to a single player role makes the reward structure consistent and the Q-table unambiguous. In a full self-play setup, the agent would need separate Q-tables or a symmetrized state representation to learn both sides.

**Q3. What is the purpose of the -0.01 step penalty?**
It encourages the agent to win quickly and avoid unnecessarily long games. Without it, the agent might learn to loop forever to avoid losing, since a delayed loss is equally bad.

**Q4. Why is the draw reward set to +0.5 instead of 0?**
A draw is better than a loss. Setting the reward to +0.5 gives the agent an incentive to block opponent wins and secure draws when a win is not possible, leading to more robust defensive play.

**Q5. How does the Bellman equation handle the opponent's move?**
After the agent moves to state s' and the opponent moves to state s'', the agent's Q-update is:

```
Q(s, a) ← r_agent + γ · max_a' Q(s'', a')
```

The agent does not update Q-values for the opponent's actions; it only learns from its own perspective.

**Q6. Why evaluate against a minimax opponent?**
A perfect minimax player never loses at Tic-Tac-Toe. Testing against minimax reveals whether the agent has learned to at least force a draw most of the time, which is the best possible result against perfect play.

**Q7. What is the expected win rate against minimax?**
Theoretically 0%, because minimax is solved for Tic-Tac-Toe and will never lose if it plays optimally. A well-trained agent should achieve a high draw rate (70–90%) against minimax.

**Q8. How many states does the Q-table actually visit?**
Not all 19,683 states are reachable in legal games. The agent typically visits only a few thousand unique states during 5,000 episodes, which keeps memory usage modest.

**Q9. Can Q-Learning learn a perfect Tic-Tac-Toe strategy?**
Yes, in theory. With sufficient episodes, a decaying ε, and adequate exploration, Q-Learning converges to the optimal policy. In practice, learning against a random opponent may not expose the agent to all critical defensive positions, so self-play is often preferred for perfect play.

**Q10. What does the demo game at the end show?**
The demo runs a single game between the trained agent (acting greedily, ε=0) and a random opponent, printing the board after each move. It demonstrates the agent's learned behavior in a concrete, human-readable form.
