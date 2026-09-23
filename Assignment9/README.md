# Assignment 9 — Reinforcement Learning in a Maze Environment

## Objective

Train an autonomous agent to explore a grid-based maze and find the goal using **Q-Learning**, a model-free reinforcement learning algorithm. The agent learns by interacting with the environment, receiving rewards and penalties, and gradually improving its policy through trial and error.

---

## Theory

### Markov Decision Process (MDP)

Reinforcement Learning (RL) problems are formally defined as **Markov Decision Processes** consisting of:

- **States (S)** — All possible configurations the agent can occupy.
- **Actions (A)** — Discrete moves available at each state.
- **Transition Function P(s' | s, a)** — Probability of reaching state s' after taking action a in state s.
- **Reward Function R(s, a, s')** — Immediate scalar feedback received after a transition.
- **Discount Factor γ (gamma)** — Trade-off between immediate and future rewards.

### Q-Learning

Q-Learning is an **off-policy, value-based** RL algorithm that learns the optimal action-value function Q*(s, a) — the maximum expected cumulative discounted reward achievable by taking action a in state s and thereafter following the optimal policy.

The update rule is:

```
Q(s, a) ← (1 - α) · Q(s, a) + α · [ r + γ · max_a' Q(s', a') ]
```

where:
- **α (alpha)** — Learning rate: controls how much new information overrides old information.
- **γ (gamma)** — Discount factor: prioritizes long-term vs. immediate rewards.
- **ε (epsilon)** — Exploration rate: probability of choosing a random action to discover new paths (ε-greedy strategy).

### Exploration vs. Exploitation

- **Exploration** — Trying new actions to discover potentially better paths.
- **Exploitation** — Using known Q-values to choose the best action.

The ε-greedy strategy balances both: with probability ε, a random action is chosen; otherwise, the action with the highest Q-value is selected. ε is typically decayed over time to shift from exploration to exploitation.

### Reward Design in Maze Navigation

| Event | Reward | Purpose |
|---|---|---|
| Step into free cell | -1 | Encourages shortest path |
| Hit a wall | -5 | Penalizes invalid moves |
| Reach goal | +100 | Strong positive signal |

---

## Environment: Grid Maze

The maze is a 10×10 binary grid where:
- `0` = free cell (agent can move here)
- `1` = wall (impassable)

The agent starts at `(0, 0)` (top-left) and must reach `(9, 9)` (bottom-right). At each step, the agent chooses one of four actions: **Up, Down, Left, Right**. If an action would move the agent into a wall or outside the grid, the agent stays in place and receives a penalty.

---

## Algorithm

1. **Initialize** Q-table with zeros (shape: 10×10×4).
2. **For each episode:**
   - Reset agent to start position.
   - Choose action using ε-greedy policy.
   - Execute action, observe reward and next state.
   - Update Q-value using the Bellman equation.
   - Decay ε.
3. **After training**, extract the greedy policy by taking `argmax_a Q(s, a)` at each state.

---

## Flowchart

```mermaid
flowchart TD
    A[Initialize Q-Table<br/>10x10x4 zeros] --> B{For each episode}
    B --> C[Reset agent to START (0,0)]
    C --> D[Choose action: ε-greedy]
    D --> E[Execute action in Maze]
    E --> F{Reached GOAL?}
    F -->|No| G[Compute reward & next state]
    G --> H[Update Q(s,a) via Bellman equation]
    H --> I[Decay ε]
    I --> D
    F -->|Yes| J[Record episode reward & steps]
    J --> B
    B -->|Done| K[Extract greedy policy]
    K --> L[Visualize: Path, Q-values, Learning Curve]
    L --> M[Evaluate policy success rate]
```

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Algorithm | Q-Learning | Simple, model-free, proven on grid worlds |
| Maze size | 10×10 | Small enough for fast training, large enough to require planning |
| Actions | 4 (Up/Down/Left/Right) | Standard cardinal directions |
| Learning rate α | 0.1 | Stable convergence without oscillation |
| Discount γ | 0.99 | Near-optimal long-term planning |
| Initial ε | 1.0 → decays to 0.05 | Full exploration early, exploitation later |
| Episodes | 800 | Sufficient for convergence on this maze |
| Max steps/episode | 200 | Prevents infinite loops in early training |

---

## How to Run

```bash
cd Assignment9
python Code.py
```

This produces three output files in the `Assignment9/` directory:
- `maze_rl_results.png` — Maze with learned path, max Q-value heatmap, and reward curve
- `maze_learning_curve.png` — Smoothed learning curve showing convergence

---

## Outputs

```
Saved maze_rl_results.png
Saved maze_learning_curve.png

Learned Policy Evaluation (greedy):
  Average steps to goal : 17.0
  Success rate          : 100.0%
  Shortest path length  : 17 steps
  Final epsilon         : 0.0500
```

> Values above are representative; results may vary slightly due to randomness in the initial Q-table and exploration, though `random_state` is fixed by using a deterministic environment.

---

## Hyperparameters

All hyperparameters are set inline in `Code.py`:

| Parameter | Value | Description |
|---|---|---|
| `alpha` | 0.1 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon` | 1.0 → 0.05 | Exploration rate (initial → min) |
| `decay` | 0.995 | Per-episode epsilon decay multiplier |
| `num_episodes` | 800 | Total training episodes |
| `max_steps` | 200 | Max steps per episode |

---

## 10 Most Likely Questions & Answers

**Q1. What is the difference between on-policy and off-policy RL?**
On-policy methods (like SARSA) learn the value of the policy being followed. Off-policy methods (like Q-Learning) learn the optimal policy independently of the agent's current behavior — it can learn from exploratory actions while still targeting the greedy policy.

**Q2. Why does the agent sometimes bounce off walls early in training?**
Early in training, ε is high (≈1.0), so the agent chooses random actions frequently. Random actions can select moves into walls, resulting in penalties and zero progress. As ε decays, this behavior diminishes.

**Q3. What does a negative Q-value mean?**
A negative Q-value for a state-action pair means that, on average, taking that action leads to a net negative return. This is expected in mazes with step penalties: the agent incurs -1 per step until it reaches the goal (+100).

**Q4. Why is gamma set close to 1?**
With γ=0.99, the agent strongly prioritizes long-term rewards. This encourages finding the shortest path to the goal rather than wandering aimlessly. A lower γ (e.g., 0.5) would make the agent short-sighted and likely get stuck in local minima.

**Q5. What is the exploration-exploitation trade-off?**
The agent must explore unknown states to discover rewards (exploration) while also leveraging known good actions to maximize returns (exploitation). ε-greedy balances this by gradually reducing randomness.

**Q6. How does the Q-table generalize to unseen states?**
Q-Learning is a tabular method — it only stores values for states explicitly visited during training. In larger or continuous environments, function approximation (e.g., deep Q-networks) is used instead.

**Q7. Why use a 10×10 maze instead of a smaller one?**
A 10×10 maze with multiple obstacles requires the agent to plan a multi-step route. Too small (e.g., 3×3) and the problem is trivial; too large and training becomes prohibitively slow without function approximation.

**Q8. What happens if epsilon is never decayed?**
The agent continues exploring indefinitely, leading to suboptimal performance at evaluation time. The learned policy never fully converges to the greedy optimal policy.

**Q9. Can Q-Learning get stuck in local optima?**
In deterministic grid worlds with proper exploration, Q-Learning converges to the global optimum. Stochasticity or insufficient exploration can cause convergence to suboptimal policies.

**Q10. How is the optimal path extracted after training?**
After training, we run the agent greedily (ε=0) from the start, always choosing `argmax_a Q(s, a)` at each step. The sequence of states visited forms the learned optimal path, visualized in `maze_rl_results.png`.
