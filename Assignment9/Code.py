import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import random

MAZE = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
])

START = (0, 0)
GOAL = (9, 9)

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]
NUM_ACTIONS = len(ACTIONS)

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.05
decay = 0.995
num_episodes = 800
max_steps = 200

q_table = np.zeros((MAZE.shape[0], MAZE.shape[1], NUM_ACTIONS))


def is_valid(state):
    r, c = state
    return 0 <= r < MAZE.shape[0] and 0 <= c < MAZE.shape[1] and MAZE[r, c] == 0


def step(state, action_idx):
    dr, dc = ACTIONS[action_idx]
    nr, nc = state[0] + dr, state[1] + dc
    next_state = (nr, nc)
    if not is_valid(next_state):
        next_state = state
        reward = -5
    elif next_state == GOAL:
        reward = 100
    else:
        reward = -1
    done = next_state == GOAL
    return next_state, reward, done


rewards_per_episode = []
steps_per_episode = []

for ep in range(num_episodes):
    state = START
    total_reward = 0
    steps = 0

    for _ in range(max_steps):
        if random.random() < epsilon:
            action = random.randint(0, NUM_ACTIONS - 1)
        else:
            action = np.argmax(q_table[state[0], state[1]])

        next_state, reward, done = step(state, action)
        r, c = state
        nr, nc = next_state

        best_next = np.max(q_table[nr, nc])
        q_table[r, c, action] = (1 - alpha) * q_table[r, c, action] + alpha * (reward + gamma * best_next)

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            break

    epsilon = max(epsilon_min, epsilon * decay)
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps)


def get_learned_path():
    path = [START]
    state = START
    visited = set([state])
    for _ in range(200):
        action = np.argmax(q_table[state[0], state[1]])
        next_state, _, done = step(state, action)
        if next_state in visited:
            break
        visited.add(next_state)
        path.append(next_state)
        state = next_state
        if done:
            break
    return path


learned_path = get_learned_path()
path_set = set(learned_path)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cmap = ListedColormap(["white", "black"])
ax = axes[0]
ax.imshow(MAZE, cmap=cmap, interpolation="none")
ax.set_xticks(np.arange(MAZE.shape[1]))
ax.set_yticks(np.arange(MAZE.shape[0]))
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.set_title("Maze Environment (1=Wall, 0=Free)")

for r in range(MAZE.shape[0]):
    for c in range(MAZE.shape[1]):
        if MAZE[r, c] == 1:
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, color="black", alpha=0.6))

ax.plot(START[1], START[0], "gs", markersize=14, label="Start")
ax.plot(GOAL[1], GOAL[0], "r*", markersize=18, label="Goal")
path_r, path_c = zip(*learned_path)
ax.plot(path_c, path_r, "b-", linewidth=2, alpha=0.8, label="Learned Path")
ax.plot(path_c, path_r, "bo", markersize=4)
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Learned Optimal Path")

ax = axes[1]
ax.plot(rewards_per_episode, color="steelblue", linewidth=1.2)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward")
ax.set_title("Reward per Episode (Q-Learning)")
ax.grid(True, linestyle="--", alpha=0.5)

ax = axes[2]
best_q = np.max(q_table, axis=2)
im = ax.imshow(best_q, cmap="viridis", interpolation="none")
ax.set_xticks(np.arange(MAZE.shape[1]))
ax.set_yticks(np.arange(MAZE.shape[0]))
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for r in range(MAZE.shape[0]):
    for c in range(MAZE.shape[1]):
        if MAZE[r, c] == 1:
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True, color="black", alpha=0.5))
        else:
            ax.text(c, r, f"{best_q[r,c]:.1f}", ha="center", va="center", fontsize=6, color="white" if best_q[r,c] < 0.5*best_q.max() else "black")
ax.set_title("Max Q-Value per State")
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig("maze_rl_results.png", dpi=150)
print("Saved maze_rl_results.png")

window = 50
moving_avg = [np.mean(rewards_per_episode[max(0, i-window+1):i+1]) for i in range(len(rewards_per_episode))]
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, color="lightblue", linewidth=1, label="Episode Reward")
plt.plot(moving_avg, color="darkblue", linewidth=2, label=f"{window}-Episode Moving Avg")
plt.axhline(y=np.mean(rewards_per_episode[-window:]), color="red", linestyle="--", label=f"Final Avg Reward: {np.mean(rewards_per_episode[-window:]):.1f}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Convergence on Maze")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("maze_learning_curve.png", dpi=150)
print("Saved maze_learning_curve.png")

def evaluate_policy(num_trials=20):
    total_steps = []
    successes = 0
    for _ in range(num_trials):
        state = START
        steps = 0
        for _ in range(max_steps):
            action = np.argmax(q_table[state[0], state[1]])
            next_state, _, done = step(state, action)
            steps += 1
            state = next_state
            if done:
                successes += 1
                break
        total_steps.append(steps)
    return np.mean(total_steps), successes / num_trials

avg_steps, success_rate = evaluate_policy()
print(f"\nLearned Policy Evaluation (greedy):")
print(f"  Average steps to goal : {avg_steps:.1f}")
print(f"  Success rate          : {success_rate*100:.1f}%")
print(f"  Shortest path length  : {len(learned_path)-1} steps")
print(f"  Final epsilon         : {epsilon:.4f}")
