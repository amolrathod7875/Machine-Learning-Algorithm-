import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# =============================================================================
# Assignment 10 — Tic-Tac-Toe using Reinforcement Learning (Q-Learning)
# =============================================================================

# ============================
# Part A: Setting up the environment
# ============================

class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9, dtype=int)   # 0=empty, 1=X (agent), -1=O (opponent)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self._get_state()

    def _get_state(self):
        return tuple(self.board)

    def get_valid_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def check_win(self, player):
        lines = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6],
        ]
        return any(all(self.board[i] == player for i in line) for line in lines)

    def is_draw(self):
        return len(self.get_valid_actions()) == 0 and not self.check_win(1) and not self.check_win(-1)

    def step(self, action):
        if self.done:
            raise ValueError("Game already finished")
        if self.board[action] != 0:
            raise ValueError(f"Invalid action {action}")

        self.board[action] = self.current_player

        if self.check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1.0 if self.current_player == 1 else -1.0
            return self._get_state(), reward, self.done, {}

        if self.is_draw():
            self.done = True
            self.winner = 0
            return self._get_state(), 0.5, self.done, {}

        self.current_player *= -1
        return self._get_state(), -0.01, self.done, {}


# ============================
# Part B: Defining the Tic-Tac-Toe game (rules + helpers)
# ============================

def print_board(board):
    symbols = {1: "X", -1: "O", 0: "_"}
    for i in range(3):
        print(" ".join(symbols[board[3 * i + j]] for j in range(3)))
    print()


def check_winner(board):
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6],
    ]
    for line in lines:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]
    return 0


# ============================
# Part C: Building the reinforcement learning model (Q-Learning)
# ============================

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, decay=0.9995):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.q_table = defaultdict(lambda: np.zeros(9))

    def choose_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_values = self.q_table[state]
        return max(valid_actions, key=lambda a: q_values[a])

    def update(self, state, action, reward, next_state, next_valid_actions, done):
        q = self.q_table[state]
        next_q = self.q_table[next_state]
        if done or not next_valid_actions:
            target = reward
        else:
            best_next = max(next_q[a] for a in next_valid_actions)
            target = reward + self.gamma * best_next
        q[action] += self.alpha * (target - q[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)


# ============================
# Part D: Training the model
# ============================

def train(agent, episodes=5000, opponent="random"):
    env = TicTacToeEnv()
    win_rates, draw_rates, loss_rates = [], [], []

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid)
            next_state, _, done_after_agent, _ = env.step(action)

            if done_after_agent:
                reward = 1.0 if env.winner == 1 else (0.5 if env.winner == 0 else -1.0)
                agent.update(state, action, reward, next_state, [], True)
                break

            if opponent == "random":
                opp_action = random.choice(env.get_valid_actions())
            elif opponent == "minimax":
                opp_action = minimax_move(env.board.copy(), -1)
            else:
                raise ValueError("Unknown opponent")

            opp_state, _, done_after_opp, _ = env.step(opp_action)

            if done_after_opp:
                reward = 1.0 if env.winner == 1 else (-1.0 if env.winner == -1 else 0.5)
                agent.update(state, action, reward, opp_state, [], True)
            else:
                reward = -0.01
                next_valid = env.get_valid_actions()
                agent.update(state, action, reward, opp_state, next_valid, False)

            state = opp_state
            done = done_after_opp

        agent.decay_epsilon()

        if (ep + 1) % 500 == 0:
            wr, dr, lr = evaluate(agent, num_games=200, opponent="random")
            win_rates.append(wr)
            draw_rates.append(dr)
            loss_rates.append(lr)
            print(f"Episode {ep+1:5d} | epsilon={agent.epsilon:.4f} | Win={wr:.2%} Draw={dr:.2%} Loss={lr:.2%}")

    return win_rates, draw_rates, loss_rates


# ============================
# Part E: Testing the model
# ============================

def evaluate(agent, num_games=500, opponent="random", verbose=False):
    env = TicTacToeEnv()
    prev_eps = agent.epsilon
    agent.epsilon = 0.0

    wins = draws = losses = 0

    for _ in range(num_games):
        state = env.reset()
        done = False

        while not done:
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid)
            next_state, reward, done_after_agent, _ = env.step(action)

            if done_after_agent:
                if reward == 1.0:
                    wins += 1
                elif reward == 0.5:
                    draws += 1
                else:
                    losses += 1
                break

            if opponent == "random":
                opp_action = random.choice(env.get_valid_actions())
            elif opponent == "minimax":
                opp_action = minimax_move(env.board.copy(), -1)
            else:
                raise ValueError("Unknown opponent")

            opp_state, opp_reward, done_after_opp, _ = env.step(opp_action)

            if done_after_opp:
                if env.winner == 1:
                    wins += 1
                elif env.winner == -1:
                    losses += 1
                else:
                    draws += 1
                done = True
            else:
                state = opp_state

    agent.epsilon = prev_eps

    if verbose:
        print(f"\nTest Results ({num_games} games vs {opponent}):")
        print(f"  Wins  : {wins:4d} ({wins/num_games:.2%})")
        print(f"  Draws : {draws:4d} ({draws/num_games:.2%})")
        print(f"  Losses: {losses:4d} ({losses/num_games:.2%})")

    return wins / num_games, draws / num_games, losses / num_games


# ============================
# Minimax opponent (for testing)
# ============================

def minimax_move(board, player):
    def minimax(board_state, depth, is_maximizing, alpha, beta):
        winner = check_winner(board_state)
        if winner == 1:
            return 10 - depth
        if winner == -1:
            return depth - 10
        if 0 not in board_state:
            return 0

        valid = [i for i in range(9) if board_state[i] == 0]

        if is_maximizing:
            best = -float("inf")
            for action in valid:
                board_state[action] = 1
                val = minimax(board_state, depth + 1, False, alpha, beta)
                board_state[action] = 0
                best = max(best, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best
        else:
            best = float("inf")
            for action in valid:
                board_state[action] = -1
                val = minimax(board_state, depth + 1, True, alpha, beta)
                board_state[action] = 0
                best = min(best, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best

    valid = [i for i in range(9) if board[i] == 0]
    best_action = valid[0]

    if player == 1:
        best_value = -float("inf")
        for action in valid:
            board[action] = player
            value = minimax(board, 0, False, -float("inf"), float("inf"))
            board[action] = 0
            if value > best_value:
                best_value = value
                best_action = action
    else:
        best_value = float("inf")
        for action in valid:
            board[action] = player
            value = minimax(board, 0, True, -float("inf"), float("inf"))
            board[action] = 0
            if value < best_value:
                best_value = value
                best_action = action

    return best_action


# ============================
# Main execution
# ============================

if __name__ == "__main__":
    print("=" * 60)
    print("Tic-Tac-Toe Reinforcement Learning (Q-Learning)")
    print("=" * 60)

    # Part A: Environment setup
    env = TicTacToeEnv()
    print("\n[A] Environment ready. Sample empty board:")
    print_board(env.board)

    # Part C: Build model
    agent = QLearningAgent(alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, decay=0.9995)
    print("[C] Q-Learning agent initialized.")
    print(f"    alpha={agent.alpha}, gamma={agent.gamma}, epsilon={agent.epsilon:.2f} -> {agent.epsilon_min}")

    # Part D: Training
    print("\n[D] Training against random opponent (5000 episodes)...")
    win_rates, draw_rates, loss_rates = train(agent, episodes=5000, opponent="random")

    # Part E: Testing
    print("\n[E] Testing trained agent...")
    wr, dr, lr = evaluate(agent, num_games=500, opponent="random", verbose=True)
    wr2, dr2, lr2 = evaluate(agent, num_games=200, opponent="minimax", verbose=True)

    # Visualization
    episodes = list(range(500, 5001, 500))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(episodes, [w * 100 for w in win_rates], marker="o", label="Win %")
    ax.plot(episodes, [d * 100 for d in draw_rates], marker="s", label="Draw %")
    ax.plot(episodes, [l * 100 for l in loss_rates], marker="^", label="Loss %")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Training Progress vs Random Opponent")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    categories = ["Random", "Minimax"]
    wins = [wr * 100, wr2 * 100]
    draws = [dr * 100, dr2 * 100]
    losses = [lr * 100, lr2 * 100]
    x = np.arange(len(categories))
    width = 0.25

    ax = axes[1]
    ax.bar(x - width, wins, width, label="Win")
    ax.bar(x, draws, width, label="Draw")
    ax.bar(x + width, losses, width, label="Loss")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Final Performance vs Opponents")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("tictactoe_results.png", dpi=150)
    print("\nSaved tictactoe_results.png")

    # Demo game
    print("\n" + "=" * 60)
    print("Demo: Trained Agent (X) vs Random Opponent (O)")
    print("=" * 60)
    env_demo = TicTacToeEnv()
    state = env_demo.reset()
    print("Initial board:")
    print_board(env_demo.board)

    step_count = 0
    while not env_demo.done:
        valid = env_demo.get_valid_actions()
        if env_demo.current_player == 1:
            action = agent.choose_action(state, valid)
            player_name = "Agent (X)"
        else:
            action = random.choice(valid)
            player_name = "Opponent (O)"

        state, reward, done, _ = env_demo.step(action)
        step_count += 1
        print(f"Step {step_count}: {player_name} plays position {action}")
        print_board(env_demo.board)

        if env_demo.done:
            break

    if env_demo.winner == 1:
        print("Result: Agent (X) wins!")
    elif env_demo.winner == -1:
        print("Result: Opponent (O) wins!")
    else:
        print("Result: Draw!")
