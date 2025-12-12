# dqn_carracing.py
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# ============================================================
# DQN LOGGER — saves CSV logs (rewards, losses, Q statistics)
# ============================================================
class DQNLogger:
    def __init__(self, logdir="logs_dqn_carracing"):
        os.makedirs(logdir, exist_ok=True)

        self.reward_file = os.path.join(logdir, "episode_rewards.csv")
        self.loss_file   = os.path.join(logdir, "losses.csv")
        self.q_file      = os.path.join(logdir, "q_values.csv")

        # Create CSV with headers
        if not os.path.exists(self.reward_file):
            with open(self.reward_file, "w", newline="") as f:
                csv.writer(f).writerow(["episode", "reward"])

        if not os.path.exists(self.loss_file):
            with open(self.loss_file, "w", newline="") as f:
                csv.writer(f).writerow(["step", "td_loss"])

        if not os.path.exists(self.q_file):
            with open(self.q_file, "w", newline="") as f:
                csv.writer(f).writerow(["step", "avg_q"])

    def log_reward(self, episode, reward):
        with open(self.reward_file, "a", newline="") as f:
            csv.writer(f).writerow([episode, reward])

    def log_loss(self, step, loss):
        with open(self.loss_file, "a", newline="") as f:
            csv.writer(f).writerow([step, loss])

    def log_q_values(self, step, avg_q):
        with open(self.q_file, "a", newline="") as f:
            csv.writer(f).writerow([step, avg_q])



# ============================================================
# Discretization Wrapper for CarRacing-v3
# ============================================================
class DiscretizedCarRacing(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = [
            np.array([0.0, 0.0, 0.0]),   # no-op
            np.array([-1.0, 0.0, 0.0]),  # left
            np.array([1.0, 0.0, 0.0]),   # right
            np.array([0.0, 1.0, 0.0]),   # gas
            np.array([0.0, 0.0, 1.0])    # brake
        ]
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, a):
        return self.actions[a]


# ============================================================
# Nature CNN
# ============================================================
class NatureCNN(nn.Module):
    def __init__(self, input_shape=(3, 96, 96)):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU()
        )
        self.output_dim = 512

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ============================================================
# DQN Network
# ============================================================
class DQN(nn.Module):
    def __init__(self, num_actions, input_shape=(3, 96, 96)):
        super().__init__()
        self.feature_extractor = NatureCNN(input_shape)
        self.q_head = nn.Linear(self.feature_extractor.output_dim, num_actions)

    def forward(self, x):
        feat = self.feature_extractor(x)
        return self.q_head(feat)


# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, n_s, d):
        self.buffer.append((s, a, r, n_s, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, n_s, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a),
            np.array(r),
            np.stack(n_s),
            np.array(d)
        )

    def __len__(self):
        return len(self.buffer)



# ============================================================
# DQN Training Function (with LOGGING)
# ============================================================
def run_DQN(
        episodes=400,
        batch_size=32,
        gamma=0.99,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=150000,
        target_update_interval=2000,
        replay_start=5000,
        max_steps=500000
    ):
    logger = DQNLogger("logs_dqn_carracing")

    env = DiscretizedCarRacing(gym.make("CarRacing-v3", continuous=False))
    obs, _ = env.reset()
    num_actions = env.action_space.n

    policy_net = DQN(num_actions).to(device)
    target_net = DQN(num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay = ReplayBuffer()

    epsilon = epsilon_start
    total_steps = 0

    def preprocess(o):
        return torch.tensor(o, dtype=torch.float32).permute(2, 0, 1)

    for ep in range(episodes):
        obs, _ = env.reset()
        state = preprocess(obs).numpy()
        ep_reward = 0

        while True:
            total_steps += 1

            # ε-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    t = torch.tensor(state).unsqueeze(0).to(device)
                    q = policy_net(t)
                    action = q.argmax(dim=1).item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess(next_obs).numpy()
            done = terminated or truncated

            ep_reward += reward
            replay.push(state, action, reward, next_state, done)
            state = next_state

            # Epsilon decay
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      np.exp(-total_steps / epsilon_decay)

            # Train when enough samples
            if len(replay) > replay_start:
                states, actions, rewards, next_states, dones = replay.sample(batch_size)

                states = torch.tensor(states).float().to(device)
                actions = torch.tensor(actions).long().to(device)
                rewards = torch.tensor(rewards).float().to(device)
                next_states = torch.tensor(next_states).float().to(device)
                dones = torch.tensor(dones).float().to(device)

                # Q(s,a)
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # target Q(s')
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    q_target = rewards + gamma * next_q * (1 - dones)

                loss = nn.MSELoss()(q_values, q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # LOGGING
                logger.log_loss(total_steps, loss.item())
                logger.log_q_values(total_steps, q_values.mean().item())

            # Update target network
            if total_steps % target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done or total_steps >= max_steps:
                print(f"Episode {ep} | Reward = {ep_reward:.1f} | ε={epsilon:.3f}")
                logger.log_reward(ep, ep_reward)
                break

        if total_steps >= max_steps:
            break

    env.close()
    torch.save(policy_net.state_dict(), "dqn_carracing.pth")
    print("Training complete — saved model: dqn_carracing.pth")



# ============================================================
# Allow this to run standalone
# ============================================================
if __name__ == "__main__":
    run_DQN()
