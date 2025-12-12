import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import cv2

# ==========================
# Nature CNN from DQN Nature Paper
# ==========================
class NatureCNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # stacked frames
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.net(x / 255.0)


# ==========================
# Replay Buffer
# ==========================
class ReplayBuffer:
    def __init__(self, size=100000):
        self.buffer = deque(maxlen=size)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch=32):
        batch = random.sample(self.buffer, batch)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r),
                np.array(s2), np.array(d))

    def __len__(self):
        return len(self.buffer)


# ==========================
# Preprocessing
# ==========================
def preprocess(obs):
    """Convert (96x96x3) to grayscale 84x84."""
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs


# ==========================
# Discretized Action Space
# ==========================
DISCRETE_ACTIONS = [
    np.array([0.0, 1.0, 0.0]),   # gas
    np.array([0.0, 0.0, 0.8]),   # brake
    np.array([-1.0, 0.6, 0.0]),  # left & gas
    np.array([1.0, 0.6, 0.0]),   # right & gas
    np.array([0.0, 0.0, 0.0])    # nothing
]


# ==========================
# Main Training Loop
# ==========================
def train_dqn(episodes=2000):

    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_actions = len(DISCRETE_ACTIONS)

    q_net = NatureCNN(num_actions).to(device)
    target_net = NatureCNN(num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    replay = ReplayBuffer()

    gamma = 0.99
    batch_size = 32
    train_start = 20000
    target_update_freq = 10000

    global_step = 0
    eps_start, eps_final, eps_decay = 1.0, 0.05, 400000

    # Initial frame buffer
    def init_stack(frame):
        return np.stack([frame] * 4, axis=0)

    for ep in range(episodes):
        obs, _ = env.reset()
        frame = preprocess(obs)
        state = init_stack(frame)

        ep_reward = 0

        done = False
        while not done:
            global_step += 1

            # Epsilon decay
            eps = eps_final + (eps_start - eps_final) * np.exp(-global_step / eps_decay)

            # Select action
            if random.random() < eps:
                action_idx = random.randint(0, num_actions - 1)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    action_idx = q_net(s).argmax().item()

            action = DISCRETE_ACTIONS[action_idx]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_frame = preprocess(next_obs)
            next_state = np.roll(state, -1, axis=0)
            next_state[-1] = next_frame

            replay.push(state, action_idx, reward, next_state, done)

            state = next_state
            ep_reward += reward

            # Training step
            if len(replay) > train_start:
                s, a, r, s2, d = replay.sample(batch_size)

                s = torch.tensor(s, dtype=torch.float32).to(device)
                a = torch.tensor(a, dtype=torch.long).to(device)
                r = torch.tensor(r, dtype=torch.float32).to(device)
                s2 = torch.tensor(s2, dtype=torch.float32).to(device)
                d = torch.tensor(d, dtype=torch.float32).to(device)

                q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                next_q = target_net(s2).max(1)[0]
                target = r + gamma * next_q * (1 - d)

                loss = nn.MSELoss()(q_vals, target.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target net
                if global_step % target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {ep} | Reward: {ep_reward:.1f} | Epsilon: {eps:.3f}")


if __name__ == "__main__":
    train_dqn()
