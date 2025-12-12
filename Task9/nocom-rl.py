import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import torch.optim as optim
import random
from collections import deque
import cv2
import os
import csv
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Discrete action mapping ---
DISCRETE_ACTIONS = [
    np.array([0, 0, 0], dtype=np.float32),   # no-op
    np.array([-1, 0, 0], dtype=np.float32),  # left
    np.array([1, 0, 0], dtype=np.float32),   # right
    np.array([0, 1, 0], dtype=np.float32),   # gas
    np.array([0, 0, 0.8], dtype=np.float32)  # brake
]

# --- Preprocessing ---
def preprocess(obs):
    obs = obs.transpose(2, 0, 1)  # HWC -> CHW
    obs = torch.tensor(obs, dtype=torch.float32) / 255.0
    return obs

# --- Actor-Critic ---
class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        C, H, W = obs_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        with torch.no_grad():
            flat_size = self.cnn(torch.zeros(1, C, H, W)).view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512), nn.ReLU()
        )
        self.actor_logits = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, obs):
        x = self.cnn(obs).view(obs.size(0), -1)
        x = self.fc(x)
        logits = self.actor_logits(x)
        value = self.critic(x)
        return logits, value

    def act(self, obs):
        obs = obs.unsqueeze(0).to(device)
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        return action_idx.item(), log_prob.squeeze(0), value.squeeze(0)

# --- PPO Agent ---
class PPOAgent():
    def __init__(self, policy, optimizer, gamma=0.99, lam=0.95, eps_clip=0.2):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip

    def compute_returns(self, rewards, values, dones):
        returns = []
        gae = 0
        next_value = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return returns

    def update(self, batch, epochs=4, batch_size=64):
        obs_tensor = torch.stack(batch['obs']).to(device)
        actions_tensor = torch.tensor(batch['actions'], dtype=torch.long, device=device)
        returns_tensor = torch.tensor(batch['returns'], dtype=torch.float32, device=device)
        old_log_probs_tensor = torch.tensor(batch['log_probs'], dtype=torch.float32, device=device)
        values_tensor = torch.tensor(batch['values'], dtype=torch.float32, device=device)

        advantage = returns_tensor - values_tensor
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        dataset_size = len(batch['obs'])
        losses = {'actor': 0, 'critic': 0, 'total':0}
        count = 0

        for _ in range(epochs):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                mb_obs = obs_tensor[idx]
                mb_actions = actions_tensor[idx]
                mb_adv = advantage[idx]
                mb_returns = returns_tensor[idx]
                mb_old_log_probs = old_log_probs_tensor[idx]

                logits_pred, value_pred = self.policy(mb_obs)
                value_pred = value_pred.squeeze()
                dist = torch.distributions.Categorical(logits=logits_pred)
                log_probs = dist.log_prob(mb_actions)

                ratio = torch.exp(log_probs - mb_old_log_probs)
                clip_adv = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv
                loss_actor = -torch.min(ratio * mb_adv, clip_adv).mean()
                loss_critic = F.mse_loss(mb_returns, value_pred)
                entropy = dist.entropy().mean()
                loss = loss_actor + 0.5 * loss_critic - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses['actor'] += loss_actor.item()
                losses['critic'] += loss_critic.item()
                losses['total'] += loss.item()
                count += 1

        for key in losses:
            losses[key] /= count
        return losses

# --- Training Loop ---
def run_PPO(env_name="CarRacing-v3", episodes=1000):
    env = gym.make(env_name, continuous=False)
    sample_obs, _ = env.reset()
    sample_obs = preprocess(sample_obs)
    obs_shape = sample_obs.shape
    action_dim = len(DISCRETE_ACTIONS)

    policy = ActorCritic(obs_shape, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    agent = PPOAgent(policy, optimizer)

    batch_size = 2048
    mini_batch_size = 64
    update_epochs = 10

    batch = {'obs': [], 'actions':[], 'rewards':[], 'values':[], 'dones':[], 'log_probs':[]}
    steps_collected = 0
    obs, _ = env.reset()
    obs = preprocess(obs)

    for episode in range(episodes):
        episode_reward = 0
        done = False
        while not done:
            action_idx, log_prob, value = policy.act(obs)
            action_np = DISCRETE_ACTIONS[action_idx]
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            next_obs = preprocess(next_obs)
            done = terminated or truncated

            batch['obs'].append(obs)
            batch['actions'].append(action_idx)
            batch['rewards'].append(reward)
            batch['values'].append(value.cpu().item())
            batch['dones'].append(done)
            batch['log_probs'].append(log_prob.item())

            episode_reward += reward
            steps_collected += 1
            obs = next_obs

            if done:
                obs, _ = env.reset()
                obs = preprocess(obs)
                print(f"Episode {episode} reward: {episode_reward:.2f}")
                episode_reward = 0
                break

        if steps_collected >= batch_size:
            returns = agent.compute_returns(batch['rewards'], batch['values'], batch['dones'])
            batch['returns'] = returns
            agent.update(batch, epochs=update_epochs, batch_size=mini_batch_size)
            batch = {k: [] for k in batch}
            steps_collected = 0

# ====================DQN===============================================================

def init_log(log_file="dqn_log.csv"):
    # If the file doesn't exist, create it with header
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "epsilon", "loss"])
            
def log_results(episode, reward, epsilon, loss, log_file="dqn_log.csv"):
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, reward, epsilon, loss])
          
def preprocess_DQN(obs):
    obs = obs.copy()   
    # turns green(grass) in background to black.
    green_mask = obs[:,:,1]>150 # high green channel
    obs[green_mask] = 0
    # Convert to grayscale
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0

class DQN(nn.Module):
    # Basic CNN neural network.
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        c, h, w = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),  # change here
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque( maxlen = capacity)
    
    def push(self, state, action, reward, next_state, done):
        # takes in all the actions of the agent. 
        self.buffer.append((
            np.array(state, copy=True, dtype=np.float32),
            action,
            reward,
            np.array(next_state, copy=True, dtype=np.float32),
            done
        ))
        
    def sample(self, batch_size):
        # randomly gets batches based on random size
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # convert each field to stacked numpy arrays
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32)
        )
    def __len__(self):
        return len(self.buffer)

class FrameStack:
    def __init__(self, k):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, obs):
        frame = preprocess_DQN(obs)  # already float32 normalized
        for _ in range(self.k):
            self.frames.append(frame.copy())
        return self._get_stack()

    def step(self, obs):
        frame = preprocess_DQN(obs)
        self.frames.append(frame.copy())
        return self._get_stack()

    def _get_stack(self):
        # Returns np.array shape (k, 84, 84) dtype float32
        return np.stack(self.frames, axis=0)

def run_DQN(env_name="CarRacing-v3", episodes=1000):

    env = gym.make(env_name, continuous=False, render_mode="rgb_array")

    ACTIONS = [
        [0.0, 1.0, 0.0],
        [-0.3, 1.0, 0.0],
        [0.3, 1.0, 0.0],
        [-0.8, 1.0, 0.0],
        [0.8, 1.0, 0.0]
    ]
    NUM_ACTIONS = len(ACTIONS)

    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY = 50000
    TRAIN_EVERY = 4

    q_net = DQN((4, 84, 84), NUM_ACTIONS).to(device)
    target_net = DQN((4, 84, 84), NUM_ACTIONS).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(50000)

    frame_stack = FrameStack(4)
    step_count = 0

    episode_rewards = []
    losses = []

    log_file = "dqn_log.csv"
    init_log(log_file)

    for episode in range(episodes):
        obs, _ = env.reset()
        state_np = frame_stack.reset(obs)
        state = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0).to(device)

        episode_reward = 0
        neg_reward_count = 0
        done = False
        episode_loss = []

        while not done:
            step_count += 1
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-step_count / EPS_DECAY)

            # Epsilon-greedy
            if random.random() < epsilon:
                action_idx = random.randrange(NUM_ACTIONS)
            else:
                with torch.no_grad():
                    action_idx = q_net(state).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(np.array(ACTIONS[action_idx], dtype=np.float32))

            done = terminated or truncated

            # Negative reward counter
            if reward < -0.05:
                neg_reward_count += 1
            else:
                neg_reward_count = 0
            if neg_reward_count > 25:
                done = True

            next_state_np = frame_stack.step(next_obs)
            next_state = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0).to(device)

            replay_buffer.push(state_np, action_idx, reward, next_state_np, done)

            state = next_state
            state_np = next_state_np
            episode_reward += reward

            # TRAIN
            if step_count % TRAIN_EVERY == 0 and len(replay_buffer) > BATCH_SIZE:
                s, a, r, s_next, d = replay_buffer.sample(BATCH_SIZE)
                s      = torch.tensor(s, dtype=torch.float32).to(device)
                s_next = torch.tensor(s_next, dtype=torch.float32).to(device)
                a      = torch.tensor(a, dtype=torch.int64).to(device)
                r      = torch.tensor(r, dtype=torch.float32).to(device)
                d      = torch.tensor(d, dtype=torch.float32).to(device)

                q = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    best = q_net(s_next).argmax(1).unsqueeze(1)
                    next_q = target_net(s_next).gather(1, best).squeeze()
                    target = r + GAMMA * next_q * (1 - d)

                loss = nn.SmoothL1Loss()(q, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

                episode_loss.append(loss.item())

                # Soft update
                tau = 0.005
                for tgt, src in zip(target_net.parameters(), q_net.parameters()):
                    tgt.data.copy_(tau * src.data + (1 - tau) * tgt.data)

        avg_loss = np.mean(episode_loss) if episode_loss else 0
        print(f"Episode {episode} | Reward: {episode_reward:.1f} | Epsilon: {epsilon:.3f} | Loss: {avg_loss:.5f}")

        log_results(episode, episode_reward, epsilon, avg_loss, log_file)





if __name__ == "__main__":
    run_PPO(episodes=10)  # or run_PPO(...) if you want to test PPO
    run_DQN(episodes=10)