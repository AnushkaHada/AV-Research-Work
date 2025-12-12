import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# ============================================================
#  Nature CNN Feature Extractor (as in DQN Nature paper)
# ============================================================

class NatureCNN(nn.Module):
    def __init__(self, input_channels=3, feature_dim=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),  # -> 32×(24×24)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),              # -> 64×(11×11)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),              # -> 64×(9×9)
            nn.ReLU(),
            nn.Flatten()
        )

        # compute CNN output size (lazy)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 96, 96)
            n_flat = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x / 255.0  # pixel normalization
        return self.linear(self.cnn(x))


# ============================================================
#  Actor-Critic using Nature CNN Encoder
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, action_dim, feature_dim=512):
        super().__init__()
        self.encoder = NatureCNN()

        self.actor_mean = nn.Linear(feature_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Linear(feature_dim, 1)

    def forward(self, obs):
        features = self.encoder(obs)
        return features

    def act(self, obs):
        features = self.encoder(obs)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        return action, logprob, dist

    def value(self, obs):
        features = self.encoder(obs)
        return self.critic(features)


# ============================================================
#  PPO Implementation
# ============================================================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t < len(rewards) - 1 else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * (1 - dones[t]) * lastgaelam
    returns = advantages + values
    return advantages, returns


# ============================================================
#  Training Loop
# ============================================================

def train_ppo(
    env_id="CarRacing-v3",
    total_steps=500_000,
    rollout_steps=2048,
    batch_size=64,
    epochs=10,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    lr=3e-4
):
    env = gym.make(env_id, continuous=True, render_mode=None)

    obs, _ = env.reset()
    obs_shape = obs.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCritic(action_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(0, total_steps, rollout_steps):
        # ---------- Collect Rollout ----------
        obses = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        values = []

        for _ in range(rollout_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            value = model.value(obs_t).item()

            action, logprob, dist = model.act(obs_t)
            action_np = action.cpu().numpy()[0]

            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            obses.append(obs)
            actions.append(action_np)
            logprobs.append(logprob.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            obs = next_obs
            if done:
                obs, _ = env.reset()

        values = np.array(values)
        advantages, returns = compute_gae(np.array(rewards), values, np.array(dones), gamma, lam)

        # ---------- Convert to tensors ----------
        obses = torch.tensor(np.array(obses), dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        old_logprobs = torch.tensor(logprobs, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---------- PPO Update ----------
        dataset_size = len(obses)
        for _ in range(epochs):
            idx = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                batch_idx = idx[start:start+batch_size]

                batch_obs = obses[batch_idx]
                batch_actions = actions[batch_idx]
                batch_oldlog = old_logprobs[batch_idx]
                batch_adv = advantages[batch_idx]
                batch_ret = returns[batch_idx]

                features = model(batch_obs)
                mean = model.actor_mean(features)
                std = torch.exp(model.actor_logstd)
                dist = Normal(mean, std)

                new_logprob = dist.log_prob(batch_actions).sum(-1)
                ratio = (new_logprob - batch_oldlog).exp()

                # PPO clip loss
                actor_loss = -torch.min(
                    ratio * batch_adv,
                    torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch_adv
                ).mean()

                critic_value = model.critic(features).squeeze()
                critic_loss = ((critic_value - batch_ret) ** 2).mean()

                entropy = dist.entropy().sum(-1).mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Step {step}/{total_steps} | Loss={loss.item():.3f}")

    env.close()


# ============================================================
#  Run Training
# ============================================================

if __name__ == "__main__":
    train_ppo()
