import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete # Import to use for checking

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Preprocessing ---
def preprocess(obs):
    # obs is (H, W, C) -> transpose to (C, H, W) for PyTorch CNN
    obs = obs.transpose(2, 0, 1)  
    # Convert to Tensor and normalize
    obs = torch.tensor(obs, dtype=torch.float32) / 255.0
    return obs

# --- Actor-Critic ---
class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        C, H, W = obs_shape
        self.cnn = nn.Sequential(
            # Standard CNN architecture for visual RL (e.g., Atari, CarRacing)
            nn.Conv2d(C, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU()
        )
        
        # Calculate the size of the flattened layer dynamically
        with torch.no_grad():
            flat = self.cnn(torch.zeros(1, C, H, W)).view(1, -1).size(1)
            
        self.fc = nn.Sequential(nn.Linear(flat, 512), nn.ReLU())
        
        # Policy output: Logits for Categorical distribution (Discrete actions)
        self.logits = nn.Linear(512, action_dim) 
        # Value output: Single scalar estimate of state value
        self.value = nn.Linear(512, 1)

    def forward(self, obs):
        x = self.cnn(obs).view(obs.size(0), -1)
        x = self.fc(x)
        return self.logits(x), self.value(x).squeeze(-1)

    def act(self, obs):
        # Adds batch dimension (1, C, H, W) and moves to device
        obs = obs.unsqueeze(0).to(device) 
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        # Returns the action as an integer (0, 1, 2, 3, or 4 for CarRacing-v3 discrete)
        return int(action.item()), dist.log_prob(action).item(), value.item()

# --- PPO Agent ---
class PPOAgent:
    def __init__(self, policy, optimizer, gamma=0.99, lam=0.95, eps_clip=0.2):
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip

    def compute_returns(self, rewards, dones, last_val):
        # Generalized Advantage Estimation (GAE) is implicitly calculated here 
        # by using the discounted returns R, which will be used to calculate Advantage 
        # (A = R - V) during the update.
        R = last_val
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (0.0 if d else 1.0)
            returns.insert(0, R)
        return returns

    def update(self, batch, epochs=4, minibatch=64):
        # Prepare batch tensors
        obs = torch.stack(batch["obs"]).to(device)
        acts = torch.tensor(batch["actions"], device=device)
        old_logp = torch.tensor(batch["logp"], device=device)
        values = torch.tensor(batch["values"], device=device)
        returns = torch.tensor(batch["returns"], device=device)

        # Calculate and normalize advantage
        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = len(obs)
        idxs = np.arange(N)

        for _ in range(epochs):
            np.random.shuffle(idxs)
            for start in range(0, N, minibatch):
                mb = idxs[start:start+minibatch]
                mb_obs = obs[mb]
                mb_act = acts[mb]
                mb_adv = adv[mb]
                mb_ret = returns[mb]
                mb_old_logp = old_logp[mb]

                logits, val = self.policy(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_act)

                # PPO Clipping Loss
                ratio = torch.exp(logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_adv

                loss_actor = -torch.min(surr1, surr2).mean() # Maximize this
                loss_critic = F.mse_loss(mb_ret, val) # Minimize this
                loss_entropy = dist.entropy().mean() # Maximize this

                # Combined Loss
                loss = loss_actor + 0.5 * loss_critic - 0.01 * loss_entropy

                # Update step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

# --- Training Loop ---
def run_PPO(env_name="CarRacing-v3", episodes=500):
    
    # *** THE FIX FOR 'int' object has no attribute 'astype' ***
    # This forces the environment to use the discrete action space 
    # compatible with the ActorCritic's integer output.
    env = gym.make(env_name, continuous=False) 
    
    # Diagnostic Check (Should print: Discrete(5))
    print(f"Environment Action Space (Check): {env.action_space}")
    if not isinstance(env.action_space, Discrete):
        print("\nFATAL ERROR: Environment is still in continuous mode. The fix was not applied or saved correctly.\n")
        return # Stop execution if the environment setup failed

    obs, _ = env.reset()
    obs = preprocess(obs)
    shape = obs.shape
    action_dim = env.action_space.n # Will be 5 (Do Nothing, Left, Right, Gas, Brake)

    policy = ActorCritic(shape, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    agent = PPOAgent(policy, optimizer)

    BATCH = 2048 # Number of total steps to collect before a PPO update
    batch = {k: [] for k in ["obs", "actions", "rewards", "values", "dones", "logp"]}
    steps = 0
    ep_reward = 0

    for ep in range(episodes):
        done = False
        while not done:
            # a will be an integer action index (0-4)
            a, lp, v = policy.act(obs) 
            # print(f"[DEBUG] Episode {ep} action idx: {a}") # Keep for monitoring

            # The environment now correctly handles the integer action 'a' 
            # because we set continuous=False during env creation.
            next_obs, r, term, trunc, _ = env.step(a) 
            done = term or trunc
            next_obs = preprocess(next_obs)

            batch["obs"].append(obs)
            batch["actions"].append(a)
            batch["rewards"].append(r)
            batch["values"].append(v)
            batch["dones"].append(done)
            batch["logp"].append(lp)

            obs = next_obs
            ep_reward += r
            steps += 1

            if done:
                print(f"Episode {ep}  Reward: {ep_reward:.1f}")
                ep_reward = 0
                obs, _ = env.reset()
                obs = preprocess(obs)

        if steps >= BATCH:
            with torch.no_grad():
                # Get the value estimate for the last state (needed for GAE)
                last_val = policy.act(obs)[2] 

            returns = agent.compute_returns(batch["rewards"], batch["dones"], last_val)
            batch["returns"] = returns

            agent.update(batch, epochs=10, minibatch=64)
            # Reset batch buffers
            batch = {k: [] for k in ["obs", "actions", "rewards", "values", "dones", "logp"]}
            steps = 0

if __name__ == "__main__":
    run_PPO()