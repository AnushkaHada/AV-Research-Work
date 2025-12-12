import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym

# Import to use for checking (optional, but good practice)
from gymnasium.spaces import Discrete, Box 

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the 5 discrete actions as continuous [steer, gas, brake] arrays
# These actions will be used to convert the integer output of the policy 
# into the float array required by the continuous CarRacing environment.
DISCRETE_ACTIONS = {
    0: np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Do Nothing
    1: np.array([-1.0, 0.0, 0.0], dtype=np.float32), # Steer Left
    2: np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Steer Right
    3: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Gas
    4: np.array([0.0, 0.0, 0.8], dtype=np.float32)   # Brake
}
ACTION_DIM = 5 # Number of discrete choices your policy will make

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
        
        # Returns the action as an integer (0, 1, 2, 3, or 4)
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
        # Calculates discounted returns
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

                loss_actor = -torch.min(surr1, surr2).mean() 
                loss_critic = F.mse_loss(mb_ret, val) 
                loss_entropy = dist.entropy().mean() 

                # Combined Loss
                loss = loss_actor + 0.5 * loss_critic - 0.01 * loss_entropy

                # Update step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

# --- Training Loop ---
def run_PPO(env_name="CarRacing-v3", episodes=500):
    
    # *** THE NEW FIX: Initialize in continuous mode and map manually ***
    # This ensures the environment step function receives a NumPy array.
    env = gym.make(env_name) 
    
    # Diagnostic Check (Should print: Box(3,))
    print(f"Environment Action Space (Check): {env.action_space}")
    if not isinstance(env.action_space, Box):
        print("\nFATAL ERROR: Environment is not in expected continuous mode. Check your gym installation.\n")
        return

    obs, _ = env.reset()
    obs = preprocess(obs)
    shape = obs.shape
    
    # Use the globally defined discrete action count
    action_dim = ACTION_DIM 

    policy = ActorCritic(shape, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    agent = PPOAgent(policy, optimizer)

    BATCH = 2048 
    batch = {k: [] for k in ["obs", "actions", "rewards", "values", "dones", "logp"]}
    steps = 0
    ep_reward = 0

    for ep in range(episodes):
        done = False
        while not done:
            # a_idx is the integer (0-4) from your policy
            a_idx, lp, v = policy.act(obs) 
            print(f"[DEBUG] Episode {ep} action idx: {a_idx}") 

            # **MANUAL MAPPING**
            # Convert the discrete action index to the continuous NumPy array
            a_continuous = DISCRETE_ACTIONS[a_idx]
            
            # Pass the NumPy array (a_continuous) to env.step()
            next_obs, r, term, trunc, _ = env.step(a_continuous) 
            done = term or trunc
            next_obs = preprocess(next_obs)

            batch["obs"].append(obs)
            batch["actions"].append(a_idx) # Store the integer index
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
                last_val = policy.act(obs)[2] 

            returns = agent.compute_returns(batch["rewards"], batch["dones"], last_val)
            batch["returns"] = returns

            agent.update(batch, epochs=10, minibatch=64)
            # Reset batch buffers
            batch = {k: [] for k in ["obs", "actions", "rewards", "values", "dones", "logp"]}
            steps = 0

if __name__ == "__main__":
    run_PPO()