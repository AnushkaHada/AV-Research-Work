'''
*Disclaimer: ChatGPT was used to debug code. Especially data type and dimension issues. 
PPO - Trains a policy network to choose certain actions, while the critic gives a value estimate 
on how good the actions are. 

Advantage - How much better an action was compared to what the model expected. 

Ratio - Compares the new policy to the old policy

Generalized Advantage Estimation - Used to smooth the advantage. 

PPO structure has been significantly revised due to several notable bugs. 
Old PPO used a bad CNN, which did not learn features properly.
This PPO uses Nature CNN which was also used by the research paper. 
It was also updated from CarRacing v2 to v3 as v2 had been depreciated. Basically v2 was not working. 
'''

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Use GPU if avalible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# PPO Logger saves CSV logs  of rewards, losses and policy stats.
class PPOLogger:
    def __init__(self, logdir="logs_ppo_carracing"):
        os.makedirs(logdir, exist_ok=True)
        
        # Creates a folder to log the results. 
        self.reward_file = os.path.join(logdir, "episode_rewards.csv")
        self.loss_file = os.path.join(logdir, "losses.csv")
        self.policy_file = os.path.join(logdir, "policy_stats.csv")

        # Create files with headers if missing
        if not os.path.exists(self.reward_file):
            with open(self.reward_file, "w", newline="") as f:
                csv.writer(f).writerow(["episode", "reward"])

        if not os.path.exists(self.loss_file):
            with open(self.loss_file, "w", newline="") as f:
                csv.writer(f).writerow(["update_idx", "actor_loss", "critic_loss", "entropy", "total_loss"])

        if not os.path.exists(self.policy_file):
            with open(self.policy_file, "w", newline="") as f:
                csv.writer(f).writerow(["update_idx", "mean_action", "std_action"])

    # Takes in the reward and episode to append it to file. 
    def log_reward(self, episode, reward):
        with open(self.reward_file, "a", newline="") as f:
            csv.writer(f).writerow([episode, reward])
                  
    # Takes in the index, losses, and entropy to append to file. 
    def log_losses(self, update_idx, losses, entropy):
        with open(self.loss_file, "a", newline="") as f:
            csv.writer(f).writerow([
                update_idx,
                losses["actor"],
                losses["critic"],
                entropy,
                losses["total"]
            ])
    # Takes in index and policy to append the mean and standard deviation of the action. 
    def log_policy_stats(self, update_idx, policy):
        mean_action = policy.actor_mean.weight.data.mean().item()
        std_action = policy.log_std.exp().mean().item()

        with open(self.policy_file, "a", newline="") as f:
            csv.writer(f).writerow([update_idx, mean_action, std_action])



def preprocess_PPO(obs):
    obs = obs.transpose(2, 0, 1) # HWC → CHW
    # converts numpy to tensor and normalizes the observations to [0, 1]
    obs = torch.tensor(obs, dtype=torch.float32) / 255.0 
    return obs.to(device) # moves to GPU


# This Actor Critic class outputs mean, standard deviation of actions, and state value
class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()

        C, H, W = obs_shape

        # Nature CNN is taken from the paper. 
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 8, stride=4),  # (96 → 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), # (23 → 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), # (10 → 8)
            nn.ReLU()
        )
        
        # Computes the flatten size automatically 
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            flat_size = self.cnn(dummy).view(1, -1).shape[1]
            
        # Fully connected layer. Feature amount 512 also taken from paper. 
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU()
        )

        # Actor gives the Gaussian policy
        self.actor_mean = nn.Linear(512, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) * -0.5) # learnable

        # Critic outputs state value. Its a singular value. 
        self.critic = nn.Linear(512, 1)
        
    # Forward pass. 
    def forward(self, obs):
        features = self.cnn(obs)
        features = features.view(features.size(0), -1)
        features = self.fc(features)

        mean = self.actor_mean(features)
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(features)

        return mean, std, value
    
    # Sampling actions
    def act(self, obs):
        # obs is (C,H,W)
        obs = obs.unsqueeze(0)  # add batch dim
        mean, std, value = self.forward(obs)
        
        # samples from guassian policy. 
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)

        # Clamps/restricts actions
        action = action[0]
        action_clamped = torch.zeros_like(action)
        action_clamped[0] = torch.tanh(action[0])     # steering (-1 to 1)
        action_clamped[1] = torch.sigmoid(action[1])  # gas  (0 to 1)
        action_clamped[2] = torch.sigmoid(action[2])  # brake  (0 to 1)

        return action_clamped, log_prob[0], value[0]


# PPO Agent decides advantage estimation, updates neural network and finds loss. 

class PPOAgent:
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
        
        # iterates backwards 
        for t in reversed(range(len(rewards))):
            
            # General advantage function formulas: 
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            
            next_value = values[t]
            returns.insert(0, gae + values[t])

        return returns

    def update(self, batch, epochs=10, batch_size=64):
        # converts list to tensors
        obs = torch.stack(batch["obs"]).to(device)
        actions = torch.stack(batch["actions"]).to(device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32, device=device)
        values = torch.tensor(batch["values"], dtype=torch.float32, device=device)
        
        # advantage function. 
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_steps = len(batch["obs"])
        losses = {"actor": 0, "critic": 0, "total": 0}
        entropy_last = 0
        count = 0
        
        # loops through epochs
        for _ in range(epochs):
            idxs = np.arange(total_steps)
            np.random.shuffle(idxs)
            
            # goes through each batch. 
            for start in range(0, total_steps, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                
                # gets the basic info from policy. 
                mean, std, value = self.policy(mb_obs)
                value = value.squeeze()
                
                
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=1)
                entropy = dist.entropy().sum(dim=1).mean().item()
                entropy_last = entropy
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                # clips rewards so PPO does not make a big update. 
                clipped = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                
                # calculates loss. 
                actor_loss = -torch.min(ratio * mb_adv, clipped * mb_adv).mean()
                critic_loss = (mb_returns - value).pow(2).mean()
                loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().sum(dim=1).mean()
                
                # Backward pass. It sets the gradient to 0,for clarity. 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # updates info eash loop 
                losses["actor"] += actor_loss.item()
                losses["critic"] += critic_loss.item()
                losses["total"] += loss.item()
                count += 1
        
        for k in losses:
            losses[k] /= count

        return losses, entropy_last


# PPO Training Loop
def run_PPO(env_name="CarRacing-v3", episodes=600):
    # creating env. 
    env = gym.make(env_name, continuous=True)
    logger = PPOLogger("logs_ppo_carracing")
    update_idx = 0

    sample_obs, _ = env.reset()
    sample_obs = preprocess_PPO(sample_obs)
    obs_shape = sample_obs.shape
    action_dim = env.action_space.shape[0]
    
    # initialize networks. 
    policy = ActorCritic(obs_shape, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    agent = PPOAgent(policy, optimizer)
    
    # Stores info of batch. 
    batch = {"obs": [], "actions": [], "rewards": [], "values": [], "dones": [], "log_probs": []}
    
    # hyperparameters.
    batch_size = 2048
    mini_batch = 64
    update_epochs = 10
    
    # resets env and preproccess observations for CNN. 
    obs, _ = env.reset()
    obs = preprocess_PPO(obs)
    steps = 0
    
    # loops through each episode. 
    for episode in range(episodes):
        ep_reward = 0 # initializes reward at start of each episode. 

        while True:
            action, log_prob, value = policy.act(obs)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            
            # Gets information of the state and preproccess the next obervations. 
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            next_obs = preprocess_PPO(next_obs)
            done = terminated or truncated 

            # Reward shaping
            reward += action_np[1] * 0.3
            
            # Stores information to batch. 
            batch["obs"].append(obs)
            batch["actions"].append(action)
            batch["rewards"].append(reward)
            batch["values"].append(value.item())
            batch["dones"].append(done)
            batch["log_probs"].append(log_prob.item())

            obs = next_obs
            ep_reward += reward
            steps += 1
            
            # Once episode is done, it is logged. 
            if done:
                logger.log_reward(episode, ep_reward)
                print(f"Episode {episode} | Reward = {ep_reward:.2f}")
                obs, _ = env.reset()
                obs = preprocess_PPO(obs)
                break

        # Update PPO, once enough steps are reached it computes the returns. 
        if steps >= batch_size:
            returns = agent.compute_returns(batch["rewards"], batch["values"], batch["dones"])
            batch["returns"] = returns
            
            # updates loss and entorpy for policy. 
            losses, entropy = agent.update(batch, epochs=update_epochs, batch_size=mini_batch)
            
            # logs info. 
            logger.log_losses(update_idx, losses, entropy)
            logger.log_policy_stats(update_idx, policy)

            batch = {k: [] for k in batch}  # clear buffer
            steps = 0
            update_idx += 1



if __name__ == "__main__":
    run_PPO()
