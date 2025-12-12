import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import torch.optim as optim
import random
from collections import deque
import cv2
import time
import os
import csv

# Use GPU if avalible. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

          
# Makes the observations ready for proccesing by PPO
def preprocess_PPO(obs):
    # CarRacing Environment gives a 96x96x3 image. 
    obs = obs.transpose(2,0,1) # converts from HWC to CHW
    obs = torch.tensor(obs, dtype=torch.float32) /255.0  # keep shape (C,H,W) and the 255 normalizes the pixel values between [0,1]
    return obs # shape: (3, 96, 96)

# Contains an Actor that decides the action and a Critic that estimates how good that action is. 
class ActorCritic(nn.Module):
    '''
        Since CarRacing is an image based environment it needs a CNN to extract features. 
        The CNN was redone using Nature CNN from stable Baselines as a refrence, since my original CNN was messed up. 
        https://github.com/openai/baselines/blob/master/baselines/ppo1/cnn_policy.py 
        # Car racing has a shape of (3, 96, 96)
    '''
    def __init__(self, obs_shape, action_dim):
        
        super().__init__()
        
        C, H, W = obs_shape  # Assume CxHxW images (channels first)
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),   # out: (32, 23, 23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # out: (64, 10, 10)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # out: (64, 8, 8)
            nn.ReLU()
        )
        
        # computes the size for flattening. 
        with torch.no_grad():
            sample = torch.zeros(1, C, H, W) 
            flat_size = self.cnn(sample).view(1,-1).size(1)
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU()
        )
        # Policy mean
        self.actor_mean = nn.Linear(512, action_dim)
        
        # initializes log_std to -0.5 to prevent high varience at the start
        self.log_std = nn.Parameter(torch.zeros(action_dim) * -0.5)
        
        self.critic = nn.Linear(512, 1) # critic returns a singular value for advantage estimation. 
    
    def forward(self, obs):
        # obs must be (B, C, H, W)
        features = self.cnn(obs)
        
        # flattens the output for the fully connected convolutional network as nn.Linear want 2D inputs, (batch_size, features).
        # shape (B, 64*8*8) = (B, 4096).
        features = features.view(features.size(0), -1)
        # reduces features from 4096 to 512(specific feature number gotten from PPO implementated by the paper.)
        # Features is now a feature vector representing the environment. Shape: (B, 512)
        features = self.fc(features)
        
        # inear layer maps the features to the mean of each action dimension. This is the mean of the guassian policy. 
        mean = self.actor_mean(features)
        # expand_as makes std the same shape as mean. Std is a learnable parameter. 
        std = self.log_std.exp().expand_as(mean)
        # Return single value per state. Shape: (B, 1)
        value = self.critic(features)
        
        return mean, std, value
    
    def act(self, obs):
        obs = obs.unsqueeze(0).to(device) # add batch dimension. 
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        # CarRacing has 3 continouse actions (steering, gar, brake), so each action has been bounded to prevent invalid values
        action = torch.clamp(action, -1.0, 1.0)
        
        log_prob = dist.log_prob(action).sum(dim = -1)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0) 


class PPOAgent():
    def __init__(self, env, policy, optimizer, gamma=0.99, lam=0.95, eps_clip=0.2):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        
        #discount factor for rewards. How much future reward matter. 
        #If 0, only care for immidiate rewards. else 1 if future rewards matter a lot.  
        self.gamma = gamma 
        
        #How much to smooth advantage (Generalized Advantage Estimation). GAE lambda parameter. 
        self.lam = lam 
        self.eps_clip = eps_clip # Clip range between old and new policy log_probabilites. 
         
    def compute_returns(self, rewards, values, dones):
        # PPO comuptes advatages backward through time. 
        returns = [] # returns are stored in list for training purposes. 
        gae = 0
        next_value = 0
        for step in reversed(range(len(rewards))):
            
            # Delta measures how suprising the reward was given the value estimate. 
            # γ is gamma or the discount factor.  
            # delta = reward + γ * V(s') - V(s)  
            # dones[step] indicates if the episode ended. 
            # If episode ended, 1, the discount factor would not be included in the calculation. 
            # If the episode did not end, then we would take into account future rewards(discount factor would be multiplied by 1, instead of 0 due to (1-0) code). 
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            
            # Gae = delta + γ * λ * (1 - done) * gae. This smoothes out advantage over multiple steps. 
            # λ controls bias/variance tradeoff. Lower means more biased, less varience
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            
            next_value = values[step] # updates value
            returns.insert(0, gae + values[step]) # gae + values[step] is th target for the critic. 
        return returns
    
    def update(self, batch, epoches = 4, batch_size = 64):
        # converts batch data to pytorch tensors. 
        # to device means GPU or CPU. torch.stack concatenates a sequence of tensors along a new dimension. So output has one more dimension then input. 
        obs_tensor = torch.stack(batch['obs']).to(device)
        actions_tensor = torch.stack(batch['actions']).to(device)
        returns_tensor = torch.tensor(batch['returns'], dtype=torch.float32, device=device)
        old_log_probs_tensor = torch.tensor(batch['log_probs'], dtype=torch.float32, device=device)
        values_tensor = torch.tensor(batch['values'], dtype=torch.float32, device=device)
        
        # computs normalized advantage. How much better an action was than expected. 
        advantage = returns_tensor - values_tensor
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        dataset_size = len(batch['obs'])
        losses = {'actor': 0, 'critic': 0, 'total':0}
        count = 0
        
        for i in range(epoches):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            
            # loops through dataset, taking a certain number of samples each time(batch_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                minibatch_idx = indices[start:end]
                
                mb_obs = obs_tensor[minibatch_idx]
                mb_actions = actions_tensor[minibatch_idx]
                mb_returns = returns_tensor[minibatch_idx]
                mb_old_log_probs = old_log_probs_tensor[minibatch_idx]
                mb_advantage = advantage[minibatch_idx]
                
                # Get the predition based on our policy
                mean_pred, std_pred, value_pred = self.policy(mb_obs)
                value_pred = value_pred.squeeze()
                
                # get distribution based on predicted mean and std
                dist = torch.distributions.Normal(mean_pred, std_pred)
                # get the log probabilites and sum them together. 
                log_probs = dist.log_prob(mb_actions).sum(dim=1)
                # get the ratio 
                ratio = torch.exp(log_probs - mb_old_log_probs)
                # clipping the advantage 
                clip_adv = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantage
                loss_actor = -torch.min(ratio * mb_advantage, clip_adv).mean()
                loss_critic = (mb_returns-value_pred).pow(2).mean()
                entropy = dist.entropy().sum(dim=1).mean()
                loss = loss_actor + 0.5 * loss_critic - 0.01 * entropy
                
                # Reset gradiet each loop
                self.optimizer.zero_grad()
                loss.backward() # backprobagate. 
                self.optimizer.step()
                
                # updating are losses dict
                losses['actor'] += loss_actor.item()
                losses['critic'] += loss_critic.item()
                losses['total'] += loss.item()
                count += 1
        for key in losses:
            losses[key] /= count
        return losses
                
                     
        
def run_PPO(env_name = "CarRacing-v3", episodes = 1000):
    env = gym.make(env_name, continuous=True)
    obs_shape = env.observation_space.shape # get shape of observations.
    action_dim = env.action_space.shape[0] # get the shape of the action space. 
    
    # Get the policy from the actor critic class
    policy = ActorCritic(obs_shape, action_dim).to(device)
    # Uses a basic optimizer.
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    agent = PPOAgent(env, policy, optimizer)
    
    # uses some basic hyperparameters that are common for RL models. 
    n_episodes = 1000
    batch_size = 2048
    mini_batch_size = 64
    update_epochs = 10 
    
    all_rewards = []
    batch = {'obs': [], 'actions':[], 'rewards':[], 'values':[], 'dones':[], 'log_probs':[]}
    steps_collected = 0
    
    # starting with clean env
    obs, _ = env.reset()
    obs = preprocess_PPO(obs)
    done = False # episode has not ended, thus done is initialized to false. 
    
    # training loop 
    for episode in range(n_episodes):
        episode_reward = 0
        
        while True: 
            action, log_prob, value = policy.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action) # Details from env after taking an action
            next_obs = preprocess_PPO(next_obs)
            done = terminated or truncated # episode is finished if conditions are met. 
            
            reward += action[1] * 0.3 # applies reward shaping. 
            
            batch['obs'].append(obs)
            batch['actions'].append(torch.tensor(action, dtype=torch.float32))
            batch['rewards'].append(reward)
            batch['values'].append(value.cpu().item())
            batch['dones'].append(done)
            batch['log_probs'].append(log_prob.item())
            
            episode_reward += reward # adding reward to total episode reward. 
            steps_collected += 1
            obs = next_obs # after taking the action, update the obs
            
            # if episode is done
            if done: 
                obs, _ = env.reset() # reset env
                obs = preprocess_PPO(obs)
                all_rewards.append(episode_reward)
                print(f"Episode {episode} reward: {episode_reward:.2f}")
                episode_reward = 0 # reset episode reward
                break
            
        if steps_collected >= batch_size: 
            returns = agent.compute_returns(batch['rewards'], batch['values'], batch['dones'])
            batch['returns'] = returns
            losses = agent.update(batch, epoches=update_epochs, batch_size=mini_batch_size)
            batch = {k: [] for k in batch}
            steps_collected = 0 # reset steps. 

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

            next_obs, reward, terminated, truncated, _ = env.step(ACTIONS[action_idx])
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
    run_DQN(episodes=10)  # or run_PPO(...) if you want to test PPO
    run_DQN(episodes=10)