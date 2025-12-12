'''
*Disclaimer: ChatGPT was used to debug code. Especially data type and dimension issues. 
DQN - Learns which action to take in each state by using neural network to find Q-values.
Q values - How good an action a is to take in state s. 
The input is the frames and the output is the Q-values for each discrete action. 
The action choosen is just argmax(Q-value)

DQN improves Q-learning by using a CNN to estimate Q-values. 
It has a replay buffer to stor past experiences to stablize training. 
Uses Nature CNN. 
'''

import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import cv2
from collections import deque

# use gpu if avalible. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# logs dqn results the same way as PPO.  
class DQNLogger:
    def __init__(self, logdir="logs_dqn_carracing"):
        os.makedirs(logdir, exist_ok=True)

        self.reward_file = os.path.join(logdir, "episode_rewards.csv")
        self.loss_file = os.path.join(logdir, "losses.csv")
        self.q_stats_file = os.path.join(logdir, "q_stats.csv")
        
        # resets log file each run to prevent bloating. 
        for f in [self.reward_file, self.loss_file, self.q_stats_file]:
            if os.path.exists(f):
                os.remove(f)
                
        # Create with headers if needed
        if not os.path.exists(self.reward_file):
            with open(self.reward_file, "w", newline="") as f:
                csv.writer(f).writerow(["episode", "reward"])

        if not os.path.exists(self.loss_file):
            with open(self.loss_file, "w", newline="") as f:
                csv.writer(f).writerow(["update_idx", "loss"])

        if not os.path.exists(self.q_stats_file):
            with open(self.q_stats_file, "w", newline="") as f:
                csv.writer(f).writerow(["update_idx", "mean_q", "max_q"])

    def log_reward(self, episode, reward):
        with open(self.reward_file, "a", newline="") as f:
            csv.writer(f).writerow([episode, reward])

    def log_loss(self, update_idx, loss):
        with open(self.loss_file, "a", newline="") as f:
            csv.writer(f).writerow([update_idx, loss])

    def log_q_stats(self, update_idx, q_values_tensor):
        q = q_values_tensor.detach().cpu().numpy()
        with open(self.q_stats_file, "a", newline="") as f:
            csv.writer(f).writerow([update_idx, q.mean(), q.max()])


# Nature CNN, similar structure to PPO nature CNN. 
class NatureCNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # final layer outputs Q-value. 
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions) # 512 is a common amount of features used for nature CNN
        )
        

    def forward(self, x):
        return self.net(x / 255.0) # normalizes



# Replay Buffer
class ReplayBuffer:
    def __init__(self, size=100000):
        self.buf = deque(maxlen=size)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d)) # stores info in dequeue.

    def sample(self, batch_size):
        # samples random mini batches.  
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r),
                np.array(s2), np.array(d))

    def __len__(self):
        return len(self.buf)


# Preprocessing converts rbg to gray scale to prevent excess noise. 
def preprocess_frame(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA) # resizes to 84 by 84 and maintains 4 frame stack
    return obs

# This is important since 4 frames provides more info then a single frame. 
def init_stack(frame):
    return np.stack([frame] * 4, axis=0)


# Discrete actions for CarRacing manually defined. 
DISCRETE_ACTIONS = [
    np.array([0.0, 1.0, 0.0]),     # Gas
    np.array([0.0, 0.0, 1.0]),     # Brake
    np.array([-1.0, 1.0, 0.0]),    # Left + gas
    np.array([1.0, 1.0, 0.0]),     # Right + gas
    np.array([0.0, 0.0, 0.0])      # Nothing
]



# DQN Training 
def run_DQN(episodes=600):
    # Even continuous=True we manually use discrete action space. 
    env = gym.make("CarRacing-v3", continuous=True, render_mode=None)

    logger = DQNLogger()

    num_actions = len(DISCRETE_ACTIONS)
    
    # creates two copies of same network. 
    q_net = NatureCNN(num_actions).to(device)
    target_net = NatureCNN(num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-4)
    replay = ReplayBuffer(200000) # large buffer for stability. 

    gamma = 0.99
    batch_size = 32
    train_start = 20000
    target_update_freq = 10000 # updates target network every x steps. 

    eps_start, eps_final, eps_decay = 1.0, 0.05, 400000
    global_step = 0
    update_idx = 0
    
    # loops through episodes. 
    for ep in range(episodes):
        # sets up clean env and gets info
        obs, _ = env.reset()
        f = preprocess_frame(obs)
        state = init_stack(f)

        ep_reward = 0
        done = False

        while not done:
            global_step += 1

            # epsilon annealing
            eps = eps_final + (eps_start - eps_final) * np.exp(-global_step / eps_decay)

            if random.random() < eps:
                action_idx = random.randint(0, num_actions - 1)
            else:
                with torch.no_grad():
                    s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = q_net(s)
                    action_idx = q_values.argmax().item()

            action = DISCRETE_ACTIONS[action_idx]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # updates frame stack. 
            next_f = preprocess_frame(next_obs)
            next_state = np.roll(state, -1, axis=0)
            next_state[-1] = next_f # newest frame. 

            replay.push(state, action_idx, reward, next_state, done)

            state = next_state
            ep_reward += reward

            # training
            if len(replay) > train_start:
                s, a, r, s2, d = replay.sample(batch_size)
                
                # convert numpy to tensor
                s = torch.tensor(s, dtype=torch.float32).to(device)
                a = torch.tensor(a, dtype=torch.long).to(device)
                r = torch.tensor(r, dtype=torch.float32).to(device)
                s2 = torch.tensor(s2, dtype=torch.float32).to(device)
                d = torch.tensor(d, dtype=torch.float32).to(device)
                
                # pridicted Q. Gather selects the predicted q for an action. 
                q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
                # computes target. 
                next_q = target_net(s2).max(1)[0]
                target = r + gamma * next_q * (1 - d) # bellman update. 

                loss = nn.MSELoss()(q, target.detach()) # detaches to avoid backpropogation through target network. 
                
                # backpropagate. 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # logs results. 
                logger.log_loss(update_idx, loss.item())
                logger.log_q_stats(update_idx, q_net(s).detach().cpu())

                update_idx += 1

                if global_step % target_update_freq == 0:
                    target_net.load_state_dict(q_net.state_dict())

        logger.log_reward(ep, ep_reward)
        print(f"Episode {ep} | Reward: {ep_reward:.1f} | Eps: {eps:.3f}")


if __name__ == "__main__":
    run_DQN()
