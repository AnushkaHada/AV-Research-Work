import pandas as pd
import matplotlib.pyplot as plt
import os

# Path relative to THIS python file
LOGDIR = "logs_dqn_carracing"

reward_file = os.path.join(LOGDIR, "episode_rewards.csv")
loss_file = os.path.join(LOGDIR, "losses.csv")
qstats_file = os.path.join(LOGDIR, "q_stats.csv")

# Load CSV files
df_rewards = pd.read_csv(reward_file)
df_losses = pd.read_csv(loss_file)
df_qstats = pd.read_csv(qstats_file)

print("\nLoaded DQN logs")
print("Rewards:", df_rewards.shape)
print("Losses:", df_losses.shape)
print("Q-stats:", df_qstats.shape)

# Episode Rewards
plt.figure(figsize=(10,5))
plt.plot(df_rewards["episode"], df_rewards["reward"])
plt.title("DQN Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("dqn_reward_plot.png")
print("Saved dqn_reward_plot.png")

# Loss vs Update
plt.figure(figsize=(10,5))
plt.plot(df_losses["update_idx"], df_losses["loss"])
plt.title("DQN Loss Over Training")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("dqn_loss_plot.png")
print("Saved dqn_loss_plot.png")

# Q-Value Statistics
plt.figure(figsize=(10,5))
plt.plot(df_qstats["update_idx"], df_qstats["mean_q"], label="Mean Q")
plt.plot(df_qstats["update_idx"], df_qstats["max_q"], label="Max Q")
plt.title("DQN Q-Value Statistics")
plt.xlabel("Update Step")
plt.ylabel("Q-value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dqn_qvalue_plot.png")
print("Saved dqn_qvalue_plot.png")

print("\nDone! Plots saved in Task9 folder.")
