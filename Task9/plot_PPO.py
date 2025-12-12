import pandas as pd
import matplotlib.pyplot as plt
import os

LOGDIR = "logs_ppo_carracing"

reward_file = os.path.join(LOGDIR, "episode_rewards.csv")
loss_file = os.path.join(LOGDIR, "losses.csv")
policy_file = os.path.join(LOGDIR, "policy_stats.csv")

# Load CSV files
df_rewards = pd.read_csv(reward_file)
df_losses = pd.read_csv(loss_file)
df_policy = pd.read_csv(policy_file)

print("Loaded logs:")
print(f"Rewards: {df_rewards.shape}")
print(f"Losses: {df_losses.shape}")
print(f"Policy stats: {df_policy.shape}")

# Plot Episode Rewards
plt.figure(figsize=(10,5))
plt.plot(df_rewards["episode"], df_rewards["reward"])
plt.title("Episode Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_rewards.png")
print("Saved plot_rewards.png")

# Plot Losses
plt.figure(figsize=(10,5))
plt.plot(df_losses["update_idx"], df_losses["actor_loss"], label="Actor Loss")
plt.plot(df_losses["update_idx"], df_losses["critic_loss"], label="Critic Loss")
plt.plot(df_losses["update_idx"], df_losses["total_loss"], label="Total Loss")
plt.title("PPO Losses")
plt.xlabel("Update Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_losses.png")
print("Saved plot_losses.png")

# Plot Entropy
plt.figure(figsize=(10,5))
plt.plot(df_losses["update_idx"], df_losses["entropy"])
plt.title("Policy Entropy Over Training")
plt.xlabel("Update Step")
plt.ylabel("Entropy")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_entropy.png")
print("Saved plot_entropy.png")

# Plot Policy Mean and Std
plt.figure(figsize=(10,5))
plt.plot(df_policy["update_idx"], df_policy["mean_action"], label="Mean Action")
plt.plot(df_policy["update_idx"], df_policy["std_action"], label="Std Dev of Action")
plt.title("Policy Action Statistics")
plt.xlabel("Update Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_policy_stats.png")
print("Saved plot_policy_stats.png")
