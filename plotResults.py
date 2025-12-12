import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dqn_log.csv")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(df['episode'], df['reward'])
plt.xlabel('Episode'); plt.ylabel('Reward'); plt.title('Episode Reward')

plt.subplot(1,2,2)
plt.plot(df['episode'], df['loss'])
plt.xlabel('Episode'); plt.ylabel('Loss'); plt.title('Episode Loss')

plt.show()



df = pd.read_csv("dqn_log.csv")
plt.plot(df['episode'], df['reward'], label="Reward")
plt.plot(df['episode'], df['loss'], label="Loss")
plt.xlabel("Episode")
plt.legend()
plt.show()
