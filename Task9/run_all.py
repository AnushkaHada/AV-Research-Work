from dqn_carracing import train_dqn
from ppo_carracing import train_ppo   # your existing PPO file

if __name__ == "__main__":
    print("Running PPO...")
    train_ppo()

    print("Running DQN...")
    train_dqn()
