from dqn_carracing import run_DQN
from ppo_carracing import run_PPO   # your existing PPO file

if __name__ == "__main__":
    print("Running PPO...")
    run_PPO()

    print("Running DQN...")
    run_DQN()
