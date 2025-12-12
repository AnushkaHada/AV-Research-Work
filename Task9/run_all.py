from DQN import run_DQN
from PPO import run_PPO   # your existing PPO file

if __name__ == "__main__":
    print("Running PPO...")
    run_PPO()

    print("Running DQN...")
    run_DQN()
