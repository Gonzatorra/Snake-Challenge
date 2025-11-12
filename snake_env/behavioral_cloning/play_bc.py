#----------------------------------------------------#
#----------------------Step 4/5----------------------#
#Play this if you want to visualize the agent playing#
#----------------------------------------------------#



#------------Import Libraries------------#
import sys
import os
import numpy as np
from stable_baselines3 import PPO




#------------Set up directories------------#
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from snake_env.snake_env.env import SnakeEnv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "..", "models")


#------------Set up basic------------#
env = SnakeEnv(render_mode="none")

#Load the hybrid model
model = PPO.load(os.path.join(models_dir, "best_model/best_model"))




#----------Evaluation----------#
NUM_EPISODES = 10
all_rewards = []
all_lengths = []

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        #get the action
        action, _ = model.predict(obs, deterministic=True)

        #Apply it in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        #Render the game
        env.render()
        print("Current snake length:", len(env.snake_position))

        all_rewards.append(total_reward)
        all_lengths.append(len(env.snake_position))

    print(f"Episode {ep+1}: Total reward = {total_reward}")

mean_reward = np.mean(all_rewards)
mean_length = np.mean(all_lengths)
max_length = max(all_lengths)
max_reward = max(all_rewards)



print(f"\nAverage reward over {NUM_EPISODES} episodes: {mean_reward:.2f}")
print(f"Average snake length over {NUM_EPISODES} episodes: {mean_length:.2f}")
print(f"Max snake length over {NUM_EPISODES} episodes: {max_length}")
print(f"Max reward over {NUM_EPISODES} episodes: {max_reward:.2f}")

env.close()