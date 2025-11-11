#----------------------------------------------------#
#----------------------Step 4/5----------------------#
#Play this if you want to visualize the agent playing#
#----------------------------------------------------#



#------------Import Libraries------------#
import sys
import os
from stable_baselines3 import PPO




#------------Set up directories------------#
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from snake_env.snake_env.env import SnakeEnv




#------------Set up basic------------#
env = SnakeEnv(render_mode="human")

#Load the hybrid model
model = PPO.load("models/ppo_snake_model_hybrid.pth")




#----------Evaluation----------#
NUM_EPISODES = 5

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


    print(f"Episode {ep+1}: Total reward = {total_reward}")

env.close()
