#----------------------------------------------------#
#----------------------Step 4/5----------------------#
#---Play this if you want to get agent statistics----#
#----------------------------------------------------#



#------------Import Libraries------------#
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import sys, os
import numpy as np




#------------Set up directories------------#
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "snake_env"))
from env import SnakeEnv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "..", "models")




#------------Set up basic------------#
def make_env():
    return Monitor(SnakeEnv(render_mode=None))
env = DummyVecEnv([make_env])

#Load VecNormalize
vecnorm_path = os.path.join(models_dir, "vecnormalize_snake.pkl")
env = VecNormalize.load(vecnorm_path, env)
env.training = False          #Do not update the statistics, just evaluation
env.norm_reward = True        #Use normalized rewards


#Load the model
model = PPO.load(os.path.join(models_dir, "ppo_snake_model_ppo"), env=env, device="cuda")





#------------Evaluation------------#
NUM_EPISODES = 20
all_rewards = []
all_lengths = []

for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = [False]
    total_reward = 0

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

    #Save the results
    snake_length = len(env.get_attr("snake_position")[0])
    all_rewards.append(total_reward)
    all_lengths.append(snake_length)

    print(f"Episode {ep+1}: Total reward = {total_reward:.2f}, Snake length = {snake_length}")

mean_reward = np.mean(all_rewards)
mean_length = np.mean(all_lengths)

print(f"\nAverage reward over {NUM_EPISODES} episodes: {mean_reward:.2f}")
print(f"Average snake length over {NUM_EPISODES} episodes: {mean_length:.2f}")

env.close()
