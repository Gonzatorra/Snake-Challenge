#----------------------------------------------------#
#-----------------------Step 3-----------------------#
#-------Train the agent with hybrid model----------#
#----------------------------------------------------#

#------------Import Libraries------------#
import os
import sys
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback

#------------Set up directories------------#
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "snake_env"))
from env import SnakeEnv

#-------------Hyperparameters--------------#
features_dim = 128
mlp_hidden = 256
n_steps = 2048
batch_size = 64
learning_rate = 1e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.03
vf_coef = 0.5
max_grad_norm = 0.5
total_timesteps = 500_000

#------Characteristics Extractor Model-----#
class MLPExtractorNormalized(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=features_dim, hidden_dim=mlp_hidden):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  #Normalization for stability
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)

#------------Set up environment------------#
def make_env():
    return Monitor(SnakeEnv())

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

#------------Set up evaluation environment------------#
def make_eval_env():
    return Monitor(SnakeEnv())

eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, clip_reward=10.0)



#-----Set up the policy-----#
policy_kwargs = dict(
    features_extractor_class=MLPExtractorNormalized,
    features_extractor_kwargs=dict(features_dim=features_dim),
)

#----------Set up the PPO model----------#
ppo_model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_steps=n_steps,
    batch_size=batch_size,
    learning_rate=learning_rate,
    gamma=gamma,
    gae_lambda=gae_lambda,
    clip_range=clip_range,
    tensorboard_log="logs/ppo_snake_tensorboard",
    ent_coef=ent_coef,
    vf_coef=vf_coef,
    max_grad_norm=max_grad_norm
)

#----------Set up EvalCallback to save the best model----------#
save_path = os.path.join(os.path.dirname(__file__), "..", "models", "best_model")
os.makedirs(save_path, exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=save_path,
    log_path="logs/ppo_snake_eval",
    eval_freq=5000,      # Eval every 5000 steps
    deterministic=True,
    render=False
)

#--------------Training--------------#
ppo_model.learn(total_timesteps=total_timesteps, callback=eval_callback)

#----Save the final model and VecNormalize----#
final_model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_snake_model_final")
ppo_model.save(final_model_path)
env.save(os.path.join(os.path.dirname(__file__), "..", "models", "vecnormalize_snake_ppo.pkl"))

print("Training complete. Best model saved during training and final model/VecNormalize saved.")
