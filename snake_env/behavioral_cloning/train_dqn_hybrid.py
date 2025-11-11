#----------------------------------------------------#
#-----------------------Step 3-----------------------#
#-------Play this to train the agent, with the-------#
#--------------------hybrid model--------------------#
#----------------------------------------------------#



#------------Import Libraries------------#
import os
import sys
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor




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
total_timesteps = 10_000





#------Characteristics Extractor Model-----#
class MLPExtractorNormalized(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=features_dim, hidden_dim=mlp_hidden):
        super().__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), #Normlaization for a better stabiility
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)





#------------Set up basic------------#
def make_env():
    return Monitor(SnakeEnv())

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0) #Normalize rewards





#-----Set up the recurrent policy----#
#Set up parameters for policy MpLstmPolicy
policy_kwargs = dict(
    features_extractor_class=MLPExtractorNormalized,
    features_extractor_kwargs=dict(features_dim=features_dim),
    lstm_hidden_size=256,   #LSTM size
    n_lstm_layers=1,
    shared_lstm=False
)



#----------Set up the model----------#
#PPO with LSTM
ppo_model = RecurrentPPO(
    "MlpLstmPolicy",
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





#--------------Training--------------#
ppo_model.learn(total_timesteps=total_timesteps)






#----Save the model and normalizer----#
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
ppo_model.save("models/ppo_snake_model_recurrent")
env.save("models/vecnormalize_snake.pkl")


print("Training complete. Model and VecNormalize saved.")
