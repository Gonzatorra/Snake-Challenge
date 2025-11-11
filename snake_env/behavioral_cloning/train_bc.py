#----------------------------------------------------#
#-----------------------Step 2-----------------------#
#----Play this to train just BC, not necessary if----#
#---------train_dqn_hybrid will be executed----------#
#----------------------------------------------------#


#------------Import Libraries------------#
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import numpy as np



#------------Set up directories------------#
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "snake_env"))
from env import SnakeEnv





#------------Set up basic------------#
#Get human data
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "human_data_snake.pkl")
with open(data_path, "rb") as f:
    human_data = pickle.load(f)

STACKED_ACTIONS = 5  # igual que en tu SnakeEnv
obs_list, actions_list = [], []






#--Data procesing and normalization--#
for d in human_data:
    obs_raw = d[0]
    head_x, head_y, apple_dx, apple_dy, snake_length = obs_raw[:5]
    prev_actions = obs_raw[5:]  #Last 5 actions for LSTM
    #Normalize data so actions are between 0 and 1
    obs_norm = [
        head_x / 500,
        head_y / 500,
        apple_dx / 500,
        apple_dy / 500,
        snake_length / 50
    ] + [a / 3 for a in prev_actions] #Normalize also the last actions
    obs_list.append(obs_norm)
    actions_list.append(d[1])

#Convert arrays into tensors
obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
actions_tensor = torch.tensor(actions_list, dtype=torch.long)






#-------------LSTM Model-------------#
input_dim = obs_tensor.shape[1]
action_dim = 4
seq_len = STACKED_ACTIONS


class LSTMPolicyClone(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)









#--------------Training--------------#
model = LSTMPolicyClone(input_dim, action_dim)


optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
batch_size = 64
epochs = 50
dataset_size = len(obs_tensor)


obs_seq = obs_tensor.unsqueeze(1).repeat(1, seq_len, 1)

for epoch in range(epochs):
    idx = torch.randperm(dataset_size)
    for i in range(0, dataset_size, batch_size):
        batch_idx = idx[i:i+batch_size]
        x_batch = obs_seq[batch_idx]
        y_batch = actions_tensor[batch_idx]

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}")







#-----------Save the model-----------#
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "..", "models", "bc_lstm_model.pth"))
print("BC LSTM model trained and saved in 'models/'")
