import torch
import torch.optim as optim
import numpy as np
from snake_env import SnakeEnv
from q_network import DQN, ReplayBuffer
import random

# Parámetros
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE = 10
NUM_EPISODES = 1000
STACKED_ACTIONS = 5

# Entorno
env = SnakeEnv()
state_dim = 5 + STACKED_ACTIONS
action_dim = env.action_space.n

# Redes
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

# Replay Buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE)
epsilon = EPS_START




def select_action(state, policy_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return q_values.argmax().item()
    




def train_step(policy_net, target_net, replay_buffer, optimizer):
    if len(replay_buffer) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
    
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones).unsqueeze(1)
    
    # Q actual
    q_values = policy_net(states).gather(1, actions)
    # Q target
    next_q_values = target_net(next_states).max(1)[0].detach().unsqueeze(1)
    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
    
    loss = torch.nn.MSELoss()(q_values, target_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



for episode in range(1, NUM_EPISODES+1):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, policy_net, epsilon, action_dim)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        train_step(policy_net, target_net, replay_buffer, optimizer)
    
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.2f}")
