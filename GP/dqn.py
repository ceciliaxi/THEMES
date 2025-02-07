import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

import torch
import torch.nn.functional as F
from evaluation import *

# Set the environment parameters
sa_dict = {'CCHS': {'act_num': 2, 'state_dim': 42}} 

# Create a custom ReplayBuffer
class CustomReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device="cpu"):
        super().__init__(buffer_size, observation_space, action_space, device)
    
    def add_offline_data(self, offline_data):
        """
        Adds offline data into the replay buffer.
        """
        for state, action, reward, next_state, done in offline_data:
            infos = {}
            self.handle_timeout_termination = False
            self.add(state, next_state, np.array(action), np.array(reward), done, infos)    

# Initialize the environment (can be any custom environment)
class CustomEnv(gym.Env):
    def __init__(self, env='CCHS'):
        super().__init__()
        self.act_num, self.state_dim = sa_dict[env]['act_num'], sa_dict[env]['state_dim']
        self.action_space = gym.spaces.Discrete(self.act_num) 
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32) 
        self.state = np.random.uniform(0, 1, self.state_dim)  # Random state initialization
        self.steps = 0  # Track number of steps

    def reset(self):
        self.state = np.random.uniform(0, 1, self.state_dim)
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1
        reward = 0 # Scalar reward
        done = self.steps >= 500
        next_state = np.random.uniform(0, 1, self.state_dim)
        info = {}
        return next_state, reward, done, info

    def render(self):
        print(f"State: {self.state}")

    def get_act_num(self): 
        return self.act_num

    def get_state_dim(self): 
        return self.state_dim

# Train the offline DQN
def train_offline_dqn(tr_data, env='CCHS', total_timesteps=100000, learning_rate=1e-6, batch_size=128, buffer_size=1000000, verbose=1, save_model=False): 
    # Prepare the offline data
    # Example offline data: A list of (state, action, reward, next_state, done) tuples
    offline_data = []
    tr_observations, tr_actions, tr_rewards, tr_terminals = tr_data
    for ep_idx in range(1, len(tr_observations)): 
        offline_data.append((tr_observations[ep_idx-1], tr_actions[ep_idx-1], np.clip(tr_rewards[ep_idx-1], 0, 1),
                             tr_observations[ep_idx], bool(tr_terminals[ep_idx])))
    
    # Wrap the environment in DummyVecEnv (Stable-Baselines3 expects this format)
    cus_env = CustomEnv(env=env)
    cus_env = DummyVecEnv([lambda: cus_env])
    
    # Initialize the ReplayBuffer and load offline data
    replay_buffer = CustomReplayBuffer(
        buffer_size=1000000,  # Size of the buffer
        observation_space=cus_env.observation_space,  # Observation space from the environment
        action_space=cus_env.action_space,  # Action space from the environment
        device="cpu"  # Using CPU (can switch to 'cuda' for GPU)
    )
    
    # Add your offline data to the buffer
    replay_buffer.add_offline_data(offline_data)
    
    # Initialize the DQN model
    model = DQN("MlpPolicy", cus_env, learning_rate=learning_rate, batch_size=batch_size, buffer_size=buffer_size, verbose=verbose)
    
    # Train the model using the replay buffer
    # In Stable-Baselines3, the replay buffer is used during training automatically
    # model.set_replay_buffer(replay_buffer)
    model.replay_buffer = replay_buffer
    
    # Train the model (here we train for 100000 timesteps)
    model.learn(total_timesteps=total_timesteps)
    
    # Step 8: Save the trained model
    if save_model: 
        model.save("dqn_offline_model")
    return model


# Evaluate the offline DQN
def test_offline_dqn(model, te_data, env='CCHS', verbo=True): 
    act_num = sa_dict[env]['act_num']
    te_observations, te_actions, _, _ = te_data

    true_lab, pred_prob, pred_lab = [], [], []
    for te_idx in range(len(te_observations)): 
        te_obs = te_observations[te_idx]
        pred_act = model.predict(te_obs)[0]

        # Use model.predict to get the action and Q-values
        with torch.no_grad():
            q_values = model.q_net(torch.tensor(te_obs, dtype=torch.float32).unsqueeze(0))  # Add batch dimension
            q_values = q_values.squeeze(0).cpu().numpy() 
        
        # Apply softmax to Q-values to get action probabilities
        action_probabilities = F.softmax(torch.tensor(q_values), dim=0).numpy()

        pred_lab.append(int(pred_act))
        true_lab.append(int(te_actions[te_idx]))
        pred_prob.append(action_probabilities)

    metrics = overall_eval(true_lab, pred_lab, pred_prob, act_num, avg_patterns='binary', verbo=True)
    return metrics