import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Actor (policy network)
class Actor(nn.Module):
    def __init__(self, state_dim, act_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh() # output: [-1, 1]
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)

# Critic (value network)
class Critic(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x, u):
        return self.net(torch.cat([x, u], dim=-1))

class ReplayBuffer:
    def __init__(self, size=100000, device='cpu'):
        self.buffer = deque(maxlen=size)
        self.device = device

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.tensor(state, dtype=torch.float32).to(self.device),
            torch.tensor(action, dtype=torch.float32).to(self.device),
            torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.tensor(next_state, dtype=torch.float32).to(self.device),
            torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim, act_dim, max_action, device='cpu'):
        self.device = device
        self.actor = Actor(state_dim, act_dim, max_action)
        self.critic = Critic(state_dim, act_dim)
        self.target_actor = Actor(state_dim, act_dim, max_action)
        self.target_critic = Critic(state_dim, act_dim)

        self.loss_func = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005
        self.capacity = 1000000

        self.replay_buffer = ReplayBuffer(size=self.capacity, device=device)
        self.batch_size = 64

        self._update_targets(tau=1.0) # equivalent to copying weights

    def _update_targets(self, tau=None):
        tau = self.tau if tau is None else tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def act(self, obs, noise=0.1):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = self.actor(obs_tensor).detach().numpy()[0]
        return np.clip(action + noise * np.random.randn(*action.shape), -self.max_action, self.max_action)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            target_action = self.target_actor(next_state)
            target_q = reward + self.gamma * (1 - done) * self.target_critic(next_state, target_action)

        current_q = self.critic(state, action)
        critic_loss = self.loss_func(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._update_targets()

def train():

    env = make_env()
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print("State dim:", state_dim)
    print("Action dim:", act_dim)
    print("Max action:", max_action)
    
    agent = DDPGAgent(state_dim, act_dim, max_action)

    episode_rewards = []
    num_episodes = 200

    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.add((obs, action, reward, next_obs, float(done)))

            agent.train()
            obs = next_obs
            total_reward += reward
            step += 1
            print("\rStep:", step, "Action:", action, "Reward:", reward, "Total reward:", total_reward)

        episode_rewards.append(total_reward)
        print(f"Episode {episode}: reward = {total_reward:.2f}")

        if (episode+1)%10 == 0:
            plt.plot(episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Progress')
            plt.savefig('training_progress.png')
            plt.close()

    env.close()