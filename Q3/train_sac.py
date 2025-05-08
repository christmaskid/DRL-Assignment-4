# This script is generated with the aid of LLM (ChatGPT4o) for a template of DDPG.
# However, it is mostly similar to the code of last homework (HW3 Q4), which was solely coded on my own (w/ ref. cited in the HW3 Q4 code).
# Reference to the DDPG algorithm can be found in the class materials.

# Some other references:
# [1] https://github.com/sfujim/TD3/blob/master/TD3.py: on the smooth regularization, network architecture design, and some other constants

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from dmc import make_dmc_env

def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env


# Actor (policy network)
class Actor(nn.Module):
    def __init__(self, state_dim, act_dim, max_action, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh() # output: [-1, 1]
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x) # scale

# Critic (value network)
class Critic(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, u):
        return self.net(torch.cat([x, u], dim=-1))
class ReplayBuffer:
    def __init__(self, size=100000, device='cpu'):
        self.buffer = deque(maxlen=size)
        self.device = device

    def add(self, item):
        self.buffer.append(item)

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

class TD3Agent:
    def __init__(self, state_dim, act_dim, max_action, hidden_dim=256, lr=3e-4, device="cuda"):
        
        self.device = device
        self.max_action = max_action

        # Two critics for minimization
        self.actor = Actor(state_dim, act_dim, max_action, hidden_dim=hidden_dim).to(self.device)
        self.critic1 = Critic(state_dim, act_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, act_dim, hidden_dim=hidden_dim).to(self.device)

        self.target_actor = Actor(state_dim, act_dim, max_action, hidden_dim=hidden_dim).to(self.device)
        self.target_critic1 = Critic(state_dim, act_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, act_dim, hidden_dim=hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

        self.gamma = 0.99
        self.tau = 0.005 # soft update
        self.batch_size = 256
        self.capacity = 1000000
        
        self.policy_noise = 0.2
        self.noise_clip = 0.5 # "c"
        self.policy_delay = 2

        self.replay_buffer = ReplayBuffer(size=self.capacity, device=device)
        self.step = 0

        # Sync at the start
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    def act(self, obs, noise=0.01):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(obs).detach().cpu().numpy()[0]
        action += noise * np.random.randn(*action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return np.nan, np.nan, np.nan

        self.step += 1
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Target policy smooth regularization
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) # "epsilon" ~ clip(N(0, sigma), -c, c)
            next_action = (self.target_actor(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = reward + self.gamma * (1 - done) * torch.min(target_q1, target_q2)

        # Update critics
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        critic_loss_1 = self.loss_func(current_q1, target_q)
        critic_loss_2 = self.loss_func(current_q2, target_q)

        self.critic_optimizer1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer2.step()

        # Delayed actor update
        if self.step % self.policy_delay == 0:
            actor_loss = -self.critic1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.target_actor, self.actor)
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)

            return actor_loss.item(), critic_loss_1.item(), critic_loss_2.item()
        
        return None, critic_loss_1.item(), critic_loss_2.item()

    def _soft_update(self, target, online):
        for t_param, param in zip(target.parameters(), online.parameters()):
            t_param.data.copy_(self.tau * param + (1 - self.tau) * t_param)

    def save(self):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict()
        }, "ckpt_q3.pt")
    
    def load(self, ckpt_name="ckpt_q3.pt"):
        state_dict = torch.load(ckpt_name, map_location=torch.device(self.device), weights_only=True)
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.target_critic1.load_state_dict(state_dict["target_critic1"])
        self.target_critic2.load_state_dict(state_dict["target_critic2"])

def train():

    env = make_env()
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print("State dim:", state_dim)
    print("Action dim:", act_dim)
    print("Max action:", max_action)
    
    agent = TD3Agent(state_dim, act_dim, max_action, hidden_dim=1024, device="cuda", lr=1e-3)

    reward_history = []
    num_episodes = 10000

    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        actor_loss_print = np.nan
        print(flush=True)

        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.add((obs, action, reward, next_obs, float(done)))

            actor_loss, critic_loss_1, critic_loss_2 = agent.train()
            obs = next_obs
            total_reward += reward
            step += 1

            if actor_loss is not None:
                actor_loss_print = actor_loss
            print(f"\rStep: {step}, Reward: {reward}, Total reward: {total_reward:.2f}, Critics loss: {critic_loss_1:.10f}, {critic_loss_2:.10f}, Actor loss: {actor_loss_print:.10f}", end="", flush=True)

        agent.save()
        reward_history.append(total_reward)
        print(f"\nEpisode {episode}: step: {step}, total reward: {total_reward:.2f}     ", flush=True)

        if (episode+1)%10 == 0:
            plt.plot(reward_history)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Progress')
            plt.savefig('training_progress.png')
            plt.close()

    env.close()

if __name__=="__main__":
    train()