# This script is generated with the aid of LLM (ChatGPT4o) for a template of DDPG.
# However, it is mostly similar to the code of last homework (HW3 Q4), which was solely coded on my own (w/ ref. cited in the HW3 Q4 code).
# Reference to the DDPG algorithm can be found in the class materials.

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
        return self.max_action * self.net(x)

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
    def __init__(self, state_dim, act_dim, max_action, device='cpu', lr=2.5e-4, hidden_dim=256):
        self.device = device
        self.actor = Actor(state_dim, act_dim, max_action, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, act_dim, hidden_dim).to(self.device)
        self.target_actor = Actor(state_dim, act_dim, max_action, hidden_dim).to(self.device)
        self.target_critic = Critic(state_dim, act_dim, hidden_dim).to(self.device)

        self.loss_func = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self.gamma = 0.99
        self.tau = 0.005
        self.capacity = 1_000_000

        self.replay_buffer = ReplayBuffer(size=self.capacity, device=device)
        self.batch_size = 64

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


    def _soft_update(self, target, online):
        for t_param, param in zip(target.parameters(), online.parameters()):
            t_param.data.copy_(self.tau * param + (1 - self.tau) * t_param)

    def act(self, obs, deterministic=False):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(obs_tensor)
        if not deterministic:
            noise = torch.distributions.Normal(0, 0.2).rsample().to(self.device)
            action = torch.clamp(action + noise * torch.rand_like(action), -self.max_action, self.max_action)
        return action.detach().cpu().numpy()[0]

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return np.nan, np.nan

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

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        return critic_loss.item(), actor_loss.item()

    def save(self):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict()
        }, "ckpt_q3_ddpg.pt")
    
    def load(self, ckpt_name="ckpt_q3_ddpg.pt"):
        state_dict = torch.load(ckpt_name, map_location=torch.device(self.device), weights_only=True)
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.target_critic.load_state_dict(state_dict["target_critic"])

def train():

    env = make_env()
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print("State dim:", state_dim)
    print("Action dim:", act_dim)
    print("Max action:", max_action)
    
    agent = DDPGAgent(state_dim, act_dim, max_action, device="cpu")

    episode_rewards = []
    num_episodes = 10000

    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        print(flush=True)

        while not done:
            action = agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.add((obs, action, reward, next_obs, float(done)))

            critic_loss, actor_loss = agent.train()
            obs = next_obs
            total_reward += reward
            step += 1
            
            print(f"\rStep: {step}, Reward: {reward}, Total reward: {total_reward},",
                f"Critic loss: {critic_loss:.8f}, Actor loss: {actor_loss:.8f}", 
                end="", flush=True)

        agent.save()
        episode_rewards.append(total_reward)
        print(f"\nEpisode {episode}: step: {step}, total reward: {total_reward:.2f}     ", flush=True)

        if (episode+1)%10 == 0:
            plt.plot(episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Progress')
            plt.savefig('training_progress_ddpg.png')
            plt.close()

    env.close()

if __name__=="__main__":
    train()
