# This script is generated with the aid of LLM (ChatGPT4o) for a template of DDPG.
# However, it is mostly similar to the code of last homework (HW3 Q4), which was solely coded on my own (w/ ref. cited in the HW3 Q4 code).
# Reference to the DDPG algorithm can be found in the class materials.

# Some other references:
# [1] https://github.com/sfujim/TD3/blob/master/TD3.py: on the smooth regularization, network architecture design, and some other constants
# [2] https://github.com/XinJingHao/TD3-Pytorch/blob/main/TD3.py: also on soem hyperparameters

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

class DoubleCritic(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net1 = Critic(state_dim, act_dim, hidden_dim)
        self.net2 = Critic(state_dim, act_dim, hidden_dim)

    def forward(self, x, u):
        return self.net1(x, u), self.net2(x, u)

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
    def __init__(self, state_dim, act_dim, max_action, hidden_dim=256, lr=1e-4, device="cuda"):
        
        self.device = device
        self.max_action = max_action

        # Two critics for minimization
        self.actor = Actor(state_dim, act_dim, max_action, hidden_dim=hidden_dim).to(self.device)
        self.critic = DoubleCritic(state_dim, act_dim, hidden_dim=hidden_dim).to(self.device)

        self.target_actor = Actor(state_dim, act_dim, max_action, hidden_dim=hidden_dim).to(self.device)
        self.target_critic = DoubleCritic(state_dim, act_dim, hidden_dim=hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()

        self.epsilon = 0.15
        self.epsilon_decay = 0.999975
        self.warmup = 10000
        self.update_every = 50

        self.gamma = 0.99
        self.tau = 0.005 # soft update
        self.batch_size = 256
        self.capacity = 1000000
        
        self.policy_noise = 0.2 * self.max_action
        self.noise_clip = 0.5 * self.max_action # "c"
        self.policy_delay = 2

        self.replay_buffer = ReplayBuffer(size=self.capacity, device=device)
        self.step = 0

        # Sync at the start
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

    def act(self, obs, deterministic=True):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(obs).detach().cpu().numpy()[0]
        if deterministic:
            return action
        else:
            noise = np.random.randn(*action.shape) * self.epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, 0.0001)
            return np.clip(action + noise, -self.max_action, self.max_action)

    def train(self):
        self.step += 1
        if self.step < self.warmup or self.step % self.update_every != 0:
            return None, None, None
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            # Target policy smooth regularization
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip) # "epsilon" ~ clip(N(0, sigma), -c, c)
            next_action = (self.target_actor(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.target_critic(next_state, next_action)
            target_q = reward + self.gamma * (1 - done) * torch.min(target_q1, target_q2)

        # Update critics
        current_q1, current_q2 = self.critic(state, action)

        critic_loss_1 = self.loss_func(current_q1, target_q)
        critic_loss_2 = self.loss_func(current_q2, target_q)
        loss = critic_loss_1 + critic_loss_2

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # Delayed actor update
        if self.step % self.policy_delay == 0:
            actor_loss = -self.critic(state, self.actor(state))[0].mean() # pick q1
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.target_actor, self.actor)
            self._soft_update(self.target_critic, self.critic)

            return actor_loss.item(), critic_loss_1.item(), critic_loss_2.item()
        
        return None, critic_loss_1.item(), critic_loss_2.item()

    def _soft_update(self, target, online):
        for t_param, param in zip(target.parameters(), online.parameters()):
            t_param.data.copy_(self.tau * param + (1 - self.tau) * t_param)

    def save(self):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }, "ckpt_q3.pt")
    
    def load(self, ckpt_name="ckpt_q3.pt"):
        state_dict = torch.load(ckpt_name, map_location=torch.device(self.device), weights_only=True)
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_actor.load_state_dict(state_dict["target_actor"])
        self.target_critic.load_state_dict(state_dict["target_critic"])

def eval(env, agent):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step = 0
    print(flush=True)
    
    while not done:
        action = agent.act(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = next_obs
        total_reward += reward
        step += 1
    print(f"[EVAL] Step = {step}, total reward = {total_reward}", flush=True)

def train(load_ckpt=None):

    env = make_env()
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print("State dim:", state_dim)
    print("Action dim:", act_dim)
    print("Max action:", max_action)
    
    agent = TD3Agent(state_dim, act_dim, max_action, device="cuda")
    if load_ckpt is not None:
        agent.load(load_ckpt)

    reward_history = []
    num_episodes = 10000

    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        step = 0
        print(flush=True)
        
        actor_loss_print, critic_loss_1_print, critic_loss_2_print = np.nan, np.nan, np.nan

        while not done:
            if agent.step < agent.warmup: # explore, [2]
                action = env.action_space.sample()
            else:
                action = agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            surrogate_reward = reward #-0.1 if reward < 1e-10 else reward # punish falling down
            agent.replay_buffer.add((obs, action, surrogate_reward, next_obs, float(done)))
            
            actor_loss, critic_loss_1, critic_loss_2 = agent.train()
            
            obs = next_obs
            total_reward += reward
            step += 1
            
            def set_not_none(x, y):
                if y is not None:
                    return y
                return x
            actor_loss_print = set_not_none(actor_loss_print, actor_loss)
            critic_loss_1_print = set_not_none(critic_loss_1_print, critic_loss_1)
            critic_loss_2_print = set_not_none(critic_loss_2_print, critic_loss_2)
        
            print(f"\rStep: {step}, Reward: {reward}, Total reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}",
                f"Critics loss: {critic_loss_1_print:.10f}, {critic_loss_2_print:.10f},",
                f"Actor loss: {actor_loss_print:.10f}", end="", flush=True)

        agent.save()
        reward_history.append(total_reward)
        print(f"\nEpisode {episode}: step: {step}, total reward: {total_reward:.2f}, epsilon: {agent.epsilon:.4f}     ", flush=True)

        if (episode+1)%10 == 0:
            if episode > 499:
                eval(env, agent)
            plt.plot(reward_history)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Progress')
            plt.savefig('training_progress.png')
            plt.close()


    env.close()

if __name__=="__main__":
    import sys
    train(load_ckpt=sys.argv[1] if len(sys.argv)>=2 else None)