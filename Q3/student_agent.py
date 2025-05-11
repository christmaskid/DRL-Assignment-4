import gymnasium as gym
import numpy as np
# from train import TD3Agent, make_env
from train_ddpg import DDPGAgent, make_env

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        
        env = make_env()
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        # self.agent = TD3Agent(state_dim, act_dim, max_action, device="cpu")
        self.agent = DDPGAgent(state_dim, act_dim, max_action, device="cpu")
        # self.agent.load(ckpt_name="ckpt_q3_ddpg_3.pt")
        # self.agent.load(ckpt_name="ckpt_q3_ddpg_server.pt")
        self.agent.load(ckpt_name="ws3/ckpt_q3_ddpg.pt")

    def act(self, observation):
        # return self.action_space.sample()
        return self.agent.act(observation, deterministic=True)
