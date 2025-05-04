import gymnasium as gym
import numpy as np
from train import DDPGAgent, make_env

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)

        env = make_env()
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        self.agent = DDPGAgent(state_dim, act_dim, max_action, device="cpu")
        self.agent.load(ckpt_name="ckpt.pt")

    def act(self, observation):
        # return self.action_space.sample()
        return self.agent.act(observation)