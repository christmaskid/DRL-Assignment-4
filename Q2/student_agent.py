import gymnasium
import numpy as np
from train import DDPGAgent, make_env

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)

        env = make_env()
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        self.agent = DDPGAgent(state_dim, act_dim, max_action, device="cpu")
        self.agent.load(ckpt_name="ckpt_q2.pt")

    def act(self, observation):
        # return self.action_space.sample()
        return self.agent.act(observation)
