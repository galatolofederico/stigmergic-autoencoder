import gym
import numpy as np

class AutoEncoderEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, world, isShared=False):
        super(AutoEncoderEnv, self).__init__()

        self.reset()
        
 
    def step(self, action):


    def reset(self):


    def render(self, rendercolors="state", mode="human", close=False):

