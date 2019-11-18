import gym
import numpy as np
import random

class StigAutoEncoderEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, size, dataset, stig_max=100, stig_mark=10, stig_tick=2, max_ticks=10):
        super(StigAutoEncoderEnv, self).__init__()
        self.dataset = dataset
        self.dataset.scale(stig_max)
        self.x_train, _ = dataset.get_train()
        self.size = size

        self.stig_max = stig_max
        self.stig_mark = stig_mark
        self.stig_tick = stig_tick
        self.max_ticks = max_ticks
        
        self.action_space = gym.spaces.Discrete(self.size)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.stig_max, 
            shape=(self.x_train.shape[1] + self.size, ), dtype=np.uint8
        )
        
        self.stig_space = None
        self.reset()


    def updateSpace(self, action):
        self.stig_space[action] += self.stig_mark
        self.stig_space -= self.stig_tick
        self.stig_space[self.stig_space < 0] = 0

    
    def getDone(self):
        return self.i >= self.max_ticks

    def computeReward(self):
        return 0

    def getObservation(self):
        return np.concatenate((self.x, self.stig_space))

    def step(self, action):
        self.i += 1
        self.updateSpace(action)
        reward = self.computeReward()
        done = self.getDone()
        observation = self.getObservation()

        return observation, reward, done, {}


    def reset(self):
        self.stig_space = np.zeros(self.size)
        self.x = self.x_train[random.randint(0, self.x_train.shape[0]-1)]
        self.i = 0

        return self.getObservation()

    def render(self, rendercolors="state", mode="human", close=False):
        pass
