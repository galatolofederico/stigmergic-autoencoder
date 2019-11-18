from stigenv import StigAutoEncoderEnv
from mnist import MNIST

from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import *
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv



dataset = MNIST()
ae = StigAutoEncoderEnv(size=10, dataset=dataset)
ae.fill()
env = DummyVecEnv(([lambda: ae]))


model = PPO2(MlpPolicy, env, n_steps=512,
        verbose=1, tensorboard_log="./tb_logs",
        #exploration_fraction=.5, prioritized_replay=True
        )

try:
    model.learn(total_timesteps=1000*1000)
except KeyboardInterrupt:
    pass

