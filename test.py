import time

from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

from MotionEnv import MotionEnv



model = PPO.load("models/MotionEnv-PPO/100000.zip")

# print(model)
env = MotionEnv()
obs, _ = env.reset()


if __name__ == "__main__":

    # print(obs)
    while True:
        action, _ = model.predict(obs)

        obs, rewards, dones, truncate, info = env.step(action)

        time.sleep(0.1)

        env.render()

        print("action: {}, reward: {}".format(action, rewards))
