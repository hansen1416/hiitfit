"""
Tips:

RL, data is collected through interactions with the environment by the agent itself, 
this could lead to vicious circle, so RL may vary from one run to another. 
always do several runs to have quantitative results.

RL are generally dependent on finding appropriate hyperparameters.
A best practice when you apply RL to a new problem is to do automatic hyperparameter optimization. 

When applying RL to a custom problem, you should always normalize the input to the agent,
and look at common preprocessing done on other environments

This reward engineering, necessitates several iterations,  
Deep Mimic combines imitation learning and reinforcement learning to do acrobatic moves.

RL is the instability of training. You can observe during training a huge drop in performance. 
This behavior is particularly present in DDPG, that's why its extension TD3 tries to tackle that issue. 
Other method, like TRPO or PPO make use of a trust region to minimize that problem by avoiding too large update.

Because most algorithms use exploration noise during training, 
you need a separate test environment to evaluate the performance of your agent at a given time. 
It is recommended to periodically evaluate your agent for n test episodes (n is usually between 5 and 20) 
and average the reward per episode to have a good estimate.
"""

import math
import os

import numpy as np
import mujoco
import mujoco.viewer
from transforms3d import quaternions
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

from utils import euclidean_distance


class MotionEnv(gym.Env):

    def __init__(self):

        super(MotionEnv, self).__init__()

        # action space is 3 continuous values, representing of 3 rotations of the shoulder
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # we use the 3 rotation of shoulder as observation, 3 for start rotations, 3 for current positions, 3 for target positions
        # future, use positions of all joints as observation
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        xml_path = os.path.join('assets', 'xml', 'scene.xml')

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        mujoco.mj_kinematics(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        self.joint_r_shoulder_p = self.model.joint("R_SHOULDER_P")  # pitch
        self.joint_r_shoulder_r = self.model.joint("R_SHOULDER_R")  # roll
        self.joint_r_shoulder_y = self.model.joint("R_SHOULDER_Y")  # yaw

        self.model.opt.gravity = (0, 0, 0)

        self.viewer = None

        self.initial_state = np.array([0, 0, 0])
        self.target_state = np.array([1, 1, 1])

        self.steps_took = 0

        print("__init__ called")

    def __del__(self):

        # self.viewer.exit()
        print("__del__ called")

    def _get_obs(self):

        return np.array([self.joint_r_shoulder_p.qpos0[0], self.joint_r_shoulder_r.qpos0[0], self.joint_r_shoulder_y.qpos0[0],
                         self.target_state[0], self.target_state[1], self.target_state[2]], dtype=np.float32)

    def step(self, action):

        self.steps_took += 1

        # scale all action to pi. and limit to 1 degree per step
        action_scaled = action * (math.pi / 180)

        start_state = np.array([self.joint_r_shoulder_p.qpos0[0], self.joint_r_shoulder_r.qpos0[0],
                                self.joint_r_shoulder_y.qpos0[0]], dtype=np.float32)

        self.joint_r_shoulder_p.qpos0[0] += action_scaled[0]
        self.joint_r_shoulder_r.qpos0[0] += action_scaled[1]
        self.joint_r_shoulder_y.qpos0[0] += action_scaled[2]

        mujoco.mj_step(self.model, self.data)

        current_state = np.array([self.joint_r_shoulder_p.qpos0[0], self.joint_r_shoulder_r.qpos0[0],
                                  self.joint_r_shoulder_y.qpos0[0]], dtype=np.float32)

        # observation is current state concatenated with target state
        observation = self._get_obs()
        # reward is the distance between current state and target state
        # if current closer to target, the reward is higher, and vice versa
        reward = euclidean_distance(
            start_state, self.target_state) - euclidean_distance(current_state, self.target_state)

        # if current state is np.close true to target state, done
        done = bool(np.isclose(current_state,
                    self.target_state, atol=0.01).all())

        # when step reach a certain number, truncate

        truncate = True if self.steps_took > 1000 else False

        return observation, reward, done, truncate, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self.model, self.data)

        self.joint_r_shoulder_p.qpos0[0] = self.initial_state[0]
        self.joint_r_shoulder_r.qpos0[0] = self.initial_state[1]
        self.joint_r_shoulder_y.qpos0[0] = self.initial_state[2]

        observation = self._get_obs()

        self.steps_took = 0

        return observation, {}

    def render(self, mode="human"):

        if self.viewer is None:

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self.viewer.cam.lookat[0] = 0  # x position
            self.viewer.cam.lookat[1] = 0  # y position
            self.viewer.cam.lookat[2] = 0  # z position
            self.viewer.cam.distance = 6  # distance from the target
            self.viewer.cam.elevation = -30  # elevation angle
            self.viewer.cam.azimuth = 180  # azimuth angle

        if mode == "human":

            self.viewer.sync()

    def close(self):

        self.viewer.close()


if __name__ == "__main__":

    env = MotionEnv()

    check_env(env)

    env.reset()

    for _ in range(40000):

        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, truncate, info = env.step(action)

        env.render()

        if done == True:
            break

    env.close()
