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
import time

import numpy as np
from dm_control import mujoco
from dm_control.mujoco.wrapper.mjbindings import enums
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import PIL.Image

from utils import JointsController, angle_between_quat


def pose_angle_diff(pose1, pose2):
    """
    calculate the angle difference between two poses
    """
    angle_diff = np.zeros(pose1.shape[0])
    for i in range(pose1.shape[0]):
        angle_diff[i] = angle_between_quat(pose1[i], pose2[i])

    return angle_diff


class MotionEnv(gym.Env):

    def __init__(self):

        super(MotionEnv, self).__init__()

        # action space is 3 continuous values, representing of 3 rotations of the shoulder
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(56,), dtype=np.float32)
        # body rotation as observation spaece, exclude worldbody, concatenate with target state, shape (62, 4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(62*4, ), dtype=np.float32)

        xml_path = os.path.join('assets', 'xml', 'humanoid_CMU.xml')

        self.physics = mujoco.Physics.from_xml_path(xml_path)

        self.physics.model.opt.gravity = [0, 0, -9.81*0]

        self.target_state = np.array([[0.71, 0.71, -0.,  -0., ],
                                      [0.71, 0.71, -0.,  -0., ],
                                      [0.71, 0.7, -0.06, 0.06],
                                      [0.7,  0.71, -0.06, 0.06],
                                      [1.,   0.01, -0.09, 0., ],
                                      [1.,   0.01, -0.09, 0., ],
                                      [0.71, 0.71, -0.,  -0., ],
                                      [0.71, 0.7,  0.06, -0.06],
                                      [0.7,  0.71, 0.06, -0.06],
                                      [1.,   0.01, 0.09, -0., ],
                                      [1.,   0.01, 0.09, -0., ],
                                      [0.71, 0.71, -0.,  -0., ],
                                      [0.71, 0.71, -0.,  -0., ],
                                      [0.71, 0.71, -0.,  -0., ],
                                      [0.71, 0.71, -0.,  -0., ],
                                      [0.71, 0.71, -0.,  -0., ],
                                      [0.71, 0.71, -0.,  -0., ],
                                      [0.71, 0.71, -0.,  -0., ],
                                      [-0.02, 0.08, -0.71, -0.7,],
                                      [-0.02, 0.08, -0.71, -0.7,],
                                      [0.63, 0.56, 0.34, 0.42],
                                      [0.63, 0.56, 0.34, 0.42],
                                      [0.63, 0.56, 0.34, 0.42],
                                      [0.74, 0.39, 0.53, 0.15],
                                      [0.71, 0.71, -0.,  -0., ],
                                      [-0.02, 0.08, 0.71, 0.7,],
                                      [-0.02, 0.08, 0.71, 0.7,],
                                      [0.63, 0.56, -0.34, -0.42],
                                      [0.63, 0.56, -0.34, -0.42],
                                      [0.63, 0.56, -0.34, -0.42],
                                      [0.74, 0.39, -0.53, -0.15],],)

        self.steps_took = 0
        self.frames = []
        self.framerate = 60

        self.scene_option = mujoco.wrapper.core.MjvOption()
        self.scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

        self.jntController = JointsController(self.physics)

        print("__init__ called")

    def __del__(self):

        # self.viewer.exit()
        print("__del__ called")

    def _get_obs(self):
        # body rotation as observation spaece, exclude worldbody,
        # concatenate with target state, shape (62, 4)
        # then flatten to (62*4, )
        return np.concatenate((self.physics.data.xquat[1:], self.target_state), axis=0, dtype=np.float32).flatten()

    def step(self, action):

        self.steps_took += 1

        # get state before taking action
        start_state = np.copy(self.physics.data.xquat[1:]).astype(np.float32)

        # scale all action to pi. and limit to 1 degree per step
        action_scaled = action * (math.pi / 180)

        # apply action to all joints
        for i in range(1, self.physics.model.njnt):
            self.jntController.set_joint_rotation(
                self.physics.model.jnt(i).name, action_scaled[i-1])

        self.physics.step()

        if (self.steps_took % self.framerate) == 0:
            pixels = self.physics.render(scene_option=self.scene_option)
            self.frames.append(PIL.Image.fromarray(pixels))

        current_state = np.copy(self.physics.data.xquat[1:]).astype(np.float32)

        # get angle difference with the target state before and after apply actions
        # if state is closer to target state after apply the action,
        # reward is positive and vice versa
        start_angle_diff = pose_angle_diff(start_state, self.target_state)
        current_angle_diff = pose_angle_diff(current_state, self.target_state)
        reward = np.sum(np.sqrt(start_angle_diff)) - \
            np.sum(np.sqrt(current_angle_diff))

        # if current state is np.close true to target state, done
        done = bool(np.isclose(current_angle_diff, np.zeros(
            current_angle_diff.shape[0]), atol=0.01).all())

        # when step reach a certain number, truncate

        truncate = True if self.steps_took > 1000 else False

        # observation is current state concatenated with target state
        observation = self._get_obs()

        return observation, reward, done, truncate, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.physics.reset()

        self.steps_took = 0
        self.frames = []

        return self._get_obs(), {}

    def render(self, mode="human"):
        # save the frames as a GIF, to path frames/{timestamp}.gif
        filename = os.path.join(
            "frames", f"{time.time()}-{self.steps_took}.gif")

        self.frames[0].save(filename, save_all=True,
                            append_images=self.frames[1:], duration=100, loop=0)

    def close(self):
        pass


if __name__ == "__main__":

    env = MotionEnv()

    check_env(env)

    env.reset()

    for _ in range(40000):

        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, truncate, info = env.step(action)

        if truncate == True:
            env.render()

        if done == True:
            env.render()
            break

    env.close()
