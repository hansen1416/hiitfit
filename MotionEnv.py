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

from utils import JointsController, angle_between_quat, euclidean_distance


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
        # difference between current joint rotation qpos0 and target state as observation space, exclude root freejoint,
        self.observation_space = spaces.Box(
            low=-1., high=1., shape=(56, ), dtype=np.float32)

        xml_path = os.path.join('assets', 'xml', 'humanoid_CMU.xml')

        self.physics = mujoco.Physics.from_xml_path(xml_path)

        self.physics.model.opt.gravity = [0, 0, -9.81*0]

        self.target_state = np.array([
            0.17,  0.,    0.,    0.,    0.,    0.,    0.,   -0.17,  0.,    0.,    0.,    0.,
            0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,
            0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,   -1.4,   0.,
            0.5,   0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,    1.4,   0.,
            0.5,   0.,    0.,    0.,    0.,    0.,    0.,    0.,
        ])

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
        # difference between current state and target state
        obs = self.jntController.get_joints_rotation() - self.target_state
        # scale to -1 to 1
        obs = obs / math.pi
        # cast data type to float32
        obs = obs.astype(np.float32)

        return obs

    def step(self, action):

        self.steps_took += 1

        # get state before taking action
        start_state = np.copy(
            self.jntController.get_joints_rotation()).astype(np.float32)

        # scale all action to pi. and limit to 1 degree per step
        action_scaled = np.round(action * (math.pi / 60), decimals=2)
        # action_scaled = action * 0.1

        # apply action to all joints, exclude root freejoint
        # so the first action is for joint 1
        self.jntController.set_all_rotations(action_scaled)

        self.physics.step()

        if (self.steps_took % self.framerate) == 0:
            pixels = self.physics.render(scene_option=self.scene_option)
            self.frames.append(PIL.Image.fromarray(pixels))

        current_state = np.copy(
            self.jntController.get_joints_rotation()).astype(np.float32)

        # get angle difference with the target state before and after apply actions
        # if state is closer to target state after apply the action,
        # reward is positive and vice versa
        start_angle_diff = euclidean_distance(start_state, self.target_state)
        current_angle_diff = euclidean_distance(
            current_state, self.target_state)
        reward = (start_angle_diff - current_angle_diff) * 1000

        # when current is close enough to target, done
        terminated = bool(current_angle_diff < 0.01)

        if terminated:
            self.render()

        # when step reach a certain number, truncate
        truncate = True if self.steps_took > 2000 else False

        # observation is current state concatenated with target state
        observation = self._get_obs()

        return observation, reward, terminated, truncate, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.physics.reset()

        # self.render()

        self.steps_took = 0
        self.frames = []

        return self._get_obs(), {}

    def render(self, mode="human"):

        if not len(self.frames):
            return

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

    # start_state = np.copy(env.physics.data.xquat[1:]).astype(np.float32)

    # print(np.round(start_state))

    # # factor = (math.pi / 180)
    # factor = 0.1

    # # env.jntController.set_joint_rotation('lfemurrz', 0.17*factor)
    # # env.jntController.set_joint_rotation('rfemurrz', -0.17*factor)

    # # env.jntController.set_joint_rotation('lhumerusrz', -1.4*factor)
    # # env.jntController.set_joint_rotation('lhumerusrx', 0.5*factor)
    # # env.jntController.set_joint_rotation('rhumerusrz', 1.4*factor)
    # # env.jntController.set_joint_rotation('rhumerusrx', 0.5*factor)

    # print(env.jntController.get_joints_rotation())

    # env.physics.step()

    # current_state = np.copy(env.physics.data.xquat[1:]).astype(np.float32)

    # print(np.round(current_state))

    # start_angle_diff = pose_angle_diff(start_state, env.target_state)

    # current_angle_diff = pose_angle_diff(current_state, env.target_state)
    # reward = np.sum(np.sqrt(start_angle_diff)) - \
    #     np.sum(np.sqrt(current_angle_diff))

    # print(reward)

    # exit()

    for _ in range(4000):

        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, truncate, info = env.step(action)

        if truncate == True:
            env.render()
            env.reset()

        # if done == True:
        #     env.render()
        #     break

    env.close()
