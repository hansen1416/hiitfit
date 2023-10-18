"""
<_MjModelActuatorViews
  acc0: array([3359.06079787])
  actadr: array([-1])
  actlimited: array([0], dtype=uint8)
  actnum: array([0])
  actrange: array([0., 0.])
  biasprm: array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
  biastype: array([0])
  cranklength: array([0.])
  ctrllimited: array([1], dtype=uint8)
  ctrlrange: array([-1.,  1.])
  dynprm: array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
  dyntype: array([0])
  forcelimited: array([0], dtype=uint8)
  forcerange: array([0., 0.])
  gainprm: array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
  gaintype: array([0])
  gear: array([120.,   0.,   0.,   0.,   0.,   0.])
  group: array([0])
  id: 5
  length0: array([0.])
  lengthrange: array([0., 0.])
  name: 'lfemurrx'
  trnid: array([ 3, -1])
  trntype: array([0])
  user: array([], dtype=float64)
>
"""


import time
import os

import numpy as np
from dm_control import mujoco
from dm_control import viewer
from dm_control import suite

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_env import specs


import PIL.Image

xml_path = os.path.join('assets', 'xml', 'humanoid_CMU.xml')
# from dm_control import viewer

"""
physics.model is <class 'dm_control.mujoco.wrapper.core.MjModel'>
physics.data is <class 'dm_control.mujoco.wrapper.core.MjData'>
"""
physics = mujoco.Physics.from_xml_path(xml_path)

"""
def action_spec(physics):
    # Returns a `BoundedArraySpec` matching the `physics` actuators.
    num_actions = physics.model.nu
    is_limited = physics.model.actuator_ctrllimited.ravel().astype(bool)
    control_range = physics.model.actuator_ctrlrange
    minima = np.full(num_actions, fill_value=-mujoco.mjMAXVAL, dtype=float)
    maxima = np.full(num_actions, fill_value=mujoco.mjMAXVAL, dtype=float)
    minima[is_limited], maxima[is_limited] = control_range[is_limited].T

    return specs.BoundedArray(
        shape=(num_actions,), dtype=float, minimum=minima, maximum=maxima)


action_space = action_spec(physics)
print(action_space)
"""

# for attr in dir(physics.model):
#     if attr.startswith('actuator'):
#         print(attr, getattr(physics.model, attr))


class JointController:

    def __init__(self, physics) -> None:
        self.physics = physics

    def set_by_name(self, name, val):
        self.physics.data.ctrl[physics.model.actuator(name).id] = val


controller = JointController(physics)

# print(physics.model.actuator('headrx'))

# print(len(physics.data.ctrl))

physics.model.opt.gravity = [0, 0, -9.81*0.1]

scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

duration = 2    # (seconds)
framerate = 30  # (Hz)


# Simulate and display video.
frames = []
physics.reset()  # Reset state and time


controller.set_by_name('lfemurrx', 0.3)
controller.set_by_name('rfemurrx', -0.3)


while physics.data.time < duration:
    physics.step()
    # Note how we collect the video frames. Because physics simulation timesteps
    # are generally much smaller than framerates (the default timestep is 2ms),
    # we don't render after each step.
    if len(frames) < physics.data.time * framerate:
        pixels = physics.render(scene_option=scene_option)
        frames.append(PIL.Image.fromarray(pixels))

# print(frames)
# Save the frames as a GIF
frames[0].save("animation.gif", save_all=True,
               append_images=frames[1:], duration=100, loop=0)
