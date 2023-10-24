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


<_MjModelJointViews
  M0: array([51.8459414 , 51.8459414 , 51.8459414 ,  8.43226075,  2.32292814, 10.52306883])
  Madr: array([ 0,  1,  3,  6, 10, 15])
  armature: array([0., 0., 0., 0., 0., 0.])
  axis: array([0., 0., 1.])
  bodyid: array([1, 1, 1, 1, 1, 1])
  damping: array([0., 0., 0., 0., 0., 0.])
  dofadr: array([0])
  frictionloss: array([0., 0., 0., 0., 0., 0.])
  group: array([0])
  id: 0
  invweight0: array([0.07443921, 0.07443921, 0.07443921, 8.29058174, 8.29058174, 8.29058174])
  jntid: array([0, 0, 0, 0, 0, 0])
  limited: array([0], dtype=uint8)
  margin: array([0.])
  name: 'root'
  parentid: array([-1,  0,  1,  2,  3,  4])
  pos: array([0., 0., 0.])
  qpos0: array([0., 0., 1., 0.70710678, 0.70710678,0., 0.])
  qpos_spring: array([0., 0., 1., 0.70710678, 0.70710678,0., 0.])
  qposadr: array([0])
  range: array([0., 0.])
  simplenum: array([0, 0, 0, 0, 0, 0])
  solimp: array([[9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e+00],
       [9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e+00],
       [9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e+00],
       [9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e+00],
       [9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e+00],
       [9.0e-01, 9.5e-01, 1.0e-03, 5.0e-01, 2.0e+00]])
  solref: array([[0.02, 1.  ],
       [0.02, 1.  ],
       [0.02, 1.  ],
       [0.02, 1.  ],
       [0.02, 1.  ],
       [0.02, 1.  ]])
  stiffness: array([0.])
  type: array([0])
  user: array([], dtype=float64)
>
"""


import os

import numpy as np
from dm_control import mujoco
from dm_control import viewer
from dm_control import suite
# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
import PIL.Image
from scipy.spatial.transform import Rotation

from utils import JointsController, euclidean_distance

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
#     # if attr.startswith('actuator'):
#     if 'jnt' in attr:
#         print(attr, getattr(physics.model, attr))


"""
actuator are the actions
data.qpos are the observations
model.jnt to set static pose

venv\Lib\site-packages\dm_control\suite\humanoid_CMU.py
venv\Lib\site-packages\dm_control\suite\base.py
venv\Lib\site-packages\dm_control\mujoco\engine.py
venv\Lib\site-packages\dm_control\rl\control.py

obs['joint_angles'] = physics.joint_angles()
obs['head_height'] = physics.head_height()
obs['extremities'] = physics.extremities()
obs['torso_vertical'] = physics.torso_vertical_orientation()
obs['com_velocity'] = physics.center_of_mass_velocity()
obs['velocity'] = physics.velocity()

xmat is a 3x3 matrix that describes the orientation of the body in space. 
It is used to transform the body’s local coordinate system to the global coordinate system 1.
it include the workdbody

xpos is a 3D vector that represents the position of the body’s center of mass 
in the global coordinate system 1.

xquat is a quaternion that describes the orientation of the body in space.
"""


# print(physics.model.actuator('headrx'))
# print(len(physics.data.ctrl))
physics.model.opt.gravity = [0, 0, -9.81*0]

# print(dir(physics))
# exit()

scene_option = mujoco.wrapper.core.MjvOption()
scene_option.flags[enums.mjtVisFlag.mjVIS_JOINT] = True

jntController = JointsController(physics)

# # list all joint names
# for i in range(1, physics.model.njnt):
#     # this doesn't include root freejoint
#     print(physics.model.jnt(i).name)

# # Cartesian orientation of body frame, include worldbody

# print(np.round(physics.data.xquat[1:]))

# # # use all joints rotation as action space, (56,)
# print(jntController.get_joints_rotation())
# print(len(jntController.get_joints_rotation()))
# exit()
# # body rotation as observation spaece, exclude worldbody, (32, 4)
# print(physics.data.xquat.shape)


duration = 2    # (seconds)
framerate = 30  # (Hz)


# Simulate and display video.
frames = []

physics.reset()  # Reset state and time

# d1 = euclidean_distance(start_state, target_state)

# a = np.round(np.copy(physics.data.ximat), decimals=2)


# d2 = euclidean_distance(jntController.get_joints_rotation(), target_state)

# print(d1, d2)

# physics.step()
# print(np.round(physics.data.ximat, decimals=2))
# exit()


# factor = math.pi / 180
# step = 0

factor = 1
step = 1


xquat = None

while physics.data.time < duration:
    physics.step()

    # Note how we collect the video frames. Because physics simulation timesteps
    # are generally much smaller than framerates (the default timestep is 2ms),
    # we don't render after each step.
    if len(frames) < physics.data.time * framerate:

        # # set the joint rotation directly to get the desired pose
        jntController.set_rotation_by_names({
            'lfemurrz': 0.17 * factor * step,
            'rfemurrz': -0.17 * factor * step,
            'lhumerusrz': -1.4 * factor * step,
            'lhumerusrx': 0.5 * factor * step,
            'rhumerusrz': 1.4 * factor * step,
            'rhumerusrx': 0.5 * factor * step,
        })

        # r_matrix = Rotation.from_matrix(physics.data.xmat[1:])

        # reshape physics.data.xmat[1:], from (n, 9) to (n, 3, 3)
        r_matrix = Rotation.from_matrix(
            physics.data.xmat[1:].reshape(-1, 3, 3))

        r_euler = r_matrix.as_euler('xyz', degrees=False)

        # print(np.round(r_matrix.as_euler('xyz', degrees=False), decimals=2))

        if xquat is None:
            xquat = np.copy(r_euler)
        else:
            dist = euclidean_distance(xquat, r_euler)

            print(dist, dist < 0.01)
            xquat = np.copy(r_euler)

        # print(physics.data.xquat)

        # step += 1

        pixels = physics.render(scene_option=scene_option)
        frames.append(PIL.Image.fromarray(pixels))

print(step)

# output `xmat`, we will use as the target for reinforcement learning
# b = np.round(physics.data.xquat, decimals=2)

# for i in range(a.shape[0]):
#     print(angle_between_quat(a[i], b[i]))

# print(a, b)

# print(frames)
# Save the frames as a GIF
frames[0].save("tmp.gif", save_all=True,
               append_images=frames[1:], duration=100, loop=0)
