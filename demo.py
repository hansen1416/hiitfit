"""
<_MjModelJointViews
  M0: array([2.88207967])
  Madr: array([21], dtype=int32)
  armature: array([0.1925])
  axis: array([0., 1., 0.])
  bodyid: array([2], dtype=int32)
  damping: array([0.2])
  dofadr: array([6], dtype=int32)
  frictionloss: array([0.])
  group: array([0], dtype=int32)
  id: 1
  invweight0: array([2.06123843])
  jntid: array([1], dtype=int32)
  limited: array([1], dtype=uint8)
  margin: array([0.])
  name: 'R_HIP_P'
  parentid: array([5], dtype=int32)
  pos: array([0., 0., 0.])
  qpos0: array([0.])
  qpos_spring: array([0.])
  qposadr: array([7], dtype=int32)
  range: array([-2.0944  ,  0.785398])
  simplenum: array([0], dtype=int32)
  solimp: array([[9.5e-01, 1.0e-03, 5.0e-01, 2.0e+00, 9.0e-01]])
  solref: array([[0.02, 1.  ]])
  stiffness: array([0.])
  type: array([3], dtype=int32)
  user: array([], dtype=float64)
>
"""


import time
import os


import numpy as np
import mujoco
import mujoco.viewer


class JointsController:

    def __init__(self, model) -> None:

        self.model = model

        self.joint_names = [
            "R_HIP_P",
            "R_HIP_R",
            "R_HIP_Y",
            "R_KNEE",
            "R_ANKLE_R",
            "R_ANKLE_P",
            "L_HIP_P",
            "L_HIP_R",
            "L_HIP_Y",
            "L_KNEE",
            "L_ANKLE_R",
            "L_ANKLE_P",
            "WAIST_Y",
            "WAIST_P",
            "WAIST_R",
            "NECK_Y",
            "NECK_R",
            "NECK_P",
            "R_SHOULDER_P",
            "R_SHOULDER_R",
            "R_SHOULDER_Y",
            "R_ELBOW_P",
            "R_ELBOW_Y",
            "R_WRIST_R",
            "R_WRIST_Y",
            "L_SHOULDER_P",
            "L_SHOULDER_R",
            "L_SHOULDER_Y",
            "L_ELBOW_P",
            "L_ELBOW_Y",
            "L_WRIST_R",
            "L_WRIST_Y",
        ]

    def get_joints_rotation(self):

        return [self.model.jnt(joint_name).qpos0[0] for joint_name in self.joint_names]

    def set_joint_rotation(self, joint_name, rotation):

        self.model.jnt(joint_name).qpos0[0] = rotation


xml_path = os.path.join('assets', 'xml', 'scene.xml')
# xml_path = os.path.join('assets', 'xml', 'human.xml')

model = mujoco.MjModel.from_xml_path(xml_path)


# # enable joint visualization option:
# scene_option = mujoco.MjvOption()
# scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True


# print('all geom names', [model.geom(i).name for i in range(model.ngeom)])

# mjData contains the state and quantities that depend on it.
# The state is made up of time, generalized positions and generalized velocities.
# These are respectively data.time, data.qpos and data.qvel.
data = mujoco.MjData(model)

# geom positions
# print(data.geom_xpos)


# derived quantities in mjData need to be explicitly propagated
mujoco.mj_kinematics(model, data)
# print('raw access:\n', data.geom_xpos)


# MuJoCo's use of generalized coordinates is the reason that calling a function (e.g. mj_forward)
# is required before rendering or reading the global poses of objects –
# Cartesian positions are derived from the generalized positions and need to be explicitly computed.
mujoco.mj_forward(model, data)

# MuJoCo uses a representation known as the "Lagrangian", "generalized" or "additive" representation,
# whereby objects have no degrees of freedom unless explicitly added using joints.
# print('Total number of DoFs in the model:', model.nv)
# print('Generalized positions:', data.qpos)
# print('Generalized velocities:', data.qvel)


model.opt.timestep = 0.01


alive_sec = 5
total_frames = alive_sec / model.opt.timestep
ellapsed_frames = 0

# print('default gravity', model.opt.gravity)
model.opt.gravity = (0, 0, -9.81*0)

controller = JointsController(model)


# By calling viewer.launch_passive(model, data).
# This function does not block, allowing user code to continue execution.
# In this mode, the user’s script is responsible for timing and advancing the physics state,
# and mouse-drag perturbations will not work unless the user explicitly synchronizes incoming events.
with mujoco.viewer.launch_passive(model, data) as viewer:

    # set camera
    viewer.cam.lookat[0] = 0  # x position
    viewer.cam.lookat[1] = 0  # y position
    viewer.cam.lookat[2] = 0  # z position
    viewer.cam.distance = 6  # distance from the target
    viewer.cam.elevation = -30  # elevation angle
    viewer.cam.azimuth = 180  # azimuth angle

    mujoco.mj_resetData(model, data)

    controller.set_joint_rotation('R_HIP_R', 0.12)
    controller.set_joint_rotation('L_HIP_R', -0.12)

    start = time.time()

    while viewer.is_running() and time.time() - start < alive_sec:

        step_start = time.time()

        mujoco.mj_step(model, data)

        print(controller.get_joints_rotation())
        print(data.qpos)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        ellapsed_frames += 1

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    print(ellapsed_frames)
