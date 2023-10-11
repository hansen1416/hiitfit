import time
import os

import itertools
import numpy as np
import mujoco
import mujoco.viewer


xml_path = os.path.join('assets', 'xml', 'scene.xml')

model = mujoco.MjModel.from_xml_path(xml_path)


# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# print('default gravity', model.opt.gravity)
# print('default timestep', model.opt.timestep)

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


# print(dir(model))

# for attr in dir(model):
#     # # property `attr` of model that contains 'joint' or 'jnt'
#     # if 'joint' in attr or 'jnt' in attr:
#     #     # use attr as a string to get the property
#     #     print(attr, getattr(model, attr))

#     if 'sensor' in attr:
#         print(attr, getattr(model, attr))


# print(model.joint())
# print(model.jnt())
# exit()


# print('default gravity', model.opt.gravity)
model.opt.gravity = (0, 0, 0)
# print('flipped gravity', model.opt.gravity)

model.opt.timestep = 0.01


alive_sec = 5
total_frames = alive_sec / model.opt.timestep
ellapsed_frames = 0

# pitch, 0 1 0, range="-3.14159 1.0472", real range="-3.14159 3.14159"
joint_r_shoulder_p = model.joint("R_SHOULDER_P")
# roll 1 0 0 range="-1.74533 0.174533", real range="-0.3 3.27"
joint_r_shoulder_r = model.joint("R_SHOULDER_R")
# yaw 0 0 1 range="-1.5708 1.5708", real range="-0.6 0.3"
joint_r_shoulder_y = model.joint("R_SHOULDER_Y")

joint_r_hip_p = model.joint("R_HIP_P")
joint_r_hip_r = model.joint("R_HIP_R")
joint_r_hip_y = model.joint("R_HIP_Y")

r_shouder_pitch_start = 1
r_shouder_roll_start = 0.5
r_shouder_yaw_start = 0.3



# r_shouder_pitch_end = 2.14
# r_shouder_roll_end = 2.27
# r_shouder_yaw_end = -0.6

# r_shouder_pitch_step = (r_shouder_pitch_end -
#                         r_shouder_pitch_start) / total_frames
# r_shouder_roll_step = (r_shouder_roll_end -
#                        r_shouder_roll_start) / total_frames
# r_shouder_yaw_step = (r_shouder_yaw_end - r_shouder_yaw_start) / total_frames


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

    # set initial position of the shoulder joints
    joint_r_shoulder_p.qpos0[0] = r_shouder_pitch_start
    joint_r_shoulder_r.qpos0[0] = r_shouder_roll_start
    joint_r_shoulder_y.qpos0[0] = r_shouder_yaw_start

    joint_r_hip_p.qpos0[0] = 1
    joint_r_hip_r.qpos0[0] = 1
    joint_r_hip_y.qpos0[0] = 1

    start = time.time()

    while viewer.is_running() and time.time() - start < alive_sec:

        step_start = time.time()

        mujoco.mj_step(model, data)

        # print(data.qpos)

        # joint_r_shoulder_p.qpos0[0] = r_shouder_pitch_start + \
        #     r_shouder_pitch_step * ellapsed_frames
        # joint_r_shoulder_r.qpos0[0] = r_shouder_roll_start + \
        #     r_shouder_roll_step * ellapsed_frames
        # joint_r_shoulder_y.qpos0[0] = r_shouder_yaw_start + \
        #     r_shouder_yaw_step * ellapsed_frames

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        ellapsed_frames += 1

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    print(ellapsed_frames)
