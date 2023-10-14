import time
import os

import itertools
import numpy as np
import mujoco
import mujoco.viewer


# xml_path = os.path.join('assets', 'xml', 'scene.xml')
xml_path = os.path.join('assets', 'xml', 'human.xml')

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

# model.opt.gravity = (0, 0, 10)
# derived quantities in mjData need to be explicitly propagated
mujoco.mj_kinematics(model, data)
# print('raw access:\n', data.geom_xpos)


# MuJoCo's use of generalized coordinates is the reason that calling a function (e.g. mj_forward)
# is required before rendering or reading the global poses of objects –
# Cartesian positions are derived from the generalized positions and need to be explicitly computed.
mujoco.mj_forward(model, data)

alive_sec = 5


joint_ankle_l = model.joint("ANKLE_L")

# print(joint_ankle_l.qpos)

# By calling viewer.launch_passive(model, data).
# This function does not block, allowing user code to continue execution.
# In this mode, the user’s script is responsible for timing and advancing the physics state,
# and mouse-drag perturbations will not work unless the user explicitly synchronizes incoming events.
with mujoco.viewer.launch_passive(model, data) as viewer:

    # set camera
    viewer.cam.lookat[0] = 0  # x position
    viewer.cam.lookat[1] = 0  # y position
    viewer.cam.lookat[2] = 0  # z position
    viewer.cam.distance = 40  # distance from the target
    viewer.cam.elevation = -30  # elevation angle
    viewer.cam.azimuth = 0  # azimuth angle

    mujoco.mj_resetData(model, data)

    start = time.time()

    joint_ankle_l.qpos0[0] = 1

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

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
