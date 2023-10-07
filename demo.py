import time
import os

import itertools
import numpy as np
import mujoco
import mujoco.viewer


xml_path = os.path.join('assets','xml', 'scene.xml')

model = mujoco.MjModel.from_xml_path(xml_path)





# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

# print('default gravity', model.opt.gravity)
# print('default timestep', model.opt.timestep)

# print('all geom names', [model.geom(i).name for i in range(model.ngeom)])
model.opt.gravity = (0,0,0)

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


# By calling viewer.launch_passive(model, data).
# This function does not block, allowing user code to continue execution.
# In this mode, the user’s script is responsible for timing and advancing the physics state,
# and mouse-drag perturbations will not work unless the user explicitly synchronizes incoming events.
with mujoco.viewer.launch_passive(model, data) as viewer:

    # set camera
    viewer.cam.lookat[0] = 0  # x position
    viewer.cam.lookat[1] = 0  # y position
    viewer.cam.lookat[2] = 0  # z position
    viewer.cam.distance = 10  # distance from the target
    viewer.cam.elevation = 0  # elevation angle
    viewer.cam.azimuth = 0  # azimuth angle

    # print('default gravity', model.opt.gravity)
    # model.opt.gravity = (0, 0, 10)
    # print('flipped gravity', model.opt.gravity)

    mujoco.mj_resetData(model, data)

    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()

    while viewer.is_running() and time.time() - start < 5:

        step_start = time.time()

        mujoco.mj_step(model, data)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)