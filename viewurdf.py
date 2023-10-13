import time
import os

import itertools
import numpy as np
import mujoco
import mujoco.viewer


# xml_path = os.path.join('.', 'humanSubject01_48dof.urdf')
xml_path = os.path.join('.', 'humanoid.urdf')

model = mujoco.MjModel.from_xml_path(xml_path)

