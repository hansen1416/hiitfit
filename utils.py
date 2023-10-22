import math

import numpy as np
from transforms3d import quaternions


class ActuatorController:

    def __init__(self, physics) -> None:
        self.physics = physics

    def set_value(self, name, val):
        self.physics.data.ctrl[self.physics.model.actuator(name).id] = val


class JointsController:

    def __init__(self, physics) -> None:
        self.physics = physics

    def get_joints_rotation(self):
        # get all joints rotation, exclude the first freejoint
        return np.round(np.array([self.physics.model.jnt(i).qpos0[0] for i in range(1, self.physics.model.njnt)]), decimals=2)

    def set_joint_rotation(self, name, rotation):
        # all joints are hinge joints, so just set the radian
        self.physics.model.jnt(name).qpos0[0] = rotation

    def set_rotation_by_names(self, rotation_dict):
        # set rotation by joint names
        for name, rotation in rotation_dict.items():
            self.physics.model.jnt(name).qpos0[0] = rotation

    def set_all_rotations(self, rotations):
        # set all joints qpos0, except the root freejoint
        for i in range(len(rotations)):
            self.physics.model.jnt(i+1).qpos0[0] = rotations[i]

    def reset_joint_rotations(self):
        # reset all joints rotation to 0
        for i in range(1, self.physics.model.njnt):
            self.physics.model.jnt(i).qpos0[0] = 0


def euclidean_distance(a, b):
    """
    calculate euclidean distance between two vectors
    """
    # calculating Euclidean distance
    # using linalg.norm()
    return np.linalg.norm(np.array(a) - np.array(b))


def manhattan_distance(a, b):
    """
    Manhattan distance 
    """
    return sum(abs(val1-val2) for val1, val2 in zip(a, b))


def chebychef_distance(vec1, vec2):
    dist = np.max(np.absolute(np.array(vec1) - np.array(vec2)))
    return dist


def cosine_similarity(v1, v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def angle_between_mat(a, b):
    a = np.copy(a).reshape(3, 3)
    b = np.copy(b).reshape(3, 3)

    # get quaternion from rotation matrix
    q1 = quaternions.mat2quat(a)
    q2 = quaternions.mat2quat(b)

    # get quternion that rotate from q1 to q2
    q = quaternions.qmult(q2, quaternions.qinverse(q1))

    # get rotation axis and angle
    _, angle = quaternions.quat2axangle(q)

    # print(axis, round(angle,2), angle < 0.001, np.isclose(angle, 0.0, atol=1e-3))
    return angle


def angle_between_quat(q1, q2):

    # get quternion that rotate from q1 to q2
    q = quaternions.qmult(q2, quaternions.qinverse(q1))

    # get rotation axis and angle
    _, angle = quaternions.quat2axangle(q)

    # print(axis, round(angle,2), angle < 0.001, np.isclose(angle, 0.0, atol=1e-3))
    return angle


if __name__ == "__main__":

    a = [1, 2, 3,]
    b = [1, 2, 3]

    res = np.isclose(a, b, atol=0.01).all()

    print(res)
