import math

import numpy as np


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


class JointsController:

    def __init__(self, physics) -> None:
        self.physics = physics

    def get_joints_rotation(self):
        # get all joints rotation, exclude the first freejoint
        return np.array([self.physics.model.jnt(i).qpos0[0] for i in range(1, self.physics.model.njnt)])

    def set_joint_rotation(self, name, rotation):
        # all joints are hinge joints, so just set the radian
        self.physics.model.jnt(name).qpos0[0] = rotation

    def reset_joint_rotations(self):
        # reset all joints rotation to 0
        for i in range(1, self.physics.model.njnt):
            self.physics.model.jnt(i).qpos0[0] = 0


if __name__ == "__main__":

    a = [1, 2, 3,]
    b = [1, 2, 3]

    res = np.isclose(a, b, atol=0.01).all()

    print(res)
