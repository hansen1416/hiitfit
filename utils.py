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


if __name__ == "__main__":

    a = [1, 2, 3,]
    b = [1, 2, 3]

    res = np.isclose(a, b, atol=0.01).all()

    print(res)
