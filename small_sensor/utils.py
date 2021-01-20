import numpy as np


def wrap_to_pi(angles_in):
    return angles_in % (2 * np.pi) - np.pi


def wrap_to_2pi(angles_in):
    return (angles_in + 2 * np.pi) % (2 * np.pi)

def smallest_diff2angles(angle_a, angle_b, rads=True):
    """
    Find the smallest angle between 2 angles
    """
    a = angle_a - angle_b
    if rads:
        return (a + np.pi) % (2*np.pi) - np.pi
    else:
        return (a + 180) % 360 - 180