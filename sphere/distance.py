#!/usr/bin/env python

__author__ = "Evripidis Gkanias"
__copyright__ = "Copyright (c) 2019, Insect Robotics Group," \
                "Institude of Perception, Action and Behaviour," \
                "School of Informatics, the University of Edinburgh"
__credits__ = ["Evripidis Gkanias"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Evripidis Gkanias"


import numpy as np
from .transform import sph2vec, vec2sph


def angle_between(ang1, ang2, sign=True):
    d = (ang1 - ang2 + np.pi) % (2 * np.pi) - np.pi
    if not sign:
        d = np.abs(d)
    return d


def angdist(v1, v2, zenith=True):
    if v1.shape[0] == 2:
        v1 = sph2vec(v1, zenith=zenith)
    if v2.shape[0] == 2:
        v2 = sph2vec(v2, zenith=zenith)
    v1 /= np.linalg.norm(v1, axis=0)
    v2 /= np.linalg.norm(v2, axis=0)

    if v1.ndim > 1 or v2.ndim > 1:
        d = np.einsum('ij,ij->j', v1, v2)
    else:
        d = np.dot(v1.T, v2)
    # if d.ndim > 1:
    #     d = d.diagonal()
    np.seterr(all='raise')
    # try:
    #     np.arccos(d)
    # except RuntimeWarning as e:
    #     print ("d is {}".format(d))
    #     print (e)
    # except FloatingPointError as e:
    #     print("d is {}".format(d))
    #     print(e)
    np.seterr(all='warn')

    # np.seterr(all='log')

    return np.absolute(np.arccos(d))


def eledist(v1, v2, zenith=True):
    if v1.shape[0] == 3:
        v1 = vec2sph(v1, zenith=zenith)
    if v2.shape[0] == 3:
        v2 = vec2sph(v2, zenith=zenith)
    d = (v1[0] - v2[0] + np.pi) % (2 * np.pi) - np.pi
    return np.absolute(d)


def azidist(v1, v2, zenith=True):
    if v1.shape[0] == 3:
        v1 = vec2sph(v1, zenith=zenith)
    if v2.shape[0] == 3:
        v2 = vec2sph(v2, zenith=zenith)
    d = (v1[1] - v2[1] + np.pi) % (2 * np.pi) - np.pi
    return np.absolute(d)

