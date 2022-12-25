##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import numpy as np
from numpy.linalg import inv, norm

def findFundamentalMat(K1, K2, R, t, normalize=True):
    """
    OpenCV appears to normalize the F matrix. It can contain
    some really large and really small values. See CSE486
    lecture 19 notes [1] for examples of the F matrix.
    Also see Epipolar Geometery, Table 9.1, last line (Cameras
    not at inf)[2] for how to calculate it.

    K1/K2: camera matrix for left/right camera
    R/T: the rotation and translation from cv2.stereoCalibrate

    [1] http://www.cse.psu.edu/~rtc12/CSE486/lecture19.pdf
    [2] http://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf
    """
    t = t.ravel()
    A = (K1 @ R.T @ t)
    C = np.array([
        [    0, -A[2], A[1]],
        [ A[2],     0,-A[0]],
        [-A[1],  A[0],    0]
    ])
    ret = inv(K2).T @ R @ K1.T @ C
    if normalize:
        ret = ret/norm(ret)
        # ret = ret/ret[2,2] # gives OpenCV answer
    return ret
