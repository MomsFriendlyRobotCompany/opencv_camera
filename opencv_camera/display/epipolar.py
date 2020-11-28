##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import cv2
import numpy as np
from ..color_space import gray2bgr

def drawEpipolarLines(l, r, lines=True, thickness=1):
    """
    Display left/right stereo images and draw epipolar lines on image pairs.
    Images should be rectified to be meaningful.

    lines: draw lines - True/False
    thickness: how thick to draw lines, default is 1 pixel
    """
    n = np.hstack((l, r))
    if len(n.shape) < 3:
        n = gray2bgr(n)
    if lines:
        h, w = n.shape[:2]
        for r in range(0, h, 20):
            cv2.line(n,(0,r), (w,r), (200,0,0), thickness)
    return n
