##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from math import atan, pi


def fov(w,f):
        """
        Returns the FOV as in degrees, given:
            w: image width (or height) in pixels
            f: focalLength (fx or fy) in pixels
        """
        return 2*atan(w/2/f) * 180/pi


params = {
    "sensor": "OV9714",
    "focalLength": 3.11,  # mm
    "area": (3936, 2460), # (w,h)
    "size": (1296, 812),  # (w,h)
}
