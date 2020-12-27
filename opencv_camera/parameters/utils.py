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
