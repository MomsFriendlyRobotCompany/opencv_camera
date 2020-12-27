##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from .utils import fov

params = {
    "sensor": "OV9714",
    "focalLength": 3.11,  # mm
    "area": (3936, 2460), # (w,h) mm
    "size": (1296, 812),  # (w,h) px
    "fov": (fov(3936, 3.11), fov(2460, 3.11),) # deg
}
