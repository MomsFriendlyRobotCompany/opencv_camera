##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from .utils import fov

# https://www.raspberrypi.org/documentation/hardware/camera/
params = {
    "sensor": "Sony IMX219",
    "resolution": 8, # Megapixels
    "focalLength": 3.04,  # mm
    "area": (3.68, 2.76), # (w,h) mm
    "size": (3280, 2464),  # (w,h) px
    "fov": (fov(3936, 3.11), fov(2460, 3.11),),
    "focalRatio": 2.0
}
