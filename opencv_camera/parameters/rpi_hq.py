##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from .utils import fov

# https://www.raspberrypi.org/documentation/hardware/camera/
params = {
    "sensor": "Sony IMX477",
    "resolution": 12.3, # Megapixels
    "focalLength": None, # depends on lens
    "area": (6.287, 4.712), # (w,h) mm
    "size": (4056, 3040),  # (w,h) px
    "fov": None # depends on lens
}
