##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import cv2
from enum import IntFlag


#                                   1   2   3   4
ColorSpace = IntFlag("ColorSpace", "bgr rgb hsv gray")

bgr2rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
bgr2gray = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray2bgr = lambda im: cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
