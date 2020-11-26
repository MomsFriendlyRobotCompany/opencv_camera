##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import cv2
from collections import namedtuple

DistortionCoefficients = namedtuple("DistortionCoefficients", "k1 k2 p1 p2 k3")

class UnDistort:
    def __init__(self, camMat, dist, w, h):
        """Remove"""
        self.camMat = camMat
        self.dist = dist
        self.size = (w,h)
        self.optCamMat, _ = cv2.getOptimalNewCameraMatrix(camMat, dist, self.size, 0)

    def set(self, camMat, dist, w, h):
        """
        Sets up the class with an Optimal Camera Matrix alpha of zero, which
        removes all unwanted pixels

        camMat: camera matrix
        dist: distortion coefficients from calibration
        w: width
        h: height
        """
        self.camMat = camMat
        self.dist = dist
        self.size = (w,h)
        self.optCamMat, _ = cv2.getOptimalNewCameraMatrix(camMat, dist, self.size, 0)

    def set_camera(self, camera):
        """
        Given a camera, setup class
        """
        self.camMat = camera.K
        self.dist = camera.distortion
        self.size = camera.size
        self.optCamMat, _ = cv2.getOptimalNewCameraMatrix(
            self.camMat,
            self.dist,
            self.size,
            0
        )

    def undistort(self, image, alpha=None):
        """
        image: an image
        alpha: values between 0 and 1 which determines the amount of unwanted
            pixels. The default is 0, but if changed, a new Optimal Camera
            Matrix is calculated for the alpha

        alpha = 0: returns undistored image with minimum unwanted pixels (image
                    pixels at corners/edges could be missing)
        alpha = 1: retains all image pixels but there will be black to make up
                    for warped image correction
        """
        # Adjust the calibrations matrix
        if alpha is not None:
            self.optCamMat, _ = cv2.getOptimalNewCameraMatrix(
                self.camMat,
                self.dist,
                self.size,
                alpha
            )
        # undistort
        return cv2.undistort(image, self.camMat, self.dist, None, self.optCamMat)
