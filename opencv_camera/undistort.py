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
    def __init__(self, K, d, h, w, R=None):
        """
        Sets up the class with an Optimal Camera Matrix alpha of zero, which
        removes all unwanted pixels

        camMat: camera matrix
        dist: distortion coefficients from calibration
        w: width
        h: height
        R: the rotation matrix from cv2.stereoCalibration(), this is optional.
        For undistorting stereo images, only use R on the right camera image
        and not the left one.
        """
        self.K = K
        self.d = d
        self.size = (w,h) # backwards
        self.shape = (h,w)
        optCamMat, _ = cv2.getOptimalNewCameraMatrix(K, d, self.size, 0)

        self.R = R
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            K,d,
            R,
            optCamMat,
            self.size, # (w,h) -- backwards # (w,h) -- backwards
            cv2.CV_32FC1
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

        Note: this is about 5x faster than using cv2.undistort(), BUT the
        self.mapx and self.mapy are EACH the same size as the image and both
        are float32. So although faster, it consumes more memory.
        """
        if self.shape != image.shape:
            raise Exception(f"Undistort set for image.shape = {self.shape}, not {image.shape}")

        if alpha is not None:
            optCamMat, _ = cv2.getOptimalNewCameraMatrix(
                self.K,
                self.d,
                self.size, # (w,h) -- backwards
                alpha
            )
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.K, self.d, self.R,
                optCamMat,
                self.size, # (w,h) -- backwards
                cv2.CV_32FC1)

        return cv2.remap(image,self.mapx,self.mapy,cv2.INTER_LINEAR)

    # def distortionMap(self):
    #     """
    #     Return a numpy array representing the image distortion
    #     """
    #     raise NotImplemented()
