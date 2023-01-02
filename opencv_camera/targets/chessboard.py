##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import cv2
import numpy as np
from opencv_camera.color_space import bgr2gray, gray2bgr


class ChessboardFinder:
    def __init__(self, size, scale):
        """
        size: pattern of chess board, tuple(rows, columns)
        scale: real-world dimension of square side, example, 2 cm (0.02 m)
        """
        self.marker_size = size
        self.marker_scale = scale
        self.type = "Chessboard"
        self.has_ids = False

    def find(self, gray, flags=None):
        """
        Given an image, this will return the corners. Optionally you can enter
        flags for the cv2.findChessboardCorners() function.

        return:
            success: (True, [corner points],)
            failure: (False, [],)
        """
        if flags is None:
            flags = 0
            flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
            flags |= cv2.CALIB_CB_FAST_CHECK
            flags |= cv2.CALIB_CB_NORMALIZE_IMAGE

        # ok,gray = cv2.threshold(gray,90,255,cv2.THRESH_BINARY)

        ret, corners = cv2.findChessboardCorners(
            gray,
            self.marker_size,
            flags=flags)

        if not ret:
            corners = None
            objp = None
        else:
            objp = self.objectPoints()

        return ret, corners, objp, None

    def objectPoints(self):
        """
        Returns a set of the target's ideal 3D feature points.
        """
        objp = np.zeros((np.prod(self.marker_size), 3), np.float32)
        objp[:, :2] = np.indices(self.marker_size).T.reshape(-1, 2)*self.marker_scale
        return objp

    def draw(self, img, corners):
        """
        Draws corners on an image for viewing/debugging
        """
        if len(img.shape) < 3:
            color = gray2bgr(img)
        else:
            color = img

        cv2.drawChessboardCorners(color, self.marker_size, corners, True)
        return color
