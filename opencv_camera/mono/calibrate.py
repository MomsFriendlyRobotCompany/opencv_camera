##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
import cv2
import attr
import time
from collections import namedtuple
from ..undistort import DistortionCoefficients
from ..color_space import bgr2gray, gray2bgr
from .camera import Camera
from tqdm import tqdm


# @attr.s(slots=True)
class CameraCalibration:
    '''
    Simple calibration class.
    '''
    # marker_type = attr.ib()
    # marker_size = attr.ib()
    # marker_scale = attr.ib(default=1)
    # save_cal_imgs = attr.ib(default=None)
    # save_objpoints = attr.ib(default=None)
    # save_imgpoints = attr.ib(default=None)

    save_cal_imgs = None

    def calibrate(self, images, board, flags=None):
        """
        images: an array of grayscale images, all assumed to be the same size.
            If images are not grayscale, then assumed to be in BGR format.
        board: an object that represents your target, i.e., Chessboard
        marker_scale: how big are your markers in the real world, example:
            checkerboard with sides 2 cm, set marker_scale=0.02 so your T matrix
            comes out in meters
        """
        self.save_cal_imgs = []

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        max_corners = board.marker_size[0]*board.marker_size[1]

        # print("Images: {} @ {}".format(len(images), images[0].shape))
        # print("{} {}".format(board.type, board.marker_size))
        # print('-'*40)
        # cnt = 0
        for cnt, gray in enumerate(tqdm(images)):
            if len(gray.shape) > 2:
                gray = bgr2gray(gray)

            # ret, corners = self.findMarkers(gray)
            ret, corners = board.find(gray)

            # If found, add object points, image points (after refining them)
            if ret:
                # imgpoints.append(corners.reshape(-1, 2))

                # get the real-world pattern of points
                objp = board.objectPoints()
                objpoints.append(objp)

                # print('[{}] + found {} of {} corners'.format(
                #     cnt, corners.size / 2, max_corners))
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
                imgpoints.append(corners.reshape(-1, 2))

                # Draw the corners
                # tmp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # cv2.drawChessboardCorners(tmp, board.marker_size, corners, True)
                # tmp = board.draw(gray, corners)
                # self.save_cal_imgs.append(tmp)
            else:
                print(f'{Fore.RED}*** Image[{cnt}] - Could not find markers ***{Fore.RESET}')

        # images size here is backwards: w,h
        h, w = images[0].shape[:2]

        # initial guess for camera matrix
        # K = None # FIXME
        f = 0.8*w
        cx, cy = w//2, h//2
        K = np.array([
            [ f,  0, cx],
            [ 0,  f, cy],
            [ 0,  0,  1]
        ])

        if flags is None:
            flags = 0
            # flags |= cv2.CALIB_THIN_PRISM_MODEL
            # flags |= cv2.CALIB_TILTED_MODEL
            # flags |= cv2.CALIB_RATIONAL_MODEL

        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), K, None, flags=flags)

        # rms, mtx, dist, rvecs, tvecs = cv2.calibrateCameraExtended(
        #     objpoints, imgpoints, (w, h), K, None)

        data = {
            'date': time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
            'markerType': board.type,
            'markerSize': board.marker_size,
            'imageSize': images[0].shape,
            'K': mtx,
            'd': dist, #DistortionCoefficients(dist),
            'rms': rms,
            'rvecs': rvecs,
            'tvecs': tvecs,
            "objpoints": objpoints,
            "imgpoints": imgpoints
        }

        cam = Camera(mtx, dist, images[0].shape[:2])

        return cam, data
