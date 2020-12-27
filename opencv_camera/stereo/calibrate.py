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
from colorama import Fore
# from glob import glob
# import yaml
# import json
# import attr
# from enum import Enum
import time
# from collections import namedtuple
from ..undistort import DistortionCoefficients
from ..color_space import bgr2gray, gray2bgr
from ..mono.calibrate import CameraCalibration
from ..mono.camera import Camera
from .camera import StereoCamera




class StereoCalibration(object):
    def __init__(self, R=None, t=None):
        """
        The frame from left to right camera is [R|t]
        """
        # self.camera_model = None
        self.save_cal_imgs = None

        # if R is None:
        #     R = np.eye(3) # no rotation between left/right camera
        # self.R = R
        #
        # if t is None:
        #     t = np.array([0.1,0,0]) # 100mm baseline
        #     t.reshape((3,1))
        # self.t = t

    # def save(self, filename, handler=pickle):
    #     if self.camera_model is None:
    #         print("no camera model to save")
    #         return
    #     with open(filename, 'wb') as f:
    #         handler.dump(self.camera_model, f)

    def calibrate(self, imgs_l, imgs_r, board, flags=None):
        """
        This will save the found markers for camera_2 (right) only in
        self.save_cal_imgs array
        """
        # so we know a little bit about the camera, so
        # start off the algorithm with a simple guess
        # h,w = imgs_l[0].shape[:2]
        # f = max(h,w)*0.8  # focal length is a function of image size in pixels
        # K = np.array([
        #     [f,0,w//2],
        #     [0,f,h//2],
        #     [0,0,1]
        # ])

        cc = CameraCalibration()
        # rms1, M1, d1, r1, t1, objpoints, imgpoints_l = cc.calibrate(imgs_l, board)
        # rms2, M2, d2, r2, t2, objpoints, imgpoints_r = cc.calibrate(imgs_r, board)

        cam, data = cc.calibrate(imgs_l, board)
        K1 = cam.K
        d1 = cam.d
        rvecs1 = data["rvecs"]
        tvecs1 = data["tvecs"]
        objpoints = data["objpoints"]
        imgpoints_l =  data["imgpoints"]

        cam, data = cc.calibrate(imgs_r, board)
        K2 = cam.K
        d2 = cam.d
        rvecs2 = data["rvecs"]
        tvecs2 = data["tvecs"]
        imgpoints_r =  data["imgpoints"]

        print(d1,d2)

        self.save_cal_imgs = cc.save_cal_imgs

        """
        CALIB_ZERO_DISPARITY: horizontal shift, cx1 == cx2
        """
        if flags is None:
            flags = 0
            # flags |= cv2.CALIB_FIX_INTRINSIC
            flags |= cv2.CALIB_ZERO_DISPARITY
            # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_ASPECT_RATIO
            # flags |= cv2.CALIB_ZERO_TANGENT_DIST
            # flags |= cv2.CALIB_RATIONAL_MODEL
            # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_K3
            # flags |= cv2.CALIB_FIX_K4
            # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        h,w = imgs_l[0].shape[:2]
        ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_l,
            imgpoints_r,
            K1, d1,
            K2, d2,
            # (w,h),
            (h,w),
            # R=self.R,
            # T=self.t,
            criteria=stereocalib_criteria,
            flags=flags)

        # d1 = d1.T[0]
        # d2 = d2.T[0]

        # print('')
        camera_model = {
            'date': time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
            'markerType': board.type,
            'markerSize': board.marker_size,
            'imageSize': imgs_l[0].shape[:2],
            # 'cameraMatrix1': K1,
            # 'cameraMatrix2': K2,
            # 'distCoeffs1': d1,
            # 'distCoeffs2': d2,
            'rvecsL': rvecs1,
            "tvecsL": tvecs1,
            'rvecsR': rvecs2,
            "tvecsR": tvecs2,
            # 'R': R,
            # 'T': T,
            # 'E': E,
            # 'F': F,
            "objpoints": objpoints,
            "imgpointsL": imgpoints_l,
            "imgpointsR": imgpoints_r,
        }

        sc = StereoCamera()
        sc.R = R
        sc.E = E
        sc.F = F
        sc.T = T
        sc.K1 = K1
        sc.K2 = K2
        sc.d1 = d1
        sc.d2 = d2

        return ret, camera_model, sc
