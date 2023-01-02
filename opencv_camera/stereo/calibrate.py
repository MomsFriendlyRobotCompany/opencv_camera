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
    save_cal_imgs = None

    # def __init__(self, R=None, t=None):
    #     """
    #     The frame from left to right camera is [R|t]
    #     """
    #     # self.camera_model = None
    #     self.save_cal_imgs = None

    #     # if R is None:
    #     #     R = np.eye(3) # no rotation between left/right camera
    #     # self.R = R
    #     #
    #     # if t is None:
    #     #     t = np.array([0.1,0,0]) # 100mm baseline
    #     #     t.reshape((3,1))
    #     # self.t = t

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

        data = cc.calibrate(imgs_l, board)
        K1 = data["K"]
        d1 = data["d"]
        rvecs1 = data["rvecs"]
        tvecs1 = data["tvecs"]
        objpts = data["objpoints"]
        imgptsL = data["imgpoints"]

        # print(objpoints)

        # time.sleep(1)

        data = cc.calibrate(imgs_r, board)
        K2 = data["K"]
        d2 = data["d"]
        rvecs2 = data["rvecs"]
        tvecs2 = data["tvecs"]
        imgptsR = data["imgpoints"]

        # print(d1,d2)

        # self.save_cal_imgs = cc.save_cal_imgs
        objpoints = []
        imgpoints_r = []
        imgpoints_l = []
        for o,l,r in zip(objpts,imgptsL,imgptsR):
            # must have all the same number of points for calibration
            if o.shape[0] == l.shape[0] == r.shape[0]:
                objpoints.append(o)
                imgpoints_r.append(r)
                imgpoints_l.append(l)
            else:
                print("bad points:", o.shape, l.shape, r.shape)

        print(f"Object Pts: {len(objpoints)}")
        print(f"Left Image Pts: {len(imgpoints_l)}")
        print(f"Right Image Pts: {len(imgpoints_r)}")

        """
        CALIB_ZERO_DISPARITY: horizontal shift, cx1 == cx2
        """
        if flags is None:
            flags = 0
            # flags |= cv2.CALIB_FIX_INTRINSIC
            flags |= cv2.CALIB_ZERO_DISPARITY
            # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_ASPECT_RATIO
            # flags |= cv2.CALIB_ZERO_TANGENT_DIST
            # flags |= cv2.CALIB_RATIONAL_MODEL
            # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
            # flags |= cv2.CALIB_FIX_K3
            # flags |= cv2.CALIB_FIX_K4
            # flags |= cv2.CALIB_FIX_K5
            # flags |= cv2.CALIB_USE_EXTRINSIC_GUESS

        stereocalib_criteria = (
            cv2.TERM_CRITERIA_MAX_ITER +
            cv2.TERM_CRITERIA_EPS,
            100,
            1e-5)

        for o in objpoints:
            print(o.shape)

        print("objpoints",type(objpoints),len(objpoints),objpoints[0].shape)
        print("imgpoints_l",type(imgpoints_l),len(imgpoints_l),imgpoints_l[0].shape)
        print("imgpoints_r",type(imgpoints_r),len(imgpoints_r),imgpoints_r[0].shape)

        cv2.utils.dumpInputArrayOfArrays(objpoints)
        cv2.utils.dumpInputArrayOfArrays(imgpoints_l)
        cv2.utils.dumpInputArrayOfArrays(imgpoints_r)

        h,w = imgs_l[0].shape[:2]
        ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_l,
            imgpoints_r,
            K1, d1,
            K2, d2,
            (w,h),
            # (h,w),
            # R=self.R,
            # T=self.t,
            # R=np.eye((3,3)),
            # T=np.array([[0.031],[0],[0]]), # FIXME
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
            'K1': K1,
            'K2': K2,
            'd1': d1,
            'd2': d2,
            'rvecsL': rvecs1,
            "tvecsL": tvecs1,
            'rvecsR': rvecs2,
            "tvecsR": tvecs2,
            'R': R,
            'T': T,
            'E': E,
            'F': F,
            "objpoints": objpoints,
            "imgpointsL": imgpoints_l,
            "imgpointsR": imgpoints_r,
        }

        return ret, camera_model
