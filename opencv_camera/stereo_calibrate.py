##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import numpy as np
np.set_printoptions(precision=1)
np.set_printoptions(suppress=True)
import cv2
# from glob import glob
# import yaml
# import json
# import attr
# from enum import Enum
import time
# from collections import namedtuple
from .undistort import DistortionCoefficients
from .color_space import bgr2gray, gray2bgr


class StereoCalibration(object):
    def __init__(self, R=None, t=None):
        """
        The frame from left to right camera is [R|t]
        """
        # self.camera_model = None
        self.save_cal_imgs = None

        if R is None:
            R = np.eye(3) # no rotation between left/right camera
        self.R = R

        if t is None:
            t = np.array([0.1,0,0]) # 100mm baseline
            t.reshape((3,1))
        self.t = t

    # def save(self, filename, handler=pickle):
    #     if self.camera_model is None:
    #         print("no camera model to save")
    #         return
    #     with open(filename, 'wb') as f:
    #         handler.dump(self.camera_model, f)

    def stereo_calibrate(self, imgs_l, imgs_r, board, flags=None):
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
        M1 = data["cameraMatrix"]
        d1 = data["distCoeffs"]
        objpoints = data["objpoints"]
        imgpoints_l =  data["imgpoints"]

        data = cc.calibrate(imgs_r, board)
        M1 = data["cameraMatrix"]
        d1 = data["distCoeffs"]
        imgpoints_l =  data["imgpoints"]

        self.save_cal_imgs = cc.save_cal_imgs

        if flags is None:
            flags = 0
            # flags |= cv2.CALIB_FIX_INTRINSIC
            flags |= cv2.CALIB_ZERO_DISPARITY
            # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS  # make an inital guess at cameraMatrix (K)
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

        ret, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_l,
            imgpoints_r,
            K1, d1,
            K2, d2,
            imgs_l[0].shape[:2],
            R=self.R,
            T=self.T,
            criteria=stereocalib_criteria,
            flags=flags)

        print('-'*50)
        print('Image: {}x{}'.format(*imgs_l[0].shape[:2]))
        print('{}: {}'.format(marker_type, marker_size))
        print('Intrinsic Camera Parameters')
        print('-'*50)
        print(' [Camera 1]')
        # print('  cameraMatrix_1', M1)
        print('  f(x,y): {:.1f} {:.1f} px'.format(K1[0,0], K1[1,1]))
        print('  principlePoint(x,y): {:.1f} {:.1f} px'.format(K1[0,2], K1[1,2]))
        print('  distCoeffs', d1[0])
        print(' [Camera 2]')
        # print('  cameraMatrix_2', M2)
        print('  f(x,y): {:.1f} {:.1f} px'.format(K2[0,0], K2[1,1]))
        print('  principlePoint(x,y): {:.1f} {:.1f} px'.format(K2[0,2], K2[1,2]))
        print('  distCoeffs', d2[0])
        print('-'*50)
        print('Extrinsic Camera Parameters')
        print('-'*50)
        print('  R', R)
        print('  T[meter]', T)
        print('  E', E)
        print('  F', F)

        # for i in range(len(r1)):
        #     print("--- pose[", i+1, "] ---")
        #     ext1, _ = cv2.Rodrigues(r1[i])
        #     ext2, _ = cv2.Rodrigues(r2[i])
        #     print('Ext1', ext1)
        #     print('Ext2', ext2)

        # print('')
        camera_model = {
            'date': time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
            'markerType': board.type,
            'markerSize': board.marker_size,
            'imageSize': imgs_l[0].shape[:2],
            'cameraMatrix1': K1,
            'cameraMatrix2': K2,
            'distCoeffs1': d1,
            'distCoeffs2': d2,
            # 'rvecs1': r1,
            # 'rvecs2': r2,
            'R': R,
            'T': T,
            'E': E,
            'F': F
        }

        return ret, camera_model
