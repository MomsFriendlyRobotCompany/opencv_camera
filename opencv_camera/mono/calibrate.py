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
import time # date saved in output
from ..color_space import bgr2gray, gray2bgr


class CameraCalibration:
    '''
    Simple calibration class.
    '''

    def findPoints(self, images, board):

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        bad_images = [] # keep track of what images failed

        if board.has_ids is True:
            tagids = []
        else:
            tagids = None

        max_corners = board.marker_size[0]*board.marker_size[1]

        for cnt, gray in enumerate(images):
            if len(gray.shape) > 2:
                gray = bgr2gray(gray)

            ok, corners, objp, ids = board.find(gray)
            if not ok:
                # bad_images.append(cnt)
                continue

            # print("board.find",ok, type(corners), len(corners), corners[0].shape, type(objp),len(objp), objp[0].shape)

            objpoints.append(objp)

            if tagids is not None:
                tagids.append(ids)

            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            # print("corners",corners.shape)
            imgpoints.append(corners.reshape(-1, 2))

        # M: number of images
        # N: number of tags found
        # tagids: list of list of ids, (M,N)
        # objpoints: list of list of 3D corner points, (M,N*4,3)
        # imgpoints: list of list of 2D corner points, (M,N*4,2)
        return objpoints, imgpoints, tagids

    def calibrate(self, images, board, flags=None):
        """
        images: an array of grayscale images, all assumed to be the same size.
            If images are not grayscale, then assumed to be in BGR format.
        board: an object that represents your target, i.e., Chessboard
        marker_scale: how big are your markers in the real world, example:
            checkerboard with sides 2 cm, set marker_scale=0.02 so your T matrix
            comes out in meters
        """
        objpoints, imgpoints, ids = self.findPoints(images, board)

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

        # not sure how much these really help
        if flags is None:
            flags = 0
            # flags |= cv2.CALIB_THIN_PRISM_MODEL
            # flags |= cv2.CALIB_TILTED_MODEL
            # flags |= cv2.CALIB_RATIONAL_MODEL

        # print(len(objpoints), len(imgpoints))
        # print(objpoints[0], imgpoints[0])

        rms, K, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.calibrateCameraExtended(
            objpoints, imgpoints, (w, h), K, None)

        data = {
            'date': time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
            'markerType': board.type,
            'markerSize': board.marker_size,
            'height': images[0].shape[0],
            'width': images[0].shape[1],
            'K': K,
            'd': dist,
            'rms': rms,
            'rvecs': rvecs,
            'tvecs': tvecs,
            "objpoints": objpoints,
            "imgpoints": imgpoints,
            # "badImages": bad_images,
            "stdint": stdDeviationsIntrinsics,
            "stdext": stdDeviationsExtrinsics,
            "perViewErr": perViewErrors,
            "height": images[0].shape[0],
            "width": images[0].shape[1]
        }

        if board.has_ids is True:
            data["ids"] = ids

        return data
