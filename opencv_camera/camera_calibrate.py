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
import attr
import time
from collections import namedtuple
from .undistort import DistortionCoefficients
from .color_space import bgr2gray, gray2bgr
# from .targets/chessboard import ChessboardFinder

# DistortionCoefficients = namedtuple("DistortionCoefficients", "k1 k2 p1 p2 k3")
# Markers = Enum('Markers', 'checkerboard circle acircle apriltag')


def calculateReprojectionError(imgpoints, objpoints, rvecs, tvecs, mtx, dist):
    """
    imgpts: features found in image, (num_imgs, 2)
    objpts: calibration known features in 3d, (num_imgs, 3)
    rvecs: rotations
    tvecs: translations
    mtx: camera matrix
    dist: distortion coefficients [k1,k2,p1,p2,k3]

    returns:
        mean_error, x_error[list], y_error[list]
    """
    imgpoints = [c.reshape(-1,2) for c in imgpoints]
    mean_error = 0
    error_x = []
    error_y = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        imgpoints2 = imgpoints2.reshape(-1,2)

        # if not all markers were found, then the norm below will fail
        if len(imgpoints[i]) != len(imgpoints2):
            continue

        error_x += list(imgpoints2[:,0] - imgpoints[i][:,0])
        error_y += list(imgpoints2[:,1] - imgpoints[i][:,1])

        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    m_error = mean_error/len(objpoints)
    return m_error, error_x, error_y


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

    def calculateReprojectionError(self, imgpoints, objpoints, rvecs, tvecs, mtx, dist):
        """
        imgpts: features found in image, (num_imgs, 2)
        objpts: calibration known features in 3d, (num_imgs, 3)
        rvecs: rotations
        tvecs: translations
        mtx: camera matrix
        dist: distortion coefficients [k1,k2,p1,p2,k3]

        returns:
            mean_error, x_error[list], y_error[list]
        """
        imgpoints = [c.reshape(-1,2) for c in imgpoints]
        mean_error = 0
        error_x = []
        error_y = []
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            imgpoints2 = imgpoints2.reshape(-1,2)

            # if not all markers were found, then the norm below will fail
            if len(imgpoints[i]) != len(imgpoints2):
                continue

            error_x += list(imgpoints2[:,0] - imgpoints[i][:,0])
            error_y += list(imgpoints2[:,1] - imgpoints[i][:,1])

            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        m_error = mean_error/len(objpoints)
        return m_error, error_x, error_y

    def calibrate(self, images, board):
        """
        images: an array of grayscale images, all assumed to be the same size
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

        print("Images: {} @ {}".format(len(images), images[0].shape))
        print("{} {}".format(board.type, board.marker_size))
        print('-'*40)
        for cnt, gray in enumerate(images):
            # orig = gray.copy()
            if len(gray.shape) > 2:
                gray = bgr2gray(gray)

            # ret, corners = self.findMarkers(gray)
            ret, corners = board.find(gray)

            # If found, add object points, image points (after refining them)
            if ret:
                imgpoints.append(corners.reshape(-1, 2))

                # get the real-world pattern of points
                objp = board.objectPoints()
                objpoints.append(objp)

                print('[{}] + found {} of {} corners'.format(
                    cnt, corners.size / 2, max_corners))
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

                # Draw the corners
                # tmp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # cv2.drawChessboardCorners(tmp, board.marker_size, corners, True)
                tmp = board.draw(gray, corners)
                self.save_cal_imgs.append(tmp)
            else:
                print(f'[{cnt}] - Could not find markers')

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
        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), K, None)
        print(f"\nRMS error: {rms}\n")
        print('-'*40)

        # print(f"rvecs: {rvecs}    tvecs: {tvecs}")
        # print(f"dist: {dist}")
        # dist = *dist[0]

        data = {
            'date': time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
            'markerType': board.type,
            'markerSize': board.marker_size,
            'imageSize': images[0].shape,
            'cameraMatrix': mtx,
            'distCoeffs': dist, #DistortionCoefficients(dist),
            'rms': rms,
            'rvecs': rvecs,
            'tvecs': tvecs,
            "objpoints": objpoints,
            "imgpoints": imgpoints
        }

        return data
