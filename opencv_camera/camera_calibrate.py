##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import numpy as np
import cv2
from glob import glob
import yaml
import json
import attr
from enum import Enum
import time
from collections import namedtuple


DistortionCoefficients = namedtuple("DistortionCoefficients", "k1 k2 p1 p2 k3")
Markers = Enum('Markers', 'checkerboard circle acircle apriltag')

@attr.s(slots=True)
class CameraCalibration:
    '''
    Simple calibration class.
    '''
    marker_type = attr.ib()
    marker_size = attr.ib()
    marker_scale = attr.ib(default=1)
    save_cal_imgs = attr.ib(default=None)
    # save_objpoints = attr.ib(default=None)
    # save_imgpoints = attr.ib(default=None)

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

    def findMarkers(self, gray):
        # Find the chess board corners or circle centers
        if self.marker_type is Markers.checkerboard:
            flags = 0
            flags |= cv2.CALIB_CB_ADAPTIVE_THRESH
            flags |= cv2.CALIB_CB_FAST_CHECK
            flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.marker_size,
                flags=flags)
        elif self.marker_type is Markers.circle:
            flags=0
            ret, corners = cv2.findCirclesGrid(gray, self.marker_size, flags=flags)
        elif self.marker_type is Markers.acircle:
            flags = cv2.CALIB_CB_ASYMMETRIC_GRID
            ret, corners = cv2.findCirclesGrid(gray, self.marker_size, flags=flags)
        else:
            raise Exception("invalid marker type: {}".format(self.marker_type))

        if not ret:
            corners = [] # didn't find any

        return ret, corners

    def calibrate(self, images):
         """
         images: an array of grayscale images, all assumed to be the same size
         marker_scale: how big are your markers in the real world, example:
            checkerboard with sides 2 cm, set marker_scale=0.02 so your T matrix
            comes out in meters
         """
         self.save_cal_imgs = []

         # Arrays to store object points and image points from all the images.
         objpoints = []  # 3d point in real world space
         imgpoints = []  # 2d points in image plane.

         max_corners = self.marker_size[0]*self.marker_size[1]

         print("Images: {} @ {}".format(len(images), images[0].shape))
         print("{} {}".format(self.marker_type, self.marker_size))
         print('-'*40)
         for cnt, gray in enumerate(images):
             # orig = gray.copy()
             if len(gray.shape) > 2:
                 gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

             ret, corners = self.findMarkers(gray)

             # If found, add object points, image points (after refining them)
             if ret:
                 imgpoints.append(corners.reshape(-1, 2))

                 # make a grid of points
                 objp = np.zeros((np.prod(self.marker_size), 3), np.float32)
                 objp[:, :2] = np.indices(self.marker_size).T.reshape(-1, 2)*self.marker_scale
                 objpoints.append(objp)

                 print('[{}] + found {} of {} corners'.format(
                     cnt, corners.size / 2, max_corners))
                 term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
                 cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

                 # Draw the corners
                 tmp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                 cv2.drawChessboardCorners(tmp, self.marker_size, corners, True)

                 # draw the axes
                 # self.drawAxes()
                 self.save_cal_imgs.append(tmp)
             else:
                 print('[{}] - Could not find markers'.format(cnt))

         # images size here is backwards: w,h
         h, w = images[0].shape[:2]
         K = None # FIXME
         rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h), K, None)
         print("RMS error: {}".format(rms))
         print('-'*40)

         dist = dist[0]

         # self.save_objpoints = objpoints
         # self.save_imgpoints = imgpoints

         data = {
             'date': time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
             'markerType': self.marker_type,
             'markerSize': self.marker_size,
             'imageSize': images[0].shape,
             'cameraMatrix': mtx,
             'distCoeffs': DistortionCoefficients(*dist),
             'rms': rms,
             'rvecs': rvecs,
             'tvecs': tvecs,
             "objpoints": objpoints,
             "imgpoints": imgpoints
        }

         return data


@attr.s(slots=True)
class UnDistort:
    dist_coeff = attr.ib()
    camera_matrix = attr.ib()
    newcameramtx = attr.ib(default=None)
    alpha = attr.ib(default=None)

    # use a calibration matrix to undistort an image
    def undistort(self, image, alpha):
        """
        image: an image

        alpha = 0: returns undistored image with minimum unwanted pixels (image
                    pixels at corners/edges could be missing)
        alpha = 1: retains all image pixels but there will be black to make up
                    for warped image correction
        """
        if (self.newcameramtx is None) or (alpha != self.alpha):
            self.alpha = alpha
            rows, cols = image.shape[:2]

            self.newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeff,
                (cols, rows),
                self.alpha)

        return cv2.undistort(
            image,
            self.camera_matrix,
            self.dist_coeff,
            None,
            self.newcameramtx)
