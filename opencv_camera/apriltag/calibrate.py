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
import time # save date in results
from ..color_space import bgr2gray, gray2bgr
from ..mono.calibrate import CameraCalibration


class ApriltagStereoCalibration:
    save_cal_imgs = None

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

        data = cc.calibrate(imgs_l, board)
        K1 = data["K"]
        d1 = data["d"]
        rvecs1 = data["rvecs"]
        tvecs1 = data["tvecs"]
        objpts = data["objpoints"]
        imgptsL = data["imgpoints"]
        idsL = data["ids"]

        data = cc.calibrate(imgs_r, board)
        K2 = data["K"]
        d2 = data["d"]
        rvecs2 = data["rvecs"]
        tvecs2 = data["tvecs"]
        imgptsR = data["imgpoints"]
        idsR = data["ids"]

        # self.save_cal_imgs = cc.save_cal_imgs
        objpoints = []
        imgpoints_r = []
        imgpoints_l = []

        # trying to be efficient in the serach instead of using
        # find() on the list
        print("Removing markers not seen in both frames:")
        totalMarkers = 0
        for img_num, (ob, idL, ipL, idR, ipR) in enumerate(zip(objpts, idsL, imgptsL, idsR, imgptsR)):
            reject = 0
            leftInfo = dict(zip(idL,tuple(zip(ipL, ob))))
            rightInfo = dict(zip(idR, ipR))
            top = []
            timl = []
            timr = []

            # for each id found in left camera, see if the marker id
            # was found in the right camera. If yes, save, if no,
            # reject the marker
            for id,(ipt,opt) in leftInfo.items():
                try:
                    iptr = rightInfo[id]
                    top.append(opt)
                    timl.append(ipt)
                    timr.append(iptr)
                except KeyError:
                    # id not found in right camera
                    reject += 1
                    continue
            objpoints.append(np.array(top))
            imgpoints_l.append(np.array(timl))
            imgpoints_r.append(np.array(timr))
            if reject > 0:
                total = max(len(idL), len(idR))
                print(f"  Image {img_num}: rejected {reject} tags of {total} tags")

            totalMarkers += len(top)

        print(f"Total markers found in BOTH cameras: {totalMarkers}")

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

        stereocalib_criteria = (
            cv2.TERM_CRITERIA_MAX_ITER +
            cv2.TERM_CRITERIA_EPS,
            100,
            1e-5)

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
            criteria=stereocalib_criteria,
            flags=flags)

        if ret is False:
            return ret, None

        camera_model = {
            'date': time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()),
            'markerType': board.type,
            'markerSize': board.marker_size,
            'height': imgs_l[0].shape[0],
            'width': imgs_l[0].shape[1],
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
