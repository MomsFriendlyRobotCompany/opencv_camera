##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from colorama import Fore
from pathlib import Path
import yaml
import numpy as np
from dataclasses import dataclass
from ..undistort import UnDistort

class UndistortStereo:
    """
    Simple class that hold the information to undistort
    a stereo image pair.
    """
    def __init__(self, K1, d1, K2,d2, h, w, R):
        self.left = UnDistort(K1,d1,h,w)
        self.right = UnDistort(K2,d2,h,w,R)

    def undistort(self, left, right):
        a = self.left.undistort(left)
        b = self.right.undistort(right)
        return a,b


@dataclass
class StereoCamera:
    """
    Nice class for holding stereo camera info and
    reading to or writing from yaml.

    K: camera matrix
    d: distortion coefficients
    R: rotation matrix between left/right camera
    T: translation vector between left/right camera
    F: fundamental matrix
    E: essential matrix

    subcripts: 1-left, 2-right
    """

    K1: np.ndarray # left camera matrix
    d1: np.ndarray # left distortion coefficients
    K2: np.ndarray # right camera matrix
    d2: np.ndarray # right distortion coefficients
    R: np.ndarray  # rotation between left/right cameras
    T: np.ndarray  # translation between left/right cameras
    F: np.ndarray  # fundamental matrix
    E: np.ndarray  # essential matrix
    # height: int # height/rows
    # width: int # width/columns

    def __str__(self):
        ms = lambda m: "    {}".format(str(m).replace('\n','\n    '))
        s = f'{Fore.BLUE}Camera 1 --------------------------\n'
        s += f'  focalLength(x,y): {self.K1[0,0]:.1f} {self.K1[1,1]:.1f} px \n'
        s += f'  principlePoint(x,y): {self.K1[0,2]:.1f} {self.K1[1,2]:.1f} px\n'
        s += f'  distortionCoeffs: {self.d1}\n'

        s += f'{Fore.GREEN}Camera 2 --------------------------\n'
        s += f'  focalLength(x,y): {self.K2[0,0]:.1f} {self.K2[1,1]:.1f} px \n'
        s += f'  principlePoint(x,y): {self.K2[0,2]:.1f} {self.K2[1,2]:.1f} px\n'
        s += f'  distortionCoeffs: {self.d2}\n'

        s += f"{Fore.MAGENTA}"
        s += 'Extrinsic Camera Parameters -------\n'
        s += f"  Translation between Left/Right Camera: {self.T}\n"
        s += f"  Rotation between Left/Right Camera:\n{ms(self.R)}\n"
        s += f"  Essential Matrix:\n{ms(self.E)}\n"
        s += f'  Fundatmental Matrix:\n{ms(self.F)}\n'
        s += f"{Fore.RESET}"
        return s

    @classmethod
    def from_yaml(cls, file):
        if not isinstance(file, Path):
            p = Path(file)
        p = p.expanduser().resolve()

        with p.open(mode="r") as fd:
            info = yaml.safe_load(fd)

        args = []
        for key in ["K1","d1","K2","d2","R","T","F","E"]:
            args.append(np.array(info[key]))

        sc = cls(*args)
        # sc = cls()

        # for key in info.keys():
        #     sc.__dict__[key] = np.array(info[key])

        return sc

    def to_yaml(self, filename):
        camera_parameters = {
            "K1": self.K1.tolist(),
            "d1": self.d1.tolist(),
            "K2": self.K2.tolist(),
            "d2": self.d2.tolist(),
            "T": self.T.tolist(),
            "R": self.R.tolist(),
            "F": self.F.tolist(),
            "E": self.E.tolist(),
            # "width": self.width,
            # "height": self.height,
        }

        p = Path(filename).expanduser().resolve()
        with p.open("w") as fd:
            yaml.safe_dump(camera_parameters, fd)

    def p1(self):
        """Returns projection matrix: P1 = K1*[I|0]"""
        return self.K1 @ np.hstack((np.eye(3), np.zeros((3,1))))

    def p2(self):
        """Returns projection matrix: P2 = K2*[R|t]"""
        return self.K2 @ np.hstack((self.R, self.T.T))

    def getUndistortion(self, h, w):
        # w = self.width
        # h = self.height
        return UndistortStereo(
            self.K1,
            self.d1,
            self.K2,
            self.d2,
            h,w,
            self.R
        )

    # def scale(self, scale):
    #     return None








# class Stereo2PointCloud:
#     def __init__(self, baseline, focalLength, h, w):
#         # self.baseline = baseline
#         # self.f = focalLength
#         # self.w = w
#         # self.h = h
#
#         t = baseline
#         f = focalLength
#         Q = np.float32([[1, 0, 0, -0.5*w],
#                         [0, 1, 0, -0.5*h],
#                         [0, 0, 0,      f],
#                         [0, 0,-1/t,    0]])
#         self.Q = Q
#         print(Q)
#
#         ch = 3 # channels
#         window_size = 5 # SADWindowSize
#         min_disp = 16*0
#         num_disp = 16*4-min_disp
# #         stereo = cv2.StereoBM_create(numDisparities=96, blockSize=5)
# #         stereo = cv2.StereoSGBM_create(0,64,21)
#         self.matcher = cv2.StereoSGBM_create(
#             minDisparity = min_disp,
#             numDisparities = num_disp,
#             blockSize = 16,          # 3-11
#             P1 = 8*ch*window_size**2,
#             P2 = 32*ch*window_size**2,
#             disp12MaxDiff = 1,
#             uniquenessRatio = 10,    # 5-15
#             speckleWindowSize = 100, # 50-200
#             speckleRange = 32         # 1-2
#         )
#
#     def to_pc(self, imgL, imgR):
#         disp = self.matcher.compute(imgL, imgR).astype(np.float32) / 16.0
# #         disp = stereo.compute(imgR, imgL).astype(np.float32) / 16.0 # <<<<<<<<
#         # dm = (disp-min_disp)/num_disp
#
#         print(f">> Computed disparity, {disp.shape} pts")
#         print(f">> Disparity max: {disp.max()}, min: {disp.min()}")
#         return disp #, dm
#
#     def reproject(self, img, disp):
#         if len(img.shape) == 2:
#             colors = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         else:
#             colors = img
#
#         # t = self.baseline
#         # f = self.f
#         # Q = np.float32([[1, 0, 0, -0.5*w],
#         #                 [0, 1, 0, -0.5*h],
#         #                 [0, 0, 0,      f],
#         #                 [0, 0,-1/t,    0]])
#         # print(Q)
#
#         points = cv2.reprojectImageTo3D(disp, self.Q)
# #         mask = disp > disp.min()  # find pts with min depth
# #         mask = disp > 1
#
#         points = points.reshape(-1, 3)
#         colors = colors.reshape(-1, 3)
#         disp = disp.reshape(-1)
#
#         print(colors.shape, points.shape, disp.shape)
#
#         # filter out NaN and inf (disparity == 0)
#         mask = (
#             (disp > disp.min()) &
#             np.all(~np.isnan(points), axis=1) &
#             np.all(~np.isinf(points), axis=1)
#         )
#
#         out_points = points[mask] # remove pts with no depth
#         out_colors = colors[mask] # remove pts with no depth
#
#         # print(f'>> Generated 3d point cloud, {out_points.shape}')
#         # print(f">> out_points {out_points.max()}, {out_points.min()}")
#
#         return out_points, out_colors
