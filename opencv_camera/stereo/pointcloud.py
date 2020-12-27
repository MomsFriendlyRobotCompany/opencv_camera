##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import cv2
import numpy as np
from ..save.pointcloud import PLY


class Stereo2PointCloud:
    def __init__(self, baseline, focalLength, h, w):
        # self.baseline = baseline
        # self.f = focalLength
        # self.w = w
        # self.h = h

        t = baseline
        f = focalLength
        Q = np.float32([[1, 0, 0, -0.5*w],
                        [0, 1, 0, -0.5*h],
                        [0, 0, 0,      f],
                        [0, 0,-1/t,    0]])
        self.Q = Q
        print(Q)

        ch = 3 # channels
        window_size = 5 # SADWindowSize
        min_disp = 16*0
        num_disp = 16*4-min_disp
#         stereo = cv2.StereoBM_create(numDisparities=96, blockSize=5)
#         stereo = cv2.StereoSGBM_create(0,64,21)
        self.matcher = cv2.StereoSGBM_create(
            minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = 16,          # 3-11
            P1 = 8*ch*window_size**2,
            P2 = 32*ch*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,    # 5-15
            speckleWindowSize = 100, # 50-200
            speckleRange = 32         # 1-2
        )

    def to_pc(self, imgL, imgR):
        disp = self.matcher.compute(imgL, imgR).astype(np.float32) / 16.0
#         disp = stereo.compute(imgR, imgL).astype(np.float32) / 16.0 # <<<<<<<<
        # dm = (disp-min_disp)/num_disp

        print(f">> Computed disparity, {disp.shape} pts")
        print(f">> Disparity max: {disp.max()}, min: {disp.min()}")
        return disp #, dm

    def reproject(self, img, disp):
        if len(img.shape) == 2:
            colors = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            colors = img

        # t = self.baseline
        # f = self.f
        # Q = np.float32([[1, 0, 0, -0.5*w],
        #                 [0, 1, 0, -0.5*h],
        #                 [0, 0, 0,      f],
        #                 [0, 0,-1/t,    0]])
        # print(Q)

        points = cv2.reprojectImageTo3D(disp, self.Q)
#         mask = disp > disp.min()  # find pts with min depth
#         mask = disp > 1

        points = points.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        disp = disp.reshape(-1)

        print(colors.shape, points.shape, disp.shape)

        # filter out NaN and inf (disparity == 0)
        mask = (
            (disp > disp.min()) &
            np.all(~np.isnan(points), axis=1) &
            np.all(~np.isinf(points), axis=1)
        )

        out_points = points[mask] # remove pts with no depth
        out_colors = colors[mask] # remove pts with no depth

        # print(f'>> Generated 3d point cloud, {out_points.shape}')
        # print(f">> out_points {out_points.max()}, {out_points.min()}")

        return out_points, out_colors


# FIXME: add pathlib here to handle home folder (~)

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


# class PLY:
#     def write(self, fn, verts, colors):
#         verts = verts.reshape(-1, 3)
#         colors = colors.reshape(-1, 3)
#         verts = np.hstack([verts, colors])
#         with open(fn, 'w') as f:
#             f.write(ply_header % dict(vert_num=len(verts)))
#             np.savetxt(f, verts, '%f %f %f %d %d %d')
