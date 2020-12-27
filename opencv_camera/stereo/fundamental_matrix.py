##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
# import cv2
# from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import inv, norm

def findFundamentalMat(K1, K2, R, t, normalize=True):
    """
    OpenCV appears to normalize the F matrix. It can contain
    some really large and really small values. See CSE486
    lecture 19 notes [1] for examples of the F matrix.
    Also see Epipolar Geometery, Table 9.1, last line (Cameras
    not at inf)[2] for how to calculate it.

    K1/K2: camera matrix for left/right camera
    R/T: the rotation and translation from cv2.stereoCalibrate

    [1] http://www.cse.psu.edu/~rtc12/CSE486/lecture19.pdf
    [2] http://www.robots.ox.ac.uk/~vgg/hzbook/hzbook2/HZepipolar.pdf
    """
    t = t.T[0]
    A = (K1 @ R.T @ t)
    C = np.array([
        [    0, -A[2], A[1]],
        [ A[2],     0,-A[0]],
        [-A[1],  A[0],    0]
    ])
    ret = inv(K2).T @ R @ K1.T @ C
    if normalize:
        ret = ret/norm(ret)
    return ret

#
# def plotEpilines(pts1, pts2, img1, img2):
#     # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
#     pts1 = np.int32(pts1)
#     pts2 = np.int32(pts2)
#     F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
#
#     # We select only inlier points
#     pts1 = pts1[mask.ravel()==1]
#     pts2 = pts2[mask.ravel()==1]
#
#     def drawlines(img1,img2,lines,pts1,pts2):
#         ''' img1 - image on which we draw the epilines for the points in img2
#             lines - corresponding epilines '''
#         r,c = img1.shape
#         img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#         img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
#         for r,pt1,pt2 in zip(lines,pts1,pts2):
#             color = tuple(np.random.randint(0,255,3).tolist())
#             x0,y0 = map(int, [0, -r[2]/r[1] ])
#             x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#             img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
#             img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
#             img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
#         return img1,img2
#
#     # Find epilines corresponding to points in right image (second image) and
#     # drawing its lines on left image
#     lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
#     lines1 = lines1.reshape(-1,3)
#     img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
#
#     # Find epilines corresponding to points in left image (first image) and
#     # drawing its lines on right image
#     lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
#     lines2 = lines2.reshape(-1,3)
#     img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
#
#     # plt.subplot(121),plt.imshow(img5)
#     # plt.subplot(122),plt.imshow(img3)
#     # plt.show()
#     return np.hstack((img5, img3))
