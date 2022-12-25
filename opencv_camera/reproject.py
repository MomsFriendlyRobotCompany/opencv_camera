##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import cv2
import numpy as np
import matplotlib.pyplot as plt


def computeReprojectionErrors(imgpoints, objpoints, rvecs, tvecs, K, D):
    """
    Uses the camera matrix (K) and the distortion coefficients to reproject the
    object points back into 3D camera space and then calculate the error between
    them and the image points that were found.

    Reference: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

    imgpoints: features found in image, (num_imgs, 2)
    objpoints: calibration known features in 3d, (num_imgs, 3)
    rvecs: rotations
    tvecs: translations
    K: camera matrix
    D: distortion coefficients [k1,k2,p1,p2,k3]

    returns:
        rms
        rms_per_view
        errors
    """
    imgpoints = [c.reshape(-1,2) for c in imgpoints]
    mean_error = None
    error_x = []
    error_y = []
    rms = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        imgpoints2 = imgpoints2.reshape(-1,2)

        # if not all markers were found, then the norm below will fail
        if len(imgpoints[i]) != len(imgpoints2):
            continue

        error_x.append(list(imgpoints2[:,0] - imgpoints[i][:,0]))
        error_y.append(list(imgpoints2[:,1] - imgpoints[i][:,1]))

        rr = np.sum((imgpoints2 - imgpoints[i])**2, axis=1)
        if mean_error is None:
            mean_error = rr
        else:
            mean_error = np.hstack((mean_error, rr))
        rr = np.sqrt(np.mean(rr))
        rms.append(rr)

    m_error = np.sqrt(np.mean(mean_error))
    return m_error, rms, [error_x, error_y]



def visualizeReprojErrors(totalRSME, rmsPerView, reprojErrs, fontSize=16,legend=False,xlim=None,ylim=None):
    fig, ax = plt.subplots()
    for i,(rms,x,y) in enumerate(zip(rmsPerView,reprojErrs[0],reprojErrs[1])):
        ax.scatter(x,y,label=f"RMSE [{i}]: {rms:0.3f}")

    # change dimensions if legend displayed
    if legend:
        ax.legend(fontsize=fontSize)
        ax.axis('equal')
    else:
        ax.set_aspect('equal', 'box')
    ax.set_title(f"Reprojection Error (in pixels), RMSE: {totalRSME:0.4f}", fontsize=fontSize)
    ax.set_xlabel("X", fontsize=fontSize)
    ax.set_ylabel("Y", fontsize=fontSize)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', labelsize=fontSize)
    ax.grid(True)
    # return fig
