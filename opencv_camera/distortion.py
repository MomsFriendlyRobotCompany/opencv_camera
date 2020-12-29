##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from numpy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt


def visualizeDistortion(K, D, h, w, fontsize=16, contourLevels=10, nstep=20):
    """
    http://amroamroamro.github.io/mexopencv/opencv/calibration_demo.html
    https://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html#details
    """
    # K = [fx 0 cx; 0 fy cy; 0 0 1]
    #
    # * focal lengths   : fx, fy
    # * aspect ratio    : a = fy/fx
    # * principal point : cx, cy
    M = K
    # D = [k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,taux,tauy]
    #
    # * radial distortion     : k1, k2, k3
    # * tangential distortion : p1, p2
    # * rational distortion   : k4, k5, k6
    # * thin prism distortion : s1, s2, s3, s4
    # * tilted distortion     : taux, tauy (ignored here)
    #
    D = D.ravel()
    d = np.zeros(14)
    d[:D.size] = D
    D = d
    k1 = D[0]
    k2 = D[1]
    p1 = D[2]
    p2 = D[3]
    k3 = D[4]

    # nstep = 20
    u,v = np.meshgrid(
        np.arange(0, w, nstep),
        np.arange(0, h, nstep)
    )

    b = np.array([
        u.ravel(),
        v.ravel(),
        np.ones(u.size)
    ])

    xyz = lstsq(M, b,rcond=None)[0]

    xp = xyz[0,:]/xyz[2,:]
    yp = xyz[1,:]/xyz[2,:]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3

    coef = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + D[5]*r2 + D[6]*r4 + D[7]*r6)
    xpp = xp*coef + 2*p1*(xp*yp) + p2*(r2+2*xp**2) + D[8]*r2 + D[9]*r4
    ypp = yp*coef + p1*(r2+2*yp**2) + 2*p2*(xp*yp) + D[11]*r2 + D[11]*r4
    u2 = M[0,0]*xpp + M[0,2]
    v2 = M[1,1]*ypp + M[1,2]
    du = u2.ravel() - u.ravel()
    dv = v2.ravel() - v.ravel()
    dr = np.hypot(du,dv).reshape(u.shape)

    fig, ax = plt.subplots()
    ax.quiver(u.ravel(), v.ravel(), du, -dv, color="dodgerblue")
    ax.plot(w//2, h//2, "x", M[0,2], M[1,2],"^", markersize=fontsize)
    CS = ax.contour(u, v, dr, colors="black", levels=contourLevels)
    ax.set_aspect('equal', 'box')
    ax.clabel(CS, inline=1, fontsize=fontsize,fmt='%0.0f')
    ax.set_title("Distortion Model", fontsize=fontsize)
    ax.set_xlabel("u", fontsize=fontsize)
    ax.set_ylabel("v", fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylim(max(v.ravel()),0)
