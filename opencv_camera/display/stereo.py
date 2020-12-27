##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import cv2
import numpy as np


def stereoOverlay(imgL, imgR, xoffset, yoffset):
    """
    Overlays the left stereo image on top of the right image. Use the offsets
    to make them match up. Note, it won't be perfect but gives you an idea.

    xoffset: >0 integer offset
    yoffset: positive or negative integer offset
    """
    if yoffset > 0:
        tmp = cv2.addWeighted(imgR[:-yoffset,:-xoffset], 0.5, imgL[yoffset:,xoffset:], 0.5,0.5)
    else:
        yoffset = abs(yoffset)
        tmp = cv2.addWeighted(imgR[yoffset:,:-xoffset], 0.5, imgL[:-yoffset,xoffset:], 0.5,0.5)

    return tmp
    # plt.imshow(tmp, cmap="gray")

def coverage(size, imgpoints):
    """
    Creates a  coverage map of detected points from calibration images. Ideally
    the entire image space should be evenly covered.

    size: image size, (rows, cols)
    imgpoints: the detected 2D image points of the calibration target

    Returns: a numpy image
    """
    y,x = size #imgs[0].shape[:2]
    tgt = 255*np.ones((y,x,3),dtype=np.uint8)

    rad = 5*max(int(y/1000),1)
    c = (0,0,255)
    for f in imgpoints:
        for x in f:
            cv2.circle(tgt, tuple(x.astype(int)),rad,c,thickness=-1)

    return tgt

def tip_sheet(imgs, width=5, scale=None):
    raise DeprecationWarning("Use mosaic instead")
    return mosaic(imgs, width, scale)

def mosaic(imgs, width=5, scale=None):
    """
    Creates a single image (mosaic) with thumb nails of
    the input images. Useful for displaying a batch of calibration
    images.

    imgs: array of grayscale images
    width: number of thumbnail images across the tip sheet
           will be
    scale: scale the resulting image down FIXME: implement
    """
    tip = None
    num = len(imgs)
    r,c = imgs[0].shape[:2]
    wid = width
    rr = int(np.floor(r/wid))
    cc = int(c*rr/r)
    row = None

    for i in range(num):
        im = cv2.resize(imgs[i],(cc,rr,),interpolation=cv2.INTER_NEAREST)

        if i == 0:
            row = im.copy()
        elif i == (num-1):
            blk = np.zeros((row.shape[0], tip.shape[1]))
            blk[:row.shape[0],:row.shape[1]] = row
            row = blk
            tip = np.vstack((tip, row))
        elif i%wid == 0:
            if tip is None:
                tip = row.copy()
            else:
                tip = np.vstack((tip, row))
            row = im.copy()
        else:
            row = np.hstack((row, im))

    return tip
