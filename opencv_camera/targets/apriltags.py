##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import cv2
from opencv_camera.color_space import bgr2gray, gray2bgr
import numpy as np
from ..apriltag.apriltag_marker import ApriltagMarker

tag_sizes = {
    'tag16h5' : 6,
    'tag25h9' : 7,
    'tag36h10': 8,
    'tag36h11': 8,
}


class ApriltagFinder:
    def __init__(self, detector, size, scale):
        """
        size: pattern of chess board, tuple(rows, columns)
        scale: real-world dimension of square side, example, 2 cm (0.02 m)
        """
        self.detector = detector
        self.marker_size = size
        self.marker_scale = scale
        self.type = "Apriltag"

    def find(self, gray, flags=None):
        """
        Given an image, this will return tag corners. The input flags are not
        used.

        return:
            success: (True, [corner points],[object points])
            failure: (False, None, None)
        """
        if len(gray[0].shape) > 2:
            raise Exception(f"Images must be grayscale, not shape: {gray[0].shape}")

        # additionally, I do a binary thresholding which greatly reduces
        # the apriltag's bad corner finding which resulted in non-square
        # tags which gave horrible calibration results.
        ok,gray = cv2.threshold(gray,90,255,cv2.THRESH_BINARY)

        tags = self.detector.detect(
            gray,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=self.marker_scale)

        if len(tags) == 0:
            return False, None, None
        #---
        # get complete listing of objpoints in a target
        opdict = self.objectPoints()

        ob = []
        tt = []
        # for each tag, get corners and obj point corners:
        for tag in tags:
            # add found objpoint to list IF tag id found in image
            obcorners = opdict[tag.tag_id]
            for oc in obcorners:
                ob.append(oc)
            cs = tag.corners
            for c in cs:
                tt.append(c)

        corners = np.array(tt, dtype=np.float32)
        objpts = np.array(ob, dtype=np.float32)
        # print("corners", corners.shape, corners.dtype)
        # print(corners)

        return True, corners, objpts

    def objectPoints(self, ofw=2):
        """
        Returns a set of the target's ideal 3D feature points.

        sz: size of board, ex: (6,9)
        ofw: offset width, ex: 2px
        """
        family = self.detector.params["families"][0]
        pix = tag_sizes[family]
        sz = self.marker_size
        scale = self.marker_scale/8
        ofr = pix+ofw
        ofc = pix+ofw
        r = sz[0]*(ofr)
        c = sz[1]*(ofc)
        b = np.ones((r,c))
        objpts = {}

        for i in range(sz[0]):     # rows
            for j in range(sz[1]): # cols
                r = i*(ofr)+ofw
                c = j*(ofc)+ofw
                x = i*sz[1]+j

                rr = r+pix
                cc = c+pix
                objpts[x] = (
                    (scale*rr,scale*c,0),
                    (scale*rr,scale*cc,0),
                    (scale*r,scale*cc,0),
                    (scale*r,scale*c,0)) # ccw - best
        return objpts

    def draw(self, img, tags):
        """
        Draws corners on an image for viewing/debugging
        """
        if len(img.shape) == 2:
            color = gray2bgr(img) #cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            color = img.copy()

        tm = ApriltagMarker()
        return tm.draw(color, tags)
