##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from opencv_camera.color_space import bgr2gray, gray2bgr
import numpy as np
import cv2


class ApriltagMarker:
    def draw(self, img, tags, id=False):
        if len(img.shape) == 2:
            color_img = gray2bgr(img)
        else:
            color_img = img.copy()

        if not isinstance(tags, list):
            tags = [tags]

        for tag in tags:
            num = tag.tag_id if id else None
            color_img = self.draw_tag(color_img, tag.corners, num)

        return color_img

    def draw_tag(self, color_img, corners, tag_id=None):
        """
        color_img: image to draw on, must be color
        corners: corner points from apriltag detector, v[0] is the
                 lower left of the tag and the point move CCW.
        """
        pts = corners.reshape((-1,1,2)).astype('int32')
        cv2.polylines(color_img,[pts],True,(0,255,0),thickness=4)

        # r = 15
        y = color_img.shape[0]
        r = max(int(y/200),1)
        c = (255,0,0)
        oc = (0,0,255)
        v = corners.astype('int32')
        cv2.circle(color_img, tuple(v[0]),r,oc,thickness=-1)
        cv2.circle(color_img, tuple(v[1]),r,c,thickness=-1)
        cv2.circle(color_img, tuple(v[2]),r,c,thickness=-1)
        cv2.circle(color_img, tuple(v[3]),r,c,thickness=-1)

        if tag_id:
            offset = int((v[1][0]-v[0][0])/4)
            cv2.putText(color_img, str(tag_id),
                        org=(v[0][0]+offset,v[0][1]-offset,),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        thickness=4,
                        color=(255, 0, 255))

        return color_img
