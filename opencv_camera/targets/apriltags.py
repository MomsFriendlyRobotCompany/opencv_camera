##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from opencv_camera.color_space import bgr2gray, gray2bgr


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
            success: (True, [corner points],)
            failure: (False, [],)
        """
        tags = []
        num = 0
        if len(images[0].shape) > 2:
            raise Exception(f"Images must be grayscale, not shape: {images[0].shape}")

        for img in gray:
            t = self.detector.detect(
                img,
                estimate_tag_pose=False,
                camera_params=None,
                tag_size=self.marker_scale)

            tags.append(t)
            # num += len(t)
            # list of tags found by detector for each image
        img_ids = [[t.tag_id for t in f] for f in tags]

        # list of searchable tag coordinates found by detector for each image
        stags = [{t.tag_id: t.corners for t in tag} for tag in tags]

        # points found in image from detector
        imgpoints = []

        # point locations on an ideal target array
        # objpoints = []

        for stag,ids,im in zip(stags, img_ids):
            op = [] # objpoints
            # ip = [] # imgpoints

            # putting the ids in order
            ids.sort()
            #s=0.0235/8 # fixme
            s=tag_size/8 # fixme
            for id in ids:
                for x in objpts[id]:
                    op.append((s*x[0],s*x[1], 0,))

                # for x in stag[id]:
                #     ip.append(x)

            #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            #ip = cv2.cornerSubPix(im, ip, (11, 11), (-1, -1), criteria)
            #img = cv2.drawChessboardCorners(im, (width, height), corners2, True)

            x = np.array(ip, dtype=np.float32)
            imgpoints.append(x)

            # x = np.array(op, dtype=np.float32)
            # objpoints.append(x)

        ret = False
        if len(imgpoints) > 0:
            ret = True

        return ret, imgpoints

    def objectPoints(self, sz, ofw=2):
        """
        Returns a set of the target's ideal 3D feature points.

        sz: ?
        ofw: ?
        """
        ofr = 8+ofw
        ofc = 8+ofw
        r = sz[0]*(ofr)
        c = sz[1]*(ofc)
        b = np.ones((r,c))
        objpts = {}

        for i in range(sz[0]):     # rows
            for j in range(sz[1]): # cols
                r = i*(ofr)+ofw
                c = j*(ofc)+ofw
                x = i*sz[1]+j

                rr = r+8
                cc = c+8
                objpts[x] = ((rr,c),(rr,cc),(r,cc),(r,c)) # ccw - best
        return objpts

    def draw(self, img, tags):
        """
        Draws corners on an image for viewing/debugging
        """:
        if len(img.shape) == 2:
            color = gray2bgr(img) #cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            color = img.copy()

        for t in tags:
                color = self.draw_tag(color, t.corners)

        return color

    def draw_tag(color_img, corners, tag_id=None):
        """
        color_img: image to draw on, must be color
        corners: corner points from apriltag detector, v[0] is the
                 lower left of the tag and the point move CCW.
        tag_id: display the tag's number ID, must be a valid integer number
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

        if tag_id is not None:
            offset = int((v[1][0]-v[0][0])/4)
            cv2.putText(color_img, str(tag_id),
                        org=(v[0][0]+offset,v[0][1]-offset,),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        thickness=4,
                        color=(255, 0, 255))

        return color_img
