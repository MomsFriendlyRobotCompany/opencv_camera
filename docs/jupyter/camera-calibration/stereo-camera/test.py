#!/usr/bin/env python3

import cv2
import time

# windows 10 (usb2/3)
# 0: black image
# usb2: color image: 1 left (as you look at it)

# macbook (usb3)
# 0: (720, 2560, 3)
# 1: is built-in camera
# can't find depth camera

# old imac (usb2)
# 0: built-in camera
# 1: nothing ... doesn't come up
camera = cv2.VideoCapture(2)

cnt = 0

while True:
    ok, frame = camera.read()
    if ok:
        print(frame.shape)
        h, w, d = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)
        # frame = cv2.resize(frame, (w//2,h//2))
        # frame = cv2.equalizeHist(frame)
        # cv2.imshow("img", cv2.resize(frame, (w//3,h//3)))
        cv2.imshow("img", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('s'):
            print(frame.shape)
            cv2.imwrite('image-{}.png'.format(cnt), frame)
            cnt += 1
        time.sleep(0.033)
    else:
        print(">> nothing: {}".format(ok))
        time.sleep(1)
camera.release()
