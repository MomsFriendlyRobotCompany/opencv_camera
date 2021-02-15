#!/usr/bin/env python3
import numpy as np
import cv2
from enum import Enum

Format = Enum('Format', 'png jpg tiff')

class ImageGrab(object):
    def __init__(self, cam=0):
        self.camera = cv2.VideoCapture(cam)
        self.format = Format.png

    def __del__(self):
        self.camera.release()

    def run(self):
        cnt = 0
        print('='*50)
        print(' [s]ave the image to file')
        print(' [q]uit the program')
        print(' ')
        while True:
            ok, img = self.camera.read()
            if ok:
                h,w = img.shape[:2]
                img = cv2.resize(img,(w//2,h//2))
                cv2.imshow('{}'.format(img.shape), img)
                key = cv2.waitKey(33)

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if self.format == Format.png:
                        fname = 'image-{}.png'.format(cnt)
                    elif self.format == Format.jpg:
                        fname = 'image-{}.jpg'.format(cnt)
                    elif self.format == Format.tiff:
                        fname = 'image-{}.tiff'.format(cnt)
                    else:
                        print("Invalid format:", self.format)
                        exit(1)

                    cnt += 1

                    cv2.imwrite(fname, img)

if __name__ == "__main__":
    ig = ImageGrab(0)
    ig.run()

    print
