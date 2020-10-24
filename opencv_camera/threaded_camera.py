##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from colorama import Fore
import io
from threading import Thread #, Lock
import time
from slurm.rate import Rate
import numpy as np
import cv2
from enum import IntFlag
import attr

#                                   1   2   3   4
ColorSpace = IntFlag("ColorSpace", "bgr rgb hsv gray")

@attr.s(slots=True)
class ThreadedCamera:
    """
    https://www.raspberrypi.org/documentation/hardware/camera/
    Raspberry Pi v2:
        resolution: 3280 x 2464 pixels
        sensor area: 3.68 mm x 2.76 mm
        pixel size: 1.12 um x 1.12 um
        video modes:1080p30, 720p60 and 640x480p60/90
        optical size: 1/4"
        driver: V4L2 driver

    c = ThreadedCamera()
    c.open(0, (640,480), 2) # starts internal loop, camera 0, RGB format
    frame = c.read()        # numpy array
    c.close()               # stops internal loop and gathers back up the thread
    """

    camera = attr.ib(default=None)
    frame = attr.ib(default=None)
    run = attr.ib(default=False)
    thread_hz = attr.ib(default=30)
    fmt = attr.ib(default=0)
    ps = attr.ib(default=None)
    # lock = attr.ib(default=Lock())


    def __del__(self):
        # self.run = False
        self.close()

    def close(self):
        self.run = False
        time.sleep(0.25)
        self.camera.release()
        self.join(0.1)
        # print('exiting CameraCV ... bye!')

    def open(self, path=0, resolution=None, fmt=ColorSpace.bgr):
        """Starts the internal loop in a thread"""
        if resolution is None:
            resolution=(480,640,)

        if fmt not in list(ColorSpace):
            raise Exception(f"Unknown color format: {fmt}")
        self.fmt = fmt

        self.run = True
        self.ps = Thread(target=self.thread_func, args=(path, resolution))
        self.ps.daemon = True
        self.ps.start()
        return self

    def read(self):
        """Returns image frame"""
        if self.frame is None:
            return False, None

        return True, self.frame

    def thread_func(self, path, resolution):
        """Internal function, do not call"""

        self.camera = cv2.VideoCapture(path)
        rate = Rate(self.thread_hz)

        if isinstance(path, int):
            rows, cols = resolution
            self.camera.set(3, cols) #cv2.CAP_PROP_FRAME_WIDTH
            self.camera.set(4, rows) #cv2.CAP_PROP_FRAME_HEIGHT

        while self.run:
            ok, img = self.camera.read()

            if ok:
                # self.lock.acquire()

                if self.fmt == ColorSpace.bgr:
                    self.frame = img.copy()
                elif self.fmt == ColorSpace.rgb:
                    self.frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                elif self.fmt == ColorSpace.rgb:
                    self.frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif self.fmt == ColorSpace.gray:
                    self.frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    raise Exception(f"Invalide colorspace: {self.fmt}")

                # self.lock.release()
            rate.sleep()

        self.close()

    def join(self, timeout=1.0):
        """
        Attempts to join() the process with the given timeout. If that fails, it calls
        terminate().
        timeout: how long to wait for join() in seconds.
        """
        if self.ps:
            self.ps.join(timeout)
        self.ps = None
