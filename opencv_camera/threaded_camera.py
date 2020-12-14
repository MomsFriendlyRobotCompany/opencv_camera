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
# from enum import IntFlag
import attr
from .color_space import ColorSpace

#                                     1   2   3   4
# ColorSpace = IntFlag("ColorSpace", "bgr rgb hsv gray")

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

    camera = attr.ib(default=None)  # opencv camera object
    frame = attr.ib(default=None)   # current frame
    run = attr.ib(default=False)    # thread loop run parameter
    thread_hz = attr.ib(default=30) # thread loop rate
    fmt = attr.ib(default=0)        # colorspact format
    ps = attr.ib(default=None)      # thread process
    # lock = attr.ib(default=Lock())


    def __del__(self):
        self.close()

    def close(self):
        self.run = False
        time.sleep(0.25)
        self.camera.release()

    def __colorspace(self):
        s = "unknown"
        if self.fmt == 1:
            s = "BGR"
        elif self.fmt == 2:
            s = "RGB"
        elif self.fmt == 4:
            s = "HSV"
        elif self.fmt == 8:
            s = "GRAY"

        return s

    def set_resolution(self, resolution):
        rows, cols = resolution
        self.camera.set(3, cols) #cv2.CAP_PROP_FRAME_WIDTH
        self.camera.set(4, rows) #cv2.CAP_PROP_FRAME_HEIGHT

    def get_resolution(self):
        cols = self.camera.get(3) #cv2.CAP_PROP_FRAME_WIDTH
        rows = self.camera.get(4) #cv2.CAP_PROP_FRAME_HEIGHT
        return (rows, cols,)

    def open(self, path=0, resolution=None, fmt=ColorSpace.bgr):
        """
        Opens the camera object and starts the internal loop in a thread
        """

        if fmt not in list(ColorSpace):
            print(f"{Fore.RED}*** Threaded Camera.Open: Unknown color format: {fmt} ***{Fore.RESET}")
            fmt = 1
        self.fmt = fmt

        self.run = True
        self.camera = cv2.VideoCapture(path)

        if resolution:
            self.set_resolution(resolution)

        print("========================")
        print(f"Opened camera: {path}")
        print(f"Resolution: {resolution}")
        print(f"Colorspace: {self.__colorspace()}")
        print("")

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

        rate = Rate(self.thread_hz)

        while self.run:
            ok, img = self.camera.read()

            if ok:
                # self.lock.acquire()

                if self.fmt == ColorSpace.bgr:
                    self.frame = img.copy()
                elif self.fmt == ColorSpace.hsv:
                    self.frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                elif self.fmt == ColorSpace.rgb:
                    self.frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif self.fmt == ColorSpace.gray:
                    self.frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    print(f"{Fore.RED}*** Threaded Camera: Unknown color format: {self.fmt}, reset to BGR ***{Fore.RESET}")
                    self.fmt = 1

            rate.sleep()
