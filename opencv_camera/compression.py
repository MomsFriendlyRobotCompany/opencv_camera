##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import numpy as np
import cv2


class Compressor:
    """
    Compressor allow you to serialize and compress an image using either JPEG
    or PNG compression.
    """

    _format = ".png"

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, fmt):
        """
        Set format to either .jpg jpg .png png
        """
        if fmt not in [".jpg", "jpg", ".png", "png"]:
            raise ValueError(f"Invalid format: {fmt}")
        if fmt.find(".") != 0:
            fmt = "." + fmt
        self._format = fmt

    def compress(self, img):
        ok, cb = cv2.imencode(self._format, img)
        if ok:
            cb = cb.tobytes()
        else:
            cb = None
        return cb

    def uncompress(self, img_bytes, shape):
        img = np.frombuffer(img_bytes, dtype=np.uint8)

        if len(shape) == 3:
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        else:
            img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

        img = img.reshape(shape)
        return img
