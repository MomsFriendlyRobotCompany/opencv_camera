##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
import cv2

def isBlurry(image, threshold=100.0):
    """
    Given an image and threshold, returns if image is blurry and its value

    Args:
      image: opencv grayscale image
      threshold: blur threshold value, below this value an image is considered
                 to be blurry
    Return:
      blurry: True/False
      value: numeric value of blurriness

    Reference: https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    val = cv2.Laplacian(image, cv2.CV_64F).var()
    val = int(val)
    if val < threshold:
        return True, val
    return False, val