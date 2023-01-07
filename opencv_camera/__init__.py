##############################################
# The MIT License (MIT)
# Copyright (c) 2014 Kevin Walchko
# see LICENSE for full details
##############################################
# -*- coding: utf-8 -*
from .targets.chessboard import ChessboardFinder
# from .targets.apriltags import ApriltagTargetFinder

from .mono.camera import Camera
from .mono.calibrate import CameraCalibration

from .stereo.camera import StereoCamera, UndistortStereo
from .stereo.calibrate import StereoCalibration
from .stereo.fundamental_matrix import findFundamentalMat

from .undistort import UnDistort
from .distortion import visualizeDistortion

# from .apriltag.apriltag_detections import visualizeTargetDetections
# from .apriltag.apriltag_marker import ApriltagMarker

from .reproject import computeReprojectionErrors
from .reproject import visualizeReprojErrors

from .display.lines import drawHorizontalLines
from .display.lines import drawEpipolarLines
from .display.stereo import mosaic
from .display.stereo import coverage
from .display.stereo import stereoOverlay

from .threaded_camera import ThreadedCamera

from .save.video import SaveVideo

from .color_space import ColorSpace
from .color_space import bgr2gray, gray2bgr
from .color_space import rgb2gray, gray2rgb
from .color_space import hsv2bgr, bgr2hsv
from .color_space import bgr2rgb, rgb2bgr

from .blurry import isBlurry

from .compression import Compressor

from importlib.metadata import version # type: ignore

__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2014 Kevin Walchko'
__author__ = 'Kevin J. Walchko'
__version__ = version("opencv_camera")
