# OpenCV Camera

![CheckPackage](https://github.com/MomsFriendlyRobotCompany/opencv_camera/workflows/CheckPackage/badge.svg)
![GitHub](https://img.shields.io/github/license/MomsFriendlyRobotCompany/opencv_camera)
[![Latest Version](https://img.shields.io/pypi/v/opencv_camera.svg)](https://pypi.python.org/pypi/opencv_camera/)
[![image](https://img.shields.io/pypi/pyversions/opencv_camera.svg)](https://pypi.python.org/pypi/opencv_camera)
[![image](https://img.shields.io/pypi/format/opencv_camera.svg)](https://pypi.python.org/pypi/opencv_camera)
![PyPI - Downloads](https://img.shields.io/pypi/dm/opencv_camera?color=aqua)

Simple threaded camera and calibration code using OpenCV. This tries to simplify some things

## Install

The preferred way to install is using `pip`:

```
pip install opencv_camera
```

## Usage

See the jupyter notebooks under the `docs` in the repository for some examples.

Online [nbviewer](nbviewer.org/github/MomsFriendlyRobotCompany/opencv_camera/tree/master/docs/jupyter/)

### Colorspace

Change between common colorspaces with:

- `bgr2gray(image)`
- `gray2bgr(image)`
- `bgr2rgb(image)`
- `rgb2bgr(image)`
- `bgr2hsv(image)`
- `hsv2bgr(image)`

### Calibration

Create a mosaic of input calibration images with `mosaic(images, width)`

![](https://github.com/MomsFriendlyRobotCompany/opencv_camera/blob/master/pics/mosaic.png?raw=true)

Calibrate a camera with:

```python
calibrator = CameraCalibration()
board = ChessboardFinder((9,6), 1)
cam, cal = calibrator.calibrate(images, board)
```

Display all of the found image points with `coverage((width, height), imagePoints)`

![](https://github.com/MomsFriendlyRobotCompany/opencv_camera/blob/master/pics/target-points.png?raw=true)

### Distortion

Use the found calibration parameters to undistort an image:

```python
un = UnDistort(cameraMatrix, distortionCoeff, w, h)
corr_img = un.undistort(image)
```

Visualize the lens distortion with:

```python
visualizeDistortion(cameraMatrix, distortCoeff, height, width)
```

![](https://github.com/MomsFriendlyRobotCompany/opencv_camera/blob/master/pics/py-dist.png?raw=true)

### Stereo

Calibrate a stereo camera with:

```python
stereoCal = StereoCalibration()
board = ChessboardFinder((9,6), 1)
ok, cm, sc = stereoCal.calibrate(imgL, imgR, board)
```

Draw epipolar lines in stereo images with `drawEpipolarLines(imgpointsL,imgpointsR,imgL,imgR)`

![](https://github.com/MomsFriendlyRobotCompany/opencv_camera/blob/master/pics/epipolar.png?raw=true)

## Apps

Use `program --help` to display switches for each of the following:

- `opencv_calibrate`: calibrate a camera
- `opencv_capture`: simple tool to capture and save images
- `opencv_mjpeg`: sets up a simple jmpeg server so you can view images in a web browser
- `udp_server x.x.x.x`: sends camera images via UDP
- `udp_client x.x.x.x`: displays UDP camera images from server

# ToDo

- [ ] Add in `apriltag` calibration
- [ ] Add pointcloud from stereo
- [x] Add parameters for known cameras
- [x] Add Jupyter notebook documentation and examples
- [x] Simplify stereo camera
- [x] Add `computeReprojectionErrors` and `visualizeReprojErrors`
- [x] Add `visualizeDistortion`
- [ ] Add `visualizeExtrinsics`
- [x] Remove `attrs`

# Change Log

| Data       | Version    | Notes                                     |
|------------|------------|-------------------------------------------|
|            | 2022.12.23 | changed version number to match date, changed to `opencv-contrib-python` |
| 2020-12-27 | 0.10.10 | added distortion and reprojection display |
| 2020-09-15 | 0.10.8 | added known camera params and general cleanup |
| 2020-08-24 | 0.10.6 | added UDP image server and client |
| 2020-07-03 | 0.10.2 | renamed and focused on camera |
| 2018-07-19 |  0.9.4 | simple clean-up and updating some things |
| 2017-10-29 |  0.9.3 | bug fixes |
| 2017-04-09 |  0.9.0 | initial python 3 support |
| 2017-03-31 |  0.7.0 | refactored and got rid of things I don't need |
| 2017-01-29 |  0.6.0 | added video capture (video and images) program |
| 2016-12-30 |  0.5.3 | typo fix |
| 2016-12-30 |  0.5.1 | refactored |
| 2016-12-11 |  0.5.0 | published to PyPi |
| 2014-3-11  |  0.2.0 | started |

# MIT License

**Copyright (c) 2014 Kevin J. Walchko**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
