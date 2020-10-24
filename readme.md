# OpenCV Camera

![CheckPackage](https://github.com/MomsFriendlyRobotCompany/opencv_camera/workflows/CheckPackage/badge.svg)
![GitHub](https://img.shields.io/github/license/MomsFriendlyRobotCompany/opencv_camera)
[![Latest Version](https://img.shields.io/pypi/v/opencv_camera.svg)](https://pypi.python.org/pypi/opencv_camera/)
[![image](https://img.shields.io/pypi/pyversions/opencv_camera.svg)](https://pypi.python.org/pypi/opencv_camera)
[![image](https://img.shields.io/pypi/format/opencv_camera.svg)](https://pypi.python.org/pypi/opencv_camera)

Simple threaded camera and calibration code.

## Install

The preferred way to install is using `pip`:

```
pip install opencv_camera
```

## Apps

Use `program --help` to display switches for each of the following:

- `opencv_calibrate`: calibrate a camera
- `opencv_capture`: simple tool to capture and save images
- `opencv_mjpeg`: sets up a simple jmpeg server so you can view images in a web browser
- `udp_server x.x.x.x`: sends camera images via UDP
- `udp_client x.x.x.x`: displays UDP camera images from server

# Change Log

| Data       | Version| Notes                                     |
|------------|--------|-------------------------------------------|
| 2018-07-19 | 0.10.6 |  added UDP image server and client |
| 2018-07-19 | 0.10.0 |  renamed and focused on camera |
| 2018-07-19 |  0.9.4 |  simple clean-up and updating some things |
| 2017-10-29 |  0.9.3 |  bug fixes |
| 2017-04-09 |  0.9.0 |  initial python 3 support |
| 2017-03-31 |  0.7.0 |  refactored and got rid of things I don't need |
| 2017-01-29 |  0.6.0 |  added video capture (video and images) program |
| 2016-12-30 |  0.5.3 |  typo fix |
| 2016-12-30 |  0.5.1 |  refactored |
| 2016-12-11 |  0.5.0 |  published to PyPi |
| 2014-3-11  |  0.2.0 |  started |

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
