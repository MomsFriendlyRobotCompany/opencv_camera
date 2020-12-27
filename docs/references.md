

# References

- [OpenCV Camera Calibration docs](https://docs.opencv.org/master/d9/d0c/group__calib3d.html)
- [mexopencv](http://amroamroamro.github.io/mexopencv/)
- [calibration demo](http://amroamroamro.github.io/mexopencv/opencv/calibration_demo.html)
- [Opencv camera cal](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d)
    - **cv2.CALIB_TILTED_MODEL:** Coefficients tauX and tauY are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the tilted sensor model and return 14 coefficients. If the flag is not set, the function computes and returns only 5 distortion coefficients.
    - **cv2.CALIB_THIN_PRISM_MODEL:** Coefficients s1, s2, s3 and s4 are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the thin prism model and return 12 coefficients. If the flag is not set, the function computes and returns only 5 distortion coefficients.
    - **cv2.CALIB_RATIONAL_MODEL:** Coefficients k4, k5, and k6 are enabled. To provide the backward compatibility, this extra flag should be explicitly specified to make the calibration function use the rational model and return 8 coefficients. If the flag is not set, the function computes and returns only 5 distortion coefficients.
